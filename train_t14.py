from share import *

import os
import signal
import sys
import requests
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
# from satellite_dataset import MyDataset
from satellite_tiles_t14 import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.callbacks import ModelCheckpoint

image_dir = './urban_data/Vector_Image_Partition/'
hint_dir = './urban_data/Vector_Image_Partition/'
data_dir = []
#for rt in [0,90,180,270]:
for r in [0,1,2]:
    for d in [0,1,2]:
        data_dir.append(f'./urban_data/Descriptions_s3/NewYork_gc_grid_r{r}_d{d}.csv')

gpu_no = 4
ck_output_file = './ckpts/checkpoints_t14'

# Configs
resume_path = './models/control_sd15_ini.ckpt'
url='https://huggingface.co/Boese0601/MagicDance/resolve/main/control_sd15_ini.ckpt?download=true'
if not os.path.exists(resume_path):
    print(f"Checkpoint not found at {resume_path}. Downloading from {url}...")
    os.makedirs(os.path.dirname(resume_path), exist_ok=True)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(resume_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded checkpoint and saved to {resume_path}.")
    else:
        raise Exception(f"Failed to download checkpoint from {url}. Status code: {response.status_code}")
else:
    print(f"Checkpoint already exists at {resume_path}.")


batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


def main():
    global model, trainer, checkpoint_callback

    device = torch.device(f"cuda:{gpu_no}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Using device: {device}")
    
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').to(device)
    model.load_state_dict(load_state_dict(resume_path, location='cpu')) #,strict=False
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.to(device)

    # CHeckpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath = ck_output_file,
        filename = '{epoch}-{step}',
        save_top_k = -1,
        every_n_epochs = 5,
        save_last = False
    )    
    os.makedirs(ck_output_file, exist_ok=True)
    
    # Misc
    dataset = MyDataset(image_dir, data_dir, hint_dir)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)

    trainer = pl.Trainer(accelerator='gpu', devices=[gpu_no], precision=32, callbacks=[logger, checkpoint_callback])
    # Macbook
    #trainer = pl.Trainer(accelerator=device.type, devices=1, precision=32, callbacks=[logger, checkpoint_callback])
    #trainer = pl.Trainer(devices=1, precision=32, callbacks=[logger])


    # Train!
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    main()