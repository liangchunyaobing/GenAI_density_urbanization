from share import *

import os
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
# from satellite_dataset import MyDataset
from satellite_tiles_density_ten import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.callbacks import ModelCheckpoint

import os
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# dir_my = '/home/yuebing/gud/Urban_Data/'# '../urban_data/'
# hint_dir = './urban_data/tencities' # '/home/yuebing/gud/Urban_Data/Vector_Image_Partition/'
# image_dir = './urban_data/tencities' # '/home/yuebing/gud/Urban_Data/Vector_Image_Partition/'
# desc_dir = './urban_data/tencities' # '/home/yuebing/gud/Urban_Data/Descriptions_s4/'
data_dir = './urban_data/tencities/train.json'

#hint_dir = "/home/gridsan/qwang/satellite_tiles_control/skeleton/16/"
#hint_dir = dir_my + 'Constraint_Images/'


ck_output_file = './ckpts_s/checkpoints_density'

# Configs
resume_path = './models/control_sd15_ini.ckpt'
# resume_path = ck_output_file + '/epoch=7-step=30047.ckpt'
# resume_path = './lightning_logs/version_24275255/checkpoints/epoch=4-step=112594.ckpt'
# resume_path = './lightning_logs/version_24305430/checkpoints/epoch=4-step=112594.ckpt'


batch_size = 8
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


def main():
    # Macbook
    # 检查 MPS 是否可用
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # torch.cuda.set_device(device)
    # print(f"Using device: {device}")
    
    
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    # model = create_model('./models/cldm_v15.yaml').to(device)
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    # Macbook
    # model.to(device)

    # 定义检查点回调函数
    checkpoint_callback = ModelCheckpoint(
        dirpath=ck_output_file,  # 保存检查点的目录
        filename='{epoch}-{step}',  # 文件名格式
        save_top_k=-1,  # 保存所有检查点
        every_n_epochs=2,  # 每隔几代保存一次
        save_last=False
    )    
    os.makedirs(ck_output_file, exist_ok=True)
    
    # Misc
    dataset = MyDataset(data_dir)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    #trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
    #trainer = pl.Trainer(accelerator='gpu', devices=1, precision=32, callbacks=[logger])
    # trainer = pl.Trainer(accelerator='gpu', devices=[2], precision=32, callbacks=[logger, checkpoint_callback])
    trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=[2,3,4,5, 6,7,8,9], precision=32, callbacks=[logger,checkpoint_callback])

    # Macbook
    #trainer = pl.Trainer(accelerator=device.type, devices=1, precision=32, callbacks=[logger])
    #trainer = pl.Trainer(devices=1, precision=32, callbacks=[logger])
    
    # Train!
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    main()
