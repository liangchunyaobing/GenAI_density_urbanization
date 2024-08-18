from share import *

import os
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
# from satellite_dataset import MyDataset
from satellite_tiles_s2_chn import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.callbacks import ModelCheckpoint

# image_dir = "/home/gridsan/qwang/satellite_images/zoom17/"
# data_dir = "/home/gridsan/qwang/JTL-transit_shared/deep_hybrid_model/data/"

#image_dir = "/home/gridsan/qwang/satellite_tiles_control/satellite_tiles/16/"
#dir_hemy = '/Users/hemingyi/Research/202405_MIT-UF-NEU_/GenerativeUrbanDesign/'
dir_hemy = '/home/yuebing/Mingyi/GenerativeUrbanDesign/Urban_Data/'
#image_dir = dir_hemy + 'Satellite_Images/'
image_dir = dir_hemy + 'Vector_Image_Partition/'

data_dir = [dir_hemy + 'Descriptions_s2/Beijing_grid_r0_d0.csv',
            dir_hemy + 'Descriptions_s2/Shanghai_grid_r0_d0.csv',
            dir_hemy + 'Descriptions_s2/Guangzhou_grid_r0_d0.csv',
            dir_hemy + 'Descriptions_s2/Guangzhou_grid_r1_d0.csv',
            dir_hemy + 'Descriptions_s2/Shenzhen_grid_r0_d0.csv',
            dir_hemy + 'Descriptions_s2/Shenzhen_grid_r1_d0.csv',
            dir_hemy + 'Descriptions_s2/Shenzhen_grid_r0_d1.csv',
            dir_hemy + 'Descriptions_s2/Shenzhen_grid_r1_d1.csv',
            dir_hemy + 'Descriptions_s2/Chengdu_grid_r0_d0.csv',
            dir_hemy + 'Descriptions_s2/Chengdu_grid_r1_d0.csv',
            dir_hemy + 'Descriptions_s2/Xi_an_grid_r0_d0.csv',
            dir_hemy + 'Descriptions_s2/Xi_an_grid_r1_d0.csv',
            dir_hemy + 'Descriptions_s2/Xi_an_grid_r0_d1.csv',
            dir_hemy + 'Descriptions_s2/Xi_an_grid_r1_d1.csv']

#hint_dir = "/home/gridsan/qwang/satellite_tiles_control/skeleton/16/"
#hint_dir = dir_hemy + 'Constraint_Images/'
hint_dir = dir_hemy + 'Vector_Image_Partition/'


# Configs
resume_path = './models/control_sd15_ini.ckpt'
# resume_path = './lightning_logs/version_24275255/checkpoints/epoch=4-step=112594.ckpt'
# resume_path = './lightning_logs/version_24305430/checkpoints/epoch=4-step=112594.ckpt'


batch_size = 3
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


def main():
    # Macbook
    # 检查 MPS 是否可用
    device = torch.device("cuda:5" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Using device: {device}")
    
    
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    #model = create_model('./models/cldm_v15.yaml').cpu()
    model = create_model('./models/cldm_v15.yaml').to(device)
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    # Macbook
    model.to(device)

    # 定义检查点回调函数
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints_s2_chn',  # 保存检查点的目录
        filename='{epoch}-{step}',  # 文件名格式
        save_top_k=-1,  # 保存所有检查点
        every_n_epochs=5,  # 每隔几代保存一次
        save_last=True
    )    
    
    # Misc
    dataset = MyDataset(image_dir, data_dir, hint_dir)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    #trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
    #trainer = pl.Trainer(accelerator='gpu', devices=1, precision=32, callbacks=[logger])
    trainer = pl.Trainer(accelerator='gpu', devices=[5], precision=32, callbacks=[logger, checkpoint_callback])
    # Macbook
    #trainer = pl.Trainer(accelerator=device.type, devices=1, precision=32, callbacks=[logger])
    #trainer = pl.Trainer(devices=1, precision=32, callbacks=[logger])
    
    # Train!
    trainer.fit(model, dataloader)

#def main():
#    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#    world_size = 1 #torch.cuda.device_count()
#    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    #mp.set_start_method('spawn')
    main()