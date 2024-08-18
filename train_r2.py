import argparse
from share import *
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
# from tutorial_dataset import MyDataset
# from satellite_dataset import MyDataset

# --------------------- #
from satellite_tiles_r2 import MyDataset

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.callbacks import ModelCheckpoint

dir_hemy = '/home/yuebing/Mingyi/GenerativeUrbanDesign/Urban_Data/'
image_dir = dir_hemy + 'Vector_Image_Partition/'
hint_dir = dir_hemy + 'Vector_Image_Partition/'

data_dir = [dir_hemy + 'Descriptions_s3/NewYork_grid_r0_d0.csv',
            dir_hemy + 'Descriptions_s3/NewYork_grid_r1_d0.csv',
            dir_hemy + 'Descriptions_s3/NewYork_grid_r2_d0.csv',
            dir_hemy + 'Descriptions_s3/NewYork_grid_r0_d1.csv',
            dir_hemy + 'Descriptions_s3/NewYork_grid_r1_d1.csv',
            dir_hemy + 'Descriptions_s3/NewYork_grid_r2_d1.csv',
            dir_hemy + 'Descriptions_s3/NewYork_grid_r0_d2.csv',
            dir_hemy + 'Descriptions_s3/NewYork_grid_r1_d2.csv',
            dir_hemy + 'Descriptions_s3/NewYork_grid_r2_d2.csv']


# Configs
resume_path = './models/control_sd15_ini.ckpt'
# resume_path = './lightning_logs/version_24275255/checkpoints/epoch=4-step=112594.ckpt'
# resume_path = './lightning_logs/version_24305430/checkpoints/epoch=4-step=112594.ckpt'


log_dir = './lightning_logs_task2'

batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


#def parse_args():
#    parser = argparse.ArgumentParser(description="Train model with PyTorch Lightning")
#    #parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model')
#    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs')
#    #parser.add_argument('--gpu', type=int, required=True, help='GPU id to use for training')
#    return parser.parse_args()

#def main_worker(rank, world_size):
def main():
    #args = parse_args()

    # 设置设备
    #device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda:4" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.cuda.set_device(device) # important
    print(f"Using device: {device}")

    # 加载模型
    model = create_model('./models/cldm_v15.yaml').to(device)
    
    # 创建模型（不加载到设备）
    #model = create_model('./models/cldm_v15.yaml')
    #print("Model created successfully")
    #
    ## 将模型的每一部分逐步加载到设备
    #for name, param in model.named_parameters():
    #    print(f"Moving {name} to {device}")
    #    param.data = param.data.to(device)
    #    if param._grad is not None:
    #        param._grad.data = param._grad.data.to(device)
    #print("Model loaded to device successfully")

    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.to(device) # important

    # 确保输出和日志目录存在
    #os.makedirs(args.log_dir, exist_ok=True)
    #os.makedirs(log_dir, exist_ok=True)

    # 数据集和数据加载器
    #dataset = MyDataset(image_dir, data_dir, hint_dir)
    #dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    ##logger = ImageLogger(batch_frequency=logger_freq, save_dir=args.log_dir)
    #logger = ImageLogger(batch_frequency=logger_freq) #, save_dir=log_dir)

    ## 设置Trainer
    #trainer = pl.Trainer(
    #    accelerator='gpu' if torch.cuda.is_available() else device.type,
    #    #devices=[args.gpu] if torch.cuda.is_available() else 1,
    #    devices=[4] if torch.cuda.is_available() else 1,
    #    precision=32,
    #    callbacks=[logger]
    #    #default_root_dir=log_dir
    #    #default_root_dir=args.log_dir 
    #)

    ## 开始训练
    #trainer.fit(model, dataloader)
    
    # 定义检查点回调函数
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',  # 保存检查点的目录
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
    trainer = pl.Trainer(accelerator='gpu', devices=[4], precision=32, callbacks=[logger, checkpoint_callback])
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
    
# 命令行参数
# python train_r2.py --output_dir /path/to/output/task2 --log_dir /path/to/logs/task2 --gpu 4
# python train_r2.py --log_dir ./lightning_logs_task2  --gpu 4