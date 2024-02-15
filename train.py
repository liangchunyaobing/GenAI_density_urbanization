from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
# from satellite_dataset import MyDataset
from satellite_tiles import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# image_dir = "/home/gridsan/qwang/satellite_images/zoom17/"
# data_dir = "/home/gridsan/qwang/JTL-transit_shared/deep_hybrid_model/data/"

image_dir = "/home/gridsan/qwang/satellite_tiles_control/satellite_tiles/16/"
data_dir = ["/home/gridsan/qwang/satellite_tiles_control/tile_descriptions/tile_descriptions_chicago_1114.csv",
            "/home/gridsan/qwang/satellite_tiles_control/tile_descriptions/tile_descriptions_chicago+5+5_1114.csv",
           "/home/gridsan/qwang/satellite_tiles_control/tile_descriptions/tile_descriptions_la+5+5_1114.csv",
           "/home/gridsan/qwang/satellite_tiles_control/tile_descriptions/tile_descriptions_la+5+0_1114.csv",
           "/home/gridsan/qwang/satellite_tiles_control/tile_descriptions/tile_descriptions_la+0+5_1114.csv",
           "/home/gridsan/qwang/satellite_tiles_control/tile_descriptions/tile_descriptions_la_1114.csv",
           "/home/gridsan/qwang/satellite_tiles_control/tile_descriptions/tile_descriptions_dallas_1114.csv",
           "/home/gridsan/qwang/satellite_tiles_control/tile_descriptions/tile_descriptions_dallas+5+5_1114.csv",
           "/home/gridsan/qwang/satellite_tiles_control/tile_descriptions/tile_descriptions_dallas+5+0_1114.csv",
           "/home/gridsan/qwang/satellite_tiles_control/tile_descriptions/tile_descriptions_dallas+0+5_1114.csv"]
hint_dir = "/home/gridsan/qwang/satellite_tiles_control/skeleton/16/"

# Configs
# resume_path = './models/control_sd15_ini.ckpt'
# resume_path = './lightning_logs/version_24275255/checkpoints/epoch=4-step=112594.ckpt'
resume_path = './lightning_logs/version_24305430/checkpoints/epoch=4-step=112594.ckpt'

batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset(image_dir, data_dir, hint_dir)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
