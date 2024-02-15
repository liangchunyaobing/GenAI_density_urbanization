from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import random
import pickle as pkl

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import pandas as pd
import matplotlib.pyplot as plt

from util_generate import process

version = '24511719'
# version = '24305430'
# version = '24275255'
epoch = '4'
step = '112594'
city_dic = {'chicago': 'Chicago', 'dallas':'Dallas', 'la':'Los Angeles'}
city = 'chicago'

desc = {}
for c in ['chicago','la','dallas']:
    desc[c] = pd.read_csv("~/satellite_tiles_control/tile_descriptions/tile_descriptions_"+c+"_1114.csv")#.set_index(['xtile','ytile'])
tile_descriptions = desc[city]

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./lightning_logs/version_'+version+'/checkpoints/epoch='+epoch+'-step='+step+'.ckpt', location='cuda'))

model = model.cuda()
from torchmetrics.image.fid import FID
fid = FID()

import os
os.makedirs("../satellite_tiles_control/generation/"+version+"_"+epoch+"-"+step+"_"+city+"/", exist_ok=True)

# i = 0

# real = []
# fake = []

for xtile, ytile, city, prompt in zip(tile_descriptions['xtile'], tile_descriptions['ytile'], tile_descriptions['city'], tile_descriptions['final_description']):
    
    xtile = str(xtile)
    ytile = str(ytile)
    zoom = "16"
    input_city = city
    output_city = city

    num_samples = 1
    image_resolution = 512
    strength = 1.0
    guess_mode = False
    detect_resolution = 512
    ddim_steps = 20
    scale = 9.0
    seed = 5354
    eta = 0.0
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    outputs = process(model, xtile, ytile, zoom, input_city, output_city, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, RGB=False);
    
    cv2.imwrite("../satellite_tiles_control/generation/"+version+"_"+epoch+"-"+step+"_"+city+"/"+xtile+'_'+ytile+'.png',outputs[2]);

    # i += 1
        
    # x = torch.tensor(outputs[1][None,:,:,:], dtype=torch.float)
    # x = einops.rearrange(x, 'b h w c -> b c h w')
    # x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    # x = x.to(torch.uint8)
    # real.append(x)

    # x = torch.tensor(outputs[2][None,:,:,:], dtype=torch.float)
    # x = einops.rearrange(x, 'b h w c -> b c h w')
    # x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    # x = x.to(torch.uint8)
    # fake.append(x)
    
    # break

# real = torch.concat(real)
# fake = torch.concat(fake)

# fid.reset()
# fid.update(real, real=True)
# fid.update(fake, real=False)

# print(fid.compute())

# with open(version+"_"+epoch+"-"+step+"_"+city+".pkl", "wb") as f:
#     pkl.dump(fake, f)
    # pkl.dump(real, f)

