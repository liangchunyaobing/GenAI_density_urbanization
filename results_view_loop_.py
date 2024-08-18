from share import *
import config

import cv2
import einops
#import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


device = torch.device('cuda:6')
print(device)  # 输出：device(type='cuda', index=1)
torch.cuda.set_device(device)
# 检查设备是否正确设置
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    print("CUDA is not available.")

city_dic={'chicago': 'Chicago', 'dallas':'Dallas', 'la':'Los Angeles'}
city = 'NewYork'

infer_stage = 2
input_city = 'NewYork'
output_city = 'NewYork'


ckpt_list = [
    #{'version': '3', 'epoch': '6', 'step': '101877'},
    #{'version': '3', 'epoch': '10', 'step': '160093'},
    #{'version': '3', 'epoch': '16', 'step': '247417'},
    #{'version': '3', 'epoch': '24', 'step': '363849'},
    {'version': '3', 'epoch': '29', 'step': '436619'}
]

dir_hemy = '/home/yuebing/Mingyi/GenerativeUrbanDesign/Urban_Data/'
image_dir = dir_hemy + 'Vector_Image_Partition/'
data_dir = [dir_hemy + 'Descriptions_s2/NewYork_grid_r0_d0.csv',
            dir_hemy + 'Descriptions_s2/NewYork_grid_r1_d0.csv',
            dir_hemy + 'Descriptions_s2/NewYork_grid_r2_d0.csv',
            dir_hemy + 'Descriptions_s2/NewYork_grid_r0_d1.csv',
            dir_hemy + 'Descriptions_s2/NewYork_grid_r1_d1.csv',
            dir_hemy + 'Descriptions_s2/NewYork_grid_r2_d1.csv',
            dir_hemy + 'Descriptions_s2/NewYork_grid_r0_d2.csv',
            dir_hemy + 'Descriptions_s2/NewYork_grid_r1_d2.csv',
            dir_hemy + 'Descriptions_s2/NewYork_grid_r2_d2.csv']

df = pd.DataFrame() 
for f in data_dir: 
    dfi = pd.read_csv(f)
    df = pd.concat([df, dfi[dfi['Train']==0]])
print(df.shape)


def load_model(ckpt_path, device):
    
    # 清理 GPU 内存
    torch.cuda.empty_cache()
    
    # 删除旧模型引用（如果存在）
    if 'model' in globals():
        del globals()['model']
    if 'ddim_sampler' in globals():
        del globals()['ddim_sampler']
    
    # 再次清理 GPU 内存
    torch.cuda.empty_cache()
    
    # 1. 创建模型实例
    model = create_model('./models/cldm_v15.yaml').cpu()
    
    # 2. 加载模型状态字典
    #ckpt_path = f'./checkpoints/epoch={epoch}-step={step}.ckpt'
    #ckpt_path = f'./version_archive/version_{version}_e{epoch}/checkpoints/epoch={epoch}-step={step}.ckpt'
    model.load_state_dict(load_state_dict(ckpt_path, location='cpu'))
    
    # 3. 将模型移动到指定设备
    model.to(device)
    
    # 4. 创建采样器实例
    ddim_sampler = DDIMSampler(model)
    
    return model, ddim_sampler

def process(infer_stage, xtile, ytile, r, d, zoom, input_city, output_city, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, v=None, RGB=True):
    
    with torch.no_grad():
        if infer_stage == 3:
            hint_name = image_dir + city + '_Stage_2_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.tif'
            img_name =  image_dir + city + '_Stage_3_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.tif' 
        elif infer_stage == 2:
            hint_name = image_dir + city + '_Stage_1_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.tif'
            img_name =  image_dir + city + '_Stage_2_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.tif' 
            
        input_image = cv2.imread(img_name)
        input_image = np.array(input_image)
        print(img_name)
        H,W,C = input_image.shape

        detected_map = cv2.imread(hint_name, cv2.IMREAD_UNCHANGED)
        detected_map = np.array(detected_map)    
        
        if detected_map.shape[2] == 4:
            # convert 4-channel source image to 3-channel
            #make mask of where the transparent bits are
            trans_mask = detected_map[:,:,3] == 0
    
            #replace areas of transparency with white and not transparent
            detected_map[trans_mask] = [255, 255, 255, 255]
    
            #new image without alpha channel...
            detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGRA2BGR)        
        
        #OpenCV read images in BGR order.
        control = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
        control = torch.from_numpy(control.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
            
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
            
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        if RGB:
            results = [x_samples[i] for i in range(num_samples)]
            detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
        else:
            results = [cv2.cvtColor(x_samples[i], cv2.COLOR_RGB2BGR) for i in range(num_samples)]

        return [detected_map] + [input_image] + results


def convert_bgr_to_rgb(image):
    """
    将 BGR 图像转换为 RGB 图像。
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


zoom = "16"
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

random_indices = [4873,2043,7119,740,424,3627,3270,2934,4604,694,2859,7202,6899,6184,5304,
                  6843,5619,801,5881,7080,4391,292,2845,3910,1979,5539,2195,2670,4959,1519,
                  3215,4139,9,3631,54,4805,3404,6939,7051,857,1070,6765,64,2534,7284,4908,
                  6404,6611,5282,5820,5357,294,4840,1630,2616,3351,6560,5906,2720,5567,1785,
                  6530,7277,5376,3179,342,5000,6726,838,1880,4453,6183,5451,642,4867,5339,
                  7128,2557,128,2650,1024,2871,60,1172,5634,363,3872,6405,1302,846,6668,5329,
                  641,3253,7038,734,5362,1497,1072,6478]


for e in range(len(ckpt_list)):
    
    version = ckpt_list[e]['version']
    epoch = ckpt_list[e]['epoch']
    step = ckpt_list[e]['step']
    if epoch == '29':
        ckpt_path = './lightning_logs/version_'+version+'/checkpoints/epoch='+epoch+'-step='+step+'.ckpt'
    else:
        ckpt_path = './version_archive/version_'+version+'_e'+epoch+'/checkpoints/epoch='+epoch+'-step='+step+'.ckpt'
    
    model, ddim_sampler = load_model(ckpt_path, device)
    
    outputs = []
    
    output_file = './output_image/stage_2_0702/allpics_epoch_' + ckpt_list[e]['epoch']
    os.makedirs(output_file, exist_ok=True)

    for i in range(df.shape[0]): #random_indices
    
        xtile = df.iloc[i]['row']
        ytile = df.iloc[i]['col']
        right = df.iloc[i]['r']
        down  = df.iloc[i]['d']
        
        prompt = df.iloc[i]['descriptions']
        #prompt = df.loc[(int(xtile),int(ytile))]['Land_use_description']
        
        RGB=True
        outputs_i = process(infer_stage, xtile, ytile, right, down, zoom, input_city, output_city, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, v='0', RGB=RGB)
        outputs.append(outputs_i)
        
        # Save each image with the desired naming convention
        image = outputs_i[2]
        filename = f'{xtile}_{ytile}_{right}_{down}.png'
        #plt.imsave(os.path.join(output_file,filename), image)
        # 将 numpy 数组转换为 PIL 图像
        image_pil = Image.fromarray(image)
        # 保存图像并设置质量和优化参数
        image_pil.save(os.path.join(output_file,filename), format='JPEG', quality=85, optimize=True)
    
    
    #fig, ax = plt.subplots(100,3, figsize=(6,240))
    ##fig, ax = plt.subplots(5,3, figsize=(6,12))
    #for x in range(100):
    #    for y in range(3):
    #        if y == 1:
    #            ax[x,y].imshow(convert_bgr_to_rgb(outputs[x][y]))
    #        else:
    #            ax[x,y].imshow(outputs[x][y])
    #
    #for ax in ax.ravel():
    #    ax.set_axis_off()
    #os.makedirs('./output_image/stage_2_0702_summary', exist_ok=True)
    #fig.savefig('./output_image/stage_2_0702_summary/output_image_e'+ckpt_list[e]['epoch']+'.png', bbox_inches='tight')