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


device = torch.device('cuda:0')
print(device)  # 输出：device(type='cuda', index=1)
torch.cuda.set_device(device)
# 检查设备是否正确设置
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    print("CUDA is not available.")

train_No = 'density'
# infer_stage = 4
# infer_num = 200
# desc = 'Description_landuse_med_hh_inc'
test_seed = 42

ckpt_directory = f'./ckpts_s/checkpoints_{train_No}'
ckpt_epoch_list = ['19', '31'] # '3', '7', '11', '15', '19', 
output_file_dir = f'./output_image/{train_No}/epoch_'

def find_epoch_files(directory, epoch_value):
    match_string = f'epoch={epoch_value}-step='
    all_files = os.listdir(directory)
    matching_files = [f for f in all_files if f.startswith(match_string)]
    return matching_files

# 查找匹配的文件
ckpt_path_list = []
for d in range(len(ckpt_epoch_list)):
    epoch_value = ckpt_epoch_list[d]
    matching_files = find_epoch_files(ckpt_directory, epoch_value)
    ckpt_path_list.append(f'{ckpt_directory}/{matching_files[0]}')

city = 'Chicago'
hint_dir = './urban_data/Chicago/hint_images/' # '/home/yuebing/gud/Urban_Data/Vector_Image_Partition/'
image_dir = './urban_data/Chicago/satellite_images/' # '/home/yuebing/gud/Urban_Data/Vector_Image_Partition/' 
desc_dir = './urban_data/Chicago/descriptions/' # '/home/yuebing/gud/Urban_Data/Descriptions_s4/'
data_dir = [desc_dir + 'Chicago_Density_r0_d0.csv',
            desc_dir + 'Chicago_Density_r1_d0.csv',
            desc_dir + 'Chicago_Density_r2_d0.csv',
            desc_dir + 'Chicago_Density_r0_d1.csv',
            desc_dir + 'Chicago_Density_r1_d1.csv',
            desc_dir + 'Chicago_Density_r2_d1.csv',
            desc_dir + 'Chicago_Density_r0_d2.csv',
            desc_dir + 'Chicago_Density_r1_d2.csv',
            desc_dir + 'Chicago_Density_r2_d2.csv'
            ]


df = pd.DataFrame() 
for f in data_dir: 
    dfi = pd.read_csv(f)
    df = pd.concat([df, dfi[(dfi['Train']==0)&(dfi['img']==1)&(dfi['streetview']==1)&(dfi['built_volume_total'].notna())&(dfi['built_volume_nres'].notna())]])
    # desc
def compute_description(row):
        return f"Satellite Image of Chicago. Total built-up volume is {int(row['built_volume_total']/1000)} thousand m3. Non-residential built-up volume is {int(row['built_volume_nres']/1000)} thousand m3."
df['Description_Density'] = df.apply(lambda row: compute_description(row), axis=1)
desc = 'Description_Density'
print("number of test data", len(df))
print(df.shape)


def split_and_sample(df, seed, column_name, k, split='quantile'):
    """
    Split a DataFrame into 4 groups based on quartiles of a specified column
    and sample k rows from each group.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column to split the DataFrame on.
        k (int): Number of samples to take from each split.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled rows from each split.
    """
    if split == "quantile":
        # Get split points
        quantiles = [0.25, 0.5, 0.75]
        split_points = df[column_name].quantile(quantiles).to_dict()
        # Assign labels to the splits
        labels = ['low', 'midlow', 'midhigh', 'high'] # 

        # 我感觉不能这么分，应该是根据最大最小值
        # Ensure the split points are sorted
        sorted_points = sorted(split_points.values())
        
        # Use pd.cut to map the values into bins
        bins = [-float('inf')] + sorted_points + [float('inf')]

    else:
        min_val = df[column_name].min()
        max_val = df[column_name].max()
    
        # Define the bins for equal intervals
        bins = [
            min_val,
            min_val + (max_val - min_val) / 4,
            min_val + (max_val - min_val) / 2,
            min_val + 3 * (max_val - min_val) / 4,
            max_val
        ]
    
    # Define the categories
    labels = ['low', 'midlow', 'midhigh', 'high']
    
    # Use pd.cut to map the values into bins
    df[f'{column_name}_category'] = pd.cut(
        df[column_name], bins=bins, labels=labels, include_lowest=True
    )
    print(df.head())
    # Sample k rows from each split
    sampled_dfs = []

    for label in labels:
        split_df = df[df[f'{column_name}_category'] == label]
        print("=======", label, len(split_df))
        # print(split_df[[column_name]].head())
        # print(label, len(split_df))
        if label == 'high' and split=='quantile':
            sample_df = split_df.sort_values(by=[column_name], ascending=False).iloc[:k].copy()
        else:
            sample_df = split_df.sample(n=k, random_state=seed, replace=False)
        sampled_dfs.append(sample_df)  # Replace=True if fewer rows exist
        print(sample_df[column_name].tolist())
    # Combine all sampled DataFrames
    sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
    return sampled_df

# df = split_and_sample(df, test_seed, 'built_volume_total', 10, split='quantile')
# print(df.head())


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

def process(xtile, ytile, r, d, zoom, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, v=None, RGB=True):
    
    with torch.no_grad():        
        if city == "Chicago":
            hint_name = hint_dir + city + '_of_Stage_1_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.tif'
            # img_name =  image_dir + city + '_ofr_Stage_2_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.tif'
            img_name =  image_dir + city + '_of_Stage_4_grid_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.jpg'

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
num_samples = 3
image_resolution = 512
strength = 1.0
guess_mode = False
detect_resolution = 512
ddim_steps = 20
scale = 9.0
seed = 5354 # [42, 1234, 5678]
eta = 0.0
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'



for e in range(len(ckpt_epoch_list)):
    
    ckpt_path = ckpt_path_list[e]
    model, ddim_sampler = load_model(ckpt_path, device)
    
    outputs = []
    
    output_file = output_file_dir + str(ckpt_epoch_list[e]) #ckpt_list[e]['epoch']
    os.makedirs(output_file, exist_ok=True)

    for i in range(len(df)): #df.shape[0] 
        print(i, )
        xtile = df.iloc[i]['row']
        ytile = df.iloc[i]['col']
        right = df.iloc[i]['r']
        down  = df.iloc[i]['d']
        
        prompt = df.iloc[i][desc]
        #prompt = df.loc[(int(xtile),int(ytile))]['Land_use_description']
        
        RGB=True
        # for seed in seeds:
        outputs_i = process(xtile, ytile, right, down, zoom, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, v='0', RGB=RGB)
        outputs.append(outputs_i)
        
        for j, image in enumerate(outputs_i[2:]):
            # Save each image with the desired naming convention
            filename = f'{xtile}_{ytile}_{right}_{down}_{j}.png'
            #plt.imsave(os.path.join(output_file,filename), image)
            # 将 numpy 数组转换为 PIL 图像
            image_pil = Image.fromarray(image)
            # 保存图像并设置质量和优化参数
            image_pil.save(os.path.join(output_file,filename), format='JPEG', quality=85, optimize=True)
    