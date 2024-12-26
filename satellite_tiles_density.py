import cv2
import glob
import numpy as np
import os
import pandas as pd
import re
from torch.utils.data import Dataset
from itertools import compress
from annotator.hed import HEDdetector
from annotator.util import HWC3
import torch

#from decimal import Decimal

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, image_dir, data_dir, hint_dir):

        self.image_dir = image_dir
        self.data_dir = data_dir
        self.hint_dir = hint_dir

        df = pd.DataFrame() 
        for f in self.data_dir: 
            #dfi = pd.read_csv(f, dtype={'left_x': str, 'top_y': str})  # 读取为字符串以确保高精度不丢失
            #dfi['left_x'] = dfi['left_x'].apply(Decimal)
            #dfi['top_y'] = dfi['top_y'].apply(Decimal)
            #df = pd.concat([df, pd.read_csv(f)])
            dfi = pd.read_csv(f)
            # pct_bl_100
            df = pd.concat([df, dfi[(dfi['Train']==1)&(dfi['img']==1)&(dfi['streetview']==1)&(dfi['built_volume_total'].notna())&(dfi['built_volume_nres'].notna())]])

        df['Description_Density'] = df.apply(lambda row: self.compute_description(row), axis=1)
        self.description = df
        
        #self.image_list_xtile = self.description['xtile'].to_list()
        #self.image_list_ytile = self.description['ytile'].to_list()       
        self.image_city = self.description['city'].to_list()
        #self.image_city = 'NewYork'

        self.image_list_xtile = self.description['row'].to_list()
        self.image_list_ytile = self.description['col'].to_list()
        #self.image_list_xcord = self.description['left_x'].to_list()
        #self.image_list_ycord = self.description['top_y'].to_list()
        self.image_list_right = self.description['r'].to_list()
        self.image_list_down = self.description['d'].to_list()

        #self.description = self.description['final_description'].to_list()
        # self.description = self.description['Description_med_hh_inc'].to_list()
        self.description = self.description['Description_Density'].to_list() # satellite_
        print(self.description[:5])

    def compute_description(self, row):
        return f"Satellite Image of Chicago. Total built-up volume is {int(row['built_volume_total']/1000)} thousand m3. Non-residential built-up volume is {int(row['built_volume_nres']/1000)} thousand m3."
    
    def __len__(self):
        return len(self.image_list_xtile)

    def __getitem__(self, item):

        xtile = self.image_list_xtile[item]
        ytile = self.image_list_ytile[item]
        #xcord = self.image_list_xcord[item]
        #ycord = self.image_list_ycord[item]
        r = self.image_list_right[item]
        d = self.image_list_down[item]
        city = self.image_city[item]
        #city = self.image_city
        if city == "Chicago":
            hint_name = self.hint_dir + city + '_of_Stage_1_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.tif'
            img_name =  self.image_dir + city + '_of_Stage_4_grid_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.jpg'

        #print('-----img_name-----', img_name)
        #print('-----hint_name-----', hint_name)

        target = cv2.imread(img_name)
        source = cv2.imread(hint_name, cv2.IMREAD_UNCHANGED)
        if source is None:
            print(hint_name)
        
        if source.shape[2] == 4:
            # convert 4-channel source image to 3-channel
            #make mask of where the transparent bits are
            trans_mask = source[:,:,3] == 0

            #replace areas of transparency with white and not transparent
            source[trans_mask] = [255, 255, 255, 255]

            #new image without alpha channel...
            source = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)

        #OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        prompt = self.description[item]
        
        # to(device)
        target = torch.tensor(target).to(device)
        source = torch.tensor(source).to(device)
        
        return dict(jpg=target, txt=prompt, hint=source)

