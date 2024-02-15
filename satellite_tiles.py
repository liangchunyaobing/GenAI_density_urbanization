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

class MyDataset(Dataset):
    def __init__(self, image_dir, data_dir, hint_dir):

        self.image_dir = image_dir
        self.data_dir = data_dir
        self.hint_dir = hint_dir
       
        df = pd.DataFrame() 
        for f in self.data_dir: 
            df = pd.concat([df, pd.read_csv(f)])

        self.description = df
        
        self.image_list_xtile = self.description['xtile'].to_list()
        self.image_list_ytile = self.description['ytile'].to_list()       
        self.image_city = self.description['city'].to_list()
 
        self.description = self.description['final_description'].to_list()
        
    def __len__(self):
        return len(self.image_list_xtile)

    def __getitem__(self, item):

        xtile = self.image_list_xtile[item]
        ytile = self.image_list_ytile[item]
        city = self.image_city[item]
 
        if '+' in str(xtile):
            img_name = self.image_dir + city + "/augment/16_" + str(xtile) + "_" + str(ytile) + ".png"
            hint_name = self.hint_dir + city + "/augment/" + str(xtile) + "/" + str(ytile) + ".png"
        else:
            img_name = self.image_dir + city + "/16_" + str(xtile) + "_" + str(ytile) + ".png"
            hint_name = self.hint_dir + city + "/" + str(xtile) + "/" + str(ytile) + ".png"
            
        target = cv2.imread(img_name)
        source = cv2.imread(hint_name, cv2.IMREAD_UNCHANGED)
        if source is None:
            print(hint_name)

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
        
        
        return dict(jpg=target, txt=prompt, hint=source)

