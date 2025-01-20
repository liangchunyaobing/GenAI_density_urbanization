import cv2
import glob
import numpy as np
import os
import pandas as pd
import re
from torch.utils.data import Dataset
from itertools import compress
import torch
import json
# from annotator.hed import HEDdetector
# from annotator.util import HWC3

# from decimal import Decimal

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, data_dir):

        self.data = []
        with open(data_dir, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)





    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # 读取图像和提示图像
        target = cv2.imread(target_filename)
        source = cv2.imread(source_filename, cv2.IMREAD_COLOR)


        # 如果提示图像是4通道，转换为3通道
        if source.shape[2] == 4:
            trans_mask = source[:, :, 3] == 0
            source[trans_mask] = [255, 255, 255, 255]
            source = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)

        # 转换颜色空间
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # 归一化
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0



        # 转换为张量并移动到设备
        # to(device)
        target = torch.tensor(target).to(device)
        source = torch.tensor(source).to(device)
        # print(target.shape,source.shape)

        return dict(jpg=target, txt=prompt, hint=source)

# 示例使用
if __name__ == "__main__":
    data_dir = './urban_data/tencities/train.json'

    # 初始化数据集
    dataset = MyDataset(data_dir)
    print(f"Dataset size: {len(dataset)}")

    # 如果数据集为空，提示用户检查
    if len(dataset) == 0:
        print("Dataset is empty. Please check if the file names in the directories match.")
    else:
        sample = dataset[0]
        print(sample["txt"])
