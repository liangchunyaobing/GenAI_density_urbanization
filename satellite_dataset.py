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
    def __init__(self, image_dir, data_dir, demo=-1):

        self.image_dir = image_dir
        self.data_dir = data_dir

        # data = pd.read_csv(data_dir + "census_tracts_filtered-1571.csv")
        # tracts = [str(s) + '_' + str(c) + '_' + str(t) for (s, c, t) in zip(data['state_fips'], data['county_fips'], data['tract_fips'])]

        image_list = glob.glob(image_dir + "*.png")
        image_list += glob.glob(image_dir + "*.jpg")
        image_df = pd.DataFrame(image_list, columns=['img_dir'])
        image_df['geoid'] = [img_name[img_name.rfind('/') + 1:img_name.rfind('_')] for img_name in image_list]

        self.demo = demo
        if demo > 0:
            demo_df, self.demo_cols = self.load_demo(data_dir)

        else:
            return

        demo_df['geoid'] = [str(s) + '_' + str(c) + '_' + str(t) for (s, c, t) in zip(demo_df['STATEFP'], demo_df['COUNTYFP'], demo_df['TRACTCE'])]

        df = pd.merge(image_df, demo_df, how='inner')
        self.image_list = df['img_dir'].to_numpy()
        self.demo = df['description'].to_numpy()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):

        img_name = self.image_list[item]
        target = cv2.imread(img_name)[44:556, 44:556, :]
        # target = cv2.imread(img_name)[172:428, 172:428, :]
        # source = cv2.imread(img_name.replace('zoom17', 'zoom17_edge'))[172:428, 172:428, :]

        image_apply = HEDdetector()
        source = image_apply(target)
        source[source < 175] = 0
        source = 255 - source
        source = HWC3(source)

        prompt = self.demo[item]

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

    def load_demo(self, data_dir):

        # Smart Locations from EPA
        epa = pd.read_csv(data_dir + "EPA_SmartLocations_CensusTract_export.csv")
        # log_tranform
        #     epa['activity_density'] = np.log(epa['activity_density'])
        #     epa['auto_oriented'] = np.log(epa['auto_oriented'])

        # Pollution PM2.5
        pm = pd.read_csv(data_dir + "PM2.5_Concentrations_2016_Illinois_Average_Annual.csv")
        pm['ctfips'] = pm['ctfips'] % 1e6
        pm['ctfips'] = pm['ctfips'].astype(int)
        demo_df = pd.merge(epa, pm, left_on=['STATEFP', 'COUNTYFP', 'TRACTCE'],
                           right_on=['statefips', 'countyfips', 'ctfips'])

        # ACS Demo
        acs = pd.read_csv(data_dir + "demo_tract.csv")
        acs['pop_density'] = acs['tot_population'] / acs['area']
        #     acs['pop_density'] = np.log(acs['pop_density'])
        demo_df = pd.merge(demo_df, acs, left_on=['COUNTYFP', 'TRACTCE'], right_on=['COUNTYA', 'TRACTA'])

        # normalize
        real_columns = ['activity_density', 'auto_oriented', 'multi_modal', 'pedestrian_oriented', 'PM2.5']
        real_columns += ['pop_density', 'inc_per_capita']

        #     return demo_df, None

        for c in real_columns:
            # demo_df[c] = (demo_df[c] - demo_df[c].mean())/demo_df[c].std()
            demo_df[c] = (demo_df[c] - demo_df[c].min()) / (demo_df[c].max() - demo_df[c].min())
            demo_df[c] = (demo_df[c] - 0.5) / 0.5

        pct_columns = ['employment_entropy', 'pop_income_entropy', 'wrk_emp_balance']
        pct_columns += ['pct25_34yrs', 'pct35_50yrs', 'pctover65yrs',
                        'pctwhite_alone', 'pct_nonwhite']
        for c in pct_columns:
            demo_df[c] = (demo_df[c] - demo_df[c].min()) / (demo_df[c].max() - demo_df[c].min())
            demo_df[c] = (demo_df[c] - 0.5) / 0.5

        #     columns = ['pop_density','inc_per_capita','pct25_34yrs','pct35_50yrs','pctover65yrs',
        #                  'pctwhite_alone','pct_nonwhite']
        columns = ['activity_density', 'multi_modal', 'pedestrian_oriented', 'PM2.5',
                   'inc_per_capita', 'employment_entropy', 'wrk_emp_balance', 'pct25_34yrs']

        description = {'activity_density+':' active', 'multi_modal+':'multi-modal',
                       'pedestrian_oriented+':'pedestrian_oriented', 'PM2.5+':'polluted',
                       'inc_per_capita+':'high-income', 'employment_entropy+':'diverse employment',
                       'wrk_emp_balance+':'balanced landuse', 'pct25_34yrs+':'young adults ',

                       'activity_density-':' quiet', 'multi_modal-':'car-dominant',
                       'pedestrian_oriented-':'', 'PM2.5-':'clean',
                       'inc_per_capita-':'low-income', 'employment_entropy-':'', 'wrk_emp_balance-':'', 'pct25_34yrs-':''}

        for c in columns:
            demo_df[c+'_label'] = ''
            demo_df.loc[demo_df[c] > 0.8, c+'_label'] = description[c+"+"]
            demo_df.loc[demo_df[c] < -0.8, c + '_label'] = description[c + "-"]

        demo_df['description'] = demo_df[[c+'_label' for c in columns]].agg(','.join, axis=1)
        demo_df['description'] = [re.sub(r'(,)\1+', r'\1',s) for s in demo_df['description']]
        demo_df['description'] = demo_df['description'].str[1:-1]
        demo_df['description'] = demo_df['description'].apply(lambda x: f"satellite image of a {x} region")

        return demo_df, columns
