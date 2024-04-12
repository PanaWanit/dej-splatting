import os
import pandas as pd
import numpy as np
import torch
from torch import nn

# Read images.txt
def read_images(images_file_path : str):
    with open(images_file_path) as f:
        images_list = [line.split() for line in f.readlines()[::2]]
        images_columns = ['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME']
        images_df = pd.DataFrame(images_list, columns = images_columns, dtype = float)
    return images_df

def read_cameras(camera_file_path:str) -> pd.DataFrame:
    columns = ['CAM_ID', 'MODEL', 'W', 'H', 'FocalX', 'FocalY', 'PrincX', 'PrincY']
    columns_type = {
        'CAM_ID': 'int',
        'MODEL': 'str',
        'W': 'int',
        'H': 'int',
        'FocalX': 'float',
        'FocalY': 'float',
        'PrincX': 'float',
        'PrincY': 'float',
    }
    with open(camera_file_path, 'r') as f:
        cam_lists = [x.split() for x in f.readlines()]
    df = pd.DataFrame(cam_lists, columns=columns).astype(columns_type)
    return df
