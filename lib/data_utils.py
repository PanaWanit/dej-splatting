import os
import pandas as pd
import numpy as np
import torch
from torch import nn

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