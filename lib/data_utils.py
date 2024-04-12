import os
import pandas as pd
import numpy as np
import torch
from torch import nn

# Read images.txt
def read_images_colmap(images_file_path : str) -> pd.DataFrame:
    with open(images_file_path) as f:
        images_list = [line.split() for line in f.readlines()[::2]]
        images_columns = ['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME']
        columns_type = {
            'IMAGE_ID': int,
            'QW': float,
            'QX': float,
            'QY': float,
            'QZ': float,
            'TX': float,
            'TY': float,
            'TZ': float,
            'CAMERA_ID': int,
            'NAME': str,
        }
        images_df = pd.DataFrame(images_list, columns = images_columns).astype(columns_type)
    return images_df

def read_cameras_colmap(camera_file_path:str) -> pd.DataFrame:
    columns = ['CAM_ID', 'MODEL', 'W', 'H', 'FocalX', 'FocalY', 'PrincX', 'PrincY']
    columns_type = {
        'CAM_ID': int,
        'MODEL': str,
        'W': int,
        'H': int,
        'FocalX': float,
        'FocalY': float,
        'PrincX': float,
        'PrincY': float,
    }
    with open(camera_file_path, 'r') as f:
        cam_lists = [x.split() for x in f.readlines()]
    df = pd.DataFrame(cam_lists, columns=columns).astype(columns_type)
    return df

def read_all(images_file_path:str, camera_file_path:str):
    # img_info_df = read_images_colmap(images_file_path)
    cam_info_df = read_cameras_colmap(camera_file_path)

    # gen instrinsic
    instrinsics = torch.zeros(cam_info_df.shape[0], 3, 3)

    instrinsics[:, 0, 0] = torch.from_numpy(cam_info_df['FocalX'].to_numpy())
    instrinsics[:, 1, 1] = torch.from_numpy(cam_info_df['FocalY'].to_numpy())
    instrinsics[:, 0, 2] = torch.from_numpy(cam_info_df['PrincX'].to_numpy())
    instrinsics[:, 1, 2] = torch.from_numpy(cam_info_df['PrincY'].to_numpy())
    instrinsics[:, 2, 2] = 1



if __name__ == '__main__':
    print(read_all('./images.txt', 'test_cam.txt'))