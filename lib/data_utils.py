import os
import pandas as pd
import numpy as np
import torch
import torch.functional as F
from torch import nn
from transformcov import gen_rotation
import imageio

# Read images.txt
def read_images_colmap(images_file_path : str):
    def read_image(image_path, scale_factor):
        image = torch.from_numpy(imageio.imread(image_path).astype(np.float32) / 255.0)
        image_size = image.shape[:2]

        if scale_factor != 1:
            image_size[0] = image_size[0] * scale_factor
            image_size[0] = image_size[1] * scale_factor
            image = F.interpolate(image.permute(0, 3, 1, 2), scale_factor=scale_factor, mode='bilinear').permute(0, 2, 3, 1)

        return image


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

    with open(images_file_path) as f:
        images_list = [line.split() for line in f.readlines()[::2]]

    df = pd.DataFrame(images_list, columns = images_columns).astype(columns_type)
    
    # Compute extrinsic
    rot = torch.from_numpy(df[['QW', 'QX', 'QY', 'QZ']].to_numpy())
    R = gen_rotation(rot)
    extrinsics = torch.empty(R.size(0), 3, 4)

    extrinsics[:, :3, :3] = R
    extrinsics[:, 0, 3] = torch.from_numpy(df['TX'].to_numpy())
    extrinsics[:, 1, 3] = torch.from_numpy(df['TY'].to_numpy())
    extrinsics[:, 2, 3] = torch.from_numpy(df['TZ'].to_numpy())
        
    return extrinsics, R, df['CAMERA_ID'].to_numpy(), df['NAME'].apply(read_image)

def read_cameras_colmap(camera_file_path:str):
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

    intrinsics = torch.zeros(df.shape[0], 3, 3)
    intrinsics[:, 0, 0] = torch.from_numpy(df['FocalX'].to_numpy())
    intrinsics[:, 1, 1] = torch.from_numpy(df['FocalY'].to_numpy())
    intrinsics[:, 0, 2] = torch.from_numpy(df['PrincX'].to_numpy())
    intrinsics[:, 1, 2] = torch.from_numpy(df['PrincY'].to_numpy())
    intrinsics[:, 2, 2] = 1
    return intrinsics, torch.from_numpy(df['W'].to_numpy()), torch.from_numpy(df['H'].to_numpy())

def read_all(images_file_path:str, camera_file_path:str):
    extrinsics, R, cam_ids = read_images_colmap(images_file_path)
    intrinsics, Ws, Hs = read_cameras_colmap(camera_file_path)

    
    for idx, cam_id in enumerate(cam_ids):
        cur_extrinsic = extrinsics[idx]
        cur_intrinsic = intrinsics[cam_id]
        cur_W = Ws[cam_id]
        cur_H = Hs[cam_id]
        w2img =  cur_intrinsic @ cur_extrinsic
        






if __name__ == '__main__':
    print(read_all('./images.txt', 'test_cam.txt'))