import os
import pandas as pd
import numpy as np
import torch
import torch.functional as F
from torch import nn
from lib.transformcov import gen_rotation
import imageio


# Read images.txt
def read_images_colmap(images_file_path: str, scale_factor: float = 1):

    images_columns = ["IMAGE_ID", "QW", "QX", "QY", "QZ", "TX", "TY", "TZ", "CAMERA_ID", "NAME"]
    columns_type = {
        "IMAGE_ID": int,
        "QW": float,
        "QX": float,
        "QY": float,
        "QZ": float,
        "TX": float,
        "TY": float,
        "TZ": float,
        "CAMERA_ID": int,
        "NAME": str,
    }

    with open(images_file_path) as f:
        images_list = [line.split() for line in f.readlines()[::2]]

    df = pd.DataFrame(images_list, columns=images_columns).astype(columns_type)

    # Compute extrinsic
    rot = torch.from_numpy(df[["QW", "QX", "QY", "QZ"]].to_numpy())
    R = gen_rotation(rot)
    extrinsics = torch.zeros(R.size(0), 4, 4)

    extrinsics[:, :3, :3] = R
    extrinsics[:, 0, 3] = torch.from_numpy(df['TX'].to_numpy())
    extrinsics[:, 1, 3] = torch.from_numpy(df['TY'].to_numpy())
    extrinsics[:, 2, 3] = torch.from_numpy(df['TZ'].to_numpy())
    extrinsics[:, 3, 3] = torch.ones_like(extrinsics[:, 3, 3])

    # Read image rgb with down sampling
    def read_image(image_path, scale_factor):
        image = torch.from_numpy(imageio.imread(image_path).astype(np.float32) / 255.0)
        image_size = image.shape[:2]

        assert scale_factor <= 1, "scale_factor must be less than 1"

        if scale_factor != 1:
            image_size[0] = image_size[0] * scale_factor
            image_size[0] = image_size[1] * scale_factor
            image = image.unsqueeze(0)
            image = F.interpolate(image.permute(0, 3, 1, 2), scale_factor=scale_factor, mode="bilinear").permute(
                0, 2, 3, 1
            )  # [B, C, H, W]
            image = image.squeeze(0)

        return image

    dir = os.path.dirname(images_file_path)
    rgbs = [read_image(os.path.join(dir, img_path), scale_factor) for img_path in df["NAME"]]

    return extrinsics, R, df["CAMERA_ID"].to_numpy()-1, rgbs, df


def read_cameras_colmap(camera_file_path: str):
    columns = ["CAM_ID", "MODEL", "W", "H", "FocalX", "FocalY", "PrincX", "PrincY"]
    columns_type = {
        "CAM_ID": int,
        "MODEL": str,
        "W": int,
        "H": int,
        "FocalX": float,
        "FocalY": float,
        "PrincX": float,
        "PrincY": float,
    }
    with open(camera_file_path, "r") as f:
        cam_lists = [x.split() for x in f.readlines()]
    df = pd.DataFrame(cam_lists, columns=columns).astype(columns_type)

    intrinsics = torch.zeros(df.shape[0], 3, 3)
    intrinsics[:, 0, 0] = torch.from_numpy(df["FocalX"].to_numpy())
    intrinsics[:, 1, 1] = torch.from_numpy(df["FocalY"].to_numpy())
    intrinsics[:, 0, 2] = torch.from_numpy(df["PrincX"].to_numpy())
    intrinsics[:, 1, 2] = torch.from_numpy(df["PrincY"].to_numpy())
    intrinsics[:, 2, 2] = 1
    return intrinsics, torch.from_numpy(df["W"].to_numpy()), torch.from_numpy(df["H"].to_numpy()), df


def read_all(images_file_path: str, camera_file_path: str):
    extrinsics, R, cam_ids, rgbs, image_df = read_images_colmap(images_file_path)
    intrinsics, Ws, Hs, camera_df = read_cameras_colmap(camera_file_path)

    properties = []

    for idx, cam_id in enumerate(cam_ids):
        cur_intrinsic = intrinsics[cam_id]
        cur_extrinsic = extrinsics[idx]

        cur_R = R[idx]
        cur_rgb = rgbs[idx]

        cur_image_df = image_df.iloc[idx]
        cur_camera_df = camera_df.iloc[cam_id]

        #w2img =  cur_intrinsic @ cur_extrinsic

        properties.append({
            'rgb': cur_rgb,
            'R': cur_R,
            'intrinsic': cur_intrinsic,
            'w2c': cur_extrinsic,
            'c2w': cur_extrinsic.inverse(),
            'image_df': cur_image_df,
            'camera_df': cur_camera_df
        })
    return properties


if __name__ == "__main__":
    print(read_all("./images.txt", "test_cam.txt"))


def read_points3D_colmap(point3D_file_path: str):
    point3D_columns = ["POINT3D_ID", "X", "Y", "Z", "R", "G", "B"]
    columns_type = {
        "POINT3D_ID": int,
        "X": float,
        "Y": float,
        "Z": float,
        "R": int,
        "G": int,
        "B": int,
    }

    with open(point3D_file_path) as f:
        point3D_list = [line.split()[:7] for line in f.readlines()]

    df = pd.DataFrame(point3D_list, columns=point3D_columns).astype(columns_type)
    return df