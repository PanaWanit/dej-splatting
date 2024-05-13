import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
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
    rot = torch.from_numpy(df[["QW", "QX", "QY", "QZ"]].to_numpy()).cuda()
    R = gen_rotation(rot)
    extrinsics = torch.zeros(R.size(0), 4, 4).cuda()

    extrinsics[:, :3, :3] = R
    extrinsics[:, 0, 3] = torch.from_numpy(df["TX"].to_numpy())
    extrinsics[:, 1, 3] = torch.from_numpy(df["TY"].to_numpy())
    extrinsics[:, 2, 3] = torch.from_numpy(df["TZ"].to_numpy())
    extrinsics[:, 3, 3] = torch.ones_like(extrinsics[:, 3, 3])

    # Read image rgb with down sampling
    def read_image(image_path, scale_factor):
        # >>>>>>>>>>>>>>>>>>>>> remove alpha channel
        image = torch.from_numpy(imageio.imread(image_path).astype(np.float32)[..., :3] / 255.0)
        image_size = list(image.size()[:2])

        assert scale_factor <= 1, "scale_factor must be less than 1"

        if scale_factor != 1:
            image_size[0] = image_size[0] * scale_factor
            image_size[1] = image_size[1] * scale_factor
            image = image.unsqueeze(0)
            image = F.interpolate(image.permute(0, 3, 1, 2), scale_factor=scale_factor, mode="bilinear").permute(
                0, 2, 3, 1
            )  # [B, C, H, W]
            image = image.cuda()

        return image

    dir = os.path.dirname(images_file_path)
    rgbs = torch.cat([read_image(os.path.join(dir, img_path), scale_factor) for img_path in df["NAME"]], dim=0)
    # print(">>>>>>>>>>>>>>>> rgb device:", rgbs.device)
    # print(">>>>>>>>>>>>>>>> rgb shape:", rgbs.shape)

    return extrinsics, R, df["CAMERA_ID"].to_numpy() - 1, rgbs, df


def read_cameras_colmap(camera_file_path: str, scale_factor: float = 1):
    # print("camera scale:", scale_factor)
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
    intrinsics[:, 0, 0] = torch.from_numpy(df["FocalX"].to_numpy()) * scale_factor
    intrinsics[:, 1, 1] = torch.from_numpy(df["FocalY"].to_numpy()) * scale_factor
    intrinsics[:, 0, 2] = torch.from_numpy(df["PrincX"].to_numpy()) * scale_factor
    intrinsics[:, 1, 2] = torch.from_numpy(df["PrincY"].to_numpy()) * scale_factor
    intrinsics[:, 2, 2] = 1
    return intrinsics, torch.from_numpy(df["W"].to_numpy()), torch.from_numpy(df["H"].to_numpy()), df


def read_all(images_file_path: str, camera_file_path: str, scale_factor: float = 1):
    extrinsics, R, cam_ids, rgbs, image_df = read_images_colmap(images_file_path, scale_factor)
    intrinsics, Ws, Hs, camera_df = read_cameras_colmap(camera_file_path, scale_factor)

    properties = []

    for idx, cam_id in enumerate(cam_ids):
        cur_intrinsic = intrinsics[cam_id]
        cur_extrinsic = extrinsics[idx]

        cur_R = R[idx]
        cur_rgb = rgbs[idx]
        print("rgb min max mean", cur_rgb.min(), cur_rgb.max(), cur_rgb.mean())

        cur_image_df = image_df.iloc[idx]
        cur_camera_df = camera_df.iloc[cam_id]

        scaledW = int(Ws[cam_id] * scale_factor)
        scaledH = int(Hs[cam_id] * scale_factor)

        properties.append(
            {
                "rgb": cur_rgb,
                "scaledW": scaledW,
                "scaledH": scaledH,
                "R": cur_R,
                "intrinsic": cur_intrinsic,
                "w2c": cur_extrinsic,
                "c2w": cur_extrinsic.inverse(),
                "image_df": cur_image_df,
                "camera_df": cur_camera_df,
                "scale_factor": scale_factor,
            }
        )
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
    df["R"] = df["R"] / 255.0
    df["G"] = df["G"] / 255.0
    df["B"] = df["B"] / 255.0
    return df
