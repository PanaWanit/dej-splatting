import torch
import torch.nn as nn
import math
import numpy as np

def to_viewpoint_camera(data):
    camera = Camera(data['w2c'], data['camera_df'], data['image_df'])
    return camera

class Camera(nn.Module):
    def __init__(self, w2c, camera_df, image_df, znear=0.1, zfar=100):
        super(Camera, self).__init__()
        device = w2c.device
        self.znear = znear
        self.trans = torch.cat(image_df['TX'], image_df['TY'], image_df['TZ']).to(device)
        self.zfar = zfar
        self.focal_x = camera_df['FocalX']
        self.focal_y = camera_df['FocalY']
        self.FoVx = focal2fov(self.focal_x, camera_df['W'])
        self.FoVy = focal2fov(self.focal_y, camera_df['H'])
        self.image_width = camera_df['W']
        self.image_height = camera_df['H']
        self.world_view_transform = w2c.transpose(0,1).to(device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(device)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
