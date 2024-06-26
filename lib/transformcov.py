import torch
import torch.nn as nn
import math

from lib.sh_utils import get_sh_color

#################### Transform ############################


def projection_ndc(points, viewmatrix, projmatrix, eps=1e-6):
    points_o = torch.hstack([points, torch.ones(points.size(0), 1, device='cuda')])  # make it homogenous
    viewmatrix = viewmatrix
    projmatrix = projmatrix
    # print(viewmatrix.shape, projmatrix.shape)
    points_h = points_o @ viewmatrix @ projmatrix  # projection

    p_w = 1.0 / (points_h[:, -1:] + eps)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    mask = p_view[:, 2] >= 0.2
    return p_proj[mask], p_view[mask]


def build_color(means3D, shs, camera, sh_degree):
    rays_o = camera.camera_center
    color = get_sh_color(sh_degree, means3D, rays_o, shs)
    return color


############################################################

#################### Covariance ############################


def gen_rotation(rot: torch.tensor):
    norm = torch.sqrt(rot[:, 0] * rot[:, 0] + rot[:, 1] * rot[:, 1] + rot[:, 2] * rot[:, 2] + rot[:, 3] * rot[:, 3])

    q = rot / norm.unsqueeze(1)

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def gen_scaling(scale: torch.tensor):
    S = torch.zeros((scale.size(0), 3, 3), device="cuda")
    S[:, 0, 0] = scale[:, 0]
    S[:, 1, 1] = scale[:, 1]
    S[:, 2, 2] = scale[:, 2]

    return S


def get_covariance_3d(R, S):
    # print('rs', R.shape, S.shape)
    RS = R @ S  # [Vec, 3, 3]
    return RS @ RS.transpose(1, 2)


def get_covariance_2d(mean3d, cov3d, w2cam, focal_x, focal_y, trans):

    tx = trans[0]
    ty = trans[1]
    tz = trans[2]

    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[:, 0, 0] = 1 / tz * focal_x
    J[:, 0, 2] = -tx / (tz * tz) * focal_x
    J[:, 1, 1] = 1 / tz * focal_y
    J[:, 1, 2] = -ty / (tz * tz) * focal_y

    W = w2cam[:3, :3]  # .T
    cov2d = J @ W @ cov3d @ W.T @ J.transpose(1, 2)

    filter = torch.eye(2, 2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter


############################################################
