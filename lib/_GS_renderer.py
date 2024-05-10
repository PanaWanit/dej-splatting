import torch
from torch import nn
from lib._sh_utils import get_sh_color
from lib.transformcov import *
import contextlib
MAX_OPACITY = torch.tensor([0.99], dtype=torch.float32, device='cuda')

def render(mean2D, depth, cov2D, color, opacity, W, H, bg):
    idx = torch.argsort(depth)
    mean2D = mean2D[idx]
    cov2D = cov2D[idx]
    color = color[idx]
    opacity = opacity[idx]

    inv = torch.inverse(cov2D)
    out = torch.zeros((W * H, 3), dtype=torch.float32, device='cuda')
    point = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij'), dim=-1).reshape(-1, 2)
    # print(point)
    dis1 = mean2D.view(1, -1, 2) - point.view(-1, 1, 2)
    N = mean2D.shape[0]

    power = -0.5 * (dis1.view(-1, N, 1, 2) @ inv @ dis1.view(-1, N, 2, 1)).squeeze()
    power = power.clamp(max = 0)
    alpha = (opacity.view(1, -1) * torch.exp(power)).clamp(max = 0.99)
    alpha[alpha < 1./255.] = 0
    tran = torch.cat((torch.ones((H*W, 1), device='cuda'), 1-alpha), dim=-1)
    tran = torch.cumprod(tran, 1)
    tran[tran < 0.001] = 0
    weight = alpha * tran[:, :-1]
    out = (weight @ color).view(W, H, 3) + bg * tran[:, -1].view(W, H, 1)
    return out.permute(1, 0, 2)

class GaussRenderer(nn.Module):
    def __init__(self, active_sh_degree=3, white_bkgd=True, width=256, height=256, **kwargs):
        super(GaussRenderer, self).__init__()

        self.USE_GPU_PYTORCH = True
        self.USE_PROFILE = False
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.W = width
        self.H = height
        self.bg = torch.zeros((self.W, self.H, 3), dtype=torch.float32, device='cuda')
        self.point = torch.stack(torch.meshgrid(torch.arange(self.W), torch.arange(self.H), indexing='ij'), dim=-1).to('cuda').view(-1, 1, 2)
    
    def render(self, camera, means2D, depths, cov2D, color, opacity):
        W = self.W
        H = self.H
        idx = torch.argsort(depths)
        means2D = means2D[idx]
        cov2D = cov2D[idx]
        color = color[idx]
        opacity = opacity[idx]

        # print(W, H)

        inv = torch.inverse(cov2D)
        out = torch.zeros((W * H, 3), dtype=torch.float32, device='cuda')
        # point = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij'), dim=-1).to('cuda').reshape(-1, 2)
        # print(point)
        dis1 = means2D.view(1, -1, 2) - self.point
        N = means2D.shape[0]

        power = -0.5 * (dis1.view(-1, N, 1, 2) @ inv @ dis1.view(-1, N, 2, 1)).squeeze()
        power = power.clamp(max = 0)
        alpha = (opacity.view(1, -1) * torch.exp(power)).clamp(max = 0.99)
        # alpha[alpha < 1./255.] = 0
        # alpha = torch.where(alpha < 1/255, torch.zeros_like(alpha), alpha)

        tran = torch.cat((torch.ones((H*W, 1), device='cuda'), 1-alpha), dim=-1)
        tran = torch.cumprod(tran, 1)
        # tran[tran < 0.001] = 0
        # tran = torch.where(tran < 0.001, torch.zeros_like(tran), tran)
        weight = alpha * tran[:, :-1]

        out = (weight @ color).view(W, H, 3) + self.bg * tran[:, -1].view(W, H, 1)
        # out = (out - out.min()) / (out.max() - out.min())
        return {
            "render": out.permute(1, 0, 2)
        }


    def forward(self, camera, pc, **kwargs):
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
        
        if self.USE_PROFILE:
            prof = profiler.record_function
        else:
            prof = contextlib.nullcontext
            
        with prof("projection"):
            mean_ndc, mean_view = projection_ndc(means3D, 
                    viewmatrix=camera.world_view_transform, 
                    projmatrix=camera.projection_matrix)
            # mean_ndc = mean_ndc[in_mask]
            # mean_view = mean_view[in_mask]
            depths = mean_view[:,2]
        
        with prof("build color"):
            color = build_color(means3D=means3D, sh_degree=3, shs=shs, camera=camera)
        
        with prof("build cov3d"):
            S = gen_scaling(scales)
            R = gen_rotation(rotations)
            cov3d = get_covariance_3d(R, S)
            
        with prof("build cov2d"):
            cov2d = get_covariance_2d(
                mean3d=means3D, 
                cov3d=cov3d, 
                w2cam=camera.world_view_transform,
                focal_x=camera.focal_x, 
                focal_y=camera.focal_y, 
                trans=camera.trans
                )

            mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
            mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
            means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
        
        with prof("render"):
            rets = self.render(
                camera = camera, 
                means2D=means2D,
                cov2D=cov2d,
                color=color,
                opacity=opacity, 
                depths=depths,
            )

        return rets

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import contextlib
    renderer = GaussRenderer(width=400, height=300)
    USE_PROFILE = False
    import lib._sh_utils as _sh_utils
    import transformcov

    w,h = 400, 300
    class cam:
        def __init__(self):
            self.image_width = 400
            self.image_height = 300
    camera = cam()
    mean2D = torch.tensor([[100, 50], [120, 50], [110, 70]], dtype=torch.float32, device='cuda')
    depth = torch.tensor([100, 200, 150], dtype=torch.float32, device='cuda')
    cov2D = torch.tensor([[[100, 0], [0, 200]], [[300, 0], [0, 100]], [[300, 0], [0, 100]]], dtype=torch.float32, device='cuda')
    color = torch.tensor([[100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=torch.float32, device='cuda')
    bg = torch.zeros((w,h,3), dtype=torch.float32, device='cuda')
    bg[:,:,2] = 0
    opacity = torch.tensor([0.3, 0.5, 0.4], dtype=torch.float32, device='cuda')

    # out = render(mean2D, depth, cov2D, color, opacity, w, h, bg)
    out = renderer.render(camera, mean2D, depth, cov2D, color, opacity)['render']
    out = (out - out.min()) / (out.max() - out.min())

    # print(out.cpu().numpy().shape)
    plt.imshow(out.cpu().numpy())
    plt.show()
    # plt.imsave('./lib/render/test_render1.png', out.cpu().numpy())
    
