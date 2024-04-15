import pdb
import torch
import torch.nn as nn
import math
from einops import reduce

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R



def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)



def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance
    # symm = strip_symmetric(actual_covariance)
    # return symm



def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
	# and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	# Additionally considers aspect / scaling of viewport.
	# Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]


def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


from lib.sh_utils import eval_sh
import torch.autograd.profiler as profiler
USE_PROFILE = False
import contextlib

class GaussRenderer(nn.Module):
    """
    A gaussian splatting renderer

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """

    def __init__(self, active_sh_degree=3, white_bkgd=False, width=256, height=256, **kwargs):
        super(GaussRenderer, self).__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = False
        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy'), dim=-1).to('cuda')
        self.W = width
        self.H = height
            
        
    
    def build_color(self, means3D, shs, camera):
        rays_o = camera.camera_center
        rays_d = means3D - rays_o
        color = eval_sh(self.active_sh_degree, shs.permute(0,2,1), rays_d)
        color = (color + 0.5).clip(min=0.0)
        return color
    
    def render(self, camera, means2D, cov2d, color, opacity, depths):
        # camera.image_width, camera.image_height = 256, 256
        # means2D = torch.tensor([[177.3733, 179.4178]], device='cuda')
        # cov2d = torch.tensor([[[14.4300,  0.2838], [0.2838, 14.4528]]], device='cuda')
        # color = torch.tensor([[0.3412, 0.2157, 0.1569]], device='cuda')
        # opacity = torch.tensor([0.1], device='cuda')
        # depths = torch.tensor([1.8539], device='cuda')
        camera.image_width, camera.image_height = self.W, self.H
        # print(camera.image_width, camera.image_height)
        # print(means2D.shape, cov2d.shape, color.shape, opacity.shape, depths.shape)

        # means2D = means2D[0:5, ...]
        # cov2d = cov2d[0:5, ...]
        # color = color[0:5, ...]
        # opacity = opacity[0:5, ...]
        # depths = depths[0:5, ...]
        print(self.H, self.W)
        print(torch.eye(2,2).to('cuda').expand(means2D.size(0),2,2).shape, cov2d.shape)
        # cov2d = torch.eye(2,2).to('cuda').expand(means2D.size(0),2,2)

        # radii = get_radius(cov2d)
        # print(radii.shape, means2D.shape)
        # rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
        
        self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to('cuda')
        self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')
        self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')

        TILE_SIZE = 64
        for h in range(0, camera.image_height, TILE_SIZE):
            for w in range(0, camera.image_width, TILE_SIZE):
                # check if the rectangle penetrate the tile
                # over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                # over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                # in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                # in_mask = torch.ones_like(in_mask) # for now
                
                # if not in_mask.sum() > 0:
                #     continue

                # P = in_mask.sum()
                tile_coord = self.pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
                sorted_depths, index = torch.sort(depths)
                sorted_means2D = means2D[index]
                sorted_cov2d = cov2d[index] # P 2 2
                sorted_conic = sorted_cov2d.inverse() # inverse of variance
                sorted_opacity = opacity[index]
                sorted_color = color[index]
                dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
                
                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                    + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
                
                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)
                tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
                tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
                # print('tile', tile_color.shape, tile_depth.shape, acc_alpha.shape)
                self.render_color[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_color.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_depth[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)
        # print('rendered')
        return {
            "render": self.render_color,
            "depth": self.render_depth,
            "alpha": self.render_alpha,
            # "visiility_filter": radii > 0,
            # "radii": radii
        }


    def forward(self, camera, pc, **kwargs):
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
        
        if USE_PROFILE:
            prof = profiler.record_function
        else:
            prof = contextlib.nullcontext
            
        with prof("projection"):
            mean_ndc, mean_view, in_mask = projection_ndc(means3D, 
                    viewmatrix=camera.world_view_transform, 
                    projmatrix=camera.projection_matrix)
            mean_ndc = mean_ndc[in_mask]
            mean_view = mean_view[in_mask]
            depths = mean_view[:,2]
        
        with prof("build color"):
            # print('cccc', shs.shape)
            color = self.build_color(means3D=means3D, shs=shs.permute(0, 2, 1), camera=camera)
        # color = None
        
        with prof("build cov3d"):
            cov3d = build_covariance_3d(scales, rotations)
            
        with prof("build cov2d"):
            cov2d = build_covariance_2d(
                mean3d=means3D, 
                cov3d=cov3d, 
                viewmatrix=camera.world_view_transform,
                fov_x=camera.FoVx, 
                fov_y=camera.FoVy, 
                focal_x=camera.focal_x, 
                focal_y=camera.focal_y)

            mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
            mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
            means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
        
        with prof("render"):
            rets = self.render(
                camera = camera, 
                means2D=means2D,
                cov2d=cov2d,
                color=color,
                opacity=opacity, 
                depths=depths,
            )

        return rets
