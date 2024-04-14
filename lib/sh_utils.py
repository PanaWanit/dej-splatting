import torch

coeff0 = 0.28209479177387814
coeff1 = 0.4886025119029199
coeff2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
coeff3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]

def get_sh_color(deg, mean, cam_pos, sh):
    print(mean.shape, cam_pos.shape, sh.shape)
    mean, cam_pos, sh = mean.cuda(), cam_pos.cuda(), sh.cuda()
    dir = cam_pos - mean
    dir = dir / torch.norm(dir, dim=1, keepdim=True)

    sh_color = coeff0 * sh[..., 0]


    if deg > 0:
        x, y, z = dir[:, 0:1], dir[:, 1:2], dir[:, 2:3]
        sh_color = (sh_color -
                coeff1 * y * sh[..., 1] +
                coeff1 * z * sh[..., 2] -
                coeff1 * x * sh[..., 3])

        if deg > 1:
            xy , xz, yz = x * y, x * z, y * z
            xx, yy, zz = x * x, y * y, z * z
            sh_color = (sh_color
            + coeff2[0] * sh[..., 4] * xy
            + coeff2[1] * sh[..., 5] * yz
            + coeff2[2] * sh[..., 6] * (2*zz - xx - yy)
            + coeff2[3] * sh[..., 7] * xz
            + coeff2[4] * sh[..., 8] * (xx - yy))

            if deg > 2:
                sh_color = (sh_color + coeff3[0] * sh[..., 9] * y * (3*xx - yy)
                + coeff3[1] * sh[..., 10] * xy * z
                + coeff3[2] * sh[..., 11] * y * (4*zz - xx - yy)
                + coeff3[3] * sh[..., 12] * z * (2*zz - 3*xx - 3*yy)
                + coeff3[4] * sh[..., 13] * x * (4*zz - xx - yy)
                + coeff3[5] * sh[..., 14] * z * (xx - yy)
                + coeff3[6] * sh[..., 15] * x * (xx - 3*yy))
    
    sh_color += 0.5
    return torch.clip(sh_color, 0)

def RGB2SH(rgb):
    return (rgb - 0.5) / coeff0

def SH2RGB(sh):
    return sh * coeff0 + 0.5

if __name__ == '__main__':
    mean = torch.tensor([[0, 0, 0], [1,-2,6]], dtype=torch.float32)
    cam_pos = torch.tensor([[2,4,9], [-6,2,3]], dtype=torch.float32)

    sh = torch.tensor([[1, 1, 2, 3,5,-2,4,-6,2], [2, 1, 2, 3,1,2,4,1,-1], [1, 1, 2, 3,1,2,4,1,-1]], dtype=torch.float32)

    # mean = np.array([[0, 0, 0], [1,-2,6]], dtype=np.float32)
    # cam_pos = np.array([[2,4,9], [-6,2,3]], dtype=np.float32)

    # sh = np.array([[1, 1, 2, 3], [1, 1, 2, 3]], dtype=np.float32)

    print(get_sh_color(2, mean, cam_pos, sh))