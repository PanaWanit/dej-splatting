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
    dir = cam_pos - mean
    dir = dir / torch.norm(dir)

    print(dir)

    sh_color = coeff0 * sh[..., 0]
    print(sh_color)

    print()
    if deg >= 1:
        sh_color += coeff1 * sh[1] * dir[..., 1]
        sh_color += coeff1 * sh[2] * dir[..., 2]
        sh_color += coeff1 * sh[3] * dir[..., 0]

mean = torch.tensor([[0, 0, 0], [1,-2,6]], dtype=torch.float32)
cam_pos = torch.tensor([[2,4,9], [-6,2,3]], dtype=torch.float32)

sh = torch.tensor([[1, 1, 2, 3], [1, 1, 2, 3]], dtype=torch.float32)

get_sh_color(1, mean, cam_pos, sh)