import torch
import matplotlib.pyplot as plt
MAX_OPACIRY = torch.tensor([1.0], dtype=torch.float32, device='cuda')

# todo:
# - fix color range [ (out - out.min()) / (out.max() - out.min()) is not what should be done ]
# - sort gs by depth

def render(mean2D, depth, cov2D, color, opacity, W, H, bg):
    idx = torch.argsort(depth)
    mean2D = mean2D[idx]
    cov2D = cov2D[idx]
    color = color[idx]
    opacity = opacity[idx]

    mi, ma = 1e9, -1e9
    inv = torch.inverse(cov2D)
    out = torch.zeros((W, H, 3), dtype=torch.float32, device='cuda')
    N = mean2D.shape[0]

    for u in range(W):
        for v in range(H):
            point = torch.tensor([u, v], dtype=torch.float32, device='cuda')
            dis1 = mean2D - point
            # print(dis1)
            # print(point)
            # print(inv)
            # print(dis1.shape, inv.shape, N)
            power = -0.5 * (dis1.view(N, 1, -1) @ inv @ dis1.view(N, -1, 1)).squeeze()
            mi = min(mi, power.min().cpu().numpy())
            ma = max(ma, power.max().cpu().numpy())
            power[power > 0] = 0
            # print(power)
            alpha = torch.min(MAX_OPACIRY, opacity * torch.exp(power))
            tran = torch.cat((torch.tensor([1], device='cuda'), 1-alpha))
            tran = torch.cumprod(tran, 0)
            weight = alpha * tran[:-1]
            if (u,v) in [(110, 50), (210, 50)]:
                print('a', alpha)
                print('t', tran)
                print('w', weight)
                print('w @ c', weight @ color)
                print()
            # print(weight.shape, color.shape)
            out[u, v, :] = weight @ color + bg[u, v, :] * (tran[-1])
    return out, mi, ma

def render2(mean2D, cov2D, color, opacity, W, H, bg):
    inv = torch.inverse(cov2D)
    out = torch.zeros((W, H, 3), dtype=torch.float32, device='cuda')
    point = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H)), dim=-1).reshape(-1, 2)

w,h = 400, 300
mean2D = torch.tensor([[100, 50], [120, 50]], dtype=torch.float32, device='cuda')
depth = torch.tensor([300, 200], dtype=torch.float32, device='cuda')
cov2D = torch.tensor([[[100, 0], [0, 200]], [[300, 0], [0, 100]]], dtype=torch.float32, device='cuda')
color = torch.tensor([[100, 0, 0], [0, 100, 0]], dtype=torch.float32, device='cuda')
bg = torch.zeros((w,h,3), dtype=torch.float32, device='cuda')
bg[:,:,2] = 0
opacity = torch.tensor([0.8, 0.8], dtype=torch.float32, device='cuda')
out, mi, ma = render(mean2D, depth, cov2D, color, opacity, w, h, bg)
out = out.permute(1, 0, 2)

out = (out - out.min()) / (out.max() - out.min())

print(mi, ma)
# plt.imshow(out.cpu().numpy())
# plt.show()
plt.imsave('./lib/render/test_fn.png', out.cpu().numpy())