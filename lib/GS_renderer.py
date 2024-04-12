import torch

def render(mean2D, cov2D, color, opacity, W, H, bg):
    inv = torch.inverse(cov2D)
    conic = torch.tensor([inv[0,0], inv[0,1], inv[1,1]], dtype=torch.float32)
    out = torch.zeros((W, H, 3), dtype=torch.float32)
    for w in range(W):
        for h in range(H):
            point = torch.tensor([[w, h]], dtype=torch.float32)
            dis1 = mean2D - point
            power = -0.5 * (inv[0, 0] * dis1[0] * dis1[0] + inv[1, 1] * dis1[1] * dis1[1]) - inv[0, 1] * dis1[0] * dis1[1]
            if power > 0:
                continue
            alpha = torch.min(0.99, opacity[0] * torch.exp(power))
            out[w, h, :] = (1 - alpha) * bg[w, h, :] + alpha * color[0]
    return out

w,h = 256,256
mean2D = torch.tensor([[0, 0]], dtype=torch.float32)
cov2D = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
color = torch.tensor([[[100, 0, 0]]], dtype=torch.float32)
bg = torch.zeros((w,h,3), dtype=torch.float32)
bg[:,:,1] = 255
opacity = torch.tensor([[0.5]], dtype=torch.float32)
out = render(mean2D, cov2D, color, opacity, w, h, bg)