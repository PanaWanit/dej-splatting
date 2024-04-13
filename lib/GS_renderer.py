import torch
MAX_OPACITY = torch.tensor([0.99], dtype=torch.float32, device='cuda')

def render(mean2D, depth, cov2D, color, opacity, W, H, bg):
    idx = torch.argsort(depth)
    mean2D = mean2D[idx]
    cov2D = cov2D[idx]
    color = color[idx]
    opacity = opacity[idx]

    inv = torch.inverse(cov2D)
    out = torch.zeros((W * H, 3), dtype=torch.float32, device='cuda')
    point = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij'), dim=-1).to('cuda').reshape(-1, 2)
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    w,h = 400, 300
    mean2D = torch.tensor([[100, 50], [120, 50], [110, 70]], dtype=torch.float32, device='cuda')
    depth = torch.tensor([100, 200, 150], dtype=torch.float32, device='cuda')
    cov2D = torch.tensor([[[100, 0], [0, 200]], [[300, 0], [0, 100]], [[300, 0], [0, 100]]], dtype=torch.float32, device='cuda')
    color = torch.tensor([[100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=torch.float32, device='cuda')
    bg = torch.zeros((w,h,3), dtype=torch.float32, device='cuda')
    bg[:,:,2] = 0
    opacity = torch.tensor([0.3, 0.5, 0.4], dtype=torch.float32, device='cuda')

    out = render(mean2D, depth, cov2D, color, opacity, w, h, bg)
    out = (out - out.min()) / (out.max() - out.min())
    # plt.imshow(out.cpu().numpy())
    # plt.show()
    print(out.cpu().numpy().shape)
    plt.imshow(out.cpu().numpy())
    plt.show()
    # plt.imsave('./lib/render/test_render1.png', out.cpu().numpy())
    
