
def rasterizer(mean, depth, color, opacity, bg_color, W, H):
    # mean: (N, 2) = x, y
    # depth: (N, 1) = z
    # color: (N, H, W, 3) = RGB
    # opacity: (N, 1) = opacity
    # bg_color: (3)
    # W, H: int