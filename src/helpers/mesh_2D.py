import math

import torch


def get_mesh_2D(x_domain, y_domain, n_points, device=torch.device("cpu"), requires_grad=True):
    mesh_size = math.isqrt(n_points)
    x_linspace = torch.linspace(x_domain[0], x_domain[1], mesh_size)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], mesh_size)
    x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    return x_grid, y_grid
