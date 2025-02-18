from typing import Callable

import torch


def exit_criterion_1D(base_x: torch.Tensor, loss_fun: Callable, tol: float):
    x = base_x.detach().clone().requires_grad_(True)

    for x1, x2 in zip(x[:-1], x[1:]):
        int_x = torch.linspace(x1.item(), x2.item(), 20).requires_grad_(True).reshape(-1, 1).to(x.device)
        int_y = loss_fun(x=int_x) ** 2
        el_loss = torch.trapezoid(int_y, int_x, dim=0) / (x2 - x1)
        if el_loss > tol:
            return False

    return True


def exit_criterion_2D(base_x: torch.Tensor, base_y: torch.Tensor, loss_fun: Callable, tol: float):
    steps = 10
    x = base_x.detach().clone().requires_grad_(True)
    y = base_y.detach().clone().requires_grad_(True)

    for x1, x2 in zip(x[:-1], x[1:]):
        int_x = torch.linspace(x1.item(), x2.item(), steps).requires_grad_(True).to(x.device)
        for y1, y2 in zip(y[:-1], y[1:]):
            int_y = torch.linspace(y1.item(), y2.item(), steps).requires_grad_(True).to(y.device)
            grid_x, grid_y = torch.meshgrid(int_x, int_y, indexing="ij")
            linear_grid_x = torch.reshape(grid_x, [-1]).reshape(-1, 1).to(x.device)
            linear_grid_y = torch.reshape(grid_y, [-1]).reshape(-1, 1).to(x.device)

            linear_grid_z = loss_fun(x=linear_grid_x, y=linear_grid_y) ** 2
            grid_z = torch.reshape(linear_grid_z, shape=(steps, steps))

            el_loss = torch.trapezoid(torch.trapezoid(grid_z, int_y, dim=0), int_x, dim=0) / ((x2 - x1) * (y2 - y1))

            if el_loss > tol:
                return False

    return True
