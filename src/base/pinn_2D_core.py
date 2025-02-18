# -*- coding: utf-8 -*-
# Slightly updated from https://github.com/pmaczuga/pinn-adaptivity/blob/master/src/pinn_core.py
from typing import Callable

import torch
from torch import nn


class PINN_2D(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """

    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh(), pinning: bool = False):
        super().__init__()

        self.pinning = pinning

        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)])
        self.act = act

    def forward(self, x, y):
        x_stack = torch.cat([x, y], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        # if requested pin the boundary conditions
        # using a surrogate model: (x - 0) * (x - L) * NN(x)
        if self.pinning:
            logits *= (x - torch.min(x)) * (x - torch.max(x)) * (y - torch.min(y)) * (y - torch.max(y))

        return logits


def f(pinn: PINN_2D, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return pinn(x, y)


def df(output_tensor: torch.Tensor, input_tensor: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output_tensor
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input_tensor,
            grad_outputs=torch.ones_like(input_tensor),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


def dfdx(pinn: PINN_2D, x: torch.Tensor, y: torch.Tensor, order: int = 1):
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(pinn, x, y)
    return df(f_value, x, order=order)


def dfdy(pinn: PINN_2D, x: torch.Tensor, y: torch.Tensor, order: int = 1):
    f_value = f(pinn, x, y)
    return df(f_value, y, order=order)


def train_model(
    nn_approximator: PINN_2D,
    loss_fn: Callable,
    device,
    learning_rate: float = 0.01,
    max_epochs: int = 1_000,
    optimizer=None,
) -> torch.Tensor:
    if optimizer is None:
        optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)

    convergence_data = torch.empty(max_epochs, device=device)

    for epoch in range(max_epochs):
        loss = loss_fn(pinn=nn_approximator)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        convergence_data[epoch] = loss.detach()

    return convergence_data
