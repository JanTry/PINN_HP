# -*- coding: utf-8 -*-
# Slightly updated from https://github.com/pmaczuga/pinn-adaptivity/blob/master/src/pinn_core.py


from typing import Callable

import torch
from torch import nn


class PINN_1D(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """

    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh(), pinning: bool = False):
        super().__init__()

        self.pinning = pinning

        self.layer_in = nn.Linear(1, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)])
        self.act = act

    def forward(self, x):
        x_stack = torch.cat([x], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        # if requested pin the boundary conditions
        # using a surrogate model: (x - 0) * (x - L) * NN(x)
        if self.pinning:
            logits *= (x - x[0]) * (x - x[-1])

        return logits


def f(nn_approximator: PINN_1D, x: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return nn_approximator(x)


def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


def dfdx(nn_approximator: PINN_1D, x: torch.Tensor, order: int = 1):
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(nn_approximator, x)
    return df(f_value, x, order=order)


def train_model(
    nn_approximator: PINN_1D,
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

        # if epoch % 1000 == 0:
        # print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

    # print(f"Final Epoch: - Loss: {float(loss):>7f}")

    return convergence_data
