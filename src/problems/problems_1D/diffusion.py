import numpy as np
import torch
from src.base.pinn_1D_core import PINN_1D, dfdx, f
from src.helpers.problem_interface import ProblemInterface1D


class DiffusionProblem1D(ProblemInterface1D):
    def __init__(self):
        self.range = [-1.0, 1.0]
        self.eps = 0.1  # Value used across all the testing so far, hence hardcoded

    def get_range(self) -> [float, float]:
        return self.range

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * (1 - torch.exp((x - 1) / self.eps)) / (1 - np.exp(-2 / self.eps)) + x - 1

    def f_inner_loss(self, x: torch.Tensor, pinn: PINN_1D) -> torch.Tensor:
        return -self.eps * dfdx(pinn, x, order=2) + dfdx(pinn, x, order=1) - 1.0

    def compute_loss(self, x: torch.Tensor, pinn: PINN_1D) -> torch.Tensor:
        # Left boundary condition
        boundary_left = x[0].reshape(-1, 1)
        assert (boundary_left == self.range[0]).all().item(), f"First point not on a boundary: {boundary_left}"
        boundary_loss_left = f(pinn, boundary_left)

        # Right boundary condition
        boundary_right = x[-1].reshape(-1, 1)
        assert (boundary_right == self.range[1]).all().item(), f"Last point not on a boundary: {boundary_right}"
        boundary_loss_right = f(pinn, boundary_right)

        interior_loss = self.f_inner_loss(x[1:-1], pinn)

        final_loss = interior_loss.pow(2).mean() + boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean()

        return final_loss
