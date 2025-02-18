import torch
from src.base.pinn_1D_core import PINN_1D, dfdx, f
from src.helpers.problem_interface import ProblemInterface1D


class Tan01Problem1D(ProblemInterface1D):
    def __init__(self):
        self.range = [0.0, torch.pi / 2.0]

    def get_range(self) -> [float, float]:
        return self.range

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tan(x - 0.1)

    def f_inner_loss(self, x: torch.Tensor, pinn: PINN_1D) -> torch.Tensor:
        return dfdx(pinn, x, order=2) + 2 * torch.sin(0.1 - x) / torch.pow(torch.cos(0.1 - x), 3)

    def compute_loss(self, x: torch.Tensor, pinn: PINN_1D) -> torch.Tensor:
        # Left boundary condition
        boundary_left = x[0].reshape(-1, 1)
        assert (boundary_left == self.range[0]).all().item(), f"First point not on a boundary: {boundary_left}"
        boundary_loss_left = f(pinn, boundary_left) + 0.10033467208545055

        # Right boundary condition
        boundary_right = x[-1].reshape(-1, 1)
        assert (boundary_right == self.range[1]).all().item(), f"Last point not on a boundary: {boundary_right}"
        boundary_loss_right = dfdx(pinn, boundary_right, order=1) - 100.33400105968446

        interior_loss = self.f_inner_loss(x[1:-1], pinn)

        final_loss = interior_loss.pow(2).mean() + boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean()

        return final_loss
