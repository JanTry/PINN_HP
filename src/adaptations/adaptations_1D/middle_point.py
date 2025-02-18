from typing import Callable

import src.params.params_1D as params
import torch
from src.adaptations.adaptations_1D.adaptation_interface import AdaptationInterface1D


class MiddlePointAdaptation1D(AdaptationInterface1D):
    """
    The old version of the adaptation.
    """

    def refine(self, loss_function: Callable, old_x: torch.Tensor):
        self.validate_problem_details()
        x = self.base_points.detach().clone().requires_grad_(True)
        n_points = x.numel()
        refined = True

        while n_points < self.max_number_of_points and refined:
            refined = False
            new_points = []
            for x1, x2 in zip(x[:-1], x[1:]):
                int_x = torch.linspace(x1.item(), x2.item(), 20).requires_grad_(True).reshape(-1, 1).to(x.device)
                int_y = loss_function(x=int_x) ** 2
                el_loss = torch.trapezoid(int_y, int_x, dim=0) / (x2 - x1)
                if el_loss > params.TOLERANCE:
                    refined = True
                    new_points.append((x1 + x2) / 2.0)
            x = torch.cat((x, torch.tensor(new_points, device=x.device))).sort()[0]
            n_points = x.numel()
        return x.reshape(-1, 1).detach().clone().requires_grad_(True)

    def __str__(self) -> str:
        return "middle_point"
