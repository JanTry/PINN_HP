from typing import Callable

import torch
from src.adaptations.adaptations_2D.adaptation_interface import AdaptationInterface2D


class NoAdaptation2D(AdaptationInterface2D):
    def refine(self, loss_function: Callable, old_x: torch.Tensor, old_y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        self.validate_problem_details()
        return old_x, old_y

    def __str__(self) -> str:
        return "no_adaptation"
