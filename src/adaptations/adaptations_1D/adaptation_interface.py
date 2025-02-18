from abc import ABC, abstractmethod
from typing import Callable

import torch


class AdaptationInterface1D(ABC):
    def __init__(self):
        # Problem details:
        self.x_range: tuple[float, float] | None = None
        self.base_points: torch.Tensor | None = None
        self.max_number_of_points: int | None = None

    @abstractmethod
    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        pass

    def set_problem_details(
        self,
        x_range: tuple[float, float],
        base_points: torch.Tensor,
        max_number_of_points: int,
    ) -> None:
        self.x_range = x_range
        self.base_points = base_points
        self.max_number_of_points = max_number_of_points

    def validate_problem_details(self) -> None:
        if self.x_range is None or self.base_points is None or self.max_number_of_points is None:
            raise ValueError("Problem details not set.")
