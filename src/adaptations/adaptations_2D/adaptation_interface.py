import math
from abc import ABC, abstractmethod
from typing import Callable

import torch


class AdaptationInterface2D(ABC):
    def __init__(self):
        # Problem details:
        self.x_range: torch.Tensor | None = None
        self.y_range: torch.Tensor | None = None
        self.base_points_x: torch.Tensor | None = None
        self.base_points_y: torch.Tensor | None = None
        self.max_number_of_points: int | None = None
        self.max_number_of_interior_points: int | None = None

    @abstractmethod
    def refine(self, loss_function: Callable, old_x: torch.Tensor, old_y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        pass

    def set_problem_details(
        self,
        x_range: torch.Tensor,
        y_range: torch.Tensor,
        base_points_x: torch.Tensor,
        base_points_y: torch.Tensor,
        max_number_of_points: int,
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.base_points_x = base_points_x
        self.base_points_y = base_points_y
        self.max_number_of_points = max_number_of_points

    def validate_problem_details(self) -> None:
        if (
            self.x_range is None
            or self.y_range is None
            or self.base_points_x is None
            or self.base_points_y is None
            or self.max_number_of_points is None
            or math.isqrt(self.max_number_of_points) ** 2 != self.max_number_of_points
        ):
            raise ValueError("Problem details not set.")
