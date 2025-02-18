from typing import Callable

import torch
from src.adaptations.adaptations_2D.adaptation_interface import AdaptationInterface2D
from src.helpers.separate_boundary_points_2D import separate_boundary_points_2D


class DensitySamplingAdaptation2D(AdaptationInterface2D):
    def __init__(
        self,
    ):
        self.device = None
        self.x_bkt_range = None
        self.y_bkt_range = None
        self.points_count_per_bucket = None
        self.boundary_x = None
        self.boundary_y = None

        self.buckets_number_range = 16
        super().__init__()

    def set_problem_details(
        self,
        x_range: torch.Tensor,
        y_range: torch.Tensor,
        base_points_x: torch.Tensor,
        base_points_y: torch.Tensor,
        max_number_of_points: int,
    ):
        self.device = base_points_x.device
        self.x_bkt_range = (
            torch.linspace(x_range[0], x_range[1], self.buckets_number_range + 1).requires_grad_(True).to(self.device)
        )
        self.y_bkt_range = (
            torch.linspace(y_range[0], y_range[1], self.buckets_number_range + 1).requires_grad_(True).to(self.device)
        )
        self.buckets_number_range = self.buckets_number_range
        self.points_count_per_bucket = None

        inner_x, inner_y, self.boundary_x, self.boundary_y = separate_boundary_points_2D(
            base_points_x, base_points_y, x_range, y_range
        )

        self.max_number_of_points = max_number_of_points
        self.max_number_of_interior_points = max_number_of_points - list(self.boundary_x.shape)[0]

    def refine(self, loss_function: Callable, old_x: torch.Tensor, old_y: torch.Tensor):
        elements_loss = self.get_elements_loss(loss_function)  # 2D tensor with loss values for each of the buckets
        self.points_count_per_bucket = self.get_points_per_bucket(elements_loss)
        new_interior_points_x, new_interior_points_y = self.sample_in_buckets()
        new_points_x = torch.cat((new_interior_points_x, self.boundary_x))
        new_points_y = torch.cat((new_interior_points_y, self.boundary_y))
        if (
            list(new_points_x.shape)[0] != self.max_number_of_points
            or list(new_points_y.shape)[0] != self.max_number_of_points
        ):
            raise ValueError(
                f"Invalid number of points. "
                f"Expected: {self.max_number_of_points}, Received: {list(new_points_x.shape)[0]} for x"
                f", and {list(new_points_y.shape)[0]} for y"
            )
        return new_points_x, new_points_y

    def get_points_per_bucket(self, elements_loss: torch.Tensor):
        points_into_buckets = torch.multinomial(
            elements_loss.flatten(), num_samples=self.max_number_of_interior_points, replacement=True
        )
        return torch.bincount(points_into_buckets, minlength=self.buckets_number_range**2).reshape(
            [self.buckets_number_range, self.buckets_number_range]
        )

    def get_elements_loss(self, loss_fun: Callable, steps=10):
        loss_values = []
        for x1, x2 in zip(self.x_bkt_range[:-1], self.x_bkt_range[1:]):
            loss_values.append([])
            int_x = torch.linspace(x1.item(), x2.item(), steps).requires_grad_(True).to(self.device)
            for y1, y2 in zip(self.y_bkt_range[:-1], self.y_bkt_range[1:]):
                int_y = torch.linspace(y1.item(), y2.item(), steps).requires_grad_(True).to(self.device)
                grid_x, grid_y = torch.meshgrid(int_x, int_y, indexing="ij")
                linear_grid_x = torch.reshape(grid_x, [-1]).reshape(-1, 1).to(self.device)
                linear_grid_y = torch.reshape(grid_y, [-1]).reshape(-1, 1).to(self.device)

                linear_grid_z = loss_fun(x=linear_grid_x, y=linear_grid_y) ** 2
                grid_z = torch.reshape(linear_grid_z, shape=(steps, steps))

                el_loss = torch.trapezoid(torch.trapezoid(grid_z, int_y, dim=0), int_x, dim=0) / ((x2 - x1) * (y2 - y1))
                loss_values[-1].append(el_loss)

        return torch.tensor(loss_values, device=self.device).detach().clone().requires_grad_(True)

    def sample_in_buckets(self):
        collocation_points_x = torch.tensor([], dtype=torch.float, device=self.device)
        collocation_points_y = torch.tensor([], dtype=torch.float, device=self.device)
        x_bkt_size = self.x_bkt_range[1] - self.x_bkt_range[0]
        y_bkt_size = self.y_bkt_range[1] - self.y_bkt_range[0]

        for i in range(self.buckets_number_range):
            for j in range(self.buckets_number_range):
                new_x = (
                    torch.rand(self.points_count_per_bucket[i][j], device=self.device) * x_bkt_size
                    + self.x_bkt_range[i]
                )
                new_y = (
                    torch.rand(self.points_count_per_bucket[i][j], device=self.device) * y_bkt_size
                    + self.y_bkt_range[j]
                )

                collocation_points_x = torch.cat((collocation_points_x, new_x))
                collocation_points_y = torch.cat((collocation_points_y, new_y))

        collocation_points_x = collocation_points_x
        collocation_points_y = collocation_points_y

        return (
            collocation_points_x.reshape((-1, 1)).detach().clone().requires_grad_(True),
            collocation_points_y.reshape((-1, 1)).detach().clone().requires_grad_(True),
        )

    def __str__(self) -> str:
        return "density_sampling"
