from typing import Callable

import src.params.params_1D as params
import torch
from src.adaptations.adaptations_1D.adaptation_interface import AdaptationInterface1D


class DensitySamplingAdaptation1D(AdaptationInterface1D):
    def refine(self, loss_function: Callable, old_x: torch.Tensor):
        self.validate_problem_details()
        mesh_element_points = self.__prepare_mesh_points(loss_function=loss_function)
        mesh_elements_loss = self.__get_elements_loss(mesh_points=mesh_element_points, loss_function=loss_function)
        bucket_indices = mesh_elements_loss.multinomial(num_samples=self.max_number_of_points - 2, replacement=True)
        points_per_element = torch.bincount(bucket_indices, minlength=torch.numel(mesh_elements_loss))
        return self.__get_points(mesh_element_points, points_per_element)

    def __prepare_mesh_points(
        self,
        loss_function: Callable,
        max_level: int = params.MAX_DEPTH,
        tol: float = params.TOLERANCE,
    ):
        self.validate_problem_details()
        x = self.base_points.detach().clone().requires_grad_(True)
        refined = True
        adaptation_level = 0

        while adaptation_level < max_level and refined:
            adaptation_level += 1
            refined = False
            new_points = []
            for x1, x2 in zip(x[:-1], x[1:]):
                int_x = torch.linspace(x1.item(), x2.item(), 20).requires_grad_(True).reshape(-1, 1).to(x.device)
                int_y = loss_function(x=int_x) ** 2
                el_loss = torch.trapezoid(int_y, int_x, dim=0) / (x2 - x1)
                if el_loss > tol:
                    refined = True
                    new_points.append((x1 + x2) / 2.0)

            x = torch.cat((x, torch.tensor(new_points, device=x.device))).sort()[0]
        return x.reshape(-1, 1).detach().clone().requires_grad_(True)

    @staticmethod
    def __get_elements_loss(mesh_points, loss_function):
        x = mesh_points.detach().clone().requires_grad_(True)

        loss_values = []

        for x1, x2 in zip(x[:-1], x[1:]):
            int_x = torch.linspace(x1.item(), x2.item(), 20).requires_grad_(True).reshape(-1, 1).to(x.device)
            int_y = loss_function(x=int_x) ** 2
            el_loss = torch.trapezoid(int_y, int_x, dim=0) / (x2 - x1)
            loss_values.append(el_loss)

        return torch.tensor(loss_values, device=mesh_points.device).detach().clone().requires_grad_(True)

    def __get_points(self, mesh_points, points_per_element):
        points = torch.tensor(self.x_range, dtype=torch.float, device=mesh_points.device)

        for i in range(torch.numel(points_per_element)):
            if points_per_element[i] != 0:
                x1, x2 = mesh_points[i], mesh_points[i + 1]
                new_ten = (x2 - x1) * torch.rand(points_per_element[i], device=mesh_points.device) + x1

                points = torch.cat((points, new_ten))
        points = points.sort()[0]
        if list(points.size())[0] != self.max_number_of_points:
            raise RuntimeError(
                f"The number of points is invalid. Expected: {self.max_number_of_points}, \
                actual: {list(points.size())[0]}"
            )
        return points.reshape(-1, 1).detach().clone().requires_grad_(True)

    def __str__(self) -> str:
        return "density_sampling"
