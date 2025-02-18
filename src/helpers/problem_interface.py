from abc import ABC, abstractmethod

import torch
from src.base.pinn_1D_core import PINN_1D
from src.base.pinn_2D_core import PINN_2D


class ProblemInterface1D(ABC):
    @abstractmethod
    def __init__(self):
        """
        Constructor used by factory method
        """
        pass

    @abstractmethod
    def get_range(self) -> [float, float]:
        """
        :return: range of x (omega)
        """
        pass

    @abstractmethod
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to be used for model validation after the whole PINN learning process is complete
        :param x: tensor with x values that need calculation
        :return: exact solution value
        """
        pass

    @abstractmethod
    def f_inner_loss(self, x: torch.Tensor, pinn: PINN_1D) -> torch.Tensor:
        """
        Calculation of loss function for points that are not on the boundary
        :param x: list of x locations of points
        :param pinn: pinn approximator
        :return: loss function values at given points
        """
        pass

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, pinn: PINN_1D) -> torch.Tensor:
        """
        Calculate final loss for all x elements
        Check the final line for the formula
        :param x: list of x locations of points. Exactly 1 point on each of the boundaries is required
        :param pinn: pinn approximator
        :return: value of the loss function (not just a sum ;) )
        """
        pass


class ProblemInterface2D(ABC):
    @abstractmethod
    def __init__(self):
        """
        Constructor used by factory method
        """
        pass

    @abstractmethod
    def get_range(self) -> (torch.Tensor, torch.Tensor):
        """
        :return: range of x (omega)
        """
        pass

    @abstractmethod
    def exact_solution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Method to be used for model validation after the whole PINN learning process is complete
        :param x: tensor with x values that need calculation
        :param y: tensor with y values that need calculation
        :return: exact solution value
        """
        pass

    @abstractmethod
    def f_inner_loss(self, x: torch.Tensor, y: torch.Tensor, pinn: PINN_2D) -> torch.Tensor:
        """
        Calculation of loss function for points that are not on the boundary
        :param x: list of x locations of points
        :param y: list of y locations of points
        :param pinn: pinn approximator
        :return: loss function values at given points
        """
        pass

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, pinn: PINN_2D) -> torch.Tensor:
        """
        Calculate final loss for all x elements
        Check the final line for the formula
        :param x: list of x locations of points
        :param y: list of y locations of points
        :param pinn: pinn approximator
        :return: value of the loss function
        """
        pass
