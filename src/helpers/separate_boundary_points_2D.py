import torch


def separate_boundary_points_2D(x: torch.Tensor, y: torch.Tensor, x_range: torch.Tensor, y_range: torch.Tensor):
    """
    :param x: x-axis locations of all points
    :param y: y-axis locations of all points
    :param x_range: range of x
    :param y_range: range of y
    :return: x and y for interior points , x and y for boundary points
    """
    on_boundary = torch.add(torch.isin(x, x_range), torch.isin(y, y_range))

    x_boundary = torch.masked_select(x, on_boundary).reshape((-1, 1))
    y_boundary = torch.masked_select(y, on_boundary).reshape((-1, 1))
    x_interior = torch.masked_select(x, ~on_boundary).reshape((-1, 1))
    y_interior = torch.masked_select(y, ~on_boundary).reshape((-1, 1))

    return x_interior, y_interior, x_boundary, y_boundary
