import os

import matplotlib
import matplotlib.pyplot as plt
import src.params.params_1D as params
import torch
from matplotlib import rc
from src.adaptations.adaptations_1D.adaptation_interface import AdaptationInterface1D
from src.base.pinn_1D_core import f
from src.enums.problems import Problems1D
from src.helpers.factories import problem_factory_1D
from src.helpers.mesh_1D import get_mesh_1D

N_ITERS_FILE = "n_iters.pt"
TIME_FILE = "exec_time.pt"
CONVERGENCE_FILE = "convergence.pt"
PINN_FILE = "pinn.pt"
POINT_DATA_FILE = "point_data.pt"


def plot_specific_run_1D(
    run_id: int,
    problem_type: Problems1D,
    adaptation: AdaptationInterface1D,
    tolerance: float = params.TOLERANCE,
    learning_rate: float = params.LEARNING_RATE,
    layers: int = params.LAYERS,
    neurons: int = params.NEURONS,
    epochs: int = params.NUMBER_EPOCHS,
    max_points: int = params.NUM_MAX_POINTS,
    plot_training_points: bool = False,
):
    device = "cpu"
    matplotlib.rcParams.update({"font.size": 15})

    path = os.path.join(
        "results_1D",
        problem_type.value,
        str(adaptation),
        f"L{layers}_N{neurons}_" f"P{max_points}_E{epochs}",
        f"LR{learning_rate}_TOL{tolerance}",
        str(run_id),
    )

    problem = problem_factory_1D(problem_type)
    x_range = problem.get_range()

    plt.rcParams["figure.dpi"] = 150
    rc("animation", html="html5")

    nn_approximator = torch.load(os.path.join(path, PINN_FILE), weights_only=False)
    convergence_data = torch.load(os.path.join(path, CONVERGENCE_FILE), weights_only=False)
    n_iters = torch.load(os.path.join(path, N_ITERS_FILE), weights_only=False)
    exec_time = torch.load(os.path.join(path, TIME_FILE), weights_only=False)

    os.makedirs(os.path.join(path, "plots", "iterations"), exist_ok=True)

    # Plot points for each iteration
    if plot_training_points:
        point_data = torch.load(os.path.join(path, POINT_DATA_FILE), weights_only=False)
        for i, p in enumerate(point_data):
            fig, ax = plt.subplots()
            ax.scatter(*(p.transpose(0, 1).cpu().detach().numpy()), s=1)
            ax.set_title(f"Points distribution iteration {i}")
            fig.savefig(os.path.join(path, "plots", "iterations", f"iteration_{i}"))
            plt.close(fig)

    # Plot the solution in a "dense" mesh
    n_x = torch.linspace(x_range[0], x_range[1], steps=1000 + 2, requires_grad=True, device=device)[1:-1].reshape(-1)
    n_x, _ = get_mesh_1D(n_x, x_range[0], x_range[1])

    y = f(nn_approximator, n_x)
    fig, ax = plt.subplots()
    ax.plot(n_x.cpu().detach().numpy(), y.cpu().detach().numpy())
    ax.scatter(n_x.cpu().detach().numpy(), y.cpu().detach().numpy(), s=1)
    ax.set_title(f"PINN solution,\n time = {exec_time:.2f}s, iterations = {n_iters}")  # noqa: E231
    fig.savefig(os.path.join(path, "plots", "pinn_solution"))
    plt.close(fig)

    # Plot exact solution
    exact_y = problem.exact_solution(n_x)
    fig, ax = plt.subplots()
    ax.plot(n_x.cpu().detach().numpy(), exact_y.cpu().detach().numpy())
    ax.set_title(f"Exact solution,\n time = {exec_time:.2f}s, iterations = {n_iters}")  # noqa: E231
    fig.savefig(os.path.join(path, "plots", "exact_solution"))
    plt.close(fig)

    # PINN and exact solutions on one plot
    fig, ax = plt.subplots()
    ax.plot(n_x.cpu().detach().numpy(), exact_y.cpu().detach().numpy(), label="Exact")
    ax.plot(n_x.cpu().detach().numpy(), y.cpu().detach().numpy(), "--", label="PINN")
    ax.legend()
    ax.set_title(f"PINN and exact solutions,\n time = {exec_time:.2f}s, iterations = {n_iters}")  # noqa: E231
    fig.savefig(os.path.join(path, "plots", "solutions"))
    plt.close(fig)

    # Plot error
    error = y - exact_y
    rc("ytick", labelsize=12)
    fig, ax = plt.subplots()
    ax.plot(n_x.cpu().detach().numpy(), error.cpu().detach().numpy())
    ax.set_title(f"Error: NN_u - exact_solution,\n time = {exec_time:.2f}s, iterations = {n_iters}")  # noqa: E231
    fig.savefig(os.path.join(path, "plots", "error"))
    plt.close(fig)

    # Plot convergence
    rc("ytick", labelsize=15)
    # Draw the convergence plot
    fig, ax = plt.subplots()
    ax.semilogy(convergence_data.cpu().detach().numpy())
    ax.set_title(f"Convergence,\n time = {exec_time:.2f}s, iterations = {n_iters}")  # noqa: E231
    fig.savefig(os.path.join(path, "plots", "convergence"))
    plt.close(fig)
