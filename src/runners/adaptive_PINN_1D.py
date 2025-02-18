import logging
import os
import time
from functools import partial

import src.params.params_1D as params
import torch
from src.adaptations.adaptations_1D.adaptation_interface import AdaptationInterface1D
from src.base.exit_criterion import exit_criterion_1D
from src.base.pinn_1D_core import PINN_1D, f, train_model
from src.enums.problems import Problems1D
from src.helpers.factories import problem_factory_1D
from src.plots.plots_1D.plot_specific_run import CONVERGENCE_FILE, N_ITERS_FILE, PINN_FILE, POINT_DATA_FILE, TIME_FILE


def train_PINN_1D(
    run_id: int,
    adaptation: AdaptationInterface1D,
    problem_type: Problems1D = params.PROBLEM,
    save_training_data: bool = True,
):
    """
    Basic 1D PINN training based on src/params/params_1D.py file

    :param run_id: run identification number. Will overwrite any previous result with the same id and save path params
    :param problem_type: problem enum value. Based on that, a proper class from src/problems will be used
    :param adaptation: adaptation
    :param save_training_data: Either save the mid-run selected points or not (results for the run are always saved)
    :return:
    """
    logging.log(
        logging.DEBUG,
        f"Starting PINN training run {run_id} for: {problem_type.value}, {str(adaptation)}",
    )

    problem = problem_factory_1D(problem=problem_type)

    x_range = problem.get_range()

    # Starting with a set of equally distributed points (ban be changed to a set of randomly selected points)
    x = torch.linspace(
        x_range[0],
        x_range[1],
        steps=params.NUM_MAX_POINTS,
        requires_grad=True,
        device=params.DEVICE,
    ).reshape(-1, 1)
    # A set of base x points
    base_x_mesh = torch.linspace(x_range[0], x_range[1], steps=params.NUM_BASE_MESH_POINTS, device=params.DEVICE)
    # A set of test points
    test_x = torch.linspace(x_range[0], x_range[1], steps=params.NUM_TEST_POINTS, device=params.DEVICE)

    adaptation.set_problem_details(
        base_points=base_x_mesh,
        x_range=x_range,
        max_number_of_points=params.NUM_MAX_POINTS,
    )

    pinn = PINN_1D(params.LAYERS, params.NEURONS, pinning=False).to(params.DEVICE)
    convergence_data = torch.empty(0)
    point_data = []
    n_iters = -1
    optimizer = torch.optim.Adamax(pinn.parameters(), lr=params.LEARNING_RATE)

    start_time = time.time()
    for i in range(params.MAX_ITERS):
        logging.log(logging.DEBUG, f"PINN training iter: {i}")
        n_iters = i

        loss_fn = partial(problem.compute_loss, x=x)

        stage_convergence_data = train_model(
            nn_approximator=pinn,
            loss_fn=loss_fn,
            device=params.DEVICE,
            learning_rate=params.LEARNING_RATE,
            max_epochs=params.NUMBER_EPOCHS,
            optimizer=optimizer,
        )

        convergence_data = torch.cat((convergence_data, stage_convergence_data.cpu()))

        if save_training_data:
            y = f(pinn, x).detach().cpu()
            plain_x = x.detach().clone().cpu()
            point_data.append(torch.stack((plain_x, y)).transpose(1, 0).reshape(-1, 2))

        loss_fn = partial(problem.f_inner_loss, pinn=pinn)

        if exit_criterion_1D(test_x, loss_fn, params.TOLERANCE):
            break

        x = adaptation.refine(loss_function=loss_fn, old_x=x)

    end_time = time.time()
    exec_time = end_time - start_time

    if n_iters == params.MAX_ITERS - 1:
        logging.log(
            logging.WARNING,
            f"The error tolerance has not been reached in {params.MAX_ITERS} iterations",
        )

    logging.log(
        logging.INFO,
        f"Adaptation: {str(adaptation)}, Problem: {problem_type.value}, Run: {run_id}, "
        f"Finished in {n_iters+1} iterations, after {exec_time}s",
    )

    # Saving results
    base_path = os.path.join(
        "results_1D",
        problem_type.value,
        str(adaptation),
        f"L{params.LAYERS}_N{params.NEURONS}_" f"P{params.NUM_MAX_POINTS}_E{params.NUMBER_EPOCHS}",
        f"LR{params.LEARNING_RATE}_TOL{params.TOLERANCE}",
    )

    path = os.path.join(base_path, str(run_id))

    os.makedirs(name=path, exist_ok=True)

    pinn = pinn.cpu()
    torch.save(pinn, os.path.join(path, PINN_FILE))
    torch.save(n_iters, os.path.join(path, N_ITERS_FILE))
    torch.save(exec_time, os.path.join(path, TIME_FILE))
    torch.save(convergence_data.detach(), os.path.join(path, CONVERGENCE_FILE))

    if save_training_data:
        torch.save(point_data, os.path.join(path, POINT_DATA_FILE))

    with open(os.path.join(path, "result.txt"), "w") as file:
        file.write(f"PROBLEM = {problem_type.value}\n")
        file.write(f"ADAPTATION = {str(adaptation)}\n")
        file.write(f"RUN_ID = {run_id}\n")
        file.write(f"DEVICE = {params.DEVICE}\n")
        file.write(f"NUM_BASE_POINTS = {params.NUM_BASE_MESH_POINTS}\n")
        file.write(f"NUM_MAX_POINTS = {params.NUM_MAX_POINTS}\n")
        file.write(f"NUMBER_EPOCHS = {params.NUMBER_EPOCHS}\n")
        file.write(f"LEARNING_RATE = {params.LEARNING_RATE}\n")
        file.write(f"LAYERS = {params.LAYERS}\n")
        file.write(f"NEURONS = {params.NEURONS}\n")
        file.write(f"TOLERANCE = {params.TOLERANCE}")
        file.write("\n")
        file.write(f"Time = {exec_time}\n")
        file.write(f"Iterations = {n_iters+1}\n")

    with open(f"{base_path}/stability.txt", "a") as stability_results_file:
        stability_results_file.write(f"{run_id};{n_iters + 1};{exec_time}\n")  # noqa: E231, E702
