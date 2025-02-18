from src.adaptations.adaptations_1D import (
    DensitySamplingAdaptation1D,
    MiddlePointAdaptation1D,
    NoAdaptation1D,
)
from src.enums.problems import Problems1D
from src.plots.plots_1D.plot_specific_run import plot_specific_run_1D
from src.runners.adaptive_PINN_1D import train_PINN_1D

# Usage example

PROBLEM_TYPES = [
    Problems1D.DIFFUSION,
    Problems1D.TAN_03,
    Problems1D.P07_01,
]

ADAPTATIONS = [
    MiddlePointAdaptation1D(),
    DensitySamplingAdaptation1D(),
    NoAdaptation1D(),
]
types_to_time = {}
NUM_RUNS = 1

for problem_type in PROBLEM_TYPES:
    for adaptation in ADAPTATIONS:
        for i in range(NUM_RUNS):
            train_PINN_1D(i, problem_type=problem_type, adaptation=adaptation)
            plot_specific_run_1D(
                run_id=i,
                problem_type=problem_type,
                adaptation=adaptation,
                plot_training_points=True,
            )
