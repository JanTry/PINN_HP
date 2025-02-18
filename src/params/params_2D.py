import logging
import sys

import torch
from src.enums.problems import Problems2D

PROBLEM = Problems2D.P07_01  # Used only if not specified in the PINN training run
MAX_POINTS_NUMBER = 40000  # DEF 40000
TEST_POINTS_NUMBER = 65536  # DEF 65536
MAX_ITERS = 1000  # DEF 1000

LAYERS = 3  # DEF 3
NEURONS = 15  # DEF 15
LEARNING_RATE = 0.005  # DEF 0.005
NUMBER_EPOCHS = 1000  # DEF 1000
TOLERANCE = 1e-3  # DEF 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_LEVEL = logging.INFO  # DEF logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s[%(levelname)s] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
logging.log(logging.DEBUG, f"Using device: {DEVICE}")
