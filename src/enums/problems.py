from enum import Enum


class Problems1D(str, Enum):
    DIFFUSION = "diffusion"  # Advection diffusion
    TAN_01 = "tan_01"  # tan(x+0.1)
    TAN_03 = "tan_03"  # tan(x+0.3)
    P07_01 = "P07_01"  # (x+0.1)^0.7
    P07_001 = "P07_001"  # (x+0.01)^0.7


class Problems2D(str, Enum):
    P07_01 = "P07_01_2D"  # (x+0.1)^0.7*(y+0.1)^0.7
    TAN_05 = "tan_05_2D"  # tan(x-0.5)*tan(y-0.5)
