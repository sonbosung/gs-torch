import torch
import numpy as np
from typing import NamedTuple
from dataclasses import dataclass

@dataclass
class BasicPointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray