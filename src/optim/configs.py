from dataclasses import dataclass
from typing import Callable, Any
from utils.scheduler import Scheduler
import torch

@dataclass
class BOConfig:
    parser: Callable
    bounds: torch.Tensor
    scheduler: Scheduler


