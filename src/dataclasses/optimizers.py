from typing import Union, List
from dataclasses import dataclass

import torch


@dataclass()
class Optimizers:
    intended_optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]
    unintended_optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]
