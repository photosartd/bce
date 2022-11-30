from typing import Union, List
from dataclasses import dataclass

import torch.nn as nn


@dataclass()
class Losses:
    intended_losses: Union[nn.Module, List[nn.Module]]
    unintended_losses: Union[nn.Module, List[nn.Module]]
