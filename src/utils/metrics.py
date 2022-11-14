from typing import Any

import torch
from torchmetrics import Metric


class ZeroMetric(Metric):
    """Metric class that returns nothing when called."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        return 0

    def compute(self):
        return 0
