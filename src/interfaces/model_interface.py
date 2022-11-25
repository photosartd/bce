import pickle as pkl
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Union

import torch
import torch.nn as nn
from torchmetrics import MetricCollection

from .saveable import Saveable
from src.utils.metrics import ZeroMetric
from src.utils.constants import ModelType


class ModelInterface(nn.Module, Saveable, ABC):
    def __init__(self,
                 loss: nn.Module,
                 device: Union[str, torch._C.device] = "cpu",
                 model_type: ModelType = ModelType.NOT_DEFINED,
                 **kwargs
                 ):
        super(ModelInterface, self).__init__()
        self.loss = loss
        self.device = device
        self.model_type: ModelType = model_type

    @abstractmethod
    def train_loop(self, **data):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, **data):
        raise NotImplementedError()

    @cached_property
    def metrics(self):
        return MetricCollection(
            [
                ZeroMetric()
            ]
        )

    def save(self, savepath, **kwargs) -> None:
        with open(savepath, "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def load(cls, path, **kwargs):
        with open(path, "rb") as f:
            model = pkl.load(f)
        return model
