from abc import ABC, abstractmethod
from functools import cached_property

import torch
import torch.nn as nn
from torchmetrics import MetricCollection

from .saveable import Saveable
from src.utils.metrics import ZeroMetric
from src.utils.constants import ModelType


class ModelInterface(nn.Module, Saveable, ABC):
    def __init__(self,
                 loss: nn.Module,
                 device: str = "cpu",
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
        torch.save(self.state_dict(), savepath)

    @classmethod
    def load(cls, path, **kwargs):
        loss = kwargs.get("loss")
        device = kwargs.get("device") if kwargs.get("device") else "cpu"
        model = cls(loss=loss,
                    device=device
                    )
        model.load_state_dict(torch.load(path))
        return model
