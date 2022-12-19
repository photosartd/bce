import pickle as pkl
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Union, Tuple, Dict, Any

import torch
import torch.nn as nn
from torchmetrics import MetricCollection

from src.interfaces import Saveable
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
    def train_loop(self, **data) -> \
            Tuple[
                Union[torch.Tensor, None],
                Dict[str, Any]
            ]:
        """
        :param data: kwargs for all data
        :return: Tuple with 2 objects:
            :loss: torch.Tensor or None if optimized inside
            :statistics: Dict with any statistics calculated
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, **data) -> \
            Tuple[
                Union[torch.Tensor, None],
                Dict[str, Any]
            ]:
        """
        :param data: kwargs for all data
        :return: Tuple with 2 objects:
            :loss: torch.Tensor or None if optimized inside
            :statistics: Dict with any statistics calculated
        """
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
