from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .saveable import Saveable


class ModelInterface(nn.Module, Saveable, ABC):
    def __init__(self,
                 loss: nn.Module,
                 **kwargs
                 ):
        super(ModelInterface, self).__init__()
        self.loss = loss

    @abstractmethod
    def train_loop(self, **data):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, **data):
        raise NotImplementedError()

    def save(self, savepath, **kwargs) -> None:
        torch.save(self.state_dict(), savepath)

    @classmethod
    def load(cls, path, **kwargs):
        optimizer = kwargs.get("optimizer")
        loss = kwargs.get("loss")
        model = cls(optimizer=optimizer,
                    loss=loss
                    )
        model.load_state_dict(torch.load(path))
        return model
