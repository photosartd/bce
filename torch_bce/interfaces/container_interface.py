import os
from abc import ABC, abstractmethod
from typing import Collection, List

import torch

from torch_bce.interfaces import ModelInterface
from torch_bce.interfaces import Saveable


class ContainerInterface(Saveable, ABC):
    """Interface for containers of models"""

    def __init__(self, models: Collection[ModelInterface] = None, **kwargs):
        self.models: Collection[ModelInterface] = models

    def __len__(self):
        return len(self.models)

    @abstractmethod
    def add(self, model: ModelInterface) -> None:
        """Should add model to self.models in proper order"""
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        """Should give interface to iterate over self.models in right order"""
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, item):
        """Should give interface to take slices of self.models"""
        raise NotImplementedError()

    def save(self, savepath, **kwargs) -> None:
        """
        Arguments:
        :param savepath: path to folder
        :param kwargs:
        :return: None
        """
        for index, model in enumerate(self):
            current_path = os.path.join(savepath, f"{index}.pkl")
            model.save(current_path)
