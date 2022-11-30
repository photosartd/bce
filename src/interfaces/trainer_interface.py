import logging
from typing import List
from abc import ABC, abstractmethod

import torch.optim
from torch.utils.data import DataLoader

from src.dataclasses import Optimizers, Losses, Models


class TrainerInterface(ABC):
    """Abstract class created for easy training of models
    Parameters should be prepared in setup stage
    Parameters:
        :param dataloaders: list of train, val, test dataloaders
        :param optimizers: optimizers for different tasks
        :param losses: losses for different tasks
        :param models: models (of subtype ModelInterface)
        :param device: where to train nets
        :param logger: logging.Logger
    """
    dataloaders: List[DataLoader]
    optimizers: Optimizers
    losses: Losses
    models: Models
    logger: logging.Logger
    device: torch._C.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, *args, **kwargs):
        self.setup_logger()
        self.logger = logging.getLogger("default_logger")
        self.kwargs = kwargs

    def setup(self, **kwargs) -> None:
        """Implement all basic operations that are needed
        Arguments:
            :param kwargs: any keyword arguments you want to supply to 'configure_*' methods
            :return: None
        """
        self.models = self.configure_models(**kwargs)
        self.optimizers = self.configure_optimizers(**kwargs)
        self.losses = self.configure_losses(**kwargs)

    @staticmethod
    def setup_logger(name="default_logger", level=logging.WARN, **kwargs) -> None:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logging.basicConfig()

    @abstractmethod
    def fit(self, **kwargs):
        """Fit method. Must implement all basic steps in abstract form
        Arguments:
            :param kwargs:
            :return: TODO
        """
        raise NotImplementedError()

    @abstractmethod
    def log_metrics(self, **kwargs):
        """Make all needed logging"""
        raise NotImplementedError()

    @abstractmethod
    def configure_models(self, **kwargs) -> Models:
        """Returns src.dataclasses.Models"""
        raise NotImplementedError()

    @abstractmethod
    def configure_dataloaders(self, **kwargs) -> List[DataLoader]:
        """Returns dataloader for task
        Arguments:
            :return: List[torch.util.data.DataLoader] - [train, val, test] (None for every if there is no)
        """
        raise NotImplementedError()

    def configure_optimizers(self, lr: float = 3e-4, weight_decay: float = 4e-5, **kwargs) -> Optimizers:
        """Returns src.dataclasses.Optimizers"""
        intended_optimizers = [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                               for model in self.models.intended_models]
        unintended_optimizers = [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                                 for model in self.models.unintended_models]
        return Optimizers(intended_optimizers, unintended_optimizers)

    def configure_losses(self, **kwargs) -> Losses:
        """Returns src.dataclasses.Losses"""
        intended_losses = [losses for losses in
                           map(lambda x: x.loss if isinstance(x.loss, list) else [x.loss], self.models.intended_models)]
        unintended_losses = [losses for losses in map(lambda x: x.loss if isinstance(x.loss, list) else [x.loss],
                                                      self.models.unintended_models)]
        return Losses(intended_losses, unintended_losses)
