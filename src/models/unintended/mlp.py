import logging
from abc import ABC
from collections import OrderedDict
from typing import List, Tuple
from functools import cached_property

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError

from src.interfaces import ModelInterface
from src.utils.constants import ModelType

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
logging.basicConfig()


class MLP(ModelInterface, ABC):
    """A usual MLP module with several layers (no unlinearity at the end)"""

    def __init__(self,
                 loss: nn.Module,
                 input_size: int,
                 output_size: int,
                 div: int = 2,
                 device: str = "cpu",
                 model_type: ModelType = ModelType.NOT_DEFINED
                 ):
        super(MLP, self).__init__(loss, device=device, model_type=model_type)
        self.input_size = input_size
        self.output_size = output_size
        self.div = div

        self.mlp = self._init_mlp(
            input_size,
            output_size,
            div
        )
        self.to(self.device)

    @staticmethod
    def _init_mlp(input_size: int,
                  output_size: int,
                  div: int
                  ):
        mlp: List[Tuple[str, nn.Module]] = []

        current_size = input_size
        next_layer_size = current_size // div
        idx = 0
        while next_layer_size > output_size:
            mlp.append((str(idx), nn.Linear(current_size, next_layer_size)))
            mlp.append((f"ReLU{idx}", nn.ReLU()))
            idx += 1
            current_size = next_layer_size
            next_layer_size //= div
        mlp.append((str(idx), nn.Linear(current_size, output_size)))
        return nn.Sequential(OrderedDict(mlp))

    def forward(self, x):
        return self.mlp(x)

    def train_loop(self,
                   train_dataloader: torch.utils.data.DataLoader,
                   optimizer
                   ):
        self.train()
        total_loss = 0

        for (x, y) in train_dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()

            y_pred = self.forward(x)
            if self.model_type == ModelType.REGRESSION:
                y_pred = y_pred.squeeze(-1)
            loss = self.loss(y_pred, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            metric = self.metrics(y_pred, y)
            logger.info(f"Metrics: {metric}")

        metrics = self.metrics.compute()

        return total_loss, metrics

    @torch.no_grad()
    def predict(self,
                test_dataloader: torch.utils.data.DataLoader,
                compute_metrics: bool = True
                ):
        self.eval()
        losses: List[float] = []
        predictions: List[torch.Tensor] = []
        mean_loss = None
        metrics = None

        for (x, y) in test_dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_pred_distr = self.forward(x)

            if self.model_type == ModelType.CLASSIFICATION:
                predictions.append(torch.argmax(self.nonlinearity(y_pred_distr), dim=-1).detach().cpu())
            elif self.model_type == ModelType.REGRESSION:
                y_pred_distr = y_pred_distr.squeeze(-1)
                predictions.append(y_pred_distr.detach().cpu())
            else:
                raise NotImplementedError("Wrong model type for MLP")

            if compute_metrics:
                loss = self.loss(y_pred_distr, y)
                losses.append(float(loss.item()))
                metric = self.metrics(y_pred_distr, y)

                logger.info(f"Metrics: {metric}")

        predictions = torch.hstack(predictions)
        if compute_metrics:
            mean_loss = np.mean(losses)
            metrics = self.metrics.compute()
            logger.info(f"Predict mean loss: {mean_loss}")
            logger.info(f"Predict metrics: {metrics}")
        self.metrics.reset()

        return predictions, mean_loss, metrics


class MLPClassifier(MLP):
    """A usual MLP module for classification with several layers (no unlinearity at the end)"""

    def __init__(self, loss: nn.Module, input_size: int, output_size: int, *args, **kwargs):
        super(MLPClassifier, self).__init__(loss, input_size, output_size, *args, **kwargs)
        self.model_type: ModelType = ModelType.CLASSIFICATION

    @cached_property
    def metrics(self):
        return MetricCollection(
            [
                MulticlassF1Score(num_classes=self.output_size),
                MulticlassAccuracy(num_classes=self.output_size),
                MulticlassPrecision(num_classes=self.output_size),
                MulticlassRecall(num_classes=self.output_size)
            ]
        )

    @property
    def nonlinearity(self) -> nn.Module:
        return nn.LogSoftmax()


class MLPRegressor(MLP):
    """A usual MLP module for regression with several layers (no nonlinearity at the end)"""

    def __init__(self, loss: nn.Module, input_size: int, output_size: int, *args, **kwargs):
        super(MLPRegressor, self).__init__(loss, input_size, output_size, *args, **kwargs)
        self.model_type: ModelType = ModelType.REGRESSION

    @cached_property
    def metrics(self):
        return MetricCollection(
            [
                MeanAbsoluteError(),
                MeanSquaredError(),
                MeanAbsolutePercentageError()
            ]
        )
