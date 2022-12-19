import logging
import unittest
from typing import Iterable
from functools import cached_property
from tempfile import TemporaryDirectory
from copy import deepcopy

import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

from src.models import GraphSAGE, MLPRegressor
from src.losses import GraphSageLoss
from src.utils.datasets import TensorSupervisedDataset
from src.trainers.gs_alignment_trainer import GSAlignmentTrainer

KWARGS = {
    "lambda_": 8,
    "alignment": "multi_step",
    "backward_transformation": "linear",
    "level": logging.INFO,
    "num_epochs": 100,
    "setup_wandb": False
}
model1 = GraphSAGE(
    loss=GraphSageLoss(),
    in_channels=256,
    hid_channels=128,
    out_channels=32,
    n_layers=2,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

model2 = GraphSAGE(
    loss=GraphSageLoss(),
    in_channels=256,
    hid_channels=256,
    out_channels=32,
    n_layers=2,
    device="cuda" if torch.cuda.is_available() else "cpu"
)


class TestGSAlignmentTrainer(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    def test_trainer_creation(self):
        trainer = GSAlignmentTrainer(
            model=model1,
            **KWARGS
        )
        self.assertTrue(model1 == trainer.current_model)
        self.assertTrue(len(trainer.models) == 1)
        self.assertTrue(len(trainer.alignment.all_backward_transformations) == 0)

    def test_can_replace_model(self):
        trainer = GSAlignmentTrainer(
            model=model1,
            **KWARGS
        )
        trainer.replace_model(model2)
        self.assertTrue(model2 == trainer.current_model)
        self.assertTrue(len(trainer.models) == 2)
        self.assertTrue(len(trainer.alignment.all_backward_transformations) == 1)

    def test_can_save_and_load(self):
        trainer = GSAlignmentTrainer(
            model=model1,
            **KWARGS
        )
        trainer.replace_model(model2)
        with TemporaryDirectory() as d:
            trainer.save(d)
            trainer2 = GSAlignmentTrainer.load(d)
            assert trainer == trainer2, \
                f"""
            Len: {len(trainer.models)}; {len(trainer2.models)}
            Alignment len: {len(trainer.alignment.all_backward_transformations)}; {len(trainer2.alignment.all_backward_transformations)}
            Embeddings shapes: {trainer.prev_embeddings.x.shape == trainer2.prev_embeddings.x.shape}
            Embeddings: {trainer.prev_embeddings.x == trainer2.prev_embeddings.x} 
            """

    @cached_property
    def get_data(self):
        """Data"""
        loader = WikiMathsDatasetLoader()  # ChickenpoxDatasetLoader()
        dataset = loader.get_dataset()
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)
        return train_dataset, test_dataset

    def check_strictly_decreasing_trend(self, data: Iterable):
        self.assertTrue(
            all(
                [
                    (x >= y) for x, y in zip(data[:-1], data[1:])
                ]
            )
        )

    def test_can_train_and_inference(self):
        train_dataset, test_dataset = self.get_data
        num_node_features = train_dataset[0].num_node_features
        """Model"""
        model = GraphSAGE(
            loss=GraphSageLoss(),
            in_channels=num_node_features,
            hid_channels=128,
            out_channels=32,
            n_layers=2,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        trainer = GSAlignmentTrainer(
            model=model,
            **KWARGS
        )
        model_loss, statistics = trainer.train(
            train_data=train_dataset[0],
            val_data=test_dataset[0],
            walk_length=3,
            sizes=(5, 2),
            batch_size=256,
            shuffle=False,
            log_stats=False
        )
        self.assertTrue(isinstance(model_loss, torch.Tensor))
        self.logger.info(f"Loss: {model_loss.item()}")
        self.logger.info(f"Statistics: {statistics}")

        predictions, inf_statistics = trainer.inference(
            data=test_dataset[0],
            walk_length=3,
            sizes=(5, 2),
            batch_size=256,
            shuffle=False,
            log_stats=False
        )
        self.assertTrue(predictions.shape[0] == test_dataset[0].num_nodes),
        f"""Predictions shape: {predictions.shape}; num nodes: {test_dataset[0].num_nodes}"""
        self.logger.info(f"Inference statistics: {inf_statistics}")

    @unittest.skip("Skipped model training test")
    def test_train_several_epochs(self):
        train_dataset, test_dataset = self.get_data
        num_node_features = train_dataset[0].num_node_features
        """Model"""
        model = GraphSAGE(
            loss=GraphSageLoss(),
            in_channels=num_node_features,
            hid_channels=128,
            out_channels=32,
            n_layers=2,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        local_kwargs = deepcopy(KWARGS)
        local_kwargs["setup_wandb"] = True
        trainer = GSAlignmentTrainer(
            model=model,
            **local_kwargs
        )
        """Training"""
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=4e-5)
        losses = []
        for epoch in range(1, 11):
            loss = 0
            time: int = 1
            for time, data in enumerate(train_dataset, 1):
                model_loss, statistics = trainer.train(
                    train_data=data,
                    val_data=test_dataset[0],
                    walk_length=3,
                    sizes=(5, 2),
                    batch_size=256,
                    shuffle=False,
                    log_stats=False
                )
                loss = loss + model_loss
            loss = loss / time
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            trainer.log_metrics(epoch, statistics)
        self.check_strictly_decreasing_trend(losses)

    #@unittest.skip("Skipped model training test")
    def test_train_intended_unintended_2stage(self):
        N_EPOCHS = 4
        train_dataset, test_dataset = self.get_data
        num_node_features = train_dataset[0].num_node_features
        """Intended model"""
        hid_channels = 128
        out_channels = 32
        intended_model = GraphSAGE(
            loss=GraphSageLoss(),
            in_channels=num_node_features,
            hid_channels=hid_channels,
            out_channels=out_channels,
            n_layers=2,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        """Unintended model"""
        unintended_model = MLPRegressor(loss=nn.MSELoss(), input_size=out_channels, output_size=1, div=3)
        unintended_opt = torch.optim.Adam(unintended_model.parameters(), lr=0.001)
        local_kwargs = deepcopy(KWARGS)
        local_kwargs["setup_wandb"] = True
        trainer = GSAlignmentTrainer(
            model=intended_model,
            **local_kwargs
        )
        """Training Intended"""
        optimizer = torch.optim.Adam(intended_model.parameters(), lr=3e-4, weight_decay=4e-5)
        losses = []
        for epoch in range(1, N_EPOCHS + 7):
            loss = 0
            time: int = 1
            for time, data in enumerate(train_dataset, 1):
                model_loss, statistics = trainer.train(
                    train_data=data,
                    val_data=test_dataset[0],
                    walk_length=3,
                    sizes=(5, 2),
                    batch_size=256,
                    shuffle=False,
                    log_stats=False
                )
                loss = loss + model_loss
            loss = loss / time
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            trainer.log_metrics(epoch, statistics)
        self.check_strictly_decreasing_trend(losses)
        """Get predictions for latest snapshot"""
        predictions, stats = trainer.inference(
            train_dataset[-1],
            walk_length=3,
            sizes=(5, 2),
            batch_size=256,
            shuffle=False,
            log_stats=False
        )
        predictions_test, stats_test = trainer.inference(
            test_dataset[0],
            walk_length=3,
            sizes=(5, 2),
            batch_size=256,
            shuffle=False,
            log_stats=False
        )
        """Training unintended"""
        unintended_dataset = TensorSupervisedDataset(
            x=predictions,
            y=train_dataset[-1].y
        )
        unintended_dataloader = DataLoader(unintended_dataset, batch_size=256, shuffle=False)

        unintended_dataset_val = TensorSupervisedDataset(
            x=predictions_test,
            y=test_dataset[0].y
        )
        unintended_dataloader_val = DataLoader(unintended_dataset_val, batch_size=256, shuffle=False)
        unintended_losses = []
        for epoch in range(1, N_EPOCHS + 97):
            loss, metrics = unintended_model.train_loop(unintended_dataloader, unintended_opt)
            preds_val, metrics_val = unintended_model.predict(unintended_dataloader_val)
            trainer.log_metrics(
                epoch=epoch,
                metrics={
                    "Unintended: train loss": metrics["mean_train_loss"],
                    "Unintended: val loss": metrics_val["mean_loss"],
                    "Unintended: MSE train": metrics["MeanSquaredError"],
                    "Unintended: MSE val": metrics_val["MeanSquaredError"],
                    "Unintended: MAE train": metrics["MeanAbsoluteError"],
                    "Unintended: MAE val": metrics_val["MeanAbsoluteError"]
                }
            )
            unintended_losses.append(metrics["mean_train_loss"])
        self.check_strictly_decreasing_trend(unintended_losses)

        """Intended model 2"""
        intended_model_new = GraphSAGE(
            loss=GraphSageLoss(),
            in_channels=num_node_features,
            hid_channels=hid_channels // 2,
            out_channels=out_channels,
            n_layers=3,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        optimizer = torch.optim.Adam(intended_model_new.parameters(), lr=3e-4, weight_decay=4e-5)
        """Replace model"""
        trainer.replace_model(
            intended_model_new,
            train_data=test_dataset[0],
            walk_length=3,
            sizes=(5, 2),
            batch_size=256,
            shuffle=False,
            log_stats=False
        )
        """Train intended again"""
        losses = []
        scheduler = torch.optim.lr_scheduler.StepLR(trainer.alignment_optimizer, step_size=5, gamma=0.1)
        for epoch in range(1, N_EPOCHS + 70):
            loss2 = 0
            time: int = 1
            for time, data in enumerate(train_dataset, 1):
                model_loss, statistics = trainer.train(
                    train_data=data,
                    val_data=test_dataset[0],
                    walk_length=3,
                    sizes=(5, 2, 1),
                    batch_size=256,
                    shuffle=False,
                    log_stats=False
                )
                loss2 = loss2 + model_loss
            loss2 = loss2 / time
            losses.append(loss2.item())
            loss2.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            """Get preds on step"""
            predictions_test, stats_test = trainer.inference(
                test_dataset[0],
                walk_length=3,
                sizes=(5, 2, 1),
                batch_size=256,
                shuffle=False,
                log_stats=False
            )
            unintended_dataset_val = TensorSupervisedDataset(
                x=predictions_test,
                y=test_dataset[0].y
            )
            unintended_dataloader_val = DataLoader(unintended_dataset_val, batch_size=256, shuffle=False)
            preds_val, metrics_val = unintended_model.predict(unintended_dataloader_val)
            trainer.log_metrics(
                epoch=epoch,
                metrics={
                    "Unintended: train loss": metrics["mean_train_loss"],
                    "Unintended: val loss": metrics_val["mean_loss"],
                    "Unintended: MSE train": metrics["MeanSquaredError"],
                    "Unintended: MSE val": metrics_val["MeanSquaredError"],
                    "Unintended: MAE train": metrics["MeanAbsoluteError"],
                    "Unintended: MAE val": metrics_val["MeanAbsoluteError"],
                    **statistics
                }
            )
        self.check_strictly_decreasing_trend(losses)


if __name__ == "__main__":
    unittest.main()
