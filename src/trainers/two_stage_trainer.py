from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Tuple

import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

from src.models import GraphSAGE, MLPRegressor, MLP
from src.dataclasses import Models
from src.utils import PositiveNegativeNeighbourSampler, seed
from src.utils.datasets import TensorSupervisedDataset, TensorDataset
from src.interfaces import ModelInterface
from src.interfaces.trainer_interface import TrainerInterface
from src.losses import GraphSageLoss, AlignmentLoss


class TwoStageTrainer(TrainerInterface, ABC):
    def __init__(self, *args, **kwargs):
        super(TwoStageTrainer, self).__init__(*args, **kwargs)
        self.datasets: tuple[Any] = self.configure_datasets(**kwargs)
        """prev_embeddings Tensor dummy initialization"""
        self.prev_embeddings: TensorDataset = TensorDataset(x=torch.rand(()))

    def setup(self, num_epochs=20, **kwargs) -> None:
        super().setup(**kwargs)
        # TODO: key to secrets
        wandb.login(key="ff032b7d576dfc8849b5c26d72c084cfc88f7b76")
        wandb.init(project="bce", reinit=True)
        config = wandb.config
        config.epochs = num_epochs * len(self.models.intended_models)

    def fit(self, num_epochs=20, **kwargs):
        self.setup(num_epochs=num_epochs)

    def log_metrics(self, epoch: int, metrics: Dict, **kwargs):
        wandb.log(metrics)
        for metric_name, metric_val in metrics.items():
            self.logger.info(f"Epoch: {epoch}| Metric {metric_name}: {metric_val}")

    @abstractmethod
    def configure_datasets(self, **kwargs):
        raise NotImplementedError()


class GraphSAGETwoStageTrainer(TwoStageTrainer):
    def __init__(self, walk_length=3, batch_size=256, *args, **kwargs):
        super(GraphSAGETwoStageTrainer, self).__init__(*args, **kwargs)
        self.walk_length: int = walk_length
        self.batch_size: int = batch_size
        self.sizes: List[int] = kwargs["sizes"]  # Throws exception is not given

    def __fit_partial(self,
                      intended_model: ModelInterface,
                      num_epochs: int,
                      model_index: int,
                      main_loss: nn.Module,
                      alignment_loss_fn: AlignmentLoss,
                      datasets,
                      disable_progress=True
                      ):
        loss: Union[torch.Tensor, int] = 0
        loss_align: Union[torch.Tensor, int] = 0
        step = 0
        loss_train_gs = 0
        loss_train_align = 0

        def train_step(dataloader, train_data):
            intended_model.train()
            losses_gs_train = []
            losses_align_train = []
            for batch_id, (batch_size, n_id, adjs) in enumerate(dataloader):
                if intended_model.n_layers < 2:
                    adjs = [adjs.to(self.device)]
                adjs = [adj.to(self.device) for adj in adjs]
                out = intended_model.forward(train_data.x[n_id], adjs)
                z_u, z_v, z_vn = out.split(out.size(0) // 3, dim=0)[:3]

                loss_curr_gsl = main_loss(z_u, z_v, z_vn)
                loss += loss_curr_gsl
                alignment_loss = self.get_alignment_loss(z_u, n_id, alignment_loss_fn, model_index)
                loss_align += alignment_loss
                losses_gs_train.append(float(loss_curr_gsl.item()) * z_u.size(0))
                losses_align_train.append(alignment_loss.item())
                step += 1


        self.initialize_intended_model(intended_model, model_index)
        """3. For epochs"""
        for epoch in tqdm(range(num_epochs), disable=disable_progress):
            loss: Union[torch.Tensor, int] = 0
            loss_align: Union[torch.Tensor, int] = 0
            step = 0
            loss_train_gs = 0
            loss_train_align = 0
            """4. Train on snapshots (0:timestamp + one_model_snapshots_len)"""
            for data in datasets:
                dataloaders, data_objects = self.get_dataloaders(data, sizes=self.sizes)
                train_dataloader, val_dataloader, test_dataloader = dataloaders



    def fit(self, num_epochs=20, disable_progress=True, **kwargs):
        super(GraphSAGETwoStageTrainer, self).fit(num_epochs=num_epochs, **kwargs)
        """1. Get len of train dataset and divide all snapshots equally for all intended tasks"""
        train_dataset_snapshots_len = len(list(self.datasets[0]))
        one_model_snapshots_len = train_dataset_snapshots_len // len(self.models.intended_models)
        """2. Cycle by timestamps and models"""
        for index, (intended_model, losses, optimizers, timestamp) in enumerate(
                zip(
                    self.models.intended_models,
                    self.losses.intended_losses,
                    self.optimizers.intended_optimizers,
                    range(0, train_dataset_snapshots_len, one_model_snapshots_len)
                )
        ):

            self.initialize_intended_model(intended_model, index)
            """3. For epochs"""
            for epoch in tqdm(range(num_epochs), disable=disable_progress):
                loss: Union[torch.Tensor, int] = 0
                loss_align: Union[torch.Tensor, int] = 0
                step = 0
                loss_train_gs = 0
                loss_train_align = 0
                """4. Train on snapshots (0:timestamp + one_model_snapshots_len)"""
                for data in self.datasets[0][:timestamp + one_model_snapshots_len]:
                    dataloaders, data_objects = self.get_dataloaders(data, sizes=self.sizes)
                    train_dataloader, val_dataloader, test_dataloader = dataloaders
                    train_data, val_data, test_data = data_objects
                    """5. All steps manually for now (TODO)"""
                    intended_model.train()
                    losses_gs_train = []
                    losses_align_train = []
                    for batch_id, (batch_size, n_id, adjs) in enumerate(train_dataloader):
                        if intended_model.n_layers < 2:
                            adjs = [adjs.to(self.device)]
                        adjs = [adj.to(self.device) for adj in adjs]
                        out = intended_model.forward(train_data.x[n_id], adjs)
                        z_u, z_v, z_vn = out.split(out.size(0) // 3, dim=0)[:3]

                        loss_curr_gsl = losses[0](z_u, z_v, z_vn)
                        loss += loss_curr_gsl
                        alignment_loss = self.get_alignment_loss(z_u, n_id, losses[1], index)
                        loss_align += alignment_loss
                        losses_gs_train.append(float(loss_curr_gsl.item()) * z_u.size(0))
                        losses_align_train.append(alignment_loss.item())
                        step += 1
                    loss_train_gs = sum(losses_gs_train) / train_data.num_nodes
                    loss_train_align = sum(losses_align_train) / (batch_id + batch_size / self.batch_size)


                    """6. Validation"""
                    intended_model.eval()
                    losses_gs_val = []
                    losses_align_val = []
                    loss_gs_val = 0
                    loss_align_val = 0
                    predictions: List[torch.Tensor] = []
                    with torch.no_grad():
                        for batch_id, (batch_size, n_id, adjs) in enumerate(val_dataloader):
                            if intended_model.n_layers < 2:
                                adjs = [adjs.to(self.device)]
                            adjs = [adj.to(self.device) for adj in adjs]
                            out = intended_model.forward(train_data.x[n_id], adjs)
                            z_u, z_v, z_vn = out.split(out.size(0) // 3, dim=0)[:3]
                            predictions.append(z_u.detach().cpu())

                            loss_curr_gsl = losses[0](z_u, z_v, z_vn)
                            loss_gs_val += loss_curr_gsl
                            alignment_loss = self.get_alignment_loss(z_u, n_id, losses[1], index)
                            loss_align_val += alignment_loss
                            losses_gs_val.append(float(loss_curr_gsl.item()) * z_u.size(0))
                            losses_align_val.append(alignment_loss.item())
                        predictions: torch.Tensor = torch.vstack(predictions)
                        loss_gs_val = sum(losses_gs_val) / val_data.num_nodes
                        loss_align_val = sum(losses_align_val) / (batch_id + batch_size / self.batch_size)

                loss = loss + loss_align / (step + 1)
                loss.backward()
                optimizers.step()
                optimizers.zero_grad()

                tensor_dataset = TensorSupervisedDataset(x=predictions, y=val_data.y)
                unintended_dataloader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=False)
                unintended_optimizer = self.optimizers.unintended_optimizers[0]
                unintended_model = self.models.unintended_models[0]
                if isinstance(unintended_model, MLP):
                    for unintended_epoch in range(1, num_epochs + 1):
                        loss_unintended, _ = unintended_model.train_loop(
                            unintended_dataloader, unintended_optimizer)
                y_pred, loss_unintended, metrics_unintended = unintended_model.predict(
                    unintended_dataloader)
                all_metrics = metrics_unintended
                all_metrics.update({"intended_gs_loss": loss_train_gs,
                                    "intended_align_loss": loss_train_align,
                                    "intended_gs_val_loss": loss_gs_val,
                                    "intended_align_val_loss": loss_align_val
                                    })
                self.log_metrics(epoch=epoch, metrics=all_metrics)

            """ N. Update embeddings of prev_dataset"""
            # TODO: check where it's ok to put all embeddings
            self.prev_embeddings = TensorDataset(x=predictions)
        wandb.finish()

    def configure_datasets(self, lags=8, train_ratio=0.8, **kwargs):
        loader = WikiMathsDatasetLoader()
        dataset = loader.get_dataset(lags=lags)
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)
        return train_dataset, test_dataset

    def configure_models(self, models: Models = None, lambda_=4, alignment="single_step", backward_transformation="linear", n_layers=2,
                         out_channels=32, lags=8, target_size=1, div=3, **kwargs) -> Models:
        if isinstance(models, Models):
            return models
        elif models is not None:
            alignment_loss = AlignmentLoss(
                lambda_=lambda_,
                backward_transformation=backward_transformation,
                alignment=alignment,
                M=out_channels,
                N=out_channels
            )
            intended_model_1 = GraphSAGE(
                loss=[GraphSageLoss(), alignment_loss],
                in_channels=lags,
                hid_channels=64,
                out_channels=out_channels,
                n_layers=n_layers,
                device=self.device
            )
            intended_model_2 = GraphSAGE(
                loss=[GraphSageLoss(), alignment_loss],
                in_channels=lags,
                hid_channels=128,
                out_channels=out_channels,
                n_layers=n_layers,
                device=self.device
            )
            unintended_model = MLPRegressor(
                loss=nn.MSELoss(),
                input_size=out_channels,
                output_size=target_size,
                div=div,
                device=self.device
            )
            return Models(
                [intended_model_1, intended_model_2],
                [unintended_model]
            )
        else:
            raise TypeError("models must be of type Models")

    def get_dataloaders(self, data, sizes, **kwargs):
        self.dataloaders = self.configure_dataloaders(data, sizes, **kwargs)
        return self.dataloaders

    @seed
    def configure_dataloaders(self, data, sizes, num_val=0.3, num_test=0.0, return_data=True, **kwargs) -> Union[
        List[DataLoader], Tuple[List[DataLoader], List[pyg.data.data.Data]]]:
        transform = T.RandomLinkSplit(num_val=num_val, num_test=num_test)
        data_train, data_val, data_test = transform(data)
        train_loader = PositiveNegativeNeighbourSampler(self.walk_length, data_train.edge_index, sizes=sizes,
                                                        batch_size=self.batch_size,
                                                        shuffle=False, num_nodes=data_train.num_nodes)
        val_loader = PositiveNegativeNeighbourSampler(self.walk_length, data_val.edge_index, sizes=sizes,
                                                      batch_size=self.batch_size,
                                                      shuffle=False, num_nodes=data_val.num_nodes)
        # TODO: think how to prevent None passing
        if not return_data:
            return [train_loader, val_loader, None]
        else:
            return [train_loader, val_loader, None], [data_train, data_val, data_test]

    def initialize_intended_model(self, intended_model: ModelInterface, index: int):
        """Initializes intended model with previous version"""
        if index == 0:
            return
        else:
            prev_intended_model = self.models.intended_models[index - 1]
            if isinstance(intended_model, GraphSAGE) and (prev_intended_model, GraphSAGE):
                for conv_old, conv_new in zip(prev_intended_model.convs, intended_model.convs):
                    if (conv_old.in_channels == conv_new.in_channels) and (
                            conv_old.out_channels == conv_new.out_channels):
                        for target_param, param in zip(conv_new.parameters(), conv_old.parameters()):
                            target_param.data.copy_(param.data)
                    else:
                        self.logger.info(f"Skipped param copying at index {index}")
            alignment_loss = self.losses.intended_losses[index][-1]
            if isinstance(alignment_loss, AlignmentLoss):
                alignment_loss.finish(M=intended_model.out_channels,
                                      N=prev_intended_model.out_channels)
            else:
                raise TypeError("Wrong loss in initialize model")

    def get_alignment_loss(self, m_k: torch.Tensor, n_id: List[int], loss_fn: nn.Module, index: int):
        """Calculates alignment loss for current intended model
        Arguments:
            :param m_k: tensor with new embeddings
            :param n_id: ID's of nodes for m_k (respective to their initial place)
            :param loss_fn: AlignmentLoss object
            :param index: current index of model
            :returns loss_value
        """
        if index == 0:
            return torch.zeros((), requires_grad=True)
        else:
            if isinstance(loss_fn, AlignmentLoss):
                old_embeddings = self.prev_embeddings[n_id[:m_k.shape[0]]]
                assert m_k.shape[0] == old_embeddings.shape[
                    0], f"m_k shape: {m_k.shape}, old shape: {old_embeddings.shape}"
                return loss_fn(m_k, old_embeddings)
            else:
                raise TypeError("Loss was not AlignmentLoss!")
