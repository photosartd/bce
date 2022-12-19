import logging
from typing import Tuple, Union, Dict

import torch
import torch_geometric as tg

from torch_bce.interfaces import ModelInterface
from torch_bce.models import GraphSAGE
from torch_bce.interfaces.trainer_interface import AlignmentTrainerInterface
from torch_bce.utils import PositiveNegativeNeighbourSampler


class GSAlignmentTrainer(AlignmentTrainerInterface):
    def __init__(self, *args, **kwargs):
        super(GSAlignmentTrainer, self).__init__(*args, **kwargs)

    def train_model(self,
                    train_data: tg.data.Data,
                    walk_length: int = 3,
                    sizes: Tuple = (5, 2),
                    batch_size: int = 256,
                    shuffle: bool = False,
                    log_stats: bool = False,
                    *args, **kwargs) -> \
            Tuple[torch.Tensor, torch.Tensor, Dict]:
        self.__asserts(train_data, sizes)
        x_train, edge_index_train = train_data.x.to(self.device), train_data.edge_index.to(self.device)
        train_loader = PositiveNegativeNeighbourSampler(walk_length, edge_index_train, sizes=sizes,
                                                        batch_size=batch_size, shuffle=shuffle,
                                                        num_nodes=train_data.num_nodes)

        assert isinstance(self.current_model, GraphSAGE), "Model was not GraphSAGE or relative"
        loss, external_loss, statistics = self.current_model.train_loop(
            x_train,
            train_loader,
            optimizer=None,
            num_nodes=train_data.num_nodes,
            optimize=False,
            external_loss_fn=self.alignment_loss
        )
        if log_stats:
            if isinstance(self.logger, logging.Logger):
                self.logger.info(statistics)
        return loss, external_loss, statistics

    def validate_model(self,
                       val_data: tg.data.Data,
                       walk_length: int = 3,
                       sizes: Tuple = (5, 2),
                       batch_size: int = 256,
                       shuffle: bool = False,
                       log_stats: bool = False,
                       *args, **kwargs) -> \
            Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        self.__asserts(val_data, sizes)
        x_val, edge_index_val = val_data.x.to(self.device), val_data.edge_index.to(self.device)
        val_loader = PositiveNegativeNeighbourSampler(walk_length, edge_index_val, sizes=sizes,
                                                      batch_size=batch_size, shuffle=shuffle,
                                                      num_nodes=val_data.num_nodes)

        assert isinstance(self.current_model, GraphSAGE), "Model was not GraphSAGE or relative"
        predictions, statistics = self.current_model.predict(
            x_val,
            val_loader,
            num_nodes=val_data.num_nodes,
            compute_loss=True,
            external_loss_fn=self.alignment_loss
        )
        if log_stats:
            if isinstance(self.logger, logging.Logger):
                self.logger.info(statistics)
        return predictions, statistics["mean_loss"], statistics["mean_external_loss"]

    def initialize_new_model(self, prev_model: ModelInterface, new_model: ModelInterface, *args, **kwargs) -> None:
        assert isinstance(prev_model, GraphSAGE) and isinstance(new_model, GraphSAGE), \
            "One or both models are not of type GraphSAGE"
        if len(prev_model.convs) != len(new_model.convs):
            self.logger.warning(
                f"""
                Previous model convs len: {len(prev_model.convs)}
                New model convs len: {len(new_model.convs)}
                Do not copy layers, so as don't know how
                """
            )
            return
        for conv_old, conv_new in zip(prev_model.convs, new_model.convs):
            if (conv_old.in_channels == conv_new.in_channels) and (
                    conv_old.out_channels == conv_new.out_channels):
                for target_param, param in zip(conv_new.parameters(), conv_old.parameters()):
                    target_param.data.copy_(param.data)
            else:
                self.logger.info(f"Skipped param copying for conv {conv_old}")

    def __asserts(self, data: tg.data.Data, sizes: Tuple) -> None:
        assert isinstance(self.current_model, GraphSAGE), "Model was not GraphSAGE or relative"
        assert self.current_model.in_channels == data.num_node_features, "Num nodes in data is not equal to in_channels"
        assert self.current_model.n_layers == len(sizes), "model.n_layers must be == sampler sizes"

    def __eq__(self, other):
        if isinstance(other, GSAlignmentTrainer) and isinstance(self.current_model, GraphSAGE):
            return len(self.models) == len(other.models) and \
                   self.current_model.n_layers == other.current_model.n_layers and \
                   self.current_model.in_channels == other.current_model.in_channels and \
                   self.current_model.hid_channels == other.current_model.hid_channels and \
                   self.current_model.out_channels == other.current_model.out_channels and \
                   len(self.alignment.alignment.all_backward_transformations) == len(
                other.alignment.alignment.all_backward_transformations) and \
                   self.prev_embeddings.x == other.prev_embeddings.x
        else:
            raise NotImplementedError()
