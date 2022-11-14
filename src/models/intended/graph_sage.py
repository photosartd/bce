import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.interfaces import ModelInterface


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
logging.basicConfig()


class GraphSAGE(ModelInterface):
    """GraphSAGE model implemented from https://towardsdatascience.com/pytorch-geometric-graph-embedding-da71d614c3a
    with small changes
    """
    def __init__(self,
                 loss: nn.Module,
                 in_channels: int,
                 hid_channels: int,
                 out_channels: int,
                 n_layers: int,
                 aggr="mean",
                 device="cpu"
                 ):
        super(GraphSAGE, self).__init__(loss=loss, device=device)
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        for i in range(n_layers - 1):
            in_channels = in_channels if i == 0 else hid_channels
            self.convs.append(SAGEConv(in_channels, hid_channels, aggr=aggr))
        in_channels = in_channels if not len(self.convs) else hid_channels
        self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))

    def forward(self, x, adjs):
        """
        :param x: features of the sampled batch
        :param adjs: list[(edge_index, e_id, size)]
        :return:
        """
        """Iterate over adjs"""
        for i, (edge_index, _, size) in enumerate(adjs):
            if self.training:
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
            else:
                x = self.convs[i](x, edge_index)
            if i != self.n_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        x_return = x if self.training else x[:size[1]]
        logger.info(f"x shape: {x_return.shape}; training: {self.training}")
        return x_return

    def train_loop(self,
              x,
              train_dataloader,
              optimizer,
              num_nodes):
        self.train()
        total_loss = 0

        for batch_size, n_id, adjs in train_dataloader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            if self.n_layers < 2:
                adjs = [adjs]
            adjs = [adj.to(self.device) for adj in adjs]
            optimizer.zero_grad()

            out = self.forward(x[n_id], adjs)
            z_u, z_v, z_vn = out.split(out.size(0) // 3, dim=0)[:3]
            loss = self.loss(z_u, z_v, z_vn)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * z_u.size(0)
        return total_loss / num_nodes

    @torch.no_grad()
    def predict(self,
                x,
                test_dataloader,
                num_nodes,
                compute_loss: bool = True
                ):
        self.eval()
        losses: List[float] = []
        predictions: List[torch.Tensor] = []
        mean_loss = None

        for batch_size, n_id, adjs in test_dataloader:
            if self.n_layers < 2:
                adjs = [adjs]
            adjs = [adj.to(self.device) for adj in adjs]
            out = self.forward(x[n_id], adjs)
            z_u, z_v, z_vn = out.split(out.size(0) // 3, dim=0)[:3]
            predictions.append(z_u.detach().cpu())
            if compute_loss:
                loss = self.loss(z_u, z_v, z_vn)
                losses.append(float(loss.item()) * z_u.size(0))

        predictions = torch.vstack(predictions)
        if compute_loss:
            mean_loss = np.sum(losses) / num_nodes
            logger.info(f"Predict mean loss: {mean_loss}")

        return predictions, mean_loss
