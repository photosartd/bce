import torch
from torch import Tensor
from torch_cluster import random_walk

from torch_geometric.loader import NeighborSampler


class PositiveNegativeNeighbourSampler(NeighborSampler):
    def __init__(self, walk_length, *args, **kwargs):
        super(PositiveNegativeNeighbourSampler, self).__init__(*args, **kwargs)
        self.walk_length = walk_length

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        """
        For each node we sample two nodes:
            z_v: positive node from nearest random walk
            z_{vn}: negative node as random
        """
        z_v_batch = random_walk(row, col, batch, self.walk_length, coalesced=False)[:, 1]
        z_vn_batch = torch.randint(low=0,
                                   high=self.adj_t.size(1),
                                   size=(batch.numel(), ),
                                   dtype=torch.long)
        batch = torch.cat([batch, z_v_batch, z_vn_batch], dim=0)
        return super(PositiveNegativeNeighbourSampler, self).sample(batch)
