import torch
from torch import Tensor
from torch_cluster import random_walk

from torch_geometric.loader import NeighborSampler


class PositiveNegativeNeighbourSampler(NeighborSampler):
    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        """
        For each node we sample two nodes:
            z_v: positive node from nearest random walk
            z_{vn}: negative node as random
        TODO: walk len as a parameter
        """
        walk_length = 1
        z_v_batch = random_walk(row, col, batch, walk_length, coalesced=False)[:, 1]
        z_vn_batch = torch.randint(low=0,
                                   high=self.adj_t.size(1),
                                   size=(batch.numel(), ),
                                   dtype=torch.long)
        batch = torch.cat([batch, z_v_batch, z_vn_batch], dim=0)
        return super(PositiveNegativeNeighbourSampler, self).sample(batch)
