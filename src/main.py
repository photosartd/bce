import logging

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import wandb

from src.models import GraphSAGE
from src.utils import PositiveNegativeNeighbourSampler
from src.losses import GraphSageLoss


logger = logging.getLogger("bce")
logger.setLevel(logging.INFO)
logging.basicConfig()


def main():
    wandb.login(key="ff032b7d576dfc8849b5c26d72c084cfc88f7b76")
    wandb.init(project="bce")
    config = wandb.config
    config.epochs = 50

    dataset = "Cora"
    path = "./data"
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    train_loader = PositiveNegativeNeighbourSampler(data.edge_index, sizes=[4], batch_size=256,
                                   shuffle=True, num_nodes=data.num_nodes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Num node features: {data.num_node_features}")
    model = GraphSAGE(loss=GraphSageLoss(), in_channels=data.num_node_features, hid_channels=64, out_channels=32,
                      n_layers=1, device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, edge_index = data.x.to(device), data.edge_index.to(device)

    num_epochs = 50

    for epoch in range(1, num_epochs + 1):
        loss = model.train_loop(x=x,
                           train_dataloader=train_loader,
                           optimizer=optimizer,
                           num_nodes=data.num_nodes)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        wandb.log({"loss": loss})

    print(f"Finished learning. Final loss: {loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()