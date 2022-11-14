import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit
import wandb

from src.models import GraphSAGE, MLPClassifier
from src.utils import PositiveNegativeNeighbourSampler
from src.utils.datasets import TensorSupervisedDataset
from src.losses import GraphSageLoss


logger = logging.getLogger("bce")
logger.setLevel(logging.INFO)
logging.basicConfig()


def main():
    num_epochs = 200
    wandb.login(key="ff032b7d576dfc8849b5c26d72c084cfc88f7b76")
    wandb.init(project="bce", reinit=True)
    config = wandb.config
    config.epochs = num_epochs
    WALK_LENGTH = 5

    dataset = "Cora"
    path = "./data"
    path_1 = "./data_1"
    dataset = Planetoid(path, dataset, transform=T.Compose([T.NormalizeFeatures(), T.RandomLinkSplit()]))
    dataset_test = Planetoid(path_1, "Cora", transform=T.Compose([T.NormalizeFeatures()]))
    #dataset = Reddit(path, transform=T.NormalizeFeatures())
    data = dataset[0]
    data_v = dataset_test[0]

    """train_loader = PositiveNegativeNeighbourSampler(data.edge_index, sizes=[4], batch_size=256,
                                   shuffle=True, num_nodes=data.num_nodes)"""
    train_loader = PositiveNegativeNeighbourSampler(WALK_LENGTH, data[0].edge_index, sizes=[5, 2], batch_size=256,
                                                    shuffle=False, num_nodes=data[0].num_nodes)
    val_loader = PositiveNegativeNeighbourSampler(WALK_LENGTH, data[1].edge_index, sizes=[5, 2], batch_size=256,
                                                  shuffle=False, num_nodes=data[1].num_nodes)
    test_loader = PositiveNegativeNeighbourSampler(WALK_LENGTH, data[2].edge_index, sizes=[5, 2], batch_size=256,
                                                   shuffle=False, num_nodes=data[2].num_nodes)
    new_val_loader = PositiveNegativeNeighbourSampler(WALK_LENGTH, data_v.edge_index, sizes=[5, 2], batch_size=256,
                                                  shuffle=False, num_nodes=data_v.num_nodes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Num node features: {data[0].num_node_features}")
    model = GraphSAGE(loss=GraphSageLoss(), in_channels=data[0].num_node_features, hid_channels=128, out_channels=32,
                      n_layers=2, device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #x, edge_index = data.x.to(device), data.edge_index.to(device)
    x_train, edge_index_train = data[0].x.to(device), data[0].edge_index.to(device)
    x_val, edge_index_val = data[1].x.to(device), data[1].edge_index.to(device)
    x_test, edge_index_test = data[2].x.to(device), data[2].edge_index.to(device)
    x_new_val, edge_index_val_new = data_v.x.to(device), data_v.edge_index.to(device)

    for epoch in range(1, num_epochs + 1):
        loss = model.train_loop(x=x_train,
                           train_dataloader=train_loader,
                           optimizer=optimizer,
                           num_nodes=data[0].num_nodes)
        predictions, loss_val = model.predict(x_val,
                                 val_loader,
                                 num_nodes=data[1].num_nodes
                                 )
        print(f'Epoch: {epoch:03d}, train loss: {loss:.4f}, val loss: {loss_val:.4f}')
        wandb.log({"loss": loss,
                   "val_loss": loss_val
                   })
    print(f"Predictions size: {predictions.shape}")
    print(f"Finished learning. Final loss: {loss_val:.4f}")
    wandb.finish()

    #get predictions for nodes
    predictions, loss = model.predict(x_new_val,
                                      new_val_loader,
                                      num_nodes=data_v.num_nodes
                                      )

    wandb.init()
    dataset = TensorSupervisedDataset(x=predictions, y=data_v.y)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    net = MLPClassifier(loss=nn.CrossEntropyLoss(), input_size=32, output_size=7, div=3)
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(1, num_epochs + 1):
        loss, metrics = net.train_loop(dataloader, optimizer)
        wandb.log({"mlp_loss": loss})
    y_pred, loss, metrics = net.predict(dataloader)
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()