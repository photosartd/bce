import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import wandb

from torch_bce.models import GraphSAGE, MLPClassifier, MLPRegressor
from torch_bce.utils import PositiveNegativeNeighbourSampler
from torch_bce.utils.datasets import TensorSupervisedDataset
from torch_bce.losses import GraphSageLoss


logger = logging.getLogger("bce")
logger.setLevel(logging.INFO)
logging.basicConfig()


def main():
    SEED = 42
    num_epochs = 10
    wandb.login(key="ff032b7d576dfc8849b5c26d72c084cfc88f7b76")
    wandb.init(project="bce", reinit=True)
    config = wandb.config
    config.epochs = num_epochs
    WALK_LENGTH = 3

    #Dataset
    loader = WikiMathsDatasetLoader()#ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(loss=GraphSageLoss(), in_channels=dataset[0].num_node_features, hid_channels=128, out_channels=32,
                      n_layers=2, device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    #mlp
    net = MLPRegressor(loss=nn.MSELoss(), input_size=32, output_size=1, div=3)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        loss = 0
        step = 0
        for data in dataset:
            torch.manual_seed(SEED)
            transform = T.RandomLinkSplit()
            data_train, data_val, data_test = transform(data)
            train_loader = PositiveNegativeNeighbourSampler(WALK_LENGTH, data_train.edge_index, sizes=[2, 1], batch_size=256,
                                                            shuffle=False, num_nodes=data_train.num_nodes)
            val_loader = PositiveNegativeNeighbourSampler(WALK_LENGTH, data_val.edge_index, sizes=[2, 1], batch_size=256,
                                                          shuffle=False, num_nodes=data_val.num_nodes)
            test_loader = PositiveNegativeNeighbourSampler(WALK_LENGTH, data_test.edge_index, sizes=[2, 1], batch_size=256,
                                                           shuffle=False, num_nodes=data_test.num_nodes)

            x_train, edge_index_train = data_train.x.to(device), data_train.edge_index.to(device)
            x_val, edge_index_val = data_val.x.to(device), data_val.edge_index.to(device)
            x_test, edge_index_test = data_test.x.to(device), data_test.edge_index.to(device)

            """loss = model.train_loop(x=x_train,
                                    train_dataloader=train_loader,
                                    optimizer=optimizer,
                                    num_nodes=data_train.num_nodes)
            predictions, loss_val = model.predict(x_val,
                                                  val_loader,
                                                  num_nodes=data_val.num_nodes
                                                  )"""
            model.train()
            losses_train = []
            for batch_size, n_id, adjs in train_loader:
                if 2 < 2:
                    adjs = [adjs]
                adjs = [adj.to(device) for adj in adjs]


                out = model.forward(x_train[n_id], adjs)
                z_u, z_v, z_vn = out.split(out.size(0) // 3, dim=0)[:3]
                loss_curr = GraphSageLoss()(z_u, z_v, z_vn)
                loss += loss_curr
                losses_train.append(float(loss_curr.item()) * z_u.size(0))
                step += 1
            ml = np.sum(losses_train) / data_train.num_nodes
            #val

            model.eval()
            losses: List[float] = []
            predictions: List[torch.Tensor] = []
            mean_loss = None
            loss_val = 0
            with torch.no_grad():
                for batch_size, n_id, adjs in val_loader:
                    if 2 < 2:
                        adjs = [adjs]
                    adjs = [adj.to(device) for adj in adjs]
                    out = model.forward(x_val[n_id], adjs)
                    z_u, z_v, z_vn = out.split(out.size(0) // 3, dim=0)[:3]
                    predictions.append(z_u.detach().cpu())
                    loss_curr = GraphSageLoss()(z_u, z_v, z_vn)
                    loss_val += loss_curr
                    losses.append(float(loss_curr.item()) * z_u.size(0))

                predictions = torch.vstack(predictions)
                mean_loss = np.sum(losses) / data_val.num_nodes
                #logger.info(f"Predict mean loss: {mean_loss}")

        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tensor_dataset = TensorSupervisedDataset(x=predictions, y=data_val.y)
        dataloader = DataLoader(tensor_dataset, batch_size=256, shuffle=False)
        for epoch in range(1, num_epochs + 1):
            loss, metrics = net.train_loop(dataloader, opt)
        y_pred, loss, metrics = net.predict(dataloader)

        print(f'Epoch: {epoch:03d}, train loss: {ml:.4f}, val loss: {mean_loss:.4f}, regression metrics: {metrics}')
        wandb.log({"loss": ml,
                   "val_loss": mean_loss,
                   "mse_loss": metrics
                   })
    print(f"Predictions size: {predictions.shape}")
    print(f"Finished learning. Final loss: {mean_loss:.4f}")
    print(f"Final regression metrics: {metrics}")
    wandb.finish()


if __name__ == "__main__":
    main()


