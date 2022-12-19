from typing import Tuple

import torch


class TensorDataset(torch.utils.data.Dataset):
    """Usual unlabelled dataset.

    Attributes:
        x: torch.Tensor
    """

    def __init__(self, x: torch.Tensor):
        self.x: torch.Tensor = x

    def __getitem__(self, index) -> torch.Tensor:
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]


class TensorSupervisedDataset(TensorDataset):
    """Usual labelled dataset.

        Attributes:
            x: torch.Tensor
            y: torch.Tensor
        """
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        super().__init__(x)
        self.y: torch.Tensor = y

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]
