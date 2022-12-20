import os
from typing import List
from functools import reduce

import torch

from torch_bce.interfaces import Saveable


class WeightsContainer(list, Saveable):
    """Container with matrixes from difference iterations/models of intended learning"""

    def __init__(self, embeddings: List[torch.Tensor] = [], **kwargs):
        assert all([isinstance(obj, torch.Tensor) for obj in embeddings]), "All objects must be torch.Tensors"
        assert all([emb_1.shape[0] == emb_2.shape[-1] for emb_1, emb_2 in zip(embeddings[:-1], embeddings[1:])]), \
            "All shapes should be so arranged matrixes are dot-multiplicable"
        super(WeightsContainer, self).__init__(embeddings, **kwargs)

    def append(self, W: torch.Tensor) -> None:
        assert isinstance(W, torch.Tensor), "Weights must be torch.Tensor"
        if self.__len__():
            assert W.shape[-1] == self[-1].shape[0], f"New matrix dim N of [M, N] must be == to" \
                                                     f"N of prev iteration [N, J]: {self[-1].shape[0]}," \
                                                     f"but was: {W.shape[-1]}"
        return super(WeightsContainer, self).append(W)

    def __getitem__(self, key) -> torch.Tensor:
        """Getting transformations for previous iterations"""
        result = super().__getitem__(key)
        if isinstance(result, list):
            if len(result) == 1:
                return result[0]
            return reduce(torch.matmul, reversed(result))
        return result

    @classmethod
    def load(cls, path, device: torch._C.device = "cpu", **kwargs):
        model_files: List[str] = sorted(os.listdir(path))
        embeddings: List[torch.Tensor] = [torch.load(os.path.join(path, current_path), map_location=device) for
                                          current_path in model_files]
        return cls(embeddings, **kwargs)

    def save(self, savepath, **kwargs) -> None:
        for index, embeddings in enumerate(self):
            current_path = os.path.join(savepath, f"{index}.pt")
            torch.save(embeddings, current_path)
