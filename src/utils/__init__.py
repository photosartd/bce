import torch

from .samplers import PositiveNegativeNeighbourSampler

SEED = 42


def seed(func):
    def wrapper(*args, **kwargs):
        torch.manual_seed(SEED)
        return func(*args, **kwargs)

    return wrapper
