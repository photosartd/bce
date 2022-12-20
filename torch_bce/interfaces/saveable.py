from abc import ABC, abstractmethod


class Saveable(ABC):

    @classmethod
    @abstractmethod
    def load(cls, path, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def save(self, savepath, **kwargs) -> None:
        raise NotImplementedError()
