import os
from typing import List, Type

from torch_bce.interfaces import ModelInterface
from torch_bce.interfaces.container_interface import ContainerInterface


class ListModelContainer(ContainerInterface):
    """Container with models stored in list"""

    def __init__(self, models: List[ModelInterface] = [], **kwargs):
        super(ListModelContainer, self).__init__(models, **kwargs)
        assert isinstance(self.models, list), "models argument was not list"

    def add(self, model: ModelInterface) -> None:
        if isinstance(self.models, list) and isinstance(model, ModelInterface):
            self.models.append(model)
        else:
            raise TypeError("Either models was not List or model was not ModelInterface")

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, item):
        if isinstance(self.models, list):
            return self.models.__getitem__(item)
        else:
            raise TypeError("Models attribute was not List")

    @classmethod
    def load(cls, path, class_type: Type[ModelInterface] = ModelInterface, **kwargs):
        """Loader for models to container.
        TODO: for now supposes only one class_type of loaded models, but may require more
        Arguments:
        :param path: path to folder with models
        :param class_type: class_type for model loading
        :param kwargs:
        :return: ListModelContainer
        """
        model_files: List[str] = sorted(os.listdir(path))
        models: List[ModelInterface] = [class_type.load(os.path.join(path, current_path)) for current_path in
                                        model_files]
        return cls(models, **kwargs)
