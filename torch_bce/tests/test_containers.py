import unittest
import copy
from tempfile import TemporaryDirectory

import torch

from torch_bce.models import GraphSAGE
from torch_bce.losses import GraphSageLoss
from torch_bce.containers import ListModelContainer

"""Examples"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels_1 = 100
in_channels_2 = 200
n_layers_1 = 2
n_layers_2 = 3
model_1 = GraphSAGE(loss=GraphSageLoss(), in_channels=in_channels_1, hid_channels=128, out_channels=32,
                    n_layers=n_layers_1, device=device)
model_2 = GraphSAGE(loss=GraphSageLoss(), in_channels=in_channels_2, hid_channels=256, out_channels=16,
                    n_layers=n_layers_2, device=device)
models = [model_1, model_2]
NUMBER_OF_MODELS = len(models)

model_container = ListModelContainer(models)


class ListModelContainerTest(unittest.TestCase):
    def test_models_is_list_of_N(self):
        self.assertEqual(len(model_container), NUMBER_OF_MODELS)

    def test_can_add(self):
        model_new = GraphSAGE(loss=GraphSageLoss(), in_channels=300, hid_channels=100, out_channels=8, n_layers=2,
                              device=device)
        model_container_cpy = copy.deepcopy(model_container)
        model_container_cpy.add(model_new)
        new_number_of_models = NUMBER_OF_MODELS + 1
        self.assertEqual(len(model_container_cpy), new_number_of_models)

    def test_iter(self):
        for index, model in enumerate(model_container):
            self.assertIs(model, model_container.models[index])

    def test_can_slice(self):
        self.assertEqual(len(model_container[:NUMBER_OF_MODELS - 1]),
                         len(model_container.models[:NUMBER_OF_MODELS - 1]))

    def test_can_save_and_load(self):
        with TemporaryDirectory() as d:
            model_container.save(d)
            new_container = model_container.load(d, GraphSAGE)
            self.assertEqual(new_container[0].n_layers, n_layers_1)
            self.assertEqual(new_container[1].n_layers, n_layers_2)


if __name__ == "__main__":
    unittest.main()
