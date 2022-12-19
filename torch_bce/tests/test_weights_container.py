import unittest
from tempfile import TemporaryDirectory

import torch

from torch_bce.containers import WeightsContainer

"""Examples"""
w_k_1 = torch.randn((128, 32))
w_k_2 = torch.randn((32, 64))
w_k_3 = torch.randn((64, 256))
weights_container = WeightsContainer([w_k_3, w_k_2, w_k_1])


class EmbeddingSpaceContainerTest(unittest.TestCase):
    def test_can_append(self):
        w_k = torch.randn((16, 128))
        self.assertIsNone(weights_container.append(w_k))

    def test_cant_append(self):
        t_4 = torch.randn((130, 130))
        with self.assertRaises(AssertionError) as context:
            weights_container.append(t_4)

    def test_get_item(self):
        self.assertTrue(weights_container[0:3].shape == (128, 256))
        self.assertTrue(torch.all(weights_container[0:3] == torch.matmul(torch.matmul(w_k_1, w_k_2), w_k_3)).item())

    def test_can_save_and_load(self):
        with TemporaryDirectory() as d:
            weights_container.save(d)
            new_emb_space_container = WeightsContainer.load(d)
            self.assertTrue(torch.all(new_emb_space_container[0] == weights_container[0]).item())
            self.assertTrue(torch.all(new_emb_space_container[1] == weights_container[1]).item())


if __name__ == "__main__":
    unittest.main()
