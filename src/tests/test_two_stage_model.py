import logging
import unittest

from src.trainers import GraphSAGETwoStageTrainer


"""Model"""
kwargs = {
    "walk_length": 3,
    "batch_size": 256,
    "sizes": [2, 1],
    "lags": 8,
    "train_ratio": 0.8,
    "lambda_": 8,
    "alignment": "single_step",
    "backward_transformation": "linear",
    "n_layers": 2,
    "out_channels": 32,
    "target_size": 1,
    "div": 3,
    "level": logging.INFO
}

model = GraphSAGETwoStageTrainer(**kwargs)


class TestGraphSAGETwoStageTrainer(unittest.TestCase):
    #@unittest.skip("Skipped model training test")
    def test_fit_works(self):
        model.fit(
            num_epochs=30,
            disable_progress=True
        )
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
