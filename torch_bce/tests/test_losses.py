import unittest
import logging

import torch
import torch.nn.functional as F

from torch_bce.losses import BackwardTransformation, AlignmentLoss, MultiStepAlignment

logger = logging.getLogger("TestAlignment")
logger.setLevel(logging.INFO)
logging.basicConfig()

"""Variables"""
"""BackwardTransformation"""
BATCH_SIZE = 128
M = 64
N = 32
m = torch.rand((BATCH_SIZE, M))
b_l_1 = BackwardTransformation(backward_transformation="linear", M=M, N=N)
b_no_1 = BackwardTransformation(backward_transformation="no_trans", M=M, N=N)

"""Alignment"""
LAMBDA = 4.0
bk_type_1 = "linear"
bk_type_2 = "no_trans"
alignment_1 = "single_step"
alignment_2 = "multi_step"


class TestBackwardTransformation(unittest.TestCase):
    def test_linear(self):
        self.assertEqual(b_l_1(m).shape, (BATCH_SIZE, N))

    def test_no_trans(self):
        # Test shape
        self.assertEqual(b_no_1(m).shape, (BATCH_SIZE, N))
        # Test the same
        self.assertTrue(torch.all(b_no_1(m) == m[:, :N]).item())


class TestAlignmentLoss(unittest.TestCase):
    def test_alignment_linear_single_step(self):
        loss = AlignmentLoss(
            lambda_=LAMBDA,
            backward_transformation=bk_type_1,
            alignment=alignment_1,
            N=N,
            M=M
        )
        m_k_1 = torch.rand((BATCH_SIZE, N))
        l = loss(m, m_k_1)
        # 1. Check shape
        self.assertEqual(l.shape, ())
        # 2. Check values
        loss_hands = LAMBDA * F.mse_loss(
            loss.backward_transformation.backward_transformation.transformation(m), m_k_1,
            reduction="sum")
        self.assertTrue(torch.all(loss_hands == l).item())
        # 3. Check finish works as expected
        self.assertTrue(len(list(loss.parameters())) == 2)
        loss.finish(M=M, N=N)
        self.assertTrue(len(list(loss.parameters())) == 2, f"was {len(list(loss.parameters()))}")
        new_loss = loss(m, m_k_1)
        self.assertFalse(torch.all(new_loss == l).item())

    def test_alignment_no_trans_single_step(self):
        loss = AlignmentLoss(
            lambda_=LAMBDA,
            backward_transformation=bk_type_2,
            alignment=alignment_1,
            N=N,
            M=M
        )
        m_k_1 = torch.rand((BATCH_SIZE, N))
        l = loss(m, m_k_1)
        # 1. Check shape
        self.assertEqual(l.shape, ())
        # 2. Check values
        loss_hands = LAMBDA * F.mse_loss(torch.matmul(m, torch.eye(max(M, N))[:M, :N]), m_k_1,
                                         reduction="sum")
        self.assertTrue(torch.all(loss_hands == l).item())
        # 3. Check finish works as expected
        self.assertTrue(len(list(loss.parameters())) == 1)
        loss.finish(M=M, N=N)
        self.assertTrue(len(list(loss.parameters())) == 1)
        new_loss = loss(m, m_k_1)
        self.assertTrue(torch.all(new_loss == l).item())
        # 4. Check single step alignment has no backward transformations
        with self.assertRaises(AttributeError):
            loss.alignment.__getattr__("all_backward_transformations")

    def test_alignment_linear_multi_step(self):
        loss = AlignmentLoss(
            lambda_=LAMBDA,
            backward_transformation=bk_type_1,
            alignment=alignment_2,
            N=N,
            M=M
        )
        NEW_N = 128
        m_k_1 = torch.rand((BATCH_SIZE, N))
        l = loss(m, m_k_1)
        # 1. Check shape
        self.assertEqual(l.shape, ())
        # 2. Check values: 1 step alignment
        loss_hands = LAMBDA * F.mse_loss(
            (loss.backward_transformation.backward_transformation.transformation(m) - m_k_1),
            torch.zeros((BATCH_SIZE, N)), reduction="sum") / 1
        self.assertTrue(torch.all(loss_hands == l).item())
        # 3. Check values: 2 step alignment
        m_k1 = torch.rand((BATCH_SIZE, NEW_N))
        self.assertTrue(len(list(loss.parameters())) == 2, f"was {len(list(loss.parameters()))}")
        logger.info(f"Multi_step alignment parameters(len={len(list(loss.parameters()))}) before .finish()")
        loss.finish(M=NEW_N, N=M)
        self.assertTrue(len(list(loss.parameters())) == 4)
        logger.info(f"Multi_step alignment parameters(len={len(list(loss.parameters()))}) after .finish()")
        # 4. Check len of all backward transformations == 1
        self.assertTrue(len(loss.alignment.all_backward_transformations) == 1)
        # 5. Assert weights container size
        if isinstance(loss.alignment, MultiStepAlignment):
            self.assertEqual(len(loss.alignment.all_backward_transformations), 1)
        l_new = loss(m_k1, m)
        # 6. Check shape new
        self.assertEqual(l_new.shape, ())
        # 7. Check hands loss
        delta = loss.backward_transformation.backward_transformation.transformation(m_k1) - m
        second_term = loss.alignment.all_backward_transformations[-1](delta)
        loss_hands_new = LAMBDA * sum(
            [F.mse_loss(delta, torch.zeros(delta.shape), reduction="sum"),
             F.mse_loss(second_term, torch.zeros(second_term.shape), reduction="sum")]
        ) / 2
        self.assertTrue(torch.all(l_new == loss_hands_new).item())


if __name__ == "__main__":
    unittest.main()
