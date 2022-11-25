import unittest

import torch
import torch.nn.functional as F

from src.losses import BackwardTransformation, AlignmentLoss, MultiStepAlignment

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
        loss_hands = LAMBDA * F.mse_loss(torch.matmul(m, loss.backward_transformation.backward_transformation.W), m_k_1)
        self.assertTrue(torch.all(loss_hands == l).item())
        # 3. Check finish works as expected
        loss.finish()
        new_loss = loss(m, m_k_1)
        self.assertTrue(torch.all(new_loss == l).item())

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
        loss_hands = LAMBDA * F.mse_loss(torch.matmul(m, torch.eye(max(M, N))[:M, :N]), m_k_1)
        self.assertTrue(torch.all(loss_hands == l).item())
        # 3. Check finish works as expected
        loss.finish()
        new_loss = loss(m, m_k_1)
        self.assertTrue(torch.all(new_loss == l).item())

    def test_alignment_linear_multi_step(self):
        loss = AlignmentLoss(
            lambda_=LAMBDA,
            backward_transformation=bk_type_1,
            alignment=alignment_2,
            N=N,
            M=M
        )
        m_k_1 = torch.rand((BATCH_SIZE, N))
        l = loss(m, m_k_1)
        # 1. Check shape
        self.assertEqual(l.shape, ())
        # 2. Check values: 1 step alignment
        loss_hands = LAMBDA * F.mse_loss(
            (torch.matmul(m, loss.backward_transformation.backward_transformation.W) - m_k_1),
            torch.zeros((BATCH_SIZE, N))) / 1
        self.assertTrue(torch.all(loss_hands == l).item())
        # 3. Check values: 2 step alignment
        NEW_N = 128
        m_k1 = torch.rand((BATCH_SIZE, NEW_N))
        loss.finish(M=NEW_N, N=M)
        # 4. Assert weights container size
        if isinstance(loss.alignment, MultiStepAlignment):
            self.assertEqual(len(loss.alignment.w_all), 1)
        l_new = loss(m_k1, m)
        # 5. Check shape new
        self.assertEqual(l_new.shape, ())
        # 6. Check hands loss
        delta = torch.matmul(
            m_k1,
            loss.backward_transformation.backward_transformation.W
        ) - m
        second_term = torch.matmul(delta, loss.alignment.w_all[-1:])
        loss_hands_new = LAMBDA * sum(
            [F.mse_loss(delta, torch.zeros(delta.shape)),
            F.mse_loss(second_term, torch.zeros(second_term.shape))]
        ) / 2
        self.assertTrue(torch.all(l_new == loss_hands_new).item())


if __name__ == "__main__":
    unittest.main()
