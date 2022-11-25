from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.containers import WeightsContainer


class GraphSageLoss(nn.Module):
    def forward(self, z_u: torch.Tensor,
                z_v: torch.Tensor,
                z_vn: torch.Tensor):
        positive_loss = F.logsigmoid((z_u * z_v).sum(-1)).mean()
        negative_loss = F.logsigmoid(-(z_u * z_vn).sum(-1)).mean()
        return -positive_loss - negative_loss


class BackwardTransformation(nn.Module):
    def __init__(self, backward_transformation: Literal["linear", "no_trans"], **kwargs):
        super(BackwardTransformation, self).__init__()
        self.backward_transformation = LinearBackwardTransformation(
            **kwargs) if backward_transformation == "linear" else LinearBackwardTransformation(no_trans=True,
                                                                                               **kwargs)

    def forward(self, m: torch.Tensor):
        return self.backward_transformation(m)


class LinearBackwardTransformation(nn.Module):
    def __init__(self, M: int, N: int, no_trans: bool = False, **kwargs):
        """
        Arguments:
            :param M: new embedding size
            :param N: old embedding size
            :param no_trans: if W will be identity matrix or not
        W = [N, M]
        """
        super(LinearBackwardTransformation, self).__init__()
        self.M = M
        self.N = N
        self.no_trans = no_trans
        self.W = torch.eye(max(M, N), requires_grad=True)[:M, :N] if no_trans else torch.randn((M, N), requires_grad=True)

    def forward(self, m: torch.Tensor):
        """Linear transformation W x m"""
        return torch.matmul(m, self.W)


class AlignmentLoss(nn.Module):
    """Abstract class for alignment loss"""

    def __init__(self, lambda_: float, backward_transformation: Literal["linear", "no_trans"],
                 alignment: Literal["single_step", "multi_step"], **kwargs):
        """
        Arguments:
            :param lambda_: alignment parameter (coefficient)
            :param backward_transformation: type of backward transformation
            :param alignment: alignment type
            :param kwargs: M, N for backward transformation
        """
        super(AlignmentLoss, self).__init__()
        self.lambda_: float = lambda_
        self.backward_transformation_type = backward_transformation
        self.backward_transformation: BackwardTransformation = BackwardTransformation(self.backward_transformation_type,
                                                                                      **kwargs)
        self.alignment_type = alignment
        self.alignment: Alignment = SingleStepAlignment(**kwargs) if alignment == "single_step" else MultiStepAlignment(
            **kwargs)

    def forward(self, m_k: torch.Tensor, m_k_1: torch.Tensor):
        """Calculation of alignment loss
        Arguments:
            :param m_k: tensor with new embeddings space
            :param m_k_1: tensor with embeddings on step before m_k
        """
        return self.lambda_ * self.alignment(self.backward_transformation(m_k), m_k_1)

    def finish(self, **kwargs) -> None:
        """Finish epoch of M_k, M_k_1. **kwargs must include kwargs for BackwardTransformation
        (in case of MultiStepLoss).
        Step 1: check if multi_step loss
        Step 2: append latest W
        Step 3: renew self.backward_transformation
        Arguments:
            :param kwargs: M, N for BackwardTransformation
            :return: None
        """
        if self.alignment_type == "single_step":
            return
        self.alignment.append(self.backward_transformation)
        self.backward_transformation = BackwardTransformation(self.backward_transformation_type, **kwargs)


class Alignment(nn.Module, ABC):
    @abstractmethod
    def append(self, backward_transformation: BackwardTransformation, **kwargs):
        raise NotImplementedError()


# TODO: implement different alignments
class SingleStepAlignment(Alignment):
    """Single step alignment as discussed in https://arxiv.org/pdf/2206.03040.pdf.
    Also called JointLinS[Loss]
    """

    def __init__(self, **kwargs):
        super(SingleStepAlignment, self).__init__()

    def forward(self, bm_k: torch.Tensor, m_k_1: torch.Tensor):
        """
        Arguments:
            :param bm_k: backward transformed embedding space of M_k
            :param m_k_1: EmbeddingSpace of M_{k - 1}
            :return: mse between them
        """
        return F.mse_loss(bm_k, m_k_1)

    def append(self, backward_transformation: BackwardTransformation, **kwargs):
        pass


class MultiStepAlignment(Alignment):
    """Multi step alignment as discussed in https://arxiv.org/pdf/2206.03040.pdf.
    Also called JointLinM[Loss]
    """

    def __init__(self, w_all: WeightsContainer = None, **kwargs):
        """
        Arguments:
            :param w_all: weights of M_{1..k-1}
            :param kwargs:
        """
        super(MultiStepAlignment, self).__init__()
        self.w_all: WeightsContainer = w_all if w_all is not None else WeightsContainer()

    def forward(self, bm_k: torch.Tensor, m_k_1: torch.Tensor):
        """
        Arguments:
            :param bm_k: backward transformed embedding space of M_k
            :param m_k_1: embedding space before latest

            :return: mse between them
        """
        # IMPORTANT: bm_k has already been transformed with w_all[-1]
        delta = (bm_k - m_k_1)
        last_term = F.mse_loss(delta, torch.zeros(delta.shape))
        losses = [last_term]
        for i in range(-1, -(len(self.w_all) + 1), -1):
            current_term = torch.matmul(delta, self.w_all[i:])
            current_term = F.mse_loss(current_term, torch.zeros(current_term.shape))
            losses.append(current_term)
        alignment_loss = sum(losses)
        k = len(self.w_all) + 1
        return alignment_loss / k

    def append(self, backward_transformation: BackwardTransformation, **kwargs):
        # TODO not extendable, need to be edited
        self.w_all.append(backward_transformation.backward_transformation.W)
