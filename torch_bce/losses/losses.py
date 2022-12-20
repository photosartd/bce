from abc import ABC, abstractmethod
from typing import Literal
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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

    @torch.no_grad()
    def inference(self, m_k: torch.Tensor):
        return self.backward_transformation(m_k)


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
        self.transformation = Parameter(torch.eye(max(M, N), requires_grad=True)[:M, :N],
                                        requires_grad=True) if no_trans else nn.Linear(M, N)

    def forward(self, m: torch.Tensor):
        """Linear transformation W x m"""
        if self.no_trans:
            return torch.matmul(m, self.transformation)
        else:
            return self.transformation(m)


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

    @torch.no_grad()
    def inference(self, m_k: torch.Tensor, ver: int = -1):
        """Calculation m_k_j
        Arguments:
            :param m_k: tensor with new embeddings space
            :param ver: embedding space version; -1 means only self.backward_transformation will be used
        """
        return m_k if ver == 0 else self.alignment.inference(bm_k=self.backward_transformation.inference(m_k), ver=ver)

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
        self.alignment.append(self.backward_transformation)
        self.backward_transformation = BackwardTransformation(self.backward_transformation_type, **kwargs)


class Alignment(nn.Module, ABC):
    @abstractmethod
    def append(self, backward_transformation: BackwardTransformation, **kwargs):
        raise NotImplementedError()

    def inference(self, bm_k: torch.Tensor, ver: int = -1, *args, **kwargs):
        """Calls inference in inner backward transformation if necessary"""
        return bm_k


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
        return F.mse_loss(bm_k, m_k_1, reduction="sum")

    def append(self, backward_transformation: BackwardTransformation, **kwargs):
        pass


class MultiStepAlignment(Alignment):
    """Multi step alignment as discussed in https://arxiv.org/pdf/2206.03040.pdf.
    Also called JointLinM[Loss]
    """

    def __init__(self, all_backward_transformations: nn.ModuleList = None, **kwargs):
        """
        Arguments:
            :param w_all: weights of M_{1..k-1}
            :param kwargs:
        """
        super(MultiStepAlignment, self).__init__()
        self.all_backward_transformations: nn.ModuleList = all_backward_transformations if \
            all_backward_transformations is not None else nn.ModuleList()

    def forward(self, bm_k: torch.Tensor, m_k_1: torch.Tensor):
        """
        Arguments:
            :param bm_k: backward transformed embedding space of M_k
            :param m_k_1: embedding space before latest

            :return: mse between them
        """
        # IMPORTANT: bm_k has already been transformed with all_backward_transformations[-1]
        delta = (bm_k - m_k_1)
        last_term = F.mse_loss(delta, torch.zeros(delta.shape), reduction="sum")
        losses = [last_term]
        for i in range(-1, -(len(self.all_backward_transformations) + 1), -1):
            current_term = reduce(
                lambda tensor, module: module(tensor),
                reversed(self.all_backward_transformations[i:]),
                delta
            )
            current_term = F.mse_loss(current_term, torch.zeros(current_term.shape), reduction="sum")
            losses.append(current_term)
        alignment_loss = sum(losses)
        k = len(self.all_backward_transformations) + 1
        return alignment_loss / k

    def inference(self, bm_k: torch.Tensor, ver: int = -1, *args, **kwargs):
        """Makes inference on input backward transformed tensor sequentially"""
        return reduce(
            lambda m_curr, bt: bt.inference(m_curr),
            self.all_backward_transformations[:ver:-1],
            bm_k
        )

    def append(self, backward_transformation: BackwardTransformation, **kwargs):
        self.all_backward_transformations.append(backward_transformation.backward_transformation.transformation)
