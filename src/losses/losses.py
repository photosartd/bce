import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSageLoss(nn.Module):
    def forward(self, z_u: torch.Tensor,
                z_v: torch.Tensor,
                z_vn: torch.Tensor):
        positive_loss = F.logsigmoid((z_u * z_v).sum(-1)).mean()
        negative_loss = F.logsigmoid(-(z_u * z_vn).sum(-1)).mean()
        return -positive_loss - negative_loss


class JointLinSingleStepLoss(nn.Module):
    def __init__(self, M, N, device):
        """
        :param M: new embedding size
        :param N: old embedding size
        W = [N, M]
        """
        self.W = torch.randn((N, M), device=device, requires_grad=True)
        self.mse_loss = nn.MSELoss()

    def forward(self, v, u):
        return self.mse_loss(torch.matmul(self.W, v), u)
