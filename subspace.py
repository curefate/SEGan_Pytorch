import numpy as np
import torch
from torch import nn


class EigenSpace(nn.Module):
    def __init__(self, res, channel, dim):
        super(EigenSpace, self).__init__()
        self.U = nn.Parameter(torch.randn(res, res, channel, dim))
        self.L = nn.Parameter(torch.randn(dim))
        self.mu = nn.Parameter(torch.randn(res, res, channel))

    def forward(self, style):
        temp = self.U[None, :] * (self.L[None, :] * style)[:, None, None, None, :]
        h = torch.sum(temp, dim=-1) + self.mu[None, :]
        return h


class EigenSpace_2style(nn.Module):
    def __init__(self, res, channel, dim):
        super(EigenSpace_2style, self).__init__()
        self.U = nn.Parameter(torch.randn(res, res, channel, dim))
        self.L = nn.Parameter(torch.randn(dim))
        self.mu = nn.Parameter(torch.randn(dim))

    def forward(self, style):
        temp = (self.L[None, :] * style)[:, None, None, None, :]
        temp = self.U[None, :] * temp
        h = torch.sum(temp, dim=[1, 2, 3]) + self.mu
        return h


class EigenSpace_onlyL(nn.Module):
    def __init__(self, dim):
        super(EigenSpace_onlyL, self).__init__()
        self.L = nn.Parameter(torch.randn(dim))

    def forward(self, style):
        out = style * self.L
        return out


if __name__ == '__main__':
    res = 16
    channel = 32
    dim = 512
    z = torch.randn(8, dim)
    print(z.shape)
    a = EigenSpace(res, channel, dim)
    b = EigenSpace_2style(res, channel, dim)
    c = EigenSpace_onlyL(dim)
    out1 = a(z)
    print(out1.shape)
    out2 = b(z)
    print(out2.shape)
    out3 = c(z)
    print(out3.shape)
