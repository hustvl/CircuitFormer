import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_mse_loss(inputs, targets, weights=None):
    loss = F.mse_loss(inputs, targets, reduce=False)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        loss2 = 0.5*delta_y2*delta_y2 - self.C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))