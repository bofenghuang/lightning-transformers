# coding=utf-8

"""
https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
"""

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super(MultiClassFocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


def multi_label_focal_loss_with_logits(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    weight: Optional[Union[int, float, List, np.ndarray]] = None,
    gamma: float = 2.0,
    reduction: str = "none",
):
    """Function that computes Binary Focal Loss."""
    if len(input_tensor.shape) != len(target_tensor.shape):
        raise ValueError(
            "Expected input shape ({}) to match target shape ({}).".format(input_tensor.shape, target_tensor.shape)
        )

    if input_tensor.size(0) != target_tensor.size(0):
        raise ValueError(
            "Expected input batch_size ({}) to match target batch_size ({}).".format(
                input_tensor.size(0), target_tensor.size(0)
            )
        )

    if input_tensor.size(1) != target_tensor.size(1):
        raise ValueError(
            "Expected input num_label ({}) to match target num_label ({}).".format(
                input_tensor.size(1), target_tensor.size(1)
            )
        )

    num_labels = input_tensor.size(1)

    if isinstance(weight, (int, float)):
        weight = torch.full((num_labels,), weight, device=input_tensor.device)
    elif isinstance(weight, (list, np.ndarray)):
        weight = torch.as_tensor(weight, device=input_tensor.device)
    else:
        # raise ValueError(f'Got wrong datatype for "weight": {type(weight)}')
        pass

    bce_loss = F.binary_cross_entropy_with_logits(
        input_tensor,
        target_tensor,
        pos_weight=weight,
        reduction="none",
    )

    pt = torch.exp(-bce_loss)
    loss_tmp = torch.pow((1 - pt), gamma) * bce_loss

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Got invalid reduction mode: {reduction}")

    return loss


class MultiLabelFocalLossWithLogits(nn.Module):
    """Criterion that computes Multi-label Focal Loss."""

    def __init__(
        self,
        weight: Optional[Union[int, float, List, np.ndarray]] = None,
        gamma: float = 2.0,
        reduction: str = "none",
    ):
        super(MultiLabelFocalLossWithLogits, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-8

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        return multi_label_focal_loss_with_logits(
            input_tensor,
            target_tensor,
            weight=self.weight,
            gamma=self.gamma,
            reduction=self.reduction,
        )


# -- obsolete
# pos_weight = true / total
def multi_label_focal_loss_with_logits_tmp(
    input_tensor,
    target_tensor,
    weight=0.5,
    gamma=2.0,
    reduction="none",
    device=torch.device("cpu"),
):
    if len(input_tensor.shape) != len(target_tensor.shape):
        raise ValueError(
            "Expected input shape ({}) to match target shape ({}).".format(input_tensor.shape, target_tensor.shape)
        )

    if input_tensor.size(0) != target_tensor.size(0):
        raise ValueError(
            "Expected input batch_size ({}) to match target batch_size ({}).".format(
                input_tensor.size(0), target_tensor.size(0)
            )
        )

    if input_tensor.size(1) != target_tensor.size(1):
        raise ValueError(
            "Expected input num_label ({}) to match target num_label ({}).".format(
                input_tensor.size(1), target_tensor.size(1)
            )
        )

    num_labels = input_tensor.size(1)

    if isinstance(weight, float) or isinstance(weight, int):
        weight = torch.full((num_labels,), weight, device=device)
    elif isinstance(weight, list):
        weight = torch.tensor(weight, device=device)
    elif isinstance(weight, np.ndarray):
        weight = torch.from_numpy(weight).to(device)
    else:
        pass

    bce_loss = F.binary_cross_entropy_with_logits(
        input_tensor,
        target_tensor,
        reduction="none",
    )

    pt = torch.exp(-bce_loss)

    weight = target_tensor * weight + (1 - target_tensor) * (1 - weight)

    loss_tmp = weight * torch.pow((1 - pt), gamma) * bce_loss

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}".format(reduction))
    return loss
