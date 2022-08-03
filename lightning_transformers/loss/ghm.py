# coding=utf-8

"""
https://github.com/libuyu/mmdetection/blob/master/mmdet/models/losses/ghm_loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GHMCLoss(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(
        self,
        bins: int = 30,
        momentum: float = 0.75,
        use_sigmoid: bool = True,
        loss_weight: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        super(GHMCLoss, self).__init__()
        # num of edges
        self.bins = bins
        # EMA (Exponential Moving Average) momentum
        self.momentum = momentum
        self.edges = torch.arange(bins + 1, device=device).float() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins, device=device)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """Calculate the GHM-C loss.

        Args:
            preds (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            targets (float tensor of size [batch_num, class_num]):
                Binary class targets for each sample.
        Returns:
            The gradient harmonized loss.
        """
        # the targets should be binary class label
        if preds.dim() != targets.dim():
            targets = F.one_hot(targets, num_classes=preds.size(-1))

        targets = targets.float()
        weights = torch.zeros_like(preds)

        # gradient length
        g = torch.abs(preds.sigmoid().detach() - targets)

        tot = max(preds.numel(), 1.0)
        # n valid bins
        n = 0
        for i in range(self.bins):
            # get examples idx whose gradient is in bins
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            # get examples num in bins
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                # set weights for examples in bins
                if self.momentum > 0:
                    # EMA
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1

        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(preds, targets, weights, reduction="mean")

        return loss * self.loss_weight


class MultiClassGHMCLoss(nn.Module):
    """Multi-class GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(
        self,
        bins: int = 30,
        momentum: float = 0.75,
        eps: float = 1e-10,
        use_softmax: bool = True,
        loss_weight: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        super(MultiClassGHMCLoss, self).__init__()
        # num of edges
        self.bins = bins
        # EMA (Exponential Moving Average) momentum
        self.momentum = momentum
        self.eps = eps
        self.edges = torch.arange(bins + 1, device=device).float() / bins
        self.edges[-1] += eps
        if momentum > 0:
            self.acc_sum = torch.zeros(bins, device=device)
        self.device = device
        self.use_softmax = use_softmax
        self.loss_weight = loss_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """Calculate the GHM-C loss.

        Args:
            preds (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            targets (float tensor of size [batch_num,]):
                Binary class targets for each sample.
        Returns:
            The gradient harmonized loss.
        """
        # binarize targets labels
        targets_ohe = F.one_hot(targets, num_classes=preds.size(-1))
        targets_ohe = targets_ohe.float()

        if self.use_softmax:
            # logits to preds
            preds = F.softmax(preds, dim=1)
            preds = torch.clamp(preds, min=self.eps, max=1 - self.eps)

        # calc the norm of gradient
        g = torch.abs(preds.detach() - targets_ohe)

        # take only true scores
        true_idx_0 = torch.arange(start=0, end=preds.size(0))
        # true_idx_1 = torch.argmax(targets_ohe, dim=1)
        true_idx_1 = targets
        g = g[true_idx_0, true_idx_1]

        weights = torch.zeros_like(g, device=self.device)

        tot = max(g.numel(), 1.0)
        # n valid bins
        n = 0
        for i in range(self.bins):
            # get examples idx whose gradient is in bins
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            # get examples num in bins
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                # set weights for examples in bins
                if self.momentum > 0:
                    # EMA
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1

        if n > 0:
            weights = weights / n

        # calc cross entropy
        loss = F.nll_loss(torch.log(preds), targets, reduction="none")
        # weighted on batch
        loss = (loss * weights).mean()

        return loss * self.loss_weight


class MultiLabelGHMCLossWithLogits(nn.Module):
    """Multi-label GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(
        self,
        bins: int = 30,
        momentum: float = 0.75,
        use_sigmoid: bool = True,
        loss_weight: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        super(MultiLabelGHMCLossWithLogits, self).__init__()
        # num of edges
        self.bins = bins
        # EMA (Exponential Moving Average) momentum
        self.momentum = momentum
        self.edges = torch.arange(bins + 1, device=device).float() / bins
        self.edges[-1] += 1e-6
        #         if momentum > 0:
        #             self.acc_sum = torch.zeros(bins, device=device)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.device = device
        self.loss_weight = loss_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """Calculate the GHM-C loss.

        Args:
            preds (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            targets (float tensor of size [batch_num, class_num]):
                Binary class targets for each sample.
        Returns:
            The gradient harmonized loss.
        """
        # the targets should be binary class label
        # TODO: not for multi label
        if preds.dim() != targets.dim():
            targets = F.one_hot(targets, num_classes=preds.size(-1))

        targets = targets.float()
        weights = torch.zeros_like(preds, device=self.device)

        # ? check shape
        num_example, num_label = preds.size()

        if not hasattr(self, "acc_sum") and self.momentum > 0:
            self.acc_sum = torch.zeros((self.bins, num_label), device=self.device)

        # gradient length
        g = torch.abs(preds.sigmoid().detach() - targets)

        # example num
        tot = max(num_example, 1.0)

        # n valid bins for each label
        n = torch.zeros(num_label, dtype=torch.long, device=self.device)
        for i in range(self.bins):
            # get examples idx whose gradient is in bins
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            # count number in region by label
            num_in_bin = inds.sum(dim=0)

            # update valid bin num
            n += torch.where(num_in_bin >= 1, 1, 0)

            if self.momentum > 0:
                # EMA
                self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin

                weights[inds] = tot / torch.stack((self.acc_sum[i],) * num_example, dim=0)[inds]
            else:
                weights[inds] = tot / torch.stack((num_in_bin,) * num_example, dim=0)[inds]

        n_expanded = torch.stack((n,) * num_example, dim=0)
        n_expanded = torch.where(n_expanded >= 1, n_expanded, 1)
        weights = weights / n_expanded

        # print(weights)

        loss = F.binary_cross_entropy_with_logits(preds, targets, weights, reduction="mean")

        return loss * self.loss_weight
