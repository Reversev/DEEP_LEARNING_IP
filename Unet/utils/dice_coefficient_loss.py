#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2022/10/13 10:49
# @Author : ''
# @FileName: dice_score.py
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt


import torch
import torch.nn as nn


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: Tensor, target: Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches or for a single mask
    if x.dim() == 2:
        # compute and average metric for single mask
        x_i, t_i = x.reshape(-1), target.reshape(-1)
        if ignore_index >= 0:
            # find not ignore_index area in mask
            roi_mask = torch.ne(t_i, ignore_index)   # element-wise compare, 0 if t_i == ignore_index else 1
            x_i, t_i = x_i[roi_mask], t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)

    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(x.shape[0]):
            dice += dice_coeff(x[i, ...], target[i, ...])
        return dice / x.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0.0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)
    return dice / input.shape[1]


def dice_loss(x: Tensor, target: Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    assert x.size() == target.size()
    x = F.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplot(1, classes + 1)
    ax[0].set_title("Input Image")
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f"Output mask (class {i+1})")
            ax[i+1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f"Output mask")
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
