#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2022/10/7 16:31
# @Author : ''
# @FileName: transforms.py
import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 如图片最短边长小于给定的size 用数据fill进行填充
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型 所以是将图像的最短边缩放到size大小
        img = F.resize(img, size)
        # 这里的插值方法在torchvision(0.9.0)后才有InterpolationMode.NEAREST
        # 如使用之前的版本插值方法需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return img, target


class RandomHorizontalFlip(object):
    def __init__(self, filp_prob):
        self.flip_prob = filp_prob

    def __call__(self, img, target):
        if random.random() < self.flip_prob:
            img = F.hflip(img)
            target = F.hflip(target)
        return img, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        img = pad_if_smaller(img, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(img, (self.size, self.size))
        img = F.crop(img, *crop_params)
        target = F.crop(target, *crop_params)
        return img, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        img = F.center_crop(img, self.size)
        target = F.center_crop(target, self.size)
        return img, target


class ToTensor(object):
    def __call__(self, img, target):
        img = F.to_tensor(img)
        target = torch.as_tensor(np.array(target), dtype=torch.float64)
        return img, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img, target


