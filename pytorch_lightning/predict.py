#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2024/2/17 19:59
# @Author : ''
# @FileName: predict.py
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
from train import CoolSystem


# functions to show an image
def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


if __name__ == '__main__':
    classes = ['ants', 'bees']
    checkpoint_dir = 'lightning_logs/version_2/checkpoints/'
    checkpoint_path = checkpoint_dir + os.listdir(checkpoint_dir)[0]
    args = {
        'num_classes': 2,
        'batch_size': 8,
        'data_dir': "./hymenoptera_data",
    }

    hparams = Namespace(**args)
    checkpoint = torch.load(checkpoint_path)
    model_infer = CoolSystem(hparams)
    model_infer.load_state_dict(checkpoint['state_dict'])

    try_dataloader = model_infer.test_dataloader()

    inputs, labels = next(iter(try_dataloader))

    # print images and ground truth
    imshow(torchvision.utils.make_grid(inputs))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(8)))

    # inference
    outputs = model_infer(inputs)

    _, preds = torch.max(outputs, dim=1)
    print(preds)
    print(torch.sum(preds == labels.data) / (labels.shape[0] * 1.0))
    print('Predicted: ', ' '.join('%5s' % classes[preds[j]] for j in range(8)))
