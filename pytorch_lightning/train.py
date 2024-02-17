#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2024/2/17 15:34
# @Author : ''
# @FileName: pytorch_lightning.py

# 1.set random seed
from pytorch_lightning import seed_everything
seed = 42
seed_everything(seed)

# the same as pytorch library
# import torch
# import numpy as np
# from random import random
# def seed_all(seed_v):
#     random.seed(seed_v)        # python
#     np.random.seed(seed_v)     # cpu
#     torch.manual_seed(seed_v)  # cpu
#     if torch.cuda.is_available():
#         print("CUDA is available")
#         torch.cuda.manual_seed(seed_v)
#         torch.cuda.manual_seed_all(seed_v)         # gpu
#         torch.backends.cudnn.deterministic = True  # needed
#         torch.backends.cudnn.benchmark = False
# seed = 42
# seed_all(seed)

import pytorch_lightning as pl
import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# if you used colab referred by google
# from google.colab import drive
# drive.mount('./content/drive')
# import os
# os.chdir("/content/drive/My Drive/")


class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()
        self.params = hparams
        self.data_dir = self.params.data_dir
        self.num_classes = self.params.num_classes

        # define the model
        arch = torchvision.models.resnet18(pretrained=True)
        num_ftrs = arch.fc.in_features

        modules = list(arch.children())[:-1]  # ResNet18 has 10 children, delete last linear layer
        self.backbone = torch.nn.Sequential(*modules)  # [bs, 512, 1, 1]
        # define new classifier with two linear layers
        self.final = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.num_classes),
            torch.nn.Softmax(dim=1))

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.final(x)
        return x

    def configure_optimizers(self):
        # REQUIRE
        optimizer = torch.optim.SGD([
            {'params': self.backbone.parameters()},
            {'params': self.final.parameters(), 'lr': 1e-2}
        ], lr=1e-3, momentum=0.9)

        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        return [optimizer], [exp_lr_scheduler]

    def training_step(self, batch, batch_idx):
        # REQUIRE
        x, y = batch
        y_hat = self.forward(x)
        # define criterion
        loss = F.cross_entropy(y_hat, y)

        _, preds = torch.max(y_hat, dim=1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)
        # set logging
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return {'loss': loss, 'train_acc': acc}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)

        self.log('val_loss', loss)
        self.log('val_acc', acc)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return {'test_loss': loss, 'test_acc': acc}

    def train_dataloader(self):
        # REQUIRED
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_set = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_set = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

        return val_loader

    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_set = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)

        return val_loader


def main(hparams):
    # initialize model
    model = CoolSystem(hparams)
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        max_epochs=hparams.epochs,
        gpus=1,
        precision=16,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.fit(model)


from argparse import Namespace

if __name__ == '__main__':
    args = {
        'num_classes': 2,
        'epochs': 100,
        'data_dir': "./hymenoptera_data",
        # 'data_dir': "/content/hymenoptera_data",  # if you use colab
    }
    hyperparams = Namespace(**args)
    main(hyperparams)
