#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2022/9/25 21:44
# @Author : ''
# @FileName: fcn.py
from collections import OrderedDict
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from src.backbone import resnet50, resnet101


class IntermediateLayerGetter(nn.ModuleDict):
    """
    function: 这个类就是获取一个Model中你指定要获取的哪些层的输出
              然后这些层的输出会在一个有序的字典中
              字典中的key就是刚开始初始化这个类传进去的 value就是feature经过指定需要层的输出
    1 在forward函数中不能对同一个模块使用两次
    2 只能调用一级子模块，不能调用二级以下的模块
    example:
        m = torchvision.models.resnet18()
        return_layers = {"layer4": "out"}
        new_m = torchvision.models._utils.IntermediateLayerGetter(m, return_layers)
        out_put = new_m(torch.rand(1, 3, 224, 224))
        -> out_put: OrderDict([('out', tensor([[[[......]]]]))])
        -> out_put["out"]: tensor([[[[......]]]])
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orgin_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # rebuild backbone, delete module we not use
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orgin_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCN(nn.Module):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]  # [h, w]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        # ConvTranspose2d is used in the origin paper, but it is froozen and
        # it is bilinear operator
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result   # ["feat4": val]  or ["feat4": val, "aux": val]


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(inter_channels, channels, kernel_size=1)
        ]
        super(FCNHead, self).__init__(*layers)  # nn.Sequential(*layers)


def fcn_resnet50(num_classes=21, aux=True, pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # load pretrain weight form resnet50 backbone
        backbone.load_state_dict(torch.load("../fcn_resnet50_coco.pth", map_location="cpu"))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)
    model = FCN(backbone, classifier, aux_classifier)
    return model


def fcn_resnet101(num_classes=21, aux=True, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # load pretrain weight form resnet101 backbone
        backbone.load_state_dict(torch.load("resnet101.pth", map_location="cpu"))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)
    model = FCN(backbone, classifier, aux_classifier)
    return model


if __name__ == "__main__":
    import torchvision
    m = torchvision.models.resnet18()
    return_layers = {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', "layer4": "out"}
    new_m = torchvision.models._utils.IntermediateLayerGetter(m, return_layers)
    out_put = new_m(torch.rand(1, 3, 224, 224))
    print(out_put)
    print([(k, v.shape) for k, v in out_put.items()])
