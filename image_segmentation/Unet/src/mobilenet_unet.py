from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import mobilenet_v3_large
from .unet import Up, OutConv
from .vgg_unet import IntermediateLayerGetter


class MobileV3Unet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super(MobileV3Unet, self).__init__()
        backbone = mobilenet_v3_large(pretrained=pretrain_backbone)

        backbone = backbone.features

        stage_indices = [1, 3, 6, 12, 15]
        self.stage_out_channels = [backbone[i].out_channels for i in stage_indices]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])
        self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        backbone_out = self.backbone(x)
        x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
        x = self.up2(x, backbone_out['stage2'])
        x = self.up3(x, backbone_out['stage1'])
        x = self.up4(x, backbone_out['stage0'])
        x = self.conv(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x
