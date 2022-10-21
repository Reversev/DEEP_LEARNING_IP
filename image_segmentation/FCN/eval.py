#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2022/10/11 10:47
# @Author : ''
# @FileName: eval.py
import os
import argparse
from src.fcn_res import *
from utils.util import evaluate
from utils.voc_seg import VOCSegmentation
import utils.transforms as T


class SegmentationPresetEval:
    def __init__(self,
                 base_size,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # T.RandomResize(base_size, base_size),
            T.RandomCrop(base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # segmentation num_classes + background
    num_classes = args.num_classes + 1

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=SegmentationPresetEval(224),
                                  txt_name="val.txt")

    num_workers = 8
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=4,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # create model
    if "19" in args.backbone:
        model = FCN8s_VGG(args.num_classes + 1)
    elif "16" in args.backbone:
        model = FCN8s_VGG16(args.num_classes + 1)
    elif "101" in args.backbone:
        model = FCN8s_ResNet(num_classes=args.num_classes + 1, backbone=args.backbone)
    else:
        model = FCN8s_ResNet(num_classes=args.num_classes + 1, backbone="resnet50")

    model.load_state_dict(torch.load(args.weights, map_location=device)["model"])
    # model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
    print(confmat)


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="../datasets/", help="VOCdevkit root")
    parser.add_argument("--weights", default="./res50_trained/model_best.pth")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--backbone", default="resnet50", type=str, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./models"):
        os.mkdir("./models")

    main(args)
