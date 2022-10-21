# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2022/10/21 11:19
# @Author: 'IReverser'
# @ FileName: predict_show.py
import os
import time
import argparse
import matplotlib.pyplot as plt
from utils.voc_seg import VOCSegmentation, inv_normalize_image, label2image
from train import get_transform
from src.fcn_res import *
import torch.nn.functional as F
from config import colormap


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(args):
    assert os.path.exists(args.weight_path), f"weights {args.weight_path} not found."
    assert os.path.exists(args.data_path), f"image {args.data_path} not found."

    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    if "19" in args.backbone:
        model = FCN8s_VGG(args.num_classes+1)
    elif "16" in args.backbone:
        model = FCN8s_VGG16(args.num_classes+1)
    elif "101" in args.backbone:
        model = FCN8s_ResNet(num_classes=args.num_classes+1, backbone=args.backbone)
    else:
        model = FCN8s_ResNet(num_classes=args.num_classes+1, backbone="resnet50")

    # load weights
    model.load_state_dict(torch.load(args.weight_path, map_location="cpu")["model"])
    model.to(device)

    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="val.txt")
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=8,
                                             num_workers=8,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    for step, (b_x, b_y) in enumerate(val_loader):
        if step > 0:
            break

    model.eval()
    b_x = b_x.float().to(device)
    b_y = b_y.long().to(device)
    t_start = time_synchronized()
    out = model(b_x)
    t_end = time_synchronized()
    print("inference time: {}".format(t_end - t_start))
    out = F.log_softmax(out, dim=1)
    pre_lab = torch.argmax(out, 1)
    # 可视化一个batch的图像，检查数据预处理 是否正确
    b_x_numpy = b_x.cpu().data.numpy()
    b_x_numpy = b_x_numpy.transpose(0, 2, 3, 1)
    b_y_numpy = b_y.cpu().data.numpy()
    pre_lab_numpy = pre_lab.cpu().data.numpy()
    plt.figure(figsize=(16, 9))
    for ii in range(4):
        plt.subplot(3, 4, ii + 1)
        plt.imshow(inv_normalize_image(b_x_numpy[ii]))
        plt.axis("off")
        plt.subplot(3, 4, ii + 5)
        plt.imshow(label2image(b_y_numpy[ii], colormap))
        plt.axis("off")
        plt.subplot(3, 4, ii + 9)
        plt.imshow(label2image(pre_lab_numpy[ii], colormap))
        plt.axis("off")
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig("test_result.png")


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="../datasets/", help="VOCdevkit root")
    parser.add_argument("--weight-path", default="./res50_trained/model_best.pth")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--backbone", default="resnet50", help="basic backbone (default: resnet50): resnet50, resnet101, vgg16, vgg19")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
