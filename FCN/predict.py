#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2022/10/11 20:41
# @Author : ''
# @FileName: predict.py
import os
import time
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from utils.voc_seg import label2image, inv_normalize_image
from src.fcn_res import *
from config import colormap


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(args):
    assert os.path.exists(args.weight_path), f"weights {args.weight_path} not found."
    assert os.path.exists(args.img_path), f"image {args.img_path} not found."

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

    # load image
    original_img = Image.open(args.img_path)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])

    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    cm = np.array(colormap).astype('uint8')

    model.eval()  # eval
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        plt.figure(figsize=(16, 9))
        img1 = img.data.numpy().transpose(0, 2, 3, 1)
        plt.subplot(121)
        img1 = inv_normalize_image(img1[0])
        plt.imshow(img1)
        plt.axis("off")

        pred = output.argmax(1)
        pred = pred.cpu().data.numpy()
        plt.subplot(122)
        mask = label2image(pred[0], cm)
        plt.imshow(mask)
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.01, wspace=0.01)
        plt.margins(0, 0)
        plt.savefig("test_result.png")


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--img-path", default="test.jpg", help="VOCdevkit root")
    parser.add_argument("--weight-path", default="./res50_trained/model_best.pth")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--backbone", default="resnet50", help="basic backbone (default: resnet50): resnet50, resnet101, vgg16, vgg19")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
