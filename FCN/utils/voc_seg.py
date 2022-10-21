#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2022/10/5 15:00
# @Author : ''
# @FileName: voc_seg.py
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from config import colormap


def inv_normalize_image(data):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])
    data = data.astype('float32') * rgb_std + rgb_mean
    return data.clip(0, 1)


def image2label(image, colormap):
    # transform labels to every elements (n) as dependence classes (n)
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(colormap):
        cm2lbl[((cm[0] * 256 + cm[1]) * 256 + cm[2])] = i
    image = np.array(image, dtype="int64")  # H, W, C
    ix = ((image[:, :, 0] * 256 + image[:, :, 1]) * 256 + image[:, :, 2])  # ix: two dimension
    image2 = cm2lbl[ix]  # index
    return image2


# transform pred to image for a label image
def label2image(prelabel, colormap):
    h, w = prelabel.shape
    prelabel = prelabel.reshape(h * w, -1)
    image = np.zeros((h * w, 3), dtype="int32")
    for ii in range(len(colormap)):
        index = np.where(prelabel == ii)  # return special index from array
        image[index, :] = colormap[ii]
    return image.reshape(h, w, 3)


class VOCSegmentation(Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str="train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "Year must be in ['2007', '2012']"
        self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(self.root), "Path: '{}' does not exist.".format(self.root)
        self.image_dir = os.path.join(self.root, "JPEGImages")
        self.mask_dir = os.path.join(self.root, "SegmentationClass")

        self.txt_path = os.path.join(self.root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(self.txt_path), "file '{}' does not exist.".format(self.txt_path)

        with open(os.path.join(self.txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(self.image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(self.mask_dir, x + ".png") for x in file_names]
        assert len(self.images) == len(self.masks)

        self.transforms = transforms

    def __getitem__(self, item):
        """
        :param item: index
        :return: tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[item]).convert("RGB")

        target = Image.open(self.masks[item]).convert("RGB")

        target = Image.fromarray(image2label(target, colormap))

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def read_images(self):
        with open(self.txt_path, 'r') as f:
            images = f.read().split()
        data = [os.path.join(self.root, 'JPEGImages', i + '.jpg') for i in images]
        label = [os.path.join(self.root, 'SegmentationClass', i + '.png') for i in images]
        return data, label

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_target = cat_list(targets, fill_value=255)
        return batched_imgs, batched_target


def cat_list(images, fill_value=0):
    # calculate common max values from channel, height and weight in a batch
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images), ) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == "__main__":
    import torchvision.models.resnet
    dataset = VOCSegmentation(voc_root="../../datasets/", year="2012", txt_name="train.txt")
