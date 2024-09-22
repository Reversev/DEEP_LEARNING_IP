#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2024/7/28 20:18
# @Author : ''
# @FileName: clip_train_cifar.py
# https://github.com/openai/CLIP
import torch
import torchvision
import clip
from tqdm import tqdm
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def accuracy(output, target, topk=(1, )):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def zeroshot_classifier(classes_names, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classes_name in tqdm(classes_names):
            texts = [template.format(classes_name) for template in templates]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings /= class_embeddings.norm()
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
# set clip model
model, preprocess = clip.load("ViT-B/32", download_root="./ckpts", device=device)
# model, preprocess = clip.load("RN50", download_root="./ckpts", device=device)
# set dataset and dataloader
train_dataset = torchvision.datasets.CIFAR100(root='./data/',
                                             train=True,
                                             download=True,
                                             transform=preprocess)
test_dataset = torchvision.datasets.CIFAR100(root='./data/',
                                            train=False,
                                            download=True,
                                            transform=preprocess)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=32,
                                          shuffle=False,
                                          num_workers=0)

classes = train_dataset.classes
print(classes)
cifar_template = ["a photo of a {}."]  # 1
# from templates import imagenet_templates
# cifar_template = imagenet_templates[:10]  # 2
# cifar_template =[
#     'a drawing of a {}.',
#     'a photo of my {}.',
#     'a photo of the {}.',
#     'a good photo of the {}.',
#     'a rendering of the {}.',
#     'a photo of a {}.',
#     'a photo of one {}.',
#     'a low resolution photo of a {}.',
#     'art of the {}.',
#     'a drawing of the {}.',
# ]  # 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}")
zeroshot_weights = zeroshot_classifier(classes, cifar_template, device)

with torch.no_grad():
    top1, top5, n = 0., 0., 0.
    for i, (images, target) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        target = target.to(device)

        # predict
        image_feat = model.encode_image(images)
        image_feat /= image_feat.norm(dim=-1, keepdim=True)
        logits = 100. * image_feat @ zeroshot_weights

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

top1 = (top1 / n) * 100
top5 = (top5 / n) * 100
print(f"Top1 accuracy: {top1:.2f}, Top5 accuracy: {top5:.2f}")

