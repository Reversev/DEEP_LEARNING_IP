#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2024/7/21 16:05
# @Author : ''
# @FileName: main.py
import numpy as np
import torch
import clip
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", download_root="./ckpts", device=device)

image = preprocess(Image.open("./bing_image_creator.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["money", "fish", "red"]).to(device)

with torch.no_grad():
    # image_feat = model.encode_image(image)  # [1 512]
    # text_feat = model.encode_text(text)     # [3 512]
    # print(image_feat.shape, text_feat.shape)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print(f"{probs}")
