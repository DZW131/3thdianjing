#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 22:53
# @Author  : Can Cui
# @File    : seg_tissue_area.py
# @Software: PyCharm
# @Comment:
import os.path
import cv2
import numpy as np
import torch
from skimage.color import rgb2hed, hed2rgb
from .kmeans_pytorch import kmeans
from torchvision import transforms as T

def auto_threshold(image):
    # k-means分离前后景
    to_tensor = T.ToTensor()
    img_hed = rgb2hed(image)
    null = np.zeros_like(img_hed[:, :, 0])
    img_to_kmeans = hed2rgb(np.stack((img_hed[:, :, 0], null, null), axis=-1))
    img_to_kmeans = (img_to_kmeans / (img_to_kmeans.max() - img_to_kmeans.min())) * 255
    img_to_kmeans = img_to_kmeans.astype(np.uint8)
    km_array = to_tensor(img_to_kmeans).permute((1, 2, 0)).reshape((-1, 3))
    status = False
    while not status:
        cluster_idx, cluster_centers, status = kmeans(X=km_array, num_clusters=2, device=torch.device('cuda:0'))
    cluster_idx = cluster_idx.reshape((img_to_kmeans.shape[0:2])).numpy()
    torch.cuda.empty_cache()

    # 判断前后景
    unique, count = np.unique(cluster_idx, return_counts=True)
    pix_dict = dict(zip(unique, count))
    foreground_id = 1
    if pix_dict[0] < pix_dict[1]:
        foreground_id = 0
    background_id = 1-foreground_id
    img_to_kmeans[cluster_idx == foreground_id] = np.array([255, 255, 255])
    img_to_kmeans[cluster_idx == background_id] = np.array([0, 0, 0])

    # 生成开闭操作的画布
    img_h, img_w = img_to_kmeans.shape[0], img_to_kmeans.shape[1]
    img_bin = np.zeros((img_h+1000, img_w+1000))
    img_bin[500:img_h+500, 500:img_w+500] = img_to_kmeans[:, :, 0]
    kernel_size_CLOSE = 15 # 10~20
    kernel_size_OPEN = 3 # 2~8
    iterations = 3
    kernel_CLOSE = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_CLOSE, kernel_size_CLOSE))
    kernel_OPEN = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_OPEN, kernel_size_OPEN))
    mask = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_CLOSE, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_OPEN, iterations=iterations)

    # 恢复正常尺寸二值图
    mask = mask[500:img_h + 500, 500:img_w + 500].astype(np.uint8)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    area_list = []
    for roi_list in contours:
        area = cv2.contourArea(roi_list)
        area_list.append(area)
    area_list = np.array(area_list, dtype=int)
    # 筛出大区域
    new_contours = []
    for i in range(area_list.size):
        if area_list[i] / np.max(area_list) >=0.05:
            new_contours.append(contours[i])

    # 丢掉极小区域
    new_contours = [contour for contour in new_contours if contour.shape[0] > 2]
    return new_contours

def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut)
    output_img = np.uint8(output_img + 0.5)  # 这句一定要加上
    return output_img

def find_tissue_countours(slide):
    try:
        scale = 64
        suffix = os.path.splitext(slide.filename)[1]
        if suffix in [".sdpc", ".mrxs"]:
            img = slide.read((0, 0), (int(slide.width), int(slide.height)), scale)
        else:
            img = np.array(slide.getThumbnail())
            img = cv2.resize(img, dsize=(int(slide.width/scale), int(slide.height/scale)), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        # 针对.mrxs去除纯黑色背景
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img[np.where(img_gray == 0)] = np.array([255,255,255])
        out = gamma(img, 0.00000005, 4.0)
        new_contours = auto_threshold(out)
        new_contours = [cnt*scale for cnt in new_contours]
    except Exception as e:
        print(e)
        new_contours = []
    return new_contours