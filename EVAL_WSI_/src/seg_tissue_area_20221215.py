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

def sort_edge(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]

def minPooling(img, K_size):
    H, W = img.shape
    # K_size = 3
    ## 255 padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.uint8)
    out[:,:] = 255 # 白底
    out[pad:pad + H, pad:pad + W] = img.copy()
    tmp = out.copy()
    for y in range(H):
        for x in range(W):
                out[pad + y, pad + x] = np.min(tmp[y:y + K_size, x:x + K_size])
    out = out[pad:pad + H, pad:pad + W].astype(np.uint8)
    return out

def auto_threshold(name, image):
    cv2.imwrite("%s_0_image.jpg" % (name), image)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(img, 5)
    # img = minPooling(img, 11) # 速度慢
    cv2.imwrite("%s_1_medianBlur.jpg" % (name), img)
    th2 = 255 - img
    cv2.imwrite("%s_2_255-img.jpg" % (name), th2)
    kernel = np.ones((10, 10), np.uint8)  # 设置小
    kernel_1 = np.ones((30, 30), np.uint8)
    kernel_2 = np.ones((40, 40), np.uint8)
    th2 = cv2.erode(th2, kernel)
    th2 = cv2.dilate(th2, kernel_1)  # 做一个连tong
    th2 = cv2.erode(th2, kernel_1)
    th2 = cv2.dilate(th2, kernel_2)
    th2 = (th2.astype(np.float32)-th2.min())/(th2.max()-th2.min())*255
    th2 = th2.astype(np.uint8)
    cv2.imwrite("%s_3_dilate.jpg" % (name), th2)
    ret, binary = cv2.threshold(th2, 130, 255, cv2.THRESH_BINARY)
    cv2.imwrite("%s_4_binary.jpg" % (name), binary)
    th2 = cv2.dilate(binary, kernel_2)
    cv2.imwrite("%s_5_dilate.jpg" % (name), th2)
    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sort_edge(th2)
    area_list = []
    for roi_list in contours:
        area = cv2.contourArea(roi_list)
        area_list.append(area)
    area_list = np.array(area_list, dtype=int)
    # 筛出大区域
    new_contours = []
    for i in range(area_list.size):
        if area_list[i] / np.max(area_list) >=0.02:
            new_contours.append(contours[i])
    # 作个连通
    mask = np.zeros_like(img)
    mask = cv2.drawContours(mask, new_contours, -1, (255, 0, 0), -1)
    mask = cv2.dilate(mask, kernel_2)
    area_max = np.max(area_list)
    kernel_size = 30
    iterations = int(area_max**0.5/10/kernel_size)
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_3, iterations=iterations)
    new_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = np.zeros_like(img)
    mask = cv2.drawContours(mask, new_contours, -1, (255, 0, 0), -1)
    cv2.imwrite("%s_6_mask.jpg"%(name), mask)
    mask = cv2.erode(mask, kernel_2)
    mask = cv2.erode(mask, kernel_2)
    mask = cv2.erode(mask, kernel_2)
    cv2.imwrite("%s_7_erode.jpg" % (name), mask)
    new_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
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
        scale=16
        img = slide.read((0,0), (slide.width, slide.height), scale)
        name = slide.filename
        name = name.split("\\")[-1]
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img[np.where(img_gray == 0)] = 255
        out = gamma(img, 0.00000005, 4.0)
        new_contours = auto_threshold(name, out)
        new_contours = [cnt*scale for cnt in new_contours]
    except Exception as e:
        print(e)
        new_contours = []
    return new_contours

