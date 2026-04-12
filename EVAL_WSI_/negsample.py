#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 16:13
# @Author  : Can Cui
# @File    : eval_multi_cell_seg.py
# @Software: PyCharm
# @Comment:
import kfb.kfbslide as kfbslide
from options import Options
import torch
import cv2
from parse_embolus import read_region_kfb
import numpy as np
import torch.nn as nn
import os, sys
import pandas as pd
from Slide.openslide_func import openSlide as di_openSlide
import openslide
import pickle

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_root)

from Core.papSmear.multi_cls_cell_seg import cal_ki67_np

global label_dict,color_dict

def walk_dir(data_dir, file_types):
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:
                if f.endswith(this_type):
                    path_list.append(os.path.join(dirpath, f))
                    break
    return path_list


def draw_center(img, center, label):
    img = img.copy()
    if len(center) != 0:
        num_center = center.shape[0]

        radius = 3
        thickness = -1

        for i in range(num_center):
            if str(label[i]) in color_dict:
                img = cv2.circle(img, (center[i, 0], center[i, 1]), radius, color_dict[str(label[i])], thickness)

        return img
    else:
        return img


from pydaily import filesystem


def sort_edge(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def OTUS_mrxs(img, result_save_sub_dir, img_name):
    img_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[np.where(img == 0)] = 255
    img_Blur = cv2.medianBlur(img, 5)
    blur = cv2.GaussianBlur(img_Blur, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th3 = 255 - th3
    # kernel = np.ones((15, 15), np.uint8)
    kernel = np.ones((20, 20), np.uint8)
    image_gray = cv2.erode(th3, kernel)
    kernel_2 = np.ones((50, 50), np.uint8)
    image_gray = cv2.dilate(image_gray, kernel_2)
    image_gray = cv2.GaussianBlur(image_gray, (15, 15), 0)
    contours = sort_edge(image_gray)
    area_list = []
    for roi_list in contours:
        area = cv2.contourArea(roi_list)
        area_list.append(area)
    area_list = np.array(area_list, dtype=float)
    area_max = np.max(area_list)
    new_contours = []
    array = np.zeros((image_gray.shape[0], image_gray.shape[1], 3))
    for i in range(area_list.size):
        if area_list[i] / area_max >= 0.1:
            cv2.drawContours(array, contours, i, (0, 0, 255), -1)
            new_contours.append(contours[i])
    x = np.array(0.7 * img_ori + 0.3 * array, dtype=int)
    img_name = img_name.replace('mrxs', 'png')
    cv2.imwrite(os.path.join(result_save_sub_dir, img_name), x)
    return new_contours


def OTUS_kfb(img, result_save_sub_dir, img_name):
    img_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_Blur = cv2.medianBlur(img, 5)
    blur = cv2.GaussianBlur(img_Blur, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th3 = 255 - th3
    # kernel = np.ones((15, 15), np.uint8)
    kernel = np.ones((15, 15), np.uint8)
    image_gray = cv2.erode(th3, kernel)
    kernel_2 = np.ones((50, 50), np.uint8)
    image_gray = cv2.dilate(image_gray, kernel_2)
    image_gray = cv2.GaussianBlur(image_gray, (15, 15), 0)
    contours = sort_edge(image_gray)
    img_name = img_name.replace('kfb', 'png')
    img_name = img_name.replace('ndpi', 'png')

    contours = sort_edge(image_gray)
    area_list = []
    for roi_list in contours:
        area = cv2.contourArea(roi_list)
        area_list.append(area)
    area_list = np.array(area_list, dtype=float)
    area_max = np.max(area_list)
    # image_gray = np.zeros((image_gray.shape[0], image_gray.shape[1]))
    array = np.zeros((image_gray.shape[0], image_gray.shape[1], 3))
    new_contours = []
    for i in range(area_list.size):
        if area_list[i] / area_max >= 0.05:
            cv2.drawContours(array, contours, i, (0, 0, 255), -1)
            new_contours.append(contours[i])
    x = np.array(0.6 * img_ori + 0.4 * array, dtype=int)
    cv2.imwrite(os.path.join(result_save_sub_dir, img_name), x)
    return new_contours

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def RGBAtoBGR(image_rgba):
    r, g, b, a = cv2.split(image_rgba)
    image_bgr = cv2.merge([r, g, b])
    return image_bgr
def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut)
    output_img = np.uint8(output_img + 0.5)  # 这句一定要加上
    return output_img
def auto_threshold(image, image_ori, type=0):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if type == 1:
        img[np.where(img == 0)] = 255
    img = cv2.medianBlur(img, 5)

    th2_ori = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th2 = 255 - th2_ori
    kernel_2 = np.ones((10, 10), np.uint8)
    th2 = cv2.dilate(th2, kernel_2)
    kernel = np.ones((30, 30), np.uint8)  # 设置小
    th2 = cv2.erode(th2, kernel)  # 做一个连tong
    th2 = cv2.dilate(th2, kernel)
    contours = sort_edge(th2)
    area_list = []
    for roi_list in contours:
        area = cv2.contourArea(roi_list)
        area_list.append(area)
    area_list = np.array(area_list, dtype=int)
    area_max = np.max(area_list)
    array = np.zeros((th2.shape[0], th2.shape[1], 3))
    new_contours = []
    for i in range(area_list.size):
        if area_list[i] / area_max >= 0.01:
            cv2.drawContours(array, contours, i, (0, 0, 255), -1)
            new_contours.append(contours[i])
    x = np.array(0.6 * image_ori + 0.4 * array, dtype=int)
    return x, new_contours
def gamma_main(img, result_save_sub_dir, img_name):
    img_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = gamma(img_ori, 0.00000005, 4.0)
    if 'mrxs' in img_name:
        out, new_contours = auto_threshold(out, img_ori, type=1)

        img_name = img_name.replace('mrxs', 'png')


    else:
        out, new_contours = auto_threshold(out, img_ori)

        img_name = img_name.replace('kfb', 'png')
        img_name = img_name.replace('ndpi', 'png')

    # cv2.imwrite(os.path.join(result_save_sub_dir, img_name), out)
    return new_contours
def count_summary(result_summary,result_save_sub_dir,kfb_name):
    total_sum={}

    total_sum['细胞总数']=np.sum(np.array(result_summary['细胞总数']),dtype=np.uint16)
    for key in label_dict.keys():
        total_sum[key]=np.sum(np.array(result_summary[key]),dtype=np.uint16)
    out_put=open(os.path.join(result_save_sub_dir,'%s.pkl'%kfb_name),'wb')
    pickle.dump(total_sum,out_put)

def split_kfb(slides_dir, crop_len, is_display):
    slide_path, slide_name = os.path.split(slides_dir)
    kfb_name, kfb_ext = os.path.splitext(slide_name)
    start_time = time.time()
    result_save_sub_dir = os.path.join(
        '/home/deepin/Desktop/Research/diyingjia/Her2/Filter/data',
        kfb_name)
    os.makedirs(result_save_sub_dir, exist_ok=True)
    if not os.path.exists(os.path.join(result_save_sub_dir, "summary_{}.csv".format(kfb_name))):
        slide_OTUS = di_openSlide(slides_dir)
        size = slide_OTUS.slide.level_dimensions[0]
        if 'ndpi' in slides_dir or 'mrxs' in slides_dir:
            slide = openslide.open_slide(slides_dir)

        else:
            slide = kfbslide.open_kfbslide(slides_dir)
        width = size[0]
        height = size[1]
        if size[0] > 2 ** 4 and size[1] > 2 ** 4:
            #try:
                cur_slide = slide_OTUS.read(location=[0, 0], size=size, scale=16)
                if 'ndpi' in slides_dir or 'kfb' in slides_dir:
                    contours = gamma_main(cur_slide, result_save_sub_dir, slide_name)
                else:
                    contours = gamma_main(cur_slide, result_save_sub_dir, slide_name)
                if contours != []:
                    print(width)
                    print(height)
                    TILE_SIZE = 1024
                    width = width - width % TILE_SIZE
                    height = height - height % TILE_SIZE
                    num_x = int(width / crop_len-1)
                    num_y = int(height / crop_len-1)
                    num_patch = 0
                    for i in range(0,num_y,2):
                        crop_start_y = i * (crop_len)
                        crop_start_x = 0
                        for j in range(0,num_x,2):
                            region_size = (crop_len, crop_len)
                            region_start = (crop_start_x, crop_start_y)
                            region_start_1=(crop_start_x, crop_start_y+1024)
                            region_start_2 = (crop_start_x+1024, crop_start_y )
                            region_start_3 = (crop_start_x + 1024, crop_start_y+1024)
                            for roi_list in contours:
                                # print(roi_list)
                                roi_list = roi_list * 16
                                flag = cv2.pointPolygonTest(roi_list, region_start, False)
                                flag_1 = cv2.pointPolygonTest(roi_list, region_start_1, False)
                                flag_2 = cv2.pointPolygonTest(roi_list, region_start_2, False)
                                flag_3 = cv2.pointPolygonTest(roi_list, region_start_3, False)
                                if flag == 1 or flag_1==1 or flag_2==1 or flag_3==1:
                                    if 'ndpi' in slides_dir or 'mrxs' in slides_dir:
                                        try:
                                            cur_region = np.array(
                                                slide.read_region(region_start, 0, region_size))
                                        except:
                                            continue
                                        else:
                                            cur_region = RGBAtoBGR(cur_region)
                                    else:
                                        cur_region = read_region_kfb(slides_dir, slide, region_start, region_size)
                                    img_name = kfb_name + '-' + str(i) + '-' + str(j) + '-' + str(num_patch) + ".png"
                                    if is_display:
                                        image = cur_region
                                        image = image[:, :, :: -1]
                                        cv2.imwrite(os.path.join(result_save_sub_dir, img_name),
                                                    image)
                            crop_start_x = (j + 1) * crop_len
                            num_patch = num_patch + 1



def count_test_summary(image, net, cell_count_dict):
    center_coords, labels = cal_ki67_np(image, net)
    total_count = 0
    reverse_dict = {}
    for k, v in label_dict.items():
        reverse_dict[v] = k
    print(reverse_dict)
    if len(labels) != 0:
        for i in range(np.max(labels) + 1):
            this_num = np.sum(labels == i)
            cell_count_dict[reverse_dict[i]] = this_num
            total_count += this_num
        cell_count_dict["细胞总数"] = total_count
        # cell_count_dict["PD-L1指数"] = \
        #     round(cell_count_dict["阳性肿瘤细胞"] / (cell_count_dict["阳性肿瘤细胞"] + cell_count_dict["阴性肿瘤细胞"] + 1e-10), 3)
    else:
        cell_count_dict["细胞总数"] = 0
        # cell_count_dict["PD-L1指数"] = 0
    return cell_count_dict, center_coords, labels


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def run_test(test_dir):
    result_save_dir = '/home/deepin/Desktop/Research/diyingjia/Her2/Filter/data'
    os.makedirs(result_save_dir, exist_ok=True)
    kfb_list = filesystem.find_ext_files(test_dir, '.kfb')
    for slide_path in kfb_list:
        image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        # if 'HE' not in image_name and 'NC' not in image_name and 'V-' not in image_name:
        print("evaluating image {}".format(slide_path))

        split_kfb(slide_path, 1024,is_display=True)




if __name__ == "__main__":
    import os

    global opt
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test_dir = "/media/deepin/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/上肿/SAMPLE"


    # run_test(test_dir, 0,
    #          weight_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_unet',
    #                                   "Model", "albu_unet_mix_1_2", "weights_epoch_55_1.303815746679902.pth"))

    # run_test(test_dir, 1,
    #          weight_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_unet',
    #                                   "Model", "att_unet_mix_1_2", "weights_epoch_30_1.795762307010591.pth"))

    run_test(test_dir)

    # run_test(test_dir, 3,
    #          weight_path= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'pytorch_unet',
    #                                    "Model", "unet_mix_1_2", "weights_epoch_125_1.5490121394395828.pth" ))
