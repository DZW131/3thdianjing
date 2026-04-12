#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 16:13
# @Author  : Can Cui
# @File    : eval_multi_cell_seg.py
# @Software: PyCharm
# @Comment:
import kfb.kfbslide as kfbslide
import os, json
from copy import deepcopy
import torch
import cv2
from parse_embolus import read_region_kfb
import numpy as np
import pdb
# from Slide.openslide_func import openSlide
# from scipy.misc import imsave, imread
import os, sys
import pandas as pd
import math
from Slide.openslide_func import openSlide as di_openSlide
import openslide
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_root)

from Core.papSmear.multi_cls_cell_seg import cal_ki67_np


def sec2min(sec):
    minute = sec // 60
    second = sec % 60
    return minute, second

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

        color_dict = {"0": (255, 0, 0),  # 阳肿
                      "1": (0, 255, 0),  # 阴肿
                      "2": (0, 0, 255),  # 阳淋
                      '3': (0, 0, 255),  # 阴淋
                      '4': (0, 255, 255),  # 阳组
                      '5': (0, 255, 255),  # 阴组
                      '6': (255, 255, 0),  # 纤维
                      '7': (255, 255, 0),  # 肺泡
                      }

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

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def RGBAtoBGR(image_rgba):
    r, g, b, a = cv2.split(image_rgba)
    image_bgr = cv2.merge([r, g, b])
    return image_bgr


def split_kfb(slides_dir, crop_len, net, test_count_dict, is_display, result_save_dir):
    slide_path, slide_name = os.path.split(slides_dir)
    kfb_name, kfb_ext = os.path.splitext(slide_name)
    result_summary = {
        'slide name': [],
        '阳性肿瘤细胞': [],
        '阴性肿瘤细胞': [],
        '阳性淋巴细胞': [],
        '阴性淋巴细胞': [],
        '阳性组织细胞': [],
        '阴性组织细胞': [],
        '纤维间质细胞': [],
        '正常肺泡细胞': [],
        '背景': [],
        '细胞总数': [],
        'PD-L1指数': [],
    }
    start_time = time.time()
    result_save_sub_dir = os.path.join(
        '/data/caijt/数据集/PDL1_Positive_cell_seg_detection/PDL1200切片/Slide/DEMO/result',
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
            try:
                cur_slide = slide_OTUS.read(location=[0, 0], size=size, scale=16)
                if 'ndpi' in slides_dir or 'kfb' in slides_dir:
                    contours = OTUS_kfb(cur_slide, result_save_sub_dir, slide_name)
                else:
                    contours = OTUS_mrxs(cur_slide, result_save_sub_dir, slide_name)
                if contours != []:
                    print(width)
                    print(height)
                    TILE_SIZE = 1024
                    width = width - width % TILE_SIZE
                    height = height - height % TILE_SIZE
                    num_x = int(width / crop_len)
                    num_y = int(height / crop_len)
                    num_patch = 0
                    for i in range(num_y):
                        crop_start_y = i * (crop_len)
                        crop_start_x = 0
                        for j in range(num_x):
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
                                    read_start_time = time.time()
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
                                    read_finish_time = time.time()
                                    print('read_region time{}'.format(read_finish_time - read_start_time))
                                    img_name = kfb_name + '-' + str(i) + '-' + str(j) + '-' + str(num_patch) + ".png"

                                    cell_count_dict, test_center_coords, test_labels = count_test_summary(cur_region,
                                                                                                          net,
                                                                                                         test_count_dict)
                                    cal_finish_time = time.time()
                                    print('cal_time {}'.format(cal_finish_time - read_finish_time))
                                    label_id_dict = {
                                        '0':'阳性肿瘤细胞',
                                        '1': '阴性肿瘤细胞',
                                        '2': '阳性淋巴细胞',
                                        '3': '阴性淋巴细胞',
                                        '4': '阳性组织细胞',
                                        '5': '阴性组织细胞',
                                        '6': '纤维间质细胞',
                                        '7': '正常肺泡细胞',
                                        '8':'背景'
                                    }
                                    for cell_index in range(len(test_center_coords)):
                                        if not cv2.pointPolygonTest(roi_list, (test_center_coords[cell_index][0],test_center_coords[cell_index][1]), False):
                                            test_center_coords.pop(cell_index)
                                            test_labels.pop(cell_index)
                                            cell_count_dict[label_id_dict[test_labels[cell_index]]]-=1
                                            #print(cell_count_dict[label_id_dict[test_labels[cell_index]]])
                                    result_summary['slide name'].append(img_name)
                                    for k, v in cell_count_dict.items():
                                        result_summary[k].append([
                                            test_count_dict[k]
                                        ])
                                    if is_display:
                                        image = cur_region
                                        # image = image[:, :, :: -1]
                                        test_res_img = draw_center(image, test_center_coords, test_labels)
                                        test_res_img = test_res_img[:, :, :: -1]
                                        cv2.imwrite(os.path.join(result_save_sub_dir, img_name),
                                                    test_res_img)
                            crop_start_x = (j + 1) * crop_len
                            num_patch = num_patch + 1
                else:
                    print(width)
                    print(height)
                    TILE_SIZE = 1024
                    width = width - width % TILE_SIZE
                    height = height - height % TILE_SIZE
                    num_x = int(width / crop_len - 1)
                    num_y = int(height / crop_len - 1)
                    num_patch = 0
                    for i in range(num_y):
                        crop_start_y = i * (crop_len)
                        crop_start_x = 0
                        for j in range(num_x):
                            region_size = (crop_len, crop_len)
                            region_start = (crop_start_x, crop_start_y)
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

                                cell_count_dict, test_center_coords, test_labels = count_test_summary(cur_region,
                                                                                                      net,
                                                                                                      test_count_dict)
                                result_summary['slide name'].append(img_name)
                                for k, v in cell_count_dict.items():
                                    result_summary[k].append([
                                        [k]
                                    ])

                                if is_display:
                                    image = cur_region
                                    # image = image[:, :, :: -1]
                                    test_res_img = draw_center(image, test_center_coords, test_labels)
                                    test_res_img = test_res_img[:, :, :: -1]
                                    cv2.imwrite(os.path.join(result_save_sub_dir, img_name),
                                                test_res_img)
                            crop_start_x = (j + 1) * crop_len
                            num_patch = num_patch + 1

            except:
                print("Fail to read region")
                flag_test = 1
            print(time.time() - start_time)
        else:

            slide = kfbslide.open_kfbslide(slides_dir)
            width, height = slide.level_dimensions[0]

            print(width)
            print(height)
            TILE_SIZE = 1024
            width = width - width % TILE_SIZE
            height = height - height % TILE_SIZE
            num_x = int(width / crop_len - 1)
            num_y = int(height / crop_len - 1)
            num_patch = 0

            for i in range(num_y):
                crop_start_y = i * (crop_len)
                crop_start_x = 0
                for j in range(num_x):
                    region_size = (crop_len, crop_len)
                    region_start = (crop_start_x, crop_start_y)
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
                    crop_start_x = (j + 1) * crop_len
                    num_patch = num_patch + 1
                    cell_count_dict, test_center_coords, test_labels = count_test_summary(cur_region, net,
                                                                                          test_count_dict)
                    result_summary['slide name'].append(img_name)
                    for k, v in cell_count_dict.items():
                        result_summary[k].append([
                            test_count_dict[k]
                        ])
                    if is_display:
                        image = cur_region

                        test_res_img = draw_center(image, test_center_coords, test_labels)
                        test_res_img = test_res_img[:, :, :: -1]
                        cv2.imwrite(os.path.join(result_save_sub_dir, img_name),
                                    test_res_img)
        df = pd.DataFrame(result_summary)
        columns = ['slide name', '阳性肿瘤细胞', '阴性肿瘤细胞', '阳性淋巴细胞', '阴性淋巴细胞', '阳性组织细胞', '阴性组织细胞', '纤维间质细胞', '正常肺泡细胞', '背景',
                   '细胞总数', 'PD-L1指数']
        df.to_csv(os.path.join(result_save_sub_dir, "summary_{}.csv".format(kfb_name)),
                  columns=columns, index=False)


def compare_with_annotations(test_dir, test_name, net, cell_count_dict=None, is_display=True):
    result_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TEST_RESULT", test_name)
    os.makedirs(result_save_dir, exist_ok=True)

    ndpi_list = filesystem.find_ext_files(test_dir, ".ndpi")
    kfb_list = filesystem.find_ext_files(test_dir, '.kfb')
    mrxs_list = filesystem.find_ext_files(test_dir, '.mrxs')
    for slide_path in kfb_list:

        image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        if 'HE' not in image_name and 'NC' not in image_name and 'V-' not in image_name:
            print("evaluating image {}".format(slide_path))

            test_count_dict = {
                '阳性肿瘤细胞': 0,
                '阴性肿瘤细胞': 0,
                '阳性淋巴细胞': 0,
                '阴性淋巴细胞': 0,
                '阳性组织细胞': 0,
                '阴性组织细胞': 0,
                '纤维间质细胞': 0,
                '正常肺泡细胞': 0,
                '背景': 0,
                '细胞总数': 0,
                'PD-L1指数': 0
            }
            # pdb.set_trace()
            split_kfb(slide_path, 1024, net, test_count_dict, is_display, result_save_dir)

    for slide_path in ndpi_list:

        image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        if 'HE' not in image_name and 'NC' not in image_name:
            print("evaluating image {}".format(slide_path))

            test_count_dict = {
                '阳性肿瘤细胞': 0,
                '阴性肿瘤细胞': 0,
                '阳性淋巴细胞': 0,
                '阴性淋巴细胞': 0,
                '阳性组织细胞': 0,
                '阴性组织细胞': 0,
                '纤维间质细胞': 0,
                '正常肺泡细胞': 0,
                '背景': 0,
                '细胞总数': 0,
                'PD-L1指数': 0
            }
            split_kfb(slide_path, 1024, net, test_count_dict, is_display, result_save_dir)

    for slide_path in mrxs_list:

        image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        if 'HE' not in image_name and 'NC' not in image_name:
            print("evaluating image {}".format(slide_path))

            test_count_dict = {
                '阳性肿瘤细胞': 0,
                '阴性肿瘤细胞': 0,
                '阳性淋巴细胞': 0,
                '阴性淋巴细胞': 0,
                '阳性组织细胞': 0,
                '阴性组织细胞': 0,
                '纤维间质细胞': 0,
                '正常肺泡细胞': 0,
                '背景': 0,
                '细胞总数': 0,
                'PD-L1指数': 0
            }
            split_kfb(slide_path, 1024, net, test_count_dict, is_display, result_save_dir)


def count_test_summary(image, net, cell_count_dict):
    center_coords, labels = cal_ki67_np(image, net)
    total_count = 0
    label_dict = {
        '阳性肿瘤细胞': 0,
        '阴性肿瘤细胞': 1,
        '阳性淋巴细胞': 2,
        '阴性淋巴细胞': 3,
        '阳性组织细胞': 4,
        '阴性组织细胞': 5,
        '纤维间质细胞': 6,
        '正常肺泡细胞': 7,
        '背景': 8,
    }

    reverse_dict = {}
    for k, v in label_dict.items():
        reverse_dict[v] = k
    # print(reverse_dict)
    if len(labels) != 0:
        for i in range(np.max(labels) + 1):
            this_num = np.sum(labels == i)
            cell_count_dict[reverse_dict[i]] = this_num
            total_count += this_num
        cell_count_dict["细胞总数"] = total_count
        cell_count_dict["PD-L1指数"] = \
            round(cell_count_dict["阳性肿瘤细胞"] / (cell_count_dict["阳性肿瘤细胞"] + cell_count_dict["阴性肿瘤细胞"] + 1e-10), 3)
    else:
        cell_count_dict["细胞总数"] = 0
        cell_count_dict["PD-L1指数"] = 0
    return cell_count_dict, center_coords, labels





def run_test(test_dir, net_name, weight_path):
    if net_name == 0:
        pass
        # from Core.models.albu_unet import AlbuNet as detnet
    elif net_name == 1:
        pass
        # from Core.models.attention_unet import AttU_Net as detnet
    elif net_name == 2:
        from Core.models.nested_unet import NestedUNet as detnet
    elif net_name == 3:
        pass
        # from Core.models.attention_unet import U_Net as detnet

    net = detnet(input_channels=3, num_classes=9)
    net_weights_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(net_weights_dict)
    net.eval()
    print('reload detection net weights from {}'.format(weight_path))

    if torch.cuda.is_available():
        net.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        print('You are using GPU')
    else:
        print('You are using CPU')

    test_name = os.path.basename(test_dir) + '_' + os.path.basename(os.path.dirname(weight_path)) + '_' + \
                os.path.splitext(os.path.basename(weight_path))[0]

    compare_with_annotations(test_dir=test_dir, test_name=test_name, net=net, is_display=True)


if __name__ == "__main__":
    allowed_net = {
        0: "albu_unet",
        1: "att_unet",
        2: "nested_unet",
        3: "unet"
    }

    # test_dir = "/data/caijt/数据集/PDL1_Positive_cell_seg_detection/PDL1200切片/Slide/DEMO"
    test_dir = '/data1/Caijt/PDL1_Parallel/'


    # run_test(test_dir, 0,
    #          weight_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_unet',
    #                                   "Model", "albu_unet_mix_1_2", "weights_epoch_55_1.303815746679902.pth"))

    # run_test(test_dir, 1,
    #          weight_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_unet',
    #                                   "Model", "att_unet_mix_1_2", "weights_epoch_30_1.795762307010591.pth"))

    # run_test(test_dir, 2,
    #          weight_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_unet',
    #                                   "Model", "nested_unet_mix_2_1_pr_512_new_dataset",
    #                                   "weights_epoch_27_2.6056071056260004.pth"))

    start_time = time.time()
    run_test(test_dir, 2,
             weight_path = "/data1/Caijt/PDL1_Parallel/Model/nested_unet_small_new_dataset/weights_epoch_168_1.095213971204228.pth")

    finish_time = time.time()
    minute, second = sec2min(finish_time- start_time)
    print('It takes {}min and {}s to finish the whole process'.format(minute, second))
    # run_test(test_dir, 3,
    #          weight_path= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'pytorch_unet',
    #                                    "Model", "unet_mix_1_2", "weights_epoch_125_1.5490121394395828.pth" ))
