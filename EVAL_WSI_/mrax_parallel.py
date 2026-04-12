#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 16:13
# @Author  : Can Cui
# @File    : eval_multi_cell_seg.py
# @Software: PyCharm
# @Comment:
import kfb.kfbslide as kfbslide
import os, json, pdb
from copy import deepcopy
import torch
import cv2
from parse_embolus import read_region_kfb
import numpy as np
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
# from Slide.openslide_func import openSlide
# from scipy.misc import imsave, imread
import os, sys
import pandas as pd
# import math
import mpi4py.MPI as MPI
from Slide.openslide_func import openSlide as di_openSlide
import openslide
from Core.models.nested_unet import NestedUNet as detnet
import random
from Core.models.HarDMSEG import HarDMSEG as detnet
#from Core.models.nested_unet import NestedUNet as detnet

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

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

        color_dict = {"0": (255, 0, 0),# 阳肿
                      "1": (0, 255, 0),  # 阴肿
                      "2": (255, 255, 0),  # 阳淋
                      '3': (255, 0, 255),  # 阴淋
                      '4': (255, 255, 0),  # 阳组
                      '5': (255, 0, 255),  # 阴组
                      '6': (255, 0, 255),  # 纤维
                      '7': (255, 0, 255),  # 肺泡
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


def get_region(slides_dir, slide, region_start, region_size):
    if 'ndpi' in slides_dir or 'mrxs' in slides_dir:
        try:
            cur_region = np.array(
                slide.read_region(region_start, 0, region_size))
        except:
            raise ValueError('ext not proper')
        else:
            cur_region = RGBAtoBGR(cur_region)
    else:
        cur_region = read_region_kfb(slides_dir, slide, region_start, region_size)

    return cur_region


def check_sanity(region_id, cur_region, test_center_coords, test_labels,result_save_sub_dir):
    image = cur_region
    test_res_img = draw_center(image, test_center_coords, test_labels)
    test_res_img = test_res_img[:, :, :: -1]
    cv2.imwrite(os.path.join(result_save_sub_dir,str(region_id)+'.png'),
                test_res_img)


def save_patch_results(cur_region, test_center_coords, test_labels, save_path):
    image = cur_region
    # image = image[:, :, :: -1]
    test_res_img = draw_center(image, test_center_coords, test_labels)
    test_res_img = test_res_img[:, :, :: -1]
    cv2.imwrite(save_path, test_res_img)


def summary(result_summary, result_save_sub_dir, kfb_name):
    df = pd.DataFrame(result_summary)
    columns = ['slide name', '阳性肿瘤细胞', '阴性肿瘤细胞', '阳性淋巴细胞', '阴性淋巴细胞', '阳性组织细胞', '阴性组织细胞', '纤维间质细胞', '正常肺泡细胞', '背景',
               '细胞总数', 'PD-L1指数']
    df.to_csv(os.path.join(result_save_sub_dir, "summary_{}.csv".format(kfb_name)),
              columns=columns, index=False)


def RGBAtoBGR(image_rgba):
    r, g, b, a = cv2.split(image_rgba)
    image_bgr = cv2.merge([r, g, b])
    return image_bgr


def local_process_per_gpu(local_data_dict, comm_rank, net, slide_path, slide, region_size,result_save_sub_dir):
    int_device = int(comm_rank)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    result_list = []
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
    region_start_list = []
    cell_count_list = []
    center_coords_list = []
    test_labels_list = []
    for key, value in local_data_dict.items():
        region_id = key
        region_start = value
        result_list.append(region_start)
        cur_region = get_region(slide_path, slide, region_start, region_size)
        # torch.from_numpy(cur_region).cuda(int_device)
        net.cuda(int_device)
        cell_count_dict, test_center_coords, test_labels = count_test_summary(cur_region, net, test_count_dict,
                                                                              int_device)
        check_sanity(region_id, cur_region, test_center_coords, test_labels,result_save_sub_dir)
        region_start_list.append(region_start)
        cell_count_list.append(cell_count_dict)
        center_coords_list.append(test_center_coords)
        test_labels_list.append(test_labels)

    # local_data_results =
    return region_start_list, cell_count_list, center_coords_list, test_labels_list


def split_kfb(slides_dir, crop_len):
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
        '/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/research/UNET/S5_NET_PDL1_CELL_DETECTION/4_Point_detection/PDL1_Parallel/result',
        kfb_name)

    os.makedirs(result_save_sub_dir, exist_ok=True)
    # if not os.path.exists(os.path.join(result_save_sub_dir, "summary_{}.csv".format(kfb_name))):
    slide_OTUS = di_openSlide(slides_dir)
    size = slide_OTUS.slide.level_dimensions[0]
    if 'ndpi' in slides_dir or 'mrxs' in slides_dir:
        slide = openslide.open_slide(slides_dir)

    else:
        slide = kfbslide.open_kfbslide(slides_dir)
    width = size[0]
    height = size[1]
    region_info_dict = {}
    if size[0] > 2 ** 4 and size[1] > 2 ** 4:
        # try:
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
                    region_start_1 = (crop_start_x, crop_start_y + 1024)
                    region_start_2 = (crop_start_x + 1024, crop_start_y)
                    region_start_3 = (crop_start_x + 1024, crop_start_y + 1024)
                    for roi_list in contours:
                        # print(roi_list)
                        roi_list = roi_list * 16
                        flag = cv2.pointPolygonTest(roi_list, region_start, False)
                        flag_1 = cv2.pointPolygonTest(roi_list, region_start_1, False)
                        flag_2 = cv2.pointPolygonTest(roi_list, region_start_2, False)
                        flag_3 = cv2.pointPolygonTest(roi_list, region_start_3, False)
                        if flag == 1 or flag_1 == 1 or flag_2 == 1 or flag_3 == 1:
                            # region_dict = {num_patch: region_start}
                            region_info_dict[num_patch] = region_start
                            # region_info_list.append(region_dict)
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
                    region_info_dict[num_patch] = region_start
                    # region_dict = {num_patch: region_start}
                    # region_info_list.append(region_dict)
                    crop_start_x = (j + 1) * crop_len
                    num_patch = num_patch + 1
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
                # region_dict = {num_patch: region_start}
                region_info_dict[num_patch] = region_start
                # region_info_list.append(region_dict)
                crop_start_x = (j + 1) * crop_len
                num_patch = num_patch + 1

    return region_info_dict,result_save_sub_dir


def compare_with_annotations(test_name, slide_path):
    result_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TEST_RESULT", test_name)
    os.makedirs(result_save_dir, exist_ok=True)

    # for slide_path in kfb_list:

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
        region_info_list,result_save_sub_dir = split_kfb(slide_path, 1024)

    return region_info_list,result_save_sub_dir


def count_test_summary(image, net, cell_count_dict, device):
    center_coords, labels = cal_ki67_np(image, net, device)
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


def split2groups(region_info_dict, comm_size):
    list_length = len(region_info_dict)
    print('list_length', list_length)
    regions_per_gpu = list_length // comm_size
    reserved_region_num = list_length % comm_size
    full_region_num = regions_per_gpu * comm_size
    group = [dict(list(region_info_dict.items())[i:i + regions_per_gpu]) for i in
             range(0, full_region_num, regions_per_gpu)]
    if reserved_region_num == 0:
        return group
    else:
        for i_region in range(reserved_region_num):
            key, value = list(region_info_dict.items())[full_region_num + i_region]
            group[i_region][key] = value
            # group[i_region].append(region_list[full_region_num + i_region])
        return group


def run_test(test_dir, slide_path, weight_path):
    # if net_name == 0:
    #     from Core.models.albu_unet import AlbuNet as detnet
    # elif net_name == 1:
    #     from Core.models.attention_unet import AttU_Net as detnet
    # elif net_name == 2:
    #     from Core.models.nested_unet import NestedUNet as detnet
    # elif net_name == 3:
    #     from Core.models.attention_unet import U_Net as detnet
    #

    # if torch.cuda.is_available():
    #     net.cuda()
    #     import torch.backends.cudnn as cudnn
    #     cudnn.benchmark = True
    #     print('You are using GPU')
    # else:
    #     print('You are using CPU')

    test_name = os.path.basename(test_dir) + '_' + os.path.basename(os.path.dirname(weight_path)) + '_' + \
                os.path.splitext(os.path.basename(weight_path))[0]

    # region_info_list, region_info_list, result_save_dir = compare_with_annotations(test_dir, test_name=test_name)
    return compare_with_annotations(test_name=test_name, slide_path=slide_path)


if __name__ == "__main__":
    allowed_net = {0: "albu_unet", 1: "att_unet", 2: "nested_unet", 3: "unet"}
    test_dir = '/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/数据集/PDL1_Positive_cell_seg_detection/PDL1200切片/Slide/PD-L1SP263原切片/SP263/协和/KFB_SLIDE'
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    weight_path = "/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/research/UNET/S5_NET_PDL1_CELL_DETECTION/4_Point_detection/pytorch_unet/Model/albunet_new_dataset_512_for_wsi/weights_epoch_93_0.7033155349696555.pth"
    ndpi_list = filesystem.find_ext_files(test_dir, ".ndpi")
    kfb_list = filesystem.find_ext_files(test_dir, '.kfb')
    mrxs_list = filesystem.find_ext_files(test_dir, '.mrxs')
    svs_list = filesystem.find_ext_files(test_dir, '.svs')

    for slide_path in kfb_list:
        image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"
        if 'HE' not in image_name and 'NC' not in image_name and 'V-' not in image_name:
            start_time = time.time()
            slide = kfbslide.open_kfbslide(slide_path)
            if comm_rank == 0:
                net = detnet(input_channels=3, num_classes=9,channel=32)
                net_weights_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
                net.load_state_dict(net_weights_dict)
                net.eval()
                region_info_dict,result_save_sub_dir = run_test(test_dir, slide_path,
                                            weight_path="/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/research/UNET/S5_NET_PDL1_CELL_DETECTION/4_Point_detection/PD-L1_test_demo_KYX_06092021/Model/nested_unet_small_new_dataset/weights_epoch_168_1.095213971204228.pth")
                region_info_dict = split2groups(region_info_dict=region_info_dict, comm_size=comm_size)
            else:
                region_info_dict = None
                net = detnet(input_channels=3, num_classes=9,channel=32)
                net_weights_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
                net.load_state_dict(net_weights_dict)
                net.eval()
                result_save_sub_dir=''
            local_data_dict = comm.scatter(region_info_dict, root=0)
            region_start_list, cell_count_list, center_coords_list, test_labels_list = local_process_per_gpu(
                local_data_dict, comm_rank, net, slide_path, slide, region_size=(1024, 1024),result_save_sub_dir=result_save_sub_dir)
            combine_region_start = comm.gather(region_start_list, root=0)
            combine_cell_count = comm.gather(cell_count_list, root=0)
            combine_center_coords = comm.gather(center_coords_list, root=0)
            combine_test_labels = comm.gather(test_labels_list, root=0)
            finish_time = time.time()
            if comm_rank == 0:
                minute, second = sec2min(finish_time - start_time)
                print('It takes {}min and {}s to finish the whole process'.format(minute, second))

    # allowed_net = {
    #     0: "albu_unet",
    #     1: "att_unet",
    #     2: "nested_unet",
    #     3: "unet"
    # }
    #
    # # test_dir = "/data/caijt/数据集/PDL1_Positive_cell_seg_detection/PDL1200切片/Slide/DEMO"
    # test_dir = '/data1/Caijt/PDL1_Parallel'
    # comm = MPI.COMM_WORLD
    # comm_rank = comm.Get_rank()
    # comm_size = comm.Get_size()
    # # print(comm_size)
    #
    # # run_test(test_dir, 0,
    # #          weight_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_unet',
    # #                                   "Model", "albu_unet_mix_1_2", "weights_epoch_55_1.303815746679902.pth"))
    #
    # # run_test(test_dir, 1,
    # #          weight_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_unet',
    # #                                   "Model", "att_unet_mix_1_2", "weights_epoch_30_1.795762307010591.pth"))
    #
    # # run_test(test_dir, 2,
    # #          weight_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_unet',
    # #                                   "Model", "nested_unet_mix_2_1_pr_512_new_dataset",
    # #                                   "weights_epoch_27_2.6056071056260004.pth"))
    #
    # start_time = time.time()
    # net = detnet(input_channels=3, num_classes=9)
    # # kfb_list = filesystem.find_ext_files(test_dir, '.kfb')
    # # slide_path = kfb_list[0]
    # slide_path = "/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/数据集/PDL1_Positive_cell_seg_detection/PDL1200切片/Slide/DEMO/18-5405 PD-L1.kfb"
    # slide = kfbslide.open_kfbslide(slide_path)
    #
    # if comm_rank == 0:
    #     region_info_dict = run_test(test_dir, slide_path,
    #                                 weight_path="/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/research/UNET/S5_NET_PDL1_CELL_DETECTION/4_Point_detection/PDL1_Parallel/Model/nested_unet_small_new_dataset/weights_epoch_168_1.095213971204228.pth")
    #     region_info_dict = split2groups(region_info_dict=region_info_dict, comm_size=comm_size)
    #
    # else:
    #     region_info_dict = None
    #
    # local_data_dict = comm.scatter(region_info_dict, root=0)
    # # local_data_list = comm.scatter(region_info_list, root = 0)
    # region_start_list, cell_count_list, center_coords_list, test_labels_list = local_process_per_gpu(local_data_dict,
    #                                                                                                  comm_rank, net,
    #                                                                                                  slide_path, slide,
    #                                                                                                  region_size=(
    #                                                                                                  1024, 1024))
    #
    # combine_region_start = comm.gather(region_start_list, root=0)
    # combine_cell_count = comm.gather(cell_count_list, root=0)
    # combine_center_coords = comm.gather(center_coords_list, root=0)
    # combine_test_labels = comm.gather(test_labels_list, root=0)
    #
    # # print(combine_data_results)
    #
    # # net = detnet(input_channels=3, num_classes=9)
    # # net_weights_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
    # # net.load_state_dict(net_weights_dict)
    # # net.eval()
    # # print('reload detection net weights from {}'.format(weight_path))
    #
    # finish_time = time.time()
    # if comm_rank == 0:
    #     minute, second = sec2min(finish_time - start_time)
    #     print('It takes {}min and {}s to finish the whole process'.format(minute, second))
    #
    # # run_test(test_dir, 3,
    # #          weight_path= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'pytorch_unet',
    # #                                    "Model", "unet_mix_1_2", "weights_epoch_125_1.5490121394395828.pth" ))
