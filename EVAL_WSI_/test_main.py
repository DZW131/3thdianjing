#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 16:13
# @Author  : YYuxin Kang
# @File    : eval_multi_cell_seg.py
# @Software: PyCharm
# @Comment:
#import kfb.kfbslide as kfbslide
import torch
import cv2
from parse_embolus import read_region_kfb
import numpy as np
import torch.backends.cudnn as cudnn
from options import Options
cudnn.benchmark = True
import os, sys
import pandas as pd
# import math
import mpi4py.MPI as MPI
from Slide.openslide_func import openSlide as di_openSlide
import openslide

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_root)

from Core.papSmear.multi_cls_cell_seg_batch import cal_ki67_np_region
from Core.models.nested_unet import NestedUNet as detnet


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

    cv2.imwrite(os.path.join(result_save_sub_dir, img_name), out)
    return new_contours


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


def check_sanity(region_id, cur_region, test_center_coords, test_labels, result_save_sub_dir):
    image = cur_region
    test_res_img = draw_center(image, test_center_coords, test_labels)
    test_res_img = test_res_img[:, :, :: -1]
    cv2.imwrite(os.path.join(result_save_sub_dir, str(region_id) + '.png'),
                test_res_img)


def check_sanity_region(region_id, cur_region, region, result_save_sub_dir):
    image = cur_region
    region_mask = np.zeros((image.shape[0], image.shape[1], 3))
    region_mask[:, :, 1] = region * 255
    image = image * 0.6 + region_mask * 0.4
    image = image[:, :, :: -1]
    result_save_sub_dir = os.path.join(result_save_sub_dir, 'region')
    os.makedirs(result_save_sub_dir, exist_ok=True)
    cv2.imwrite(os.path.join(result_save_sub_dir, str(region_id) + '.png'),
                image)


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


def local_process_per_gpu(local_data_dict, comm_rank, net, slide_path, slide, region_size, result_save_sub_dir,
                          slide_name):
    int_device = int(comm_rank)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    result_list = []

    region_start_list = []

    center_coords_list = []
    test_labels_list = []
    test_count_dict = {'阳性肿瘤细胞': 0, '阴性肿瘤细胞': 0, '阳性淋巴细胞': 0, '阴性淋巴细胞': 0, '阳性组织细胞': 0, '阴性组织细胞': 0, '纤维间质细胞': 0,
                       '正常肺泡细胞': 0, '背景': 0, '细胞总数': 0, 'PD-L1指数': 0}
    for key, value in local_data_dict.items():
        slide_name_now = slide_name + '-' + str(key) + '.png'
        region_id = key
        region_start = value
        result_list.append(region_start)
        cur_region = get_region(slide_path, slide, region_start, region_size)
        # torch.from_numpy(cur_region).cuda(int_device)
        net.cuda(int_device)
        time_now = time.time()
        cell_count_dict, test_center_coords, test_labels, region = count_test_summary(cur_region, net, test_count_dict,
                                                                                      int_device)
        time_end = time.time()
        print(time_end - time_now)
        check_sanity(region_id, cur_region, test_center_coords, test_labels, result_save_sub_dir)
        check_sanity_region(region_id, cur_region, region, result_save_sub_dir)
        print(1)
        region_start_list.append(region_start)
        center_coords_list.append(test_center_coords)
        test_labels_list.append(test_labels)
    return region_start_list, center_coords_list, test_labels_list


def split_kfb(slides_dir, crop_len, result_save_sub_dir):
    slide_path, slide_name = os.path.split(slides_dir)

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
        contours = gamma_main(cur_slide, result_save_sub_dir, slide_name)

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

                    region_start = (crop_start_x, crop_start_y)
                    region_start_1 = (crop_start_x, crop_start_y + 1024)
                    region_start_2 = (crop_start_x + 1024, crop_start_y)
                    region_start_3 = (crop_start_x + 1024, crop_start_y + 1024)
                    for roi_list in contours:

                        roi_list = roi_list * 16
                        flag = cv2.pointPolygonTest(roi_list, region_start, False)
                        flag_1 = cv2.pointPolygonTest(roi_list, region_start_1, False)
                        flag_2 = cv2.pointPolygonTest(roi_list, region_start_2, False)
                        flag_3 = cv2.pointPolygonTest(roi_list, region_start_3, False)
                        if flag == 1 or flag_1 == 1 or flag_2 == 1 or flag_3 == 1:
                            region_info_dict[num_patch] = region_start
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
                    region_start = (crop_start_x, crop_start_y)
                    region_info_dict[num_patch] = region_start
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
                region_start = (crop_start_x, crop_start_y)
                region_info_dict[num_patch] = region_start
                crop_start_x = (j + 1) * crop_len
                num_patch = num_patch + 1

    return region_info_dict, result_save_sub_dir


def compare_with_annotations(test_name, slide_path, result_save_sub_dir):
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
        region_info_list, result_save_sub_dir = split_kfb(slide_path, 1024, result_save_sub_dir)

    return region_info_list


def count_test_summary(image, net, cell_count_dict, device):
    center_coords, labels, region = cal_ki67_np_region(image, net, device)
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
    return cell_count_dict, center_coords, labels, region


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
        return group


def run_test(test_dir, slide_path, weight_path, result_save_sub_dir):
    test_name = os.path.basename(test_dir) + '_' + os.path.basename(os.path.dirname(weight_path)) + '_' + \
                os.path.splitext(os.path.basename(weight_path))[0]

    return compare_with_annotations(test_name=test_name, slide_path=slide_path, result_save_sub_dir=result_save_sub_dir)


def save_summary(combine_test_labels, result_save_sub_dir, kfb_name, comm_size):
    label_dict = {'阳性肿瘤细胞': 0, '阴性肿瘤细胞': 1, '阳性淋巴细胞': 2, '阴性淋巴细胞': 3, '阳性组织细胞': 4, '阴性组织细胞': 5, '纤维间质细胞': 6,
                  '正常肺泡细胞': 7, '背景': 8, }
    result_summary = {'slide name': '', '阳性肿瘤细胞': 0, '阴性肿瘤细胞': 0, '阳性淋巴细胞': 0, '阴性淋巴细胞': 0, '阳性组织细胞': 0, '阴性组织细胞': 0,
                      '纤维间质细胞': 0,
                      '正常肺泡细胞': 0, '背景': 0, '细胞总数': 0, 'PD-L1指数': 0}
    reverse_dict = {}
    total_count = 0
    for k, v in label_dict.items():
        reverse_dict[v] = k
    result_summary['slide name'] = kfb_name
    if combine_test_labels is not None:
        if comm_size != 1:
            for i in range(comm_size):
                for j in range(len(combine_test_labels[i])):
                    if combine_test_labels[i][j] != []:
                        for k in range(np.max(combine_test_labels[i][j]) + 1):
                            this_num = np.sum(combine_test_labels[i][j] == k)
                            result_summary[reverse_dict[k]] += this_num
                            total_count += this_num
        else:
            i = 0
            for j in range(len(combine_test_labels[i])):
                if combine_test_labels[i][j] != []:
                    for k in range(np.max(combine_test_labels[i][j]) + 1):
                        this_num = np.sum(combine_test_labels[i][j] == k)
                        result_summary[reverse_dict[k]] += this_num
                        total_count += this_num
        result_summary["细胞总数"] = total_count
        result_summary["PD-L1指数"] = round(
            result_summary["阳性肿瘤细胞"] / (result_summary["阳性肿瘤细胞"] + result_summary["阴性肿瘤细胞"] + 1e-10), 3)
        df = pd.DataFrame(result_summary, index=[0])
        columns = ['slide name', '阳性肿瘤细胞', '阴性肿瘤细胞', '阳性淋巴细胞', '阴性淋巴细胞', '阳性组织细胞', '阴性组织细胞', '纤维间质细胞', '正常肺泡细胞', '背景',
                   '细胞总数', 'PD-L1指数']
        df.to_csv(os.path.join(result_save_sub_dir, "summary_{}.csv".format(kfb_name)),
                  columns=columns, index=True)


def processing(comm, comm_rank, comm_size, slide_path, weight_path):
    _, slide_name = os.path.split(slide_path)
    kfb_name, kfb_ext = os.path.splitext(slide_name)
    result_save_sub_dir = os.path.join(result_save_dir, kfb_name)
    os.makedirs(result_save_sub_dir, exist_ok=True)

    if not os.path.exists(
            os.path.join(result_save_sub_dir, "summary_{}.csv".format(kfb_name))):
        start_time = time.time()
        print("evaluating image {}".format(slide_path))
        if 'ndpi' in slide_path or 'mrxs' in slide_path:
            slide = openslide.open_slide(slide_path)
        else:
            slide = kfbslide.open_kfbslide(slide_path)
        if comm_rank == 0:
            # net = detnet(input_channels=3, num_classes=9, channel=32)
            net = detnet(input_channels=3, num_classes=12)
            net_weights_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)

            net.load_state_dict(net_weights_dict)
            net.eval()
            region_info_dict = run_test(test_dir, slide_path, weight_path=weight_path,
                                        result_save_sub_dir=result_save_sub_dir)
            region_info_dict = split2groups(region_info_dict=region_info_dict,
                                            comm_size=comm_size)
        else:
            region_info_dict = None
            # net = detnet(input_channels=3, num_classes=9, channel=32)
            net = detnet(input_channels=3, num_classes=12)
            net_weights_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
            net.load_state_dict(net_weights_dict)
            net.eval()

        local_data_dict = comm.scatter(region_info_dict, root=0)
        region_start_list, center_coords_list, test_labels_list = local_process_per_gpu(local_data_dict, comm_rank, net,
                                                                                        slide_path, slide,
                                                                                        region_size=(1024, 1024),
                                                                                        result_save_sub_dir=result_save_sub_dir,
                                                                                        slide_name=kfb_name)
        combine_region_start = comm.gather(region_start_list, root=0)
        combine_center_coords = comm.gather(center_coords_list, root=0)
        combine_test_labels = comm.gather(test_labels_list, root=0)
        finish_time = time.time()
        if comm_rank == 0:
            minute, second = sec2min(finish_time - start_time)
            print('It takes {}min and {}s to finish the whole process'.format(minute, second))
        try:
          save_summary(combine_test_labels, result_save_sub_dir, kfb_name, comm_size)
        except:
            print("Fail to save summary")


if __name__ == "__main__":

    test_dir = '/media/deepin/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/上肿'
    comm = MPI.COMM_WORLD

    comm_rank = comm.Get_rank()
    print(comm_rank)
    comm_size = comm.Get_size()
    print(comm_size)

    result_save_dir = '/media/kd/Seagate Backup Plus Drive/Seagate/PD-L1_cell_detection_test_result_WSI'
    result_save_dir = os.path.join(result_save_dir, "TEST_RESULT")
    os.makedirs(result_save_dir, exist_ok=True)

    weight_path = "/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/research/UNET/S5_NET_PDL1_CELL_DETECTION/4_Point_detection/pytorch_unet/Model/nested_unet_select_dataset_512_for_wsi_12calss_with_out_xuanwu/weights_epoch_252_1.45360054330128.pth"
    ndpi_list = filesystem.find_ext_files(test_dir, ".ndpi")
    kfb_list = filesystem.find_ext_files(test_dir, '.kfb')
    mrxs_list = filesystem.find_ext_files(test_dir, '.mrxs')
    svs_list = filesystem.find_ext_files(test_dir, '.svs')

    for slide_path in kfb_list:
        _, slide_name = os.path.split(slide_path)
        kfb_name, kfb_ext = os.path.splitext(slide_name)
        region_name = kfb_name + ".png"
        if 'HE' not in region_name and 'NC' not in region_name and 'V-' not in region_name:
            processing(comm, comm_rank, comm_size, slide_path, weight_path)

    for slide_path in ndpi_list:

        kfb_name, kfb_ext = os.path.splitext(slide_name)
        region_name = kfb_name + ".png"
        if 'HE' not in region_name and 'NC' not in region_name:
            processing(comm, comm_rank, comm_size, slide_path, weight_path)

    for slide_path in mrxs_list:
        _, slide_name = os.path.split(slide_path)
        kfb_name, kfb_ext = os.path.splitext(slide_name)
        region_name = kfb_name + ".png"
        if 'HE' not in region_name and 'NC' not in region_name:
            processing(comm,
                       comm_rank,
                       comm_size,
                       slide_path,
                       weight_path)
