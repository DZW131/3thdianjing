#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 16:17
# @Author  : Can Cui
# @File    : run_np.py
# @Software: PyCharm
import os, sys

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
alg_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, proj_root)
sys.path.insert(0, alg_root)
# from settings import lib_path
# sys.path.insert(0, lib_path)
import cv2

# import mpi4py.MPI as MPI
# from mpi4py import MPI
import torch
import numpy as np
import subprocess
import json
from itertools import chain
import argparse
from Slide.dispatch import openSlide
from src.utils import delete_prev_json, split_patches, split2groups, dump_results, filter_points, filter_contours
from src.seg_tissue_area import find_tissue_countours
from src.multi_cls_cell_det import cal_region_deeplab, cal_bxr_np
from models.waternet.detr import build_model
from models.deeplab.deeplab import DeepLab

import copy
import math


def load_model(default_device, slide_mpp):
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model')

    if slide_mpp == 0.250001:
        # r_net_weights_path = os.path.join(model_dir, 'bxr_deeplab_0805.pth')
        r_net_weights_path = os.path.join(model_dir, 'bxr_deeplab_mpp=0.25.pth')
    else:
        # r_net_weights_path = os.path.join(model_dir, 'bxr_deeplab_1031_mpp=0.5.pth')
        # r_net_weights_path = os.path.join(model_dir, 'bxr_deeplab_20230412_mpp=0.5.pth')
        # r_net_weights_path = r"/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/code/region_level_train/run/beiertongbxr_region/checkpoint.pth.tar"
        # r_net_weights_path = r"/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/code/region_level_train/run/qidai_region/0412/model_best.pth.tar"
        # r_net_weights_path = r"/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/code/region_level_train/model_bkp/0.6223_model_best.pth.tar"
        # r_net_weights_path = r"/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/code/region_level_train/run/taimo_region/0617/model_best.pth.tar"
        # r_net_weights_path = r"/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/code/region_level_train/run/taimo_region/0617/experiment_1/checkpoint.pth.tar"
        r_net_weights_path = r"D:\msqtry\region_level_train\run\prostate_tls\384_192_2\model_best.pth.tar"
        # r_net_weights_path = os.path.join(model_dir, 'bxr_deeplab_mpp=0.5.pth')
    r_net = DeepLab(num_classes=class_num,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=True,
                    freeze_bn=False)
    r_net_weights_dict = torch.load(r_net_weights_path, map_location=lambda storage, loc: storage)['state_dict']
    # r_net_weights_dict = torch.load(r_net_weights_path, map_location=lambda storage, loc: storage)
    weights_dict = {}
    for k, v in r_net_weights_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    r_net_weights_dict = weights_dict

    r_net.load_state_dict(r_net_weights_dict)
    r_net.eval()

    if torch.cuda.is_available():
        r_net.cuda(default_device)
        print('{} are using GPU - {}'.format(os.getpid(), default_device))

    return r_net


def read_clear_img(slide, scale=16, read_scale=2, split_num=8):
    """
    本函数用于读取高尺度的图，resize到指定尺度
    @param slide: 切片对象
    @param scale: 要得到的切片尺度
    @param read_scale: 要读取的切片尺度，越低却清晰
    @param split_num: 增大划分数，会显著地降低内存消耗
    @return:读到的图片
    """
    # split_num = 8
    w_thumbnail = math.ceil(slide.width / scale)
    h_thumbnail = math.ceil(slide.height / scale)
    w_patch_thumbnail = math.ceil(w_thumbnail / split_num)
    h_patch_thumbnail = math.ceil(h_thumbnail / split_num)

    # read_scale = 1
    resize_scale_ratio = read_scale / scale
    # w_read = math.ceil(slide.width / read_scale)
    # h_read = math.ceil(slide.height / read_scale)
    w_read = math.ceil(slide.width)
    h_read = math.ceil(slide.height)
    w_patch_read = math.ceil(w_read / split_num)
    h_patch_read = math.ceil(h_read / split_num)

    thumbnail = np.zeros((h_thumbnail, w_thumbnail, 3), dtype=np.uint8)
    from tqdm import tqdm
    for row in tqdm(range(split_num)):
        for col in range(split_num):
            if (col + 1 >= split_num):
                cur_w_patch_read = w_read - w_patch_read * col
            else:
                cur_w_patch_read = w_patch_read
            if (row + 1 >= split_num):
                cur_h_patch_read = h_read - h_patch_read * row
            else:
                cur_h_patch_read = h_patch_read
            x_read = w_patch_read * col
            y_read = h_patch_read * row
            patch = slide.read((x_read, y_read), size=(cur_w_patch_read, cur_h_patch_read), scale=read_scale)
            patch = cv2.resize(patch, dsize=None, fx=resize_scale_ratio, fy=resize_scale_ratio,
                               interpolation=cv2.INTER_LINEAR)

            x_thumbnail = w_patch_thumbnail * col
            y_thumbnail = h_patch_thumbnail * row
            thumbnail[y_thumbnail:y_thumbnail + patch.shape[0], x_thumbnail:x_thumbnail + patch.shape[1], :] = patch
    return thumbnail

def draw_center(img, center, label,radius=10,thickness=6):
    img = img.copy()
    if len(center) != 0:
        num_center = center.shape[0]

        for i in range(num_center):
            if str(label[i]) in color_dict:
                img = cv2.circle(img, (center[i, 0] , center[i, 1]), radius, color_dict[str(label[i])], thickness = -1)


        return img
    else:
        return img
def analysis_wsi(wsi_cell_center_coords, wsi_cell_labels, str_crop_indexs, thumbnail, scale=16.0, radius=2, thickness=1):
    from collections import defaultdict
    cell_count_dict = defaultdict(list)

    # 全场图作画
    try:
        # wsi_cell_center_coords = np.concatenate(wsi_cell_center_coords, axis=0)
        wsi_cell_center_coords = wsi_cell_center_coords // scale
        wsi_cell_center_coords = wsi_cell_center_coords.astype(np.int32)
        # wsi_cell_labels = np.concatenate(wsi_cell_labels, axis=0)
        wsi_cell_labels = wsi_cell_labels.astype(np.int)
        # wsi_cell_image = draw_center(thumbnail, wsi_cell_center_coords, wsi_cell_labels, radius=10, thickness=6)
        wsi_cell_image = draw_center(thumbnail, wsi_cell_center_coords, wsi_cell_labels, radius=2, thickness=1)
        wsi_cell_image = wsi_cell_image[:, :, ::-1]
    except:
        wsi_cell_image = thumbnail
    return  wsi_cell_image


def cal_np(slide_path, x_coords=[], y_coords=[]):
    result_root = os.path.dirname(slide_path)
    # slide_path = '"' + slide_path + '"'
    result_file = 'np_result.json'

    if os.path.exists(os.path.join(result_root, result_file)):
        returncode = 0
    else:
        delete_prev_json(result_root, result_file)
        gpu_num = torch.cuda.device_count()
        num_process_per_gpu = 1
        command = ['mpiexec', '-np', str(gpu_num * num_process_per_gpu), 'python', '-m',
                   'Algorithms.NP.run_np']
        if sys.platform == 'linux':
            command[3] = 'python3'
            command.insert(1, '--allow-run-as-root')
        command.append('--slide_path {}'.format(slide_path))
        command.append('--roi_coords')
        command.append(json.dumps(x_coords, separators=(',', ':')))
        command.append(json.dumps(y_coords, separators=(',', ':')))
        command_str = ' '.join(command)
        if sys.platform == 'win32':
            bat_name = 'run_{}.bat'.format(os.path.splitext(os.path.basename(slide_path))[0])
            with open(os.path.join(alg_root, bat_name), 'w', encoding='gbk') as f:
                f.write(command_str)
        elif sys.platform == 'linux':
            bat_name = 'run_{}.sh'.format(os.path.splitext(os.path.basename(slide_path))[0])
            with open(os.path.join(alg_root, bat_name), 'w', encoding='utf-8') as f:
                f.write(command_str)
            os.chmod(os.path.join(alg_root, bat_name), os.stat.S_IRWXU)

        status = subprocess.Popen(os.path.join(alg_root, bat_name), cwd=proj_root, shell=True, env=os.environ.copy())

        status.communicate()
        returncode = status.returncode
        # returncode = status.wait()

    if returncode == 0:
        # if os.path.exists(os.path.join(alg_root, bat_name)):
        #     os.remove(os.path.join(alg_root, bat_name))

        with open(os.path.join(result_root, result_file), 'r') as res_f:
            result = json.load(res_f)

            center_x_coords = np.array(result['centers']['x'])
            center_y_coords = np.array(result['centers']['y'])
            cls_labels_np = np.array(result['cls_labels'])
            region_contour_ls = result['region_contours']
            region_label_ls = result['region_labels']
            total_area = result['total_area']

            center_coords_np = np.concatenate(
                (np.expand_dims(center_x_coords, axis=-1), np.expand_dims(center_y_coords, axis=-1)), axis=-1)

            # draw
            slide = openSlide(slide_path)
            save_result_folder = "result"
            slide_name = os.path.basename(slide_path)
            os.makedirs(os.path.join(save_result_folder, slide_name), exist_ok=True)
            scale = 16
            img_ori = slide.read(scale=scale)
            wsi_cell_image_bg = np.zeros((int(slide.height / scale), int(slide.width / scale), 3))
            wsi_cell_image_bg[:, :, :] = 0
            wsi_cell_image = analysis_wsi(center_coords_np, cls_labels_np, [],wsi_cell_image_bg, scale=scale, radius=2, thickness=1)
            wsi_cell_image = wsi_cell_image[:, :, ::-1]  # BGR2RGB
            roi_color = [Hex_to_RGB("#00FFBB"), Hex_to_RGB("#B821BB"), Hex_to_RGB("#3D1CE1")]
            cv2.imwrite(os.path.join(save_result_folder, slide_name, slide_name + "_cell.jpg"),
                        wsi_cell_image[:, :, ::-1])
            for class_ind in range(1, 4):
                contours = [(np.array(region_contour_ls[r_index]) / scale).astype(np.int32)[:, None, :] for r_index, cl
                            in enumerate(region_label_ls) if cl == class_ind]
                wsi_cell_image = cv2.drawContours(wsi_cell_image, contours, -1, roi_color[class_ind - 1], 3)
                wsi_cell_image_white = cv2.drawContours(wsi_cell_image_bg, contours, -1, roi_color[class_ind - 1], 3)
            cv2.imwrite(os.path.join(save_result_folder, slide_name, slide_name + "_region.jpg"),
                        wsi_cell_image_bg[:, :, ::-1])
            cv2.imwrite(os.path.join(save_result_folder, slide_name, slide_name + "_cell&region.jpg"),
                        wsi_cell_image[:, :, ::-1])
            cv2.imwrite(os.path.join(save_result_folder, slide_name, slide_name + "_img.jpg"), img_ori[:, :, ::-1])
    else:
        raise ValueError('run subprocess failed')
    # print("len(region_contour_ls):", len(region_contour_ls))
    # print("region_label_ls:", region_label_ls)

    return center_coords_np, cls_labels_np, region_contour_ls, region_label_ls, total_area


def compute_process(slide_path, x_coords=[], y_coords=[], patch_size=(384, 384)):
    comm = MPI.COMM_WORLD
    comm_rank, comm_size = comm.Get_rank(), comm.Get_size()
    gpu_num = torch.cuda.device_count()
    int_device = int(comm_rank % gpu_num)

    slide = openSlide(slide_path)
    slide_mpp = slide.mpp
    standard_region_mpp = 4
    if slide_mpp is not None:
        region_resize_ratio = slide_mpp / standard_region_mpp
    else:
        slide_mpp = 0.250001
        standard_cell_mpp = 0.25
        standard_region_mpp = 0.25
        cell_resize_ratio = 1
        region_resize_ratio = 1
    # cell_resize_ratio = 1
    # region_resize_ratio = 1
    r_net = load_model(int_device, slide_mpp)
    patch_size_ori = patch_size
    region_patch_size = (int(patch_size[0] / region_resize_ratio), int(patch_size[1] / region_resize_ratio))

    if comm_rank == 0:
        if len(x_coords) > 0:
            contours = [np.expand_dims(np.stack([x_coords, y_coords], axis=1), axis=1)]
        else:
            contours = find_tissue_countours(slide)
        R_region_info_dict = split_patches(slide, patch_w=region_patch_size[0], patch_h=region_patch_size[1],
                                           contours=contours)
        R_region_info_dict = split2groups(region_info_dict=R_region_info_dict, comm_size=comm_size)

    else:
        C_region_info_dict = None
        R_region_info_dict = None

    R_local_data_dict = comm.scatter(R_region_info_dict, root=0)
    mean, std = np.load(os.path.join(alg_root, 'mean_std.npy'))

    region_contours_list = []
    region_labels_list = []

    for key, value in R_local_data_dict.items():
        try:
            region_start = value
            cur_region = slide.read(([region_start[0], region_start[1]]), (region_patch_size[0], region_patch_size[1]),
                                    1 / region_resize_ratio)
            cur_region = cur_region.astype(np.uint8)
            cur_shape = (cur_region.shape[1], cur_region.shape[0])
            if (cur_shape[0] != patch_size_ori[0] or cur_shape[1] != patch_size_ori[1]) \
                    and abs(cur_shape[0] - patch_size_ori[0]) < 2 and abs(cur_shape[1] - patch_size_ori[1]) < 2:
                cur_region = cv2.resize(cur_region, dsize=patch_size_ori, interpolation=cv2.INTER_CUBIC)
            result_contours, result_contours_labels = cal_region_deeplab(cur_region, r_net, int_device, patch_size=512)
            if len(result_contours) > 0:
                for contour_idx in range(len(result_contours)):
                    for cell_idx in range(len(result_contours[contour_idx])):
                        result_contours[contour_idx][cell_idx][0] = \
                            result_contours[contour_idx][cell_idx][0] / region_resize_ratio + region_start[0]
                        result_contours[contour_idx][cell_idx][1] = \
                            result_contours[contour_idx][cell_idx][1] / region_resize_ratio + region_start[1]
                region_contours_list.extend(result_contours)
                region_labels_list.extend(result_contours_labels)
        except Exception as e:
            print(e, value)

    combine_region_contours = comm.gather(region_contours_list, root=0)
    combine_region_labels = comm.gather(region_labels_list, root=0)
    comm.Barrier()

    if comm_rank == 0:

        if combine_region_contours:
            combine_region_contours = list(chain(*combine_region_contours))
            combine_region_labels = list(chain(*combine_region_labels))
        else:
            combine_region_contours, combine_region_labels = [], []


        scale = 32
        # thumbnail = read_clear_img(slide, scale=scale, read_scale=1, split_num=8)
        thumbnail = slide.read(scale=scale)

        region_coords_all_resize = [(np.array(i, dtype=np.int32) / float(scale)).astype(np.int32) for i in
                                    combine_region_contours]
        region_labels_all_resize = np.array(combine_region_labels, dtype=np.int32)

        for img_index in range(1, 4):
            if img_index == 2:
                thumbnail = np.zeros_like(thumbnail, dtype=np.int32)
            elif img_index == 3:
                thumbnail = np.zeros_like(thumbnail, dtype=np.int32)
                thumbnail[:, :, :] = 255
            else:
                pass
            # 画区域mask
            result_contours_list = [np.array(i, dtype=np.int32)[:, None, :] for i in region_coords_all_resize]
            region_labels_all_nparr = np.array(region_labels_all_resize, dtype=np.int32)
            mask = np.zeros_like(thumbnail, dtype=np.int32)
            for class_region in range(1, class_num):
                class_bool_index = region_labels_all_nparr == class_region
                class_result_contours_list = [result_contours_list[index] for index, i in enumerate(class_bool_index) if
                                              i == True]
                color = Hex_to_RGB(display_color_dict[get_keys(roi_label_dict, class_region)])
                mask = cv2.drawContours(mask, class_result_contours_list, -1, color, -1)

            # 画区域
            alpha = 0.5  # 前景色块透明度
            cur_region_region_color = copy.deepcopy(thumbnail)

            # 将mask画在原图上
            region_bool_index = np.sum(mask, axis=2) > 0
            cur_region_region_color[region_bool_index] = cur_region_region_color[region_bool_index] * (1 - alpha)
            mask = mask * alpha
            cur_region_region_color = cur_region_region_color + mask

            for class_region in range(1, class_num):
                class_bool_index = region_labels_all_nparr == class_region
                class_result_contours_list = [result_contours_list[index] for index, i in enumerate(class_bool_index) if
                                              i == True]
                color = Hex_to_RGB(display_color_dict[get_keys(roi_label_dict, class_region)])
                thickness = 3
                cur_region_region_color = cv2.drawContours(cur_region_region_color, class_result_contours_list, -1,
                                                           color,
                                                           thickness)
            # cv2.imwrite("./cur_region_region_color.jpg", cur_region_region_color[:, :, ::-1])
            save_path = os.path.join(".", save_folder, slide_name + "_{}_2.jpg".format(img_index))
            cv2.imwrite(save_path, cur_region_region_color[:, :, ::-1])


def Hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    rgb = (r, g, b)
    return rgb


def get_keys(d, value):
    return [k for k, v in d.items() if v == value][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--slide_path', type=str, default="/data1/Caijt/PDL1_Parallel/A080 PD-L1 V+.kfb",
                        help='Slide Path')
    parser.add_argument('--roi_coords', type=str, nargs=2, default=None)

    # * Model
    parser.add_argument('--num_classes', type=int, default=4,
                        help="Number of cell categories")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--row', default=2, type=int, help="number of anchor points per row")
    parser.add_argument('--col', default=2, type=int, help="number of anchor points per column")

    args = parser.parse_args()
    slide_path = args.slide_path
    # mean, std = np.load('mean_std.npy')
    # slide_path = r'D:\迈杰PDL1已分析\30301-F3749-IHC.kfb'

    roi_coords = args.roi_coords
    if roi_coords is not None:
        x_coords, y_coords = json.loads(roi_coords[0]), json.loads(roi_coords[1])
    else:
        x_coords, y_coords = [], []

    # image_folder = r"/home/zhouzhihao/pathological_data/zhengdasan_qidai/taimo/20240524/image"
    image_folder = r"/D:\msqtry\region_level_train\EVAL_WSI_1GPU\test_wsi_patch\TestJpg\VisWSI\level4"
    slide_path_list = [os.path.join(image_folder, i) for i in os.listdir(image_folder)]
    print(slide_path_list)
    # save_folder = r"patch_result"
    save_folder = r"D:\msqtry\region_level_train\EVAL_WSI_1GPU\test_wsi_patch\TestJpg\VisWSI\level4_pre"
    display_color_dict = {
        "eTLS":  "#ff0000",
        'pTLS': "#0000ff",
        'sTLS': "#ff8000"
    }
    roi_label_dict = {
                  "eTLS": 1,
                  "pTLS": 2,
                  "sTLS": 3
                  }
    color_card = {name: Hex_to_RGB(display_color_dict[name]) for name in display_color_dict.keys()}
    color_dict = {str(i): color_card[name] for i, name in enumerate(roi_label_dict.keys())}
    class_num = 3+1
    for slide_path in slide_path_list:
        slide_name = os.path.basename(slide_path)
        # save_folder = os.path.join(".", save_folder, slide_name)
        os.makedirs(save_folder, exist_ok=True)
        ori_save_folder = save_folder + "_img"
        # os.makedirs(ori_save_folder, exist_ok=True)
        # try:
        compute_process(slide_path, x_coords=x_coords, y_coords=y_coords)
        # except Exception as e:
        #     print(e)
        #     continue
