import numpy as np
from imageio import imsave
import openslide
import os
import json
import cv2
from Slide_0.openslide_func import openSlide as di_openSlide
from Slide.dispatch import openSlide
import random
from tqdm import tqdm
import deepdish as dd
import re


def natural_sort_key(s):
    """提取文件名中的数字用于排序"""
    numbers = re.findall(r'\d+', os.path.basename(s))
    return [int(num) if num else 0 for num in numbers] or [s]


def get_level_dim_dict(slide_path):
    """获取WSI图像各层级的尺寸信息"""
    level_dim_dict = {}
    slide = di_openSlide(slide_path).slide
    dims = slide.level_dimensions
    downsamples = slide.level_downsamples
    for i in range(len(downsamples)):
        level_dim_dict[i] = (dims[i], downsamples[i])
    return level_dim_dict


def get_contours(slide_path, level, ext='.kfb'):
    """获取标注轮廓信息"""
    dir_name = os.path.dirname(slide_path)
    json_file = slide_path.replace(ext, '.json')
    roi_contours = []
    roi_labels = []

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            info_dict = json.load(f)
    except:
        for encoding in ['gbk', 'GBK', 'utf-8']:
            try:
                with open(json_file, 'r', encoding=encoding) as f:
                    info_dict = json.load(f)
                break
            except:
                continue
        else:
            raise Exception(f"无法读取JSON文件: {json_file}")

    roilist = info_dict.get('annotation', [])
    print(f'{len(roilist)} roi regions are labeled')

    for roi_dict in roilist:
        path = roi_dict.get("position")
        try:
            remark_name = roi_dict["label"]
            remark = label_dict[remark_name]
        except:
            continue

        x_coords_list = path["x"]
        y_coords_list = path["y"]
        corrds = list(zip(x_coords_list, y_coords_list))
        corrds_np = np.array(corrds, dtype=np.int32)
        roi_contours.append(corrds_np)
        roi_labels.append(remark)

    return roi_contours, roi_labels


def read_thumbnail_in_chunks(slide, level, dim, chunk_size=4096):
    """分块读取WSI图像"""
    w, h = dim
    thumbnail = np.zeros((h, w, 4), dtype=np.uint8)

    for y in range(0, h, chunk_size):
        for x in range(0, w, chunk_size):
            read_width = min(chunk_size, w - x)
            read_height = min(chunk_size, h - y)
            tile = slide.read_region((x * (2 ** level), y * (2 ** level)), level, (read_width, read_height))
            thumbnail[y:y + read_height, x:x + read_width] = np.array(tile)

    return thumbnail[:, :, :3]


def create_patch_visualization(patch, patch_label, label_num, label_color):
    """创建patch的可视化图像"""
    temp_mask = 0.4 * (1 - patch_label[:, :, 0] / 255)
    temp_mask[temp_mask == 0] = 1
    temp_mask = np.repeat(temp_mask[:, :, np.newaxis], 3, axis=-1)

    vis_image = temp_mask * patch

    for i in range(1, label_num):
        cur_label = patch_label[:, :, i]
        cur_label = cur_label[:, :, np.newaxis] // 255
        cur_label = np.repeat(cur_label, 3, -1)
        cur_color = list(label_color.values())[i - 1]
        label_region_color = cur_label * cur_color
        vis_image = vis_image + 0.6 * label_region_color

    return vis_image.astype(np.uint8)


def split_patches(image, mask, save_dir, save_ind, label_num, label_color, randomrate=0.01):
    """分割图像为小块并保存可视化结果"""
    h, w = image.shape[0], image.shape[1]
    patch_size = 384
    h_overlap = w_overlap = 192
    f_index = 0

    # 处理mask
    mask[:, :, 0] = 255 - np.sum(mask[:, :, 1:], axis=2)
    temp_mask = 0.4 * (1 - mask[:, :, 0] / 255)
    temp_mask[temp_mask == 0] = 1
    temp_mask = np.repeat(temp_mask[:, :, np.newaxis], 3, axis=-1)
    image_and_label = temp_mask * image

    # 保存WSI级别的可视化结果
    cv2.imwrite(os.path.join(save_dir, 'VisWSI', f"{save_ind}_original.jpg"), image)
    cv2.imwrite(os.path.join(save_dir, 'VisWSI', f"{save_ind}_label.jpg"), image_and_label)
    cv2.imwrite(os.path.join(save_dir, 'VisWSI', f"{save_ind}_mask.jpg"), 255 - mask[:, :, 0])

    # 打开文件句柄
    with open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'train.txt'), 'a') as f_train, \
            open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'val.txt'), 'a') as f_val, \
            open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'test.txt'), 'a') as f_test:

        for x in range(0, h - patch_size + 1, patch_size - h_overlap):
            for y in range(0, w - patch_size + 1, patch_size - w_overlap):
                patch = image[x:x + patch_size, y:y + patch_size, :]
                if np.sum(patch) < 20:
                    continue

                patch_label = mask[x:x + patch_size, y:y + patch_size, :]
                if np.sum(patch_label[:, :, 1:]) < 2000 and random.random() >= randomrate:
                    continue

                # 创建并保存patch可视化结果
                patch_vis = create_patch_visualization(patch, patch_label, label_num, label_color)
                cv2.imwrite(os.path.join(save_dir, 'VisPatch', f"{save_ind}_{f_index}.jpg"), patch_vis)

                # 保存patch图像和标签
                cv2.imwrite(os.path.join(save_dir, 'JPEGImages', f"{save_ind}_{f_index}.jpg"), patch)
                patch_label_class = np.argmax(patch_label, axis=-1)
                cv2.imwrite(os.path.join(save_dir, 'SegmentationClass', f"{save_ind}_{f_index}.png"),
                            patch_label_class.astype(np.uint8))

                # 写入训练/验证/测试集
                if f_index % 4 == 0:
                    f_val.write(f"{save_ind}_{f_index}\n")
                    f_test.write(f"{save_ind}_{f_index}\n")
                else:
                    f_train.write(f"{save_ind}_{f_index}\n")
                f_index += 1


def vis_anno(slide_path, roi_contours, roi_labels, level, save_dir, index, label_num, label_color):
    """处理单个WSI图像的标注可视化"""
    level_dim_dict_local = get_level_dim_dict(slide_path)
    dim = level_dim_dict_local[level][0]

    if 'ndpi' in slide_path or 'mrxs' in slide_path or 'sdpc' in slide_path:
        with openslide.OpenSlide(slide_path) as slide:
            thumbnail = read_thumbnail_in_chunks(slide, level, dim)

    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
    h, w = thumbnail.shape[0], thumbnail.shape[1]

    # 使用uint8初始化mask
    mask_fortrain = np.zeros((h, w, label_num), dtype=np.uint8)

    # 分块处理mask
    chunk_size = 4096
    for y_start in range(0, h, chunk_size):
        y_end = min(y_start + chunk_size, h)
        for x_start in range(0, w, chunk_size):
            x_end = min(x_start + chunk_size, w)

            chunk_mask = np.zeros((y_end - y_start, x_end - x_start, label_num), dtype=np.uint8)

            for i, roi_contour in enumerate(roi_contours):
                label = roi_labels[i]
                x_coord = (roi_contour[:, 0] / (2 ** level)).astype(np.int32)
                y_coord = (roi_contour[:, 1] / (2 ** level)).astype(np.int32)
                coords_np = np.stack((x_coord, y_coord), axis=1)

                # 处理当前块的轮廓
                chunk_coords = coords_np - [x_start, y_start]
                if np.any((chunk_coords[:, 0] >= 0) & (chunk_coords[:, 0] < (x_end - x_start)) &
                          (chunk_coords[:, 1] >= 0) & (chunk_coords[:, 1] < (y_end - y_start))):
                    cur_mask = np.zeros((y_end - y_start, x_end - x_start, 3), dtype=np.uint8)
                    cv2.drawContours(cur_mask, [chunk_coords], -1, (255, 0, 0), cv2.FILLED)
                    chunk_mask[:, :, label] += np.sum(cur_mask, axis=-1)

            mask_fortrain[y_start:y_end, x_start:x_end] = chunk_mask

    # 将mask值限制在0-255范围内
    mask_fortrain = np.clip(mask_fortrain, 0, 255)

    # 处理patches
    split_patches(thumbnail, mask_fortrain, save_dir=save_dir, save_ind=index,
                  label_num=label_num, label_color=label_color)


if __name__ == '__main__':
    # 配置参数
    save_dir = r'D:\msqtry\region_level_train\EVAL_WSI_1GPU\383_192_level2'
    img_folder = r'F:\mashiqi_中大_2411_前列腺癌三级淋巴结构识别\20241103'

    label_dict = {
        "eTLS": 1,
        "pTLS": 2,
        "sTLS": 3
    }

    label_color = {
        'eTLS': (0, 255, 0),
        'pTLS': (255, 0, 0),
        'sTLS': (128, 128, 0)
    }

    label_num = len(label_dict) + 1  # 加1是为了包含背景类
    randomrate = 0.01

    # 创建必要的目录
    for subdir in ['JPEGImages', 'SegmentationClass',
                   os.path.join('ImageSets', 'Segmentation'),
                   'VisWSI', 'VisPatch']:
        os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)

    level = 2
    index = 0

    # 处理所有mrxs文件
    ext_list = ['.mrxs']
    for ext in ext_list:
        from pydaily import filesystem

        imglist = filesystem.find_ext_files(img_folder, ext)
        imglist = list(set([os.path.abspath(p) for p in imglist]))
        # 按文件名排序
        imglist.sort(key=natural_sort_key)

        print(f"Total files to process: {len(imglist)}")
        for slide_path in tqdm(imglist, desc="Processing WSI files"):
            # 检查对应的json文件是否存在
            json_path = slide_path.replace(ext, '.json')
            if not os.path.exists(json_path):
                print(f"跳过 {os.path.basename(slide_path)}: 未找到对应的json文件")
                continue

            try:
                roi_contours, roi_labels = get_contours(json_path, level)
                if not roi_contours:
                    print(f"跳过 {os.path.basename(slide_path)}: 未找到标注信息")
                    continue

                print(f"处理文件 {index + 1}/{len(imglist)}: {os.path.basename(slide_path)}")
                vis_anno(slide_path, roi_contours, roi_labels, level=level,
                         save_dir=save_dir, index=index,
                         label_num=label_num, label_color=label_color)
                index += 1

            except Exception as e:
                print(f"处理 {os.path.basename(slide_path)} 时发生错误: {str(e)}")
                continue

    print("处理完成!")