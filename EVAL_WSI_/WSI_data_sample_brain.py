import numpy as np
import cv2
import json
import os
import random
from tqdm import tqdm
from imageio import imsave
from PIL import Image  # 使用PIL读取TIF图像
import tifffile

# 类别字典，确保类别从1开始，并且为连续的整数
label_color_dict = {
    (0, 170, 255): 1,  #天蓝色对应类别1
    (1, 248, 231): 2,  # 青色对应类别2
    (250, 204, 0): 3,  # 金黄色对应类别3
    (61, 192, 5): 4,
    (132, 255, 0):5,   #草绿色
    (146, 156,30):6,  #绿豆色
    (255, 0, 0):7,     #红色
    (225, 255, 0):8, #黄色
}


def get_contours(slide_path):
    """
    获取json文件中的标注轮廓，并根据颜色确定类别
    """
    index_file = slide_path.replace('.tif', '.json')
    roi_contours = []
    roi_labels = []

    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            info_dict = json.load(f)
    except Exception as e:
        print(f"Error reading json file: {e}")
        return [], []

    roilist = info_dict.get('annotation', [])
    print(f'{len(roilist)} roi regions are labeled in {slide_path}')

    for roi_dict in roilist:
        path = roi_dict.get("position", {})
        stroke_color = roi_dict.get("strokeColor", "#ffffff")  # 默认颜色为白色

        # 将颜色代码转为RGB值
        color = hex_to_rgb(stroke_color)

        label_id = label_color_dict.get(color, None)
        if label_id is None:
            continue  # 如果颜色不在字典中，跳过此区域

        x_coords_list = path.get("x", [])
        y_coords_list = path.get("y", [])
        corrds = list(zip(x_coords_list, y_coords_list))

        corrds_np = np.array(corrds, dtype=np.int32)
        roi_contours.append(corrds_np)
        roi_labels.append(label_id)

    return roi_contours, roi_labels


def hex_to_rgb(hex_color):
    """
    将十六进制颜色值转换为RGB元组
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def convert_to_label_map(slide_path, roi_contours, roi_labels, save_dir):
    """
    将轮廓标注转化为标签图，并保存
    """
    image = tifffile.imread(slide_path)  # 使用tifffile读取TIF图像
    dim = image.shape[:2]  # 获取图像的高宽

    mask_for_train = np.zeros((dim[0], dim[1], len(label_color_dict) + 1), dtype=np.uint8)

    for i, roi_contour in enumerate(roi_contours):
        label_id = roi_labels[i]
        coords = np.array(roi_contour)

        # 创建空白掩码
        cur_mask = np.zeros_like(image)
        cur_mask = cv2.drawContours(cur_mask, [coords], -1, (255, 0, 0), cv2.FILLED)  # 画轮廓

        # 将轮廓填充到mask中
        mask_for_train[:, :, label_id] += np.sum(cur_mask, axis=-1)

    mask_for_train = np.clip(mask_for_train, 0, 255)
    save_label_path = os.path.join(save_dir, f"{os.path.basename(slide_path)}_label_map.png")
    imsave(save_label_path, mask_for_train)


def split_patches_and_save(image, mask, save_dir, patch_size=384, overlap=192):
    """
    将大图分割成小patch，并保存
    """
    h, w = image.shape[:2]
    f_index = 0

    for x in range(0, h - patch_size + 1, patch_size - overlap):
        for y in range(0, w - patch_size + 1, patch_size - overlap):
            patch = image[x:x + patch_size, y:y + patch_size, :]
            patch_mask = mask[x:x + patch_size, y:y + patch_size, :]

            if np.sum(patch) < 20:  # 若没有有效内容，则跳过
                continue

            patch_label_class = np.argmax(patch_mask, axis=-1)

            # 保存图像与标签
            patch_image_path = os.path.join(save_dir, 'JPEGImages', f"{f_index}.jpg")
            patch_mask_path = os.path.join(save_dir, 'SegmentationClass', f"{f_index}.png")

            cv2.imwrite(patch_image_path, patch)
            cv2.imwrite(patch_mask_path, patch_label_class)

            f_index += 1


def process_slide_and_annotations(slide_path, save_dir):
    """
    处理整个slide及其标注，保存切片和标注
    """
    roi_contours, roi_labels = get_contours(slide_path)

    if len(roi_contours) == 0:
        print(f"无标注：{slide_path}")
        return

    convert_to_label_map(slide_path, roi_contours, roi_labels, save_dir)

    image = tifffile.imread(slide_path)  # 使用tifffile读取TIF图像
    mask_for_train = np.zeros((image.shape[0], image.shape[1], len(label_color_dict) + 1), dtype=np.uint8)

    # 生成标签图并切片
    split_patches_and_save(image, mask_for_train, save_dir)


def main():
    save_dir = '/path/to/save'
    slide_folder = '/path/to/slides'

    if not os.path.exists(os.path.join(save_dir, 'ImageSets')):
        os.makedirs(os.path.join(save_dir, 'JPEGImages'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'SegmentationClass'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'ImageSets', 'Segmentation'), exist_ok=True)

    # 获取所有slide文件路径
    slide_files = [os.path.join(slide_folder, f) for f in os.listdir(slide_folder) if f.endswith('.tif')]

    for slide_path in tqdm(slide_files):
        process_slide_and_annotations(slide_path, save_dir)


if __name__ == '__main__':
    main()