import numpy as np
from imageio import imsave
import os, json
import cv2
import random
from tqdm import tqdm
import math
import shutil
from PIL import Image


def validate_json_file(json_path):
    """
    验证JSON文件是否有效
    Args:
        json_path: JSON文件路径
    Returns:
        bool: 文件是否有效
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查必要的字段
        if 'annotation' not in data:
            return False

        # 检查是否有有效的标注
        annotations = data['annotation']
        if not isinstance(annotations, list) or len(annotations) == 0:
            return False

        # 检查标注格式
        for annotation in annotations:
            if 'position' not in annotation or 'label' not in annotation:
                return False

            position = annotation['position']
            if 'x' not in position or 'y' not in position:
                return False

            # 检查坐标是否为有效列表
            if not isinstance(position['x'], list) or not isinstance(position['y'], list):
                return False

            if len(position['x']) != len(position['y']) or len(position['x']) < 3:
                return False

        return True
    except Exception as e:
        print(f"JSON验证错误: {e}")
        return False


def get_contours(json_path):
    """
    从JSON文件中读取标注信息
    Args:
        json_path: JSON文件的完整路径
    """
    roi_contours = []
    roi_labels = []

    # 尝试使用不同编码打开文件
    encodings = ['utf-8', 'gbk', 'GBK']
    info_dict = None

    for encoding in encodings:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                info_dict = json.load(f)
                break
        except:
            continue

    if info_dict is None:
        raise Exception(f"无法解析JSON文件: {json_path}")

    roilist = info_dict['annotation']
    print(f'{len(roilist)} 个标注区域在 {os.path.basename(json_path)}')

    # 颜色到类别的映射
    color_to_label = {
        '#FF0000': 1,  # 严重损伤线粒体
        '#3DC005': 2,  # 中度损伤线粒体
        '#01F8E7': 3,  # 健康线粒体
        '#00FFB7': 4,  # 自噬线粒体
        '#E1FF00': 5,  # 肌浆网
        '#9D00FF': 6,  # 闰盘
        '#EE00FF': 7,  # Z线样物质堆积
        '#FFCC00': 8,  # 脂滴
        '#FF7700': 9,  # 糖原颗粒
        '#00AAFF': 10,  # Z线
        '#14699D': 11,  # T管
        '#84FF00': 12,  # M线
        '#929C1E': 13  # 心肌侧管
    }

    for i, roi_dict in enumerate(roilist):
        path = roi_dict["position"]
        try:
            color = roi_dict["label"]  # 现在label是颜色值
            if color not in color_to_label:
                print(f"警告: 未知颜色 {color} 在 {os.path.basename(json_path)}")
                continue
            remark = color_to_label[color]
        except:
            continue

        x_coords_list = path["x"]
        y_coords_list = path["y"]
        corrds = list(zip(x_coords_list, y_coords_list))

        corrds_np = np.array(corrds, dtype=np.int32)
        roi_contours.append(corrds_np)
        roi_labels.append(remark)

    return roi_contours, roi_labels


def visualize_patch(image, mask, save_path):
    """
    创建并保存图像和掩码的可视化
    Args:
        image: 原始图像
        mask: 分割掩码
        save_path: 保存路径
    """
    # 确保图像是BGR格式（OpenCV默认）
    if image.shape[2] == 3:
        vis_img = image.copy()
    else:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 创建彩色掩码
    color_mask = np.zeros_like(vis_img)

    # 标签颜色映射
    colors = [
        (0, 0, 0),  # 背景 (黑色)
        (0, 0, 255),  # 严重损伤线粒体 (红色)
        (5, 192, 61),  # 中度损伤线粒体 (绿色)
        (231, 248, 1),  # 健康线粒体 (青色)
        (183, 255, 0),  # 自噬线粒体 (浅绿)
        (0, 255, 225),  # 肌浆网 (黄色)
        (255, 0, 157),  # 闰盘 (紫色)
        (255, 0, 238),  # Z线样物质堆积 (粉色)
        (0, 204, 255),  # 脂滴 (橙色)
        (0, 119, 255),  # 糖原颗粒 (橙红)
        (255, 170, 0),  # Z线 (蓝色)
        (157, 105, 20),  # T管 (深蓝)
        (0, 255, 132),  # M线 (绿色)
        (30, 156, 146)  # 心肌侧管 (青绿)
    ]

    # 为每个类别创建彩色掩码
    for i in range(1, len(colors)):  # 跳过背景
        mask_i = (mask == i).astype(np.uint8) * 255
        color_mask_i = np.zeros_like(vis_img)
        color_mask_i[mask_i > 0] = colors[i]

        # 添加半透明覆盖
        alpha = 0.5
        vis_img = cv2.addWeighted(vis_img, 1, color_mask_i, alpha, 0)
        color_mask = cv2.add(color_mask, color_mask_i)

    # 创建拼接图像
    h, w = image.shape[:2]
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    combined[:, :w] = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    combined[:, w:2 * w] = color_mask
    combined[:, 2 * w:] = vis_img

    # 保存可视化结果
    cv2.imwrite(save_path, combined)
    return combined


def visualize_wsi(slide_path, json_path, roi_contours, roi_labels, level, save_path):
    """
    创建并保存WSI的缩略图可视化
    Args:
        slide_path: WSI文件路径
        json_path: JSON标注文件路径
        roi_contours: 轮廓数据
        roi_labels: 标签数据
        level: 层级
        save_path: 保存路径
    """
    try:
        # 对于TIF文件，生成一个缩略图
        # 先复制到临时目录
        temp_dir = os.path.dirname(save_path)
        os.makedirs(temp_dir, exist_ok=True)
        local_path = os.path.join(temp_dir, 'temp_' + os.path.basename(slide_path))

        try:
            shutil.copy2(slide_path, local_path)
            try:
                # 尝试使用PIL读取
                with Image.open(local_path) as img:
                    image = np.array(img)
                    if len(image.shape) == 2:  # 如果是灰度图
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    elif image.shape[2] == 4:  # 如果有Alpha通道
                        image = image[:, :, :3]
                    elif image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except:
                # 如果PIL失败，尝试OpenCV
                image = cv2.imread(local_path)

            if image is None:
                print(f"无法读取图像进行可视化: {slide_path}")
                return False
        except Exception as e:
            print(f"复制或读取文件失败: {e}")
            return False
        finally:
            # 清理临时文件
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except:
                    pass

        # 如果图像太大，进行缩放
        max_size = 2000
        h, w = image.shape[:2]
        scale = min(max_size / w, max_size / h)

        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))

            # 缩放轮廓坐标
            scaled_contours = []
            for contour in roi_contours:
                scaled_contour = contour * scale
                scaled_contours.append(scaled_contour.astype(np.int32))

            roi_contours = scaled_contours

        # 绘制轮廓
        vis_img = image.copy()

        # 标签颜色映射
        colors = [
            (0, 0, 0),  # 背景 (黑色)
            (0, 0, 255),  # 严重损伤线粒体 (红色)
            (5, 192, 61),  # 中度损伤线粒体 (绿色)
            (231, 248, 1),  # 健康线粒体 (青色)
            (183, 255, 0),  # 自噬线粒体 (浅绿)
            (0, 255, 225),  # 肌浆网 (黄色)
            (255, 0, 157),  # 闰盘 (紫色)
            (255, 0, 238),  # Z线样物质堆积 (粉色)
            (0, 204, 255),  # 脂滴 (橙色)
            (0, 119, 255),  # 糖原颗粒 (橙红)
            (255, 170, 0),  # Z线 (蓝色)
            (157, 105, 20),  # T管 (深蓝)
            (0, 255, 132),  # M线 (绿色)
            (30, 156, 146)  # 心肌侧管 (青绿)
        ]

        # 绘制每个轮廓
        for i, contour in enumerate(roi_contours):
            color = colors[roi_labels[i]]
            cv2.drawContours(vis_img, [contour], -1, color, 2)

        # 添加图例
        legend_height = 25 * 13  # 每个类别一行
        legend = np.ones((legend_height, 300, 3), dtype=np.uint8) * 255

        label_names = [
            "严重损伤线粒体", "中度损伤线粒体", "健康线粒体", "自噬线粒体",
            "肌浆网", "闰盘", "Z线样物质堆积", "脂滴", "糖原颗粒",
            "Z线", "T管", "M线", "心肌侧管"
        ]

        for i, name in enumerate(label_names):
            y = 25 * (i + 1) - 5
            color = colors[i + 1]
            cv2.rectangle(legend, (10, y - 15), (40, y + 5), color, -1)
            cv2.putText(legend, name, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 拼接图像和图例
        h, w = vis_img.shape[:2]
        lh, lw = legend.shape[:2]

        if h >= lh:
            padded_legend = np.ones((h, lw, 3), dtype=np.uint8) * 255
            padded_legend[:lh, :lw] = legend
            combined = np.hstack((vis_img, padded_legend))
        else:
            padded_vis = np.ones((lh, w, 3), dtype=np.uint8) * 255
            padded_vis[:h, :w] = vis_img
            combined = np.hstack((padded_vis, legend))

        # 保存可视化结果
        cv2.imwrite(save_path, combined)
        print(f"WSI可视化已保存到: {save_path}")
        return True

    except Exception as e:
        print(f"创建WSI可视化时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_tif_file(tif_path, json_path, save_dir, index, patch_size=384, overlap=150, randomrate=0.01, label_num=14):
    """处理单个TIF文件及其JSON标注"""
    try:
        # 首先复制TIF文件到本地临时目录
        temp_dir = os.path.join(save_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        local_tif_path = os.path.join(temp_dir, os.path.basename(tif_path))
        print(f"复制 {tif_path} 到 {local_tif_path}")

        try:
            shutil.copy2(tif_path, local_tif_path)
        except Exception as e:
            print(f"复制文件失败: {e}")
            return False

        # 尝试使用PIL读取图像
        try:
            with Image.open(local_tif_path) as img:
                width, height = img.size
                print(f"成功读取图像: {width} x {height}")
                image = np.array(img)
                if len(image.shape) == 2:  # 如果是灰度图
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4:  # 如果有Alpha通道
                    image = image[:, :, :3]
                elif image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"PIL无法读取图像: {e}")
            try:
                # 如果PIL失败，尝试OpenCV
                image = cv2.imread(local_tif_path)
                if image is None:
                    print(f"OpenCV也无法读取图像")
                    return False
                height, width = image.shape[:2]
                print(f"使用OpenCV读取: {width} x {height}")
            except Exception as e2:
                print(f"OpenCV读取错误: {e2}")
                return False

        # 读取标注
        roi_contours, roi_labels = get_contours(json_path)
        if len(roi_contours) == 0:
            print(f"没有找到标注")
            return False

        print(f"处理图像: {os.path.basename(tif_path)} ({len(roi_contours)} 个标注区域)")

        # 创建WSI可视化
        wsi_vis_path = os.path.join(save_dir, 'VisWSI', f"{index}_{os.path.basename(tif_path)[:-4]}.jpg")
        visualize_wsi(tif_path, json_path, roi_contours, roi_labels, 0, wsi_vis_path)

        # 处理图像块
        # 为了防止内存溢出，我们将图像分割成块处理
        block_size = 5000
        n_blocks_h = math.ceil(height / block_size)
        n_blocks_w = math.ceil(width / block_size)

        f_train = open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'train.txt'), 'a')
        f_val = open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'val.txt'), 'a')
        f_test = open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'test.txt'), 'a')

        patches_created = 0

        for block_y in range(n_blocks_h):
            for block_x in range(n_blocks_w):
                # 计算块坐标
                start_x = block_x * block_size
                start_y = block_y * block_size
                end_x = min(start_x + block_size, width)
                end_y = min(start_y + block_size, height)
                block_width = end_x - start_x
                block_height = end_y - start_y

                # 提取图像块
                block = image[start_y:end_y, start_x:end_x]

                # 创建掩码
                mask_block = np.zeros((block_height, block_width, label_num))

                # 处理ROI
                for i, roi_contour in enumerate(roi_contours):
                    # 坐标不需要缩放
                    x_coord = np.array([int(c[0]) for c in roi_contour])
                    y_coord = np.array([int(c[1]) for c in roi_contour])

                    # 调整坐标相对于块
                    x_coord_rel = x_coord - start_x
                    y_coord_rel = y_coord - start_y

                    coords_np = np.concatenate((x_coord_rel[:, np.newaxis], y_coord_rel[:, np.newaxis]), axis=1)

                    # 只处理与块相交的ROI
                    if np.any((x_coord >= start_x) & (x_coord < end_x)) and np.any(
                            (y_coord >= start_y) & (y_coord < end_y)):
                        cur_mask = np.zeros((block_height, block_width, 3), dtype=np.uint8)
                        coords_np = [coords_np, ]
                        cv2.drawContours(cur_mask, coords_np, -1, (255, 255, 255), cv2.FILLED)
                        mask_block[:, :, roi_labels[i]] += cur_mask[:, :, 0]

                mask_block = np.clip(mask_block, 0, 255)

                # 设置背景通道
                mask_block[:, :, 0] = 255 - np.sum(mask_block[:, :, 1:], axis=2)
                mask_block = np.clip(mask_block[:, :, 0:], 0, 255)

                # 处理patch
                f_index = 0
                for x in range(0, block_height - patch_size + 1, patch_size - overlap):
                    if x + patch_size > block_height:
                        continue

                    for y in range(0, block_width - patch_size + 1, patch_size - overlap):
                        if y + patch_size > block_width:
                            continue

                        patch = block[x:x + patch_size, y:y + patch_size]
                        if np.sum(patch) < 20:  # 跳过几乎全黑的patch
                            continue

                        patch_label = mask_block[x:x + patch_size, y:y + patch_size]
                        if np.sum(patch_label[:, :, 1:]) < 2000 and random.random() >= randomrate:
                            continue  # 对于标注较少的区域，随机抽样

                        patch_label_class = np.argmax(patch_label, axis=-1).astype(np.uint8)

                        # 保存patch和标签
                        global_index = f"{index}_{block_x}_{block_y}_{f_index}"

                        # 保存到数据集
                        patch_path = os.path.join(save_dir, 'JPEGImages', f"{global_index}.jpg")
                        label_path = os.path.join(save_dir, 'SegmentationClass', f"{global_index}.png")

                        cv2.imwrite(patch_path, patch)
                        cv2.imwrite(label_path, patch_label_class)

                        # 添加patch可视化
                        patch_vis_path = os.path.join(save_dir, 'VisPatch', f"{global_index}.jpg")
                        visualize_patch(patch, patch_label_class, patch_vis_path)

                        if f_index % 4 == 0:
                            f_val.write(f"{global_index}\n")
                            f_test.write(f"{global_index}\n")
                        else:
                            f_train.write(f"{global_index}\n")

                        f_index += 1
                        patches_created += 1

        f_train.close()
        f_val.close()
        f_test.close()

        # 清理临时文件
        try:
            os.remove(local_tif_path)
        except:
            pass

        print(f"成功从 {os.path.basename(tif_path)} 创建了 {patches_created} 个patch")
        return patches_created > 0

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"处理文件时出错: {e}")
        return False


def find_image_files(json_path, supported_extensions):
    """查找与JSON文件对应的图像文件"""
    json_name = os.path.splitext(os.path.basename(json_path))[0]
    json_dir = os.path.dirname(json_path)

    for ext in supported_extensions:
        img_path = os.path.join(json_dir, json_name + ext)
        if os.path.exists(img_path):
            return img_path

    return None


def getFileList(dir_path, ext=None):
    """递归获取目录中的所有文件"""
    file_list = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if ext is None or file.endswith(ext):
                file_list.append(os.path.join(root, file))

    return file_list


if __name__ == '__main__':
    save_dir = r'D:\msqtry\muscle indicator\patch_pengzhang_75lap'
    img_folder = r'\\10.15.20.69\homes\sunchenhao_69\北京阜外医院_250225_心肌病科研\250513-北京阜外医院心肌病科研\image'

    # 标签字典
    label_dict = {
        "严重损伤线粒体": 1,
        "中度损伤线粒体": 2,
        "健康线粒体": 3,
        "自噬线粒体": 4,
        "肌浆网": 5,
        "闰盘": 6,
        "Z线样物质堆积": 7,
        "脂滴": 8,
        "糖原颗粒": 9,
        "Z线": 10,
        "T管": 11,
        "M线": 12,
        "心肌侧管": 13
    }

    # 更新颜色映射
    label_color = {
        '严重损伤线粒体': (0, 0, 255),  # #FF0000
        '中度损伤线粒体': (5, 192, 61),  # #3DC005
        '健康线粒体': (231, 248, 1),  # #01F8E7
        '自噬线粒体': (183, 255, 0),  # #00FFB7
        '肌浆网': (0, 255, 225),  # #E1FF00
        '闰盘': (255, 0, 157),  # #9D00FF
        'Z线样物质堆积': (255, 0, 238),  # #EE00FF
        '脂滴': (0, 204, 255),  # #FFCC00
        '糖原颗粒': (0, 119, 255),  # #FF7700
        'Z线': (255, 170, 0),  # #00AAFF
        'T管': (157, 105, 20),  # #14699D
        'M线': (0, 255, 132),  # #84FF00
        '心肌侧管': (30, 156, 146)  # #929C1E
    }

    # 类别数量
    label_num = 13 + 1  # 13个目标类别 + 1个背景
    randomrate = 0.01

    # 创建必要的目录
    os.makedirs(os.path.join(save_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'SegmentationClass'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ImageSets', 'Segmentation'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'VisWSI'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'VisPatch'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'temp'), exist_ok=True)

    # 清空训练集划分文件
    train_file = os.path.join(save_dir, 'ImageSets', 'Segmentation', 'train.txt')
    val_file = os.path.join(save_dir, 'ImageSets', 'Segmentation', 'val.txt')
    test_file = os.path.join(save_dir, 'ImageSets', 'Segmentation', 'test.txt')

    for file_path in [train_file, val_file, test_file]:
        with open(file_path, 'w') as f:
            pass  # 清空文件

    # 找到所有JSON文件
    json_list = getFileList(img_folder, '.json')
    print(f"找到 {len(json_list)} 个JSON文件")

    # 支持的图像格式
    supported_extensions = ['.tif', '.tiff']

    # 统计处理情况
    processed_count = 0
    skipped_no_image = 0
    skipped_no_annotation = 0
    skipped_invalid_json = 0
    skipped_processing_error = 0

    # 处理每个JSON文件
    for index, json_path in enumerate(tqdm(json_list, desc="处理JSON文件")):
        # 验证JSON文件
        if not validate_json_file(json_path):
            print(f"警告: JSON文件 {os.path.basename(json_path)} 格式无效或损坏")
            skipped_invalid_json += 1
            continue

        # 查找对应的图像文件
        img_path = find_image_files(json_path, supported_extensions)
        if img_path is None:
            print(f"警告: JSON文件 {os.path.basename(json_path)} 没有找到对应的图像文件")
            skipped_no_image += 1
            continue

        # 处理图像和标注
        print(f"处理: {os.path.basename(json_path)} -> {os.path.basename(img_path)}")

        if process_tif_file(img_path, json_path, save_dir, index, label_num=label_num, randomrate=randomrate):
            processed_count += 1
        else:
            skipped_processing_error += 1

    # 清理临时目录
    temp_dir = os.path.join(save_dir, 'temp')
    try:
        shutil.rmtree(temp_dir)
    except:
        print(f"清理临时目录失败: {temp_dir}")

    # 输出处理统计
    print(f"\n处理完成统计:")
    print(f"总JSON文件数: {len(json_list)}")
    print(f"成功处理: {processed_count}")
    print(f"跳过(无效JSON): {skipped_invalid_json}")
    print(f"跳过(无对应图像): {skipped_no_image}")
    print(f"跳过(处理错误): {skipped_processing_error}")
    print(f"处理成功率: {processed_count / len(json_list) * 100:.2f}%")

    if processed_count > 0:
        print(f"\n成功处理的文件已保存到: {save_dir}")
        print(f"生成的训练数据包括:")
        print(f"  - 图像文件: {os.path.join(save_dir, 'JPEGImages')}")
        print(f"  - 标注文件: {os.path.join(save_dir, 'SegmentationClass')}")
        print(f"  - 数据集列表: {os.path.join(save_dir, 'ImageSets', 'Segmentation')}")
        print(f"  - WSI可视化: {os.path.join(save_dir, 'VisWSI')}")
        print(f"  - Patch可视化: {os.path.join(save_dir, 'VisPatch')}")