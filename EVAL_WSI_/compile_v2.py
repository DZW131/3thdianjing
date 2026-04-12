import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import pandas as pd
from models.deeplab.deeplab import DeepLab

# Constants for image dimensions and pixel-to-area conversion
PIXEL_TO_MICROMETER_SQ = 1  # 修改为与实际匹配的转换系数
class_num = 3 + 1  # eTLS, pTLS, sTLS 以及背景，共4类


def load_model(default_device):
    """
    加载DeepLab模型，返回网络实例。
    """
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    try:
        print("[INFO] Loading model...")
        r_net_weights_path = r'D:\msqtry\region_level_train\run\prostate_tls\384_192_2\model_best.pth.tar'
        print(f"[INFO] Model weights path: {r_net_weights_path}")

        # 初始化DeepLab
        r_net = DeepLab(num_classes=class_num, backbone='resnet', output_stride=16,
                        sync_bn=True, freeze_bn=False)
        print("[INFO] DeepLab model initialized.")

        print("[INFO] Loading weights...")
        r_net_weights_dict = torch.load(
            r_net_weights_path,
            map_location=lambda storage, loc: storage
        )['state_dict']

        # 去除多卡训练时的 'module.' 前缀
        weights_dict = {k.replace('module.', ''): v for k, v in r_net_weights_dict.items()}
        r_net.load_state_dict(weights_dict)
        r_net.eval()

        if torch.cuda.is_available():
            r_net.cuda(default_device)
            print(f"[INFO] Model loaded and using GPU: {default_device}")
        else:
            print("[INFO] Model loaded and using CPU")

        return r_net
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        raise


def process_image_block(image_block, net):
    """
    对单个图像块进行预测，返回分类索引（单通道）。
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    data_variable = preprocess(image_block).unsqueeze(0)

    if next(net.parameters()).is_cuda:
        data_variable = data_variable.cuda()

    with torch.no_grad():
        result = net(data_variable)
        result = result[0].cpu().numpy()
        region_ind = np.argmax(result, axis=0)

    return region_ind


def generate_region_mask(image, net, image_name=None, save_dir=None, block_size=384, min_area=2000):
    """
    对输入图像进行分块预测、轮廓合并，并在图像上绘制可视化结果。
    """
    try:
        print(f"[INFO] Processing image: {image_name}")
        h, w = image.shape[0], image.shape[1]
        print(f"[INFO] Image dimensions: {h}x{w}")

        # 定义各类对应的颜色 (B, G, R)
        label_color = {
            'eTLS': (0, 255, 0),  # 暗绿色
            'pTLS': (255, 0, 0),  # 暗红色（BGR格式里红色是(0,0,255)，此处稍微变暗）
            'sTLS': (128, 128, 0)  # 暗橄榄色
        }

        # 类别ID到名称的映射（跟你的网络输出对应）
        class_mapping = {1: 'eTLS', 2: 'pTLS', 3: 'sTLS'}

        # 用于存放整张图的预测结果
        mask = np.zeros((h, w), dtype=np.uint8)

        # 用于统计每个类别的矩形框数量
        class_box_count = {'eTLS': 0, 'pTLS': 0, 'sTLS': 0}

        # 逐块预测
        for row in range(0, h, block_size):
            for col in range(0, w, block_size):
                print(f"[INFO] Processing block: ({row}, {col})")
                block = image[row:row + block_size, col:col + block_size, :]
                region_ind = process_image_block(block, net)
                mask[row:row + block_size, col:col + block_size] = region_ind

        # 生成叠加在原图上的可视化结果图
        image_and_label = (0.8 * image).astype(np.uint8)

        # 收集所有类别轮廓，准备合并
        all_contours = []
        category_pixel_count = {'eTLS': 0, 'pTLS': 0, 'sTLS': 0}  # 新增的统计每个类别的像素数量
        for i, (label_name, color) in enumerate(label_color.items(), start=1):
            print(f"[INFO] Processing label {i} ({label_name})...")
            cur_label = (mask == i).astype(np.uint8)

            # 提取该类别的外部轮廓
            contours, _ = cv2.findContours(cur_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # 计算轮廓面积（像素）
                contour_area = cv2.contourArea(contour)
                # 换算成实际微米^2
                area_in_square_micrometers = contour_area * PIXEL_TO_MICROMETER_SQ

                if area_in_square_micrometers < min_area:
                    continue  # 面积太小，跳过

                # 更新类别矩形框计数
                class_box_count[label_name] += 1

                # 更新类别的像素数
                category_pixel_count[label_name] += contour_area

                # 将轮廓与其所属类别加入列表
                all_contours.append((contour, label_name))

        # 根据类别优先级进行排序（假设eTLS < pTLS < sTLS）
        merged_contours = merge_overlapping_contours(all_contours)

        # 计算分割区域的像素数量
        total_pixels = h * w  # 总像素数
        for contour, label_name in merged_contours:
            # 用对应颜色填充轮廓区域
            color_bgr = label_color[label_name]
            cv2.drawContours(image_and_label, [contour], -1, color_bgr, thickness=-1)

            # 画外接矩形 + 类别标签
            x, y, w_box, h_box = cv2.boundingRect(contour)
            cv2.rectangle(image_and_label, (x, y), (x + w_box, y + h_box), (255, 255, 255), thickness=2)
            cv2.putText(
                image_and_label,
                label_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=2
            )

            # 计算该轮廓对应的分割区域像素数量
            contour_area = cv2.contourArea(contour)
            region_pixels = contour_area
            region_ratio = region_pixels / total_pixels  # 占总图像的比例

            # 在矩形框上显示像素数量和占比
            text = f"{region_pixels} px ({region_ratio * 100:.2f}%)"
            cv2.putText(
                image_and_label,
                text,
                (x, y + h_box + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=2
            )

        # 在图像的左上角显示类别矩形框统计信息
        stats_text = "\n".join([f"{label}: {count} boxes" for label, count in class_box_count.items()])
        y_offset = 30  # 设置一个偏移量来调整统计信息的位置
        for line in stats_text.split('\n'):
            cv2.putText(
                image_and_label,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,  # 增大字体
                color=(255, 255, 0),  # 黄色
                thickness=3
            )
            y_offset += 60  # 每行之间增加一定的间隔

        # 保存结果
        if image_name is not None and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{image_name.split('.')[0]}_predict.png")
            print(f"[INFO] Saving processed image to: {save_path}")
            cv2.imwrite(save_path, image_and_label[:, :, ::-1])

        # 返回类别像素统计，用于输出到Excel
        return category_pixel_count, total_pixels

    except Exception as e:
        print(f"[ERROR] Error processing image {image_name}: {e}")
        raise


def merge_overlapping_contours(all_contours):
    """
    将输入的所有轮廓按类别优先级进行排序，并合并有重叠的轮廓。
    合并方式：用 np.concatenate 在数组层面合并，而不再做逐元素加法。
    """
    # 定义类别优先级列表
    priority_list = ['eTLS', 'pTLS', 'sTLS']
    # 按类别优先级排序（index越小越优先）
    all_contours.sort(key=lambda x: priority_list.index(x[1]))

    merged_contours = []

    while all_contours:
        contour, label_name = all_contours.pop(0)
        merged = False

        for idx, (existing_contour, existing_label) in enumerate(merged_contours):
            # 如果与已有轮廓发生重叠，就进行合并
            if is_overlapping(existing_contour, contour):
                # 用 np.concatenate 合并两个轮廓的坐标点
                new_contour = np.concatenate((existing_contour, contour), axis=0)
                # 更新此处的类别为新的 label_name（基于优先级后弹出的轮廓）
                merged_contours[idx] = (new_contour, label_name)
                merged = True
                break

        if not merged:
            merged_contours.append((contour, label_name))

    return merged_contours


def is_overlapping(contour1, contour2):
    """
    判断两个轮廓是否在外接矩形层面发生重叠。
    """
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)

    # 如果两个外接框在水平方向或竖直方向上没有相交，则认为不重叠
    if x1 + w1 < x2 or x2 + w2 < x1:
        return False
    if y1 + h1 < y2 or y2 + h2 < y1:
        return False

    return True


def get_jpg_files(folder):
    """
    扫描指定文件夹及其子文件夹，返回所有以.jpg或.JPG结尾的文件路径列表。
    """
    print(f"[INFO] Scanning folder for JPG images: {folder}")
    jpg_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    print(f"[INFO] Found {len(jpg_files)} JPG images.")
    return jpg_files


def save_to_excel(data, output_path):
    """
    将图像名称、类别像素统计及占比保存到Excel文件。
    """
    df = pd.DataFrame(data,
                      columns=["Image Name", "eTLS Pixels", "eTLS %", "pTLS Pixels", "pTLS %", "sTLS Pixels", "sTLS %"])
    df.to_excel(output_path, index=False)
    print(f"[INFO] Excel file saved to: {output_path}")


if __name__ == '__main__':
    try:
        print("[INFO] Starting program...")

        print(f"[INFO] Number of classes: {class_num}")

        print("[INFO] Loading model...")
        r_net = load_model(default_device=0)

        save_dir = r'\\10.15.20.69\homes\sunchenhao_69\TCGA\result_v1'
        if not os.path.exists(save_dir):
            print(f"[INFO] Creating save directory: {save_dir}")
            os.makedirs(save_dir)

        img_folder = r'\\10.15.20.69\homes\sunchenhao_69\TCGA\jpg'
        print(f"[INFO] Image folder: {img_folder}")

        imglist = get_jpg_files(img_folder)
        if not imglist:
            print("[ERROR] No JPG images found in the specified folder. Exiting...")
            raise FileNotFoundError("No JPG files found.")

        print(f"[INFO] Found {len(imglist)} images. Starting processing...")

        # 用于保存统计数据的列表
        all_image_data = []

        for img_path in imglist:
            print(f"[INFO] Reading image: {img_path}")
            image = cv2.imread(img_path)

            if image is None:
                print(f"[ERROR] Failed to read image: {img_path}. Skipping...")
                continue

            # 原图是BGR格式，这里转为RGB再送入模型预测（看你的网络训练时的预处理）
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            category_pixel_count, total_pixels = generate_region_mask(image, r_net,
                                                                      image_name=os.path.basename(img_path),
                                                                      save_dir=save_dir)

            # 计算每个类别的百分比
            row = [os.path.basename(img_path)]
            for label in ['eTLS', 'pTLS', 'sTLS']:
                pixel_count = category_pixel_count.get(label, 0)
                percentage = (pixel_count / total_pixels) * 100
                row.append(f"{pixel_count}")
                row.append(f"{percentage:.4f}%")

            all_image_data.append(row)

        # 将所有数据保存到Excel文件
        excel_output_path = r'\\10.15.20.69\homes\sunchenhao_69\TCGA\result_v1\output_image_statistics.xlsx'
        save_to_excel(all_image_data, excel_output_path)

        print("[INFO] All images processed and saved successfully.")

    except Exception as e:
        print(f"[ERROR] Error in main program: {e}")


