import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
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
            'eTLS': (0, 255, 0),    # 暗绿色
            'pTLS': (255, 0, 0),    # 暗红色（BGR格式里红色是(0,0,255)，此处稍微变暗）
            'sTLS': (128, 128, 0)   # 暗橄榄色
        }

        # 类别ID到名称的映射（跟你的网络输出对应）
        class_mapping = {1: 'eTLS', 2: 'pTLS', 3: 'sTLS'}

        # 用于存放整张图的预测结果
        mask = np.zeros((h, w), dtype=np.uint8)

        # 逐块预测
        for row in range(0, h, block_size):
            for col in range(0, w, block_size):
                print(f"[INFO] Processing block: ({row}, {col})")
                block = image[row:row + block_size, col:col + block_size, :]
                region_ind = process_image_block(block, net)
                mask[row:row + block_size, col:col + block_size] = region_ind

        # 生成叠加在原图上的可视化结果图
        # 这里先乘以 0.8，再转换为uint8类型，防止画图时出现类型不匹配
        image_and_label = (0.8 * image).astype(np.uint8)

        # 收集所有类别轮廓，准备合并
        all_contours = []
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

                # 将轮廓与其所属类别加入列表
                all_contours.append((contour, label_name))

        # 根据类别优先级进行排序（假设eTLS < pTLS < sTLS）
        merged_contours = merge_overlapping_contours(all_contours)

        # 在图像上对合并后的轮廓进行上色，并绘制外接矩形及类别标签
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

        # 保存结果
        if image_name is not None and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{image_name.split('.')[0]}_predict.png")
            print(f"[INFO] Saving processed image to: {save_path}")
            # 注意：原始代码里是 image_and_label[:, :, ::-1]，如果你想保持RGB模式，可直接写 image_and_label
            # 如果需要BGR保存，则继续使用[:, :, ::-1]。看你的需求而定，这里以BGR保存举例：
            cv2.imwrite(save_path, image_and_label[:, :, ::-1])

        print(f"[INFO] Image {image_name} processed successfully.")
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

if __name__ == '__main__':
    try:
        print("[INFO] Starting program...")

        print(f"[INFO] Number of classes: {class_num}")

        print("[INFO] Loading model...")
        r_net = load_model(default_device=0)

        save_dir = r'\\10.15.20.69\homes\sunchenhao_69\TCGA\result'
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
        for img_path in imglist:
            print(f"[INFO] Reading image: {img_path}")
            image = cv2.imread(img_path)

            if image is None:
                print(f"[ERROR] Failed to read image: {img_path}. Skipping...")
                continue

            # 原图是BGR格式，这里转为RGB再送入模型预测（看你的网络训练时的预处理）
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            generate_region_mask(image, r_net, image_name=os.path.basename(img_path), save_dir=save_dir)

        print("[INFO] All images processed successfully.")

    except Exception as e:
        print(f"[ERROR] Error in main program: {e}")