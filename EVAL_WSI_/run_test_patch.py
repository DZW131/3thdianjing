import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from models.deeplab.deeplab import DeepLab


def load_model(default_device):
    """
    加载预训练的 DeepLab 模型
    """
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    r_net_weights_path = r'D:\msqtry\region_level_train\run\prostate_tls\384_192_2\model_best.pth.tar'
    print(f"Loading model weights from: {r_net_weights_path}")

    if not os.path.exists(r_net_weights_path):
        raise FileNotFoundError(f"Model weights file not found: {r_net_weights_path}")

    # 初始化模型
    r_net = DeepLab(num_classes=class_num, backbone='resnet', output_stride=16, sync_bn=True, freeze_bn=False)
    r_net_weights_dict = torch.load(r_net_weights_path, map_location=lambda storage, loc: storage)['state_dict']

    # 处理多 GPU 训练的权重问题
    weights_dict = {}
    for k, v in r_net_weights_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    r_net.load_state_dict(weights_dict)
    r_net.eval()

    # 放置到 GPU 上
    if torch.cuda.is_available():
        r_net.cuda(default_device)
        print(f"{os.getpid()} are using GPU - {default_device}")
    else:
        print("GPU not available. Running on CPU.")

    return r_net


def generate_region_mask(image, net, patch_size=384, overlap=384, batch_size=1, image_name=None, save_dir=None):
    """
    对输入图像进行预测并生成标注后的图像
    """
    print(f"Starting inference on image: {image_name} with shape {image.shape}")

    # 定义颜色映射
    label_color = {
        'eTLS': (0, 255, 0),
        'pTLS': (255, 0, 0),
        'sTLS': (128, 128, 0)
    }

    mask = np.ones((image.shape[0], image.shape[1])) * 7

    try:
        # 推理
        with torch.no_grad():
            data_variable = transforms.ToTensor()(image)
            Norm_ = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            data_variable = Norm_(data_variable).unsqueeze(0)

            if net.parameters().__next__().is_cuda:
                data_variable = data_variable.cuda(net.parameters().__next__().get_device())

            print("Performing model inference...")
            result = net(data_variable)
            print(f"Inference completed. Model output shape: {result.shape}")

            # 处理推理结果
            result = torch.softmax(result, dim=1)[0, :, :, :].cpu().numpy()
            result = result.transpose((1, 2, 0))
            region_ind = np.argmax(result, axis=-1)
            mask[:, :] = region_ind
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # 绘制标注图像
    try:
        image_and_label = 0.4 * image
        for i in range(1, 4):  # 类别 1, 2, 3
            cur_label = np.where(mask == i, 1, 0)
            cur_label = cur_label[:, :, np.newaxis]
            cur_label = np.repeat(cur_label, 3, -1)
            cur_color = list(label_color.values())[i - 1]
            label_region_color = cur_label * cur_color
            label_region_color = np.asarray(label_region_color, np.uint8)
            image_and_label = image_and_label + 0.6 * label_region_color

        if image_name is not None:
            output_path = os.path.join(save_dir, image_name.split('.')[0] + '_predict.png')
            cv2.imwrite(output_path, image_and_label[:, :, ::-1])
            print(f"Saved annotated image to: {output_path}")
    except Exception as e:
        print(f"Error during annotation or saving: {e}")


def process_images(img_folder, save_dir, r_net):
    """
    处理输入文件夹中的所有图片
    """
    print(f"Looking for images in folder: {img_folder}")

    # 确保输入路径和输出路径存在
    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"Image folder not found: {img_folder}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created output directory: {save_dir}")

    imglist = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg')]
    print(f"Found {len(imglist)} images in folder {img_folder}")

    if len(imglist) == 0:
        print("No images found to process!")
        return

    # 遍历每张图片并处理
    for idx, img_path in enumerate(imglist):
        print(f"Processing image {idx + 1}/{len(imglist)}: {img_path}")
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue

            generate_region_mask(image, r_net, patch_size=384, overlap=384, batch_size=1,
                                 image_name=os.path.basename(img_path), save_dir=save_dir)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == '__main__':
    class_num = 3 + 1  # 3 类 + 背景
    save_dir = os.path.abspath('./svs test/pre_svs')  # 输出目录
    img_folder = r'D:\msqtry\region_level_train\svs test\WSI_svs'

    # 加载模型
    r_net = load_model(default_device=0)

    # 处理图片
    process_images(img_folder, save_dir, r_net)