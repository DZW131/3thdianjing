import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from models.deeplab.deeplab import DeepLab


def load_model(default_device):
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    try:
        print("[INFO] Loading model...")
        r_net_weights_path = r'D:\msqtry\region_level_train\run\prostate_tls\384_192_2\model_best.pth.tar'
        print(f"[INFO] Model weights path: {r_net_weights_path}")

        r_net = DeepLab(num_classes=class_num, backbone='resnet', output_stride=16, sync_bn=True, freeze_bn=False)
        print("[INFO] DeepLab model initialized.")

        print("[INFO] Loading weights...")
        r_net_weights_dict = torch.load(r_net_weights_path, map_location=lambda storage, loc: storage)['state_dict']

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


def generate_region_mask(image, net, image_name=None, save_dir=None, block_size=512):
    try:
        print(f"[INFO] Processing image: {image_name}")
        h, w = image.shape[0], image.shape[1]
        print(f"[INFO] Image dimensions: {h}x{w}")

        label_color = {
            'eTLS': (0, 100, 0),  # Dark Green
            'pTLS': (100, 0, 0),  # Dark Red
            'sTLS': (100, 100, 0)  # Dark Olive
        }

        class_mapping = {1: 'eTLS', 2: 'pTLS', 3: 'sTLS'}
        mask = np.zeros((h, w), dtype=np.uint8)

        # Process image in blocks
        for row in range(0, h, block_size):
            for col in range(0, w, block_size):
                print(f"[INFO] Processing block: ({row}, {col})")
                block = image[row:row + block_size, col:col + block_size, :]
                region_ind = process_image_block(block, net)
                mask[row:row + block_size, col:col + block_size] = region_ind

        # Initialize the overlay image
        image_and_label = 0.8 * image
        for i, (label_name, color) in enumerate(label_color.items(), start=1):
            print(f"[INFO] Processing label {i} ({label_name})...")
            cur_label = (mask == i).astype(np.uint8)

            # Generate overlay
            cur_label_expanded = np.expand_dims(cur_label, axis=-1)
            cur_label_expanded = np.repeat(cur_label_expanded, 3, axis=-1)
            label_region_color = cur_label_expanded * np.array(color, dtype=np.uint8)
            image_and_label = image_and_label + 1.0 * label_region_color

            # Find contours and draw bounding boxes
            contours, _ = cv2.findContours(cur_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Draw a white bounding box with thicker lines
                cv2.rectangle(image_and_label, (x, y), (x + w, y + h), (255, 255, 255), thickness=3)

                # Add white text label with bold font
                cv2.putText(
                    image_and_label,
                    label_name,
                    (x, y - 10),  # Position text above the rectangle
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,  # Adjust font size
                    color=(255, 255, 255),  # White text
                    thickness=2  # Thicker text for better visibility
                )

        # Save the result
        if image_name is not None and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{image_name.split('.')[0]}_predict.png")
            print(f"[INFO] Saving processed image to: {save_path}")
            cv2.imwrite(save_path, image_and_label[:, :, ::-1])

        print(f"[INFO] Image {image_name} processed successfully.")
    except Exception as e:
        print(f"[ERROR] Error processing image {image_name}: {e}")
        raise


def get_jpg_files(folder):
    print(f"[INFO] Scanning folder for JPG images: {folder}")
    jpg_files = [os.path.join(root, file)
                 for root, _, files in os.walk(folder)
                 for file in files if file.lower().endswith('.jpg')]
    print(f"[INFO] Found {len(jpg_files)} JPG images.")
    return jpg_files


if __name__ == '__main__':
    try:
        print("[INFO] Starting program...")

        class_num = 3 + 1
        print(f"[INFO] Number of classes: {class_num}")

        print("[INFO] Loading model...")
        r_net = load_model(default_device=0)

        save_dir = r'D:\msqtry\region_level_train\svs test\pre_svs_v0'
        if not os.path.exists(save_dir):
            print(f"[INFO] Creating save directory: {save_dir}")
            os.makedirs(save_dir)

        img_folder = r'D:\msqtry\region_level_train\svs test\WSI_svs'
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

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            generate_region_mask(image, r_net, image_name=os.path.basename(img_path), save_dir=save_dir)

        print("[INFO] All images processed successfully.")

    except Exception as e:
        print(f"[ERROR] Error in main program: {e}")