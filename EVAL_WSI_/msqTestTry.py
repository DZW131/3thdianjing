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
        r_net_weights_path = r'D:\msqtry\region_level_train\run\prostate_tls\deeplab-resnet\model_best.pth.tar'
        print(f"[INFO] Model weights path: {r_net_weights_path}")

        r_net = DeepLab(num_classes=class_num, backbone='resnet', output_stride=16, sync_bn=True, freeze_bn=False)
        print("[INFO] DeepLab model initialized.")

        print("[INFO] Loading weights...")
        r_net_weights_dict = torch.load(r_net_weights_path, map_location=lambda storage, loc: storage)['state_dict']

        # Remove 'module.' prefix if it exists in state_dict keys
        weights_dict = {k.replace('module.', ''): v for k, v in r_net_weights_dict.items()}
        r_net.load_state_dict(weights_dict)
        r_net.eval()

        # Move model to the correct device
        if torch.cuda.is_available():
            r_net.cuda(default_device)
            print(f"[INFO] Model loaded and using GPU: {default_device}")
        else:
            print("[INFO] Model loaded and using CPU")

        return r_net
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        raise


def generate_region_mask(image, net, image_name=None, save_dir=None):
    try:
        print(f"[INFO] Processing image: {image_name}")
        h, w = image.shape[0], image.shape[1]
        print(f"[INFO] Image dimensions: {h}x{w}")

        # Define class-to-color mapping
        label_color = {
            'eTLS': (0, 255, 0),  # Green
            'pTLS': (255, 0, 0),  # Red
            'sTLS': (128, 128, 0)  # Olive
        }

        mask = np.zeros((h, w), dtype=np.uint8)  # Initialize mask for class indices

        with torch.no_grad():
            # Step 1: Preprocess the image
            print("[INFO] Converting image to tensor...")
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            data_variable = preprocess(image).unsqueeze(0)  # Add batch dimension

            # Move input to GPU if model is on GPU
            if next(net.parameters()).is_cuda:
                data_variable = data_variable.cuda()

            # Step 2: Run inference
            print("[INFO] Running inference on the model...")
            result = net(data_variable)
            print(f"[DEBUG] Raw model output shape: {result.shape}")

            # Step 3: Postprocess model output
            result = result[0].cpu().numpy()  # Remove batch dimension, move to CPU
            region_ind = np.argmax(result, axis=0)  # Class index for each pixel

            # Debug: Print unique region indices
            print(f"[DEBUG] Unique region indices in output: {np.unique(region_ind)}")
            mask[:, :] = region_ind

        # Step 4: Generate overlay image
        image_and_label = 0.6 * image  # Blend original image with segmentation
        for i, color in enumerate(label_color.values(), start=1):
            print(f"[INFO] Processing label {i}...")
            cur_label = (mask == i).astype(np.uint8)  # Binary mask for current class
            cur_label = np.expand_dims(cur_label, axis=-1)  # Add channel dimension
            cur_label = np.repeat(cur_label, 3, axis=-1)  # Convert to 3-channel
            label_region_color = cur_label * np.array(color, dtype=np.uint8)
            image_and_label = image_and_label + 0.4 * label_region_color  # Adjust overlay weight

        # Step 5: Save the result
        if image_name is not None and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)  # Create save directory if it doesn't exist
            save_path = os.path.join(save_dir, f"{image_name.split('.')[0]}_predict.png")
            print(f"[INFO] Saving processed image to: {save_path}")
            cv2.imwrite(save_path, image_and_label[:, :, ::-1])  # Convert RGB to BGR for OpenCV

        print(f"[INFO] Image {image_name} processed successfully.")
    except Exception as e:
        print(f"[ERROR] Error processing image {image_name}: {e}")
        raise


def get_jpg_files(folder):
    """
    获取文件夹中的所有 JPG 文件
    """
    print(f"[INFO] Scanning folder for JPG images: {folder}")
    jpg_files = [os.path.join(root, file)
                 for root, _, files in os.walk(folder)
                 for file in files if file.lower().endswith('.jpg')]
    print(f"[INFO] Found {len(jpg_files)} JPG images.")
    return jpg_files


if __name__ == '__main__':
    try:
        print("[INFO] Starting program...")

        class_num = 3 + 1  # Number of classes including background
        print(f"[INFO] Number of classes: {class_num}")

        print("[INFO] Loading model...")
        r_net = load_model(default_device=0)

        save_dir = r'D:\msqtry\region_level_train\svs test\pre_svs'
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

            # Convert BGR to RGB for processing
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            generate_region_mask(image, r_net, image_name=os.path.basename(img_path), save_dir=save_dir)

        print("[INFO] All images processed successfully.")

    except Exception as e:
        print(f"[ERROR] Error in main program: {e}")