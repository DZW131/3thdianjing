from PIL import Image
from torchvision import transforms

import kfb.kfbslide as kfbslide
import torch
import cv2
from parse_embolus import read_region_kfb
import numpy as np
import os, sys
import pandas as pd
from Slide.openslide_func import openSlide as di_openSlide
import openslide
from regon_segmentation import deep_region_segmentation as deep_region_segmentation
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_root)

from Core.papSmear.multi_cls_cell_seg import cal_ki67_np

device = 'cuda' if torch.cuda.is_available() else 'cpu'



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



import time


def RGBAtoBGR(image_rgba):
    r, g, b, a = cv2.split(image_rgba)
    image_bgr = cv2.merge([r, g, b])
    return image_bgr


def split_kfb(slides_dir, crop_len, net, is_display, result_save_dir):
    slide_path, slide_name = os.path.split(slides_dir)
    kfb_name, kfb_ext = os.path.splitext(slide_name)
    result_summary = {
        'slide name': [],
        '阳性肿瘤细胞': [],
        '阴性肿瘤细胞': [],
        '阳性淋巴细胞': [],
        '阴性淋巴细胞': [],
        '阳性组织细胞': [],
        '阴性组织细胞': [],
        '纤维间质细胞': [],
        '正常肺泡细胞': [],
        '背景': [],
        '细胞总数': [],
        'PD-L1指数': [],
    }
    start_time = time.time()
    result_save_sub_dir = os.path.join(
        '/media/kd/Seagate Backup Plus Drive/Seagate/PD-L1_cell_detection_test_result_WSI',
        kfb_name)
    os.makedirs(result_save_sub_dir, exist_ok=True)
    if not os.path.exists(os.path.join(result_save_sub_dir, "summary_{}.csv".format(kfb_name))):
        slide_OTUS = di_openSlide(slides_dir)
        size = slide_OTUS.slide.level_dimensions[0]
        if 'ndpi' in slides_dir or 'mrxs' in slides_dir or 'svs' in slides_dir:
            slide = openslide.open_slide(slides_dir)

        else:
            slide = kfbslide.open_kfbslide(slides_dir)

        width = size[0]
        height = size[1]
        if 1:

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
                            region_size = (crop_len, crop_len)
                            region_start = (crop_start_x, crop_start_y)
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
                            if 'ndpi' in slides_dir or 'mrxs' in slides_dir:
                                try:
                                    cur_region = np.array(
                                        slide.read_region(region_start, 0, region_size))
                                except:
                                    continue
                                else:
                                    cur_region = RGBAtoBGR(cur_region)
                            else:
                                cur_region = read_region_kfb(slides_dir, slide, region_start, region_size)
                                img_name = kfb_name + '-' + str(i) + '-' + str(j) + '-' + str(num_patch) + ".png"

                                cell_count_dict, test_center_coords, test_labels = count_test_summary(cur_region,
                                                                                                      net,
                                                                                                      test_count_dict)
                                negative_index = np.where(test_labels == 1)
                                image = cur_region
                                image = image[:, :, :: -1]
                                rgb_mean, rgb_std = (0.790, 0.609, 0.806), (0.169, 0.189, 0.108)
                                transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(rgb_mean, rgb_std)
                                ])
                                if negative_index != []:
                                    negative_center_pos = test_center_coords[negative_index]
                                    negative_index = negative_index[0]
                                    negative_index_new = []
                                    for nega_item in range(negative_index.size):
                                        center_pos_x = negative_center_pos[nega_item, 0]
                                        center_pos_y = negative_center_pos[nega_item, 1]
                                        left_pos_x = center_pos_x - 32
                                        left_pos_y = center_pos_y - 32
                                        right_pos_x = center_pos_x + 32
                                        right_pos_y = center_pos_y + 32
                                        if left_pos_x >= 0 and left_pos_y >= 0 and right_pos_x <= 1024 and right_pos_y <= 1024 and len(
                                                negative_index_new) == 0:
                                            image_cur = transform(Image.fromarray(
                                                image[left_pos_y:right_pos_y, left_pos_x:right_pos_x]))
                                            image_cur = image_cur.unsqueeze(0).to(device)
                                            negative_index_new.append(negative_index[nega_item])
                                        elif left_pos_x >= 0 and left_pos_y >= 0 and right_pos_x <= 1024 and right_pos_y <= 1024 and len(
                                                negative_index_new) != 0:
                                            image_cur_now = transform(
                                                Image.fromarray(
                                                    image[left_pos_y:right_pos_y, left_pos_x:right_pos_x]))
                                            image_cur_now = image_cur_now.unsqueeze(0).to(device)
                                            image_cur = torch.cat((image_cur, image_cur_now), 0)
                                            negative_index_new.append(negative_index[nega_item])

                                    # start

                                result_summary['slide name'].append(img_name)
                                for k, v in cell_count_dict.items():
                                    # result_summary[k].append((test_count_dict[k] - annotation_count_dict[k]))
                                    # result_summary[k].append((annotation_count_dict[k] - test_count_dict[k]))
                                    if k != '阴性肿瘤细胞' and k != '纤维间质细胞':
                                        result_summary[k].append([
                                            # annotation_count_dict[k],
                                            test_count_dict[k]
                                            # round(test_count_dict[k] - annotation_count_dict[k], 3)
                                        ])
                                    elif k == '阴性肿瘤细胞':
                                        result_summary[k].append([
                                            # annotation_count_dict[k],
                                            np.where(test_labels == 1)[0].size
                                            # round(test_count_dict[k] - annotation_count_dict[k], 3)
                                        ])
                                    elif k == '纤维间质细胞':
                                        result_summary[k].append([
                                            # annotation_count_dict[k],
                                            np.where(test_labels == 6)[0].size
                                            # round(test_count_dict[k] - annotation_count_dict[k], 3)
                                        ])
                                print(result_summary)
                                if is_display:
                                    image = cur_region
                                    test_res_img = draw_center(image, test_center_coords, test_labels)
                                    test_res_img = test_res_img[:, :, :: -1]
                                    cv2.imwrite(os.path.join(result_save_sub_dir, img_name),
                                                test_res_img)
                            crop_start_x = (j + 1) * crop_len
                            num_patch = num_patch + 1


        print(time.time() - start_time)
        df = pd.DataFrame(result_summary)
        columns = ['slide name', '阳性肿瘤细胞', '阴性肿瘤细胞', '阳性淋巴细胞', '阴性淋巴细胞', '阳性组织细胞', '阴性组织细胞', '纤维间质细胞', '正常肺泡细胞', '背景',
                   '细胞总数', 'PD-L1指数']
        df.to_csv(os.path.join(result_save_sub_dir, "summary_{}.csv".format(kfb_name)),
                  columns=columns, index=False)


def compare_with_annotations(test_data_dir, test_name, net, cell_count_dict=None, is_display=True):
    result_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TEST_RESULT", test_name)
    os.makedirs(result_save_dir, exist_ok=True)

    ndpi_list = filesystem.find_ext_files(test_dir, ".ndpi")
    kfb_list = filesystem.find_ext_files(test_dir, '.kfb')
    mrxs_list = filesystem.find_ext_files(test_dir, '.mrxs')
    svs_list = filesystem.find_ext_files(test_dir, '.svs')
    for slide_path in kfb_list:

        image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        if 'HE' not in image_name and 'NC' not in image_name and 'V-' not in image_name:
            print("evaluating image {}".format(slide_path))

            split_kfb(slide_path, 1024, net, is_display, result_save_dir)

    for slide_path in ndpi_list:

        image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        if 'HE' not in image_name and 'NC' not in image_name:
            print("evaluating image {}".format(slide_path))

            split_kfb(slide_path, 1024, net, is_display, result_save_dir)

    for slide_path in mrxs_list:

        image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        if 'HE' not in image_name and 'NC' not in image_name:
            print("evaluating image {}".format(slide_path))

            split_kfb(slide_path, 1024, net, is_display, result_save_dir)
    for slide_path in svs_list:

        image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".jpg"

        if 'h' not in image_name and 'n' not in image_name:
            print("evaluating image {}".format(slide_path))

            split_kfb(slide_path, 1024, net, is_display, result_save_dir)


def count_test_summary(image, net, cell_count_dict):
    center_coords, labels = cal_ki67_np(image, net)
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
    return cell_count_dict, center_coords, labels


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def run_test(test_dir, net_name, weight_path):
    if net_name == 0:
        from Core.models.albu_unet import AlbuNet as detnet
        net = detnet(input_channels=3, num_classes=9)
    elif net_name == 1:
        from Core.models.attention_unet import AttU_Net as detnet
        net = detnet(input_channels=3, num_classes=9)
    elif net_name == 2:
        from Core.models.nested_unet import NestedUNet as detnet
        net = detnet(input_channels=3, num_classes=9)
    elif net_name == 3:
        from Core.models.attention_unet import U_Net as detnet
        net = detnet(input_channels=3, num_classes=9)
    elif net_name==4:
        from Core.models.attention_unet import R2AttU_Net as detnet
        net = detnet(input_channels=3, num_classes=9)
    elif net_name==5:
        from Core.models.node_unet_models import ConvODEUNet as detModel
        net = detModel(input_channels=3, num_filters=16, output_dim=9, time_dependent=True,
                       non_linearity='lrelu', adjoint=True, tol=1e-2)
    net_weights_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(net_weights_dict)
    net.eval()
    print('reload detection net weights from {}'.format(weight_path))

    if torch.cuda.is_available():
        net.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        print('You are using GPU')
    else:
        print('You are using CPU')

    test_name = os.path.basename(test_dir) + '_' + os.path.basename(os.path.dirname(weight_path)) + '_' + \
                os.path.splitext(os.path.basename(weight_path))[0]

    compare_with_annotations(test_data_dir=test_dir, test_name=test_name, net=net, is_display=True)


if __name__ == "__main__":
    allowed_net = {
        0: "albu_unet",
        1: "att_unet",
        2: "nested_unet",
        3: "unet"
    }

    test_dir = "/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/数据集/PDL1_Positive_cell_seg_detection/PDL1200切片/Slide/PD-L1SP263原切片/SP263/上肿/上肿 鳞癌"


    run_test(test_dir, 2,
             weight_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_unet',
                                      "Model", "nested_unet_small_new_dataset_512_for_wsi",
                                      "weights_epoch_7_0.8042927390190121.pth"))



