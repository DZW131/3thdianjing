#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 16:13
# @Author  : Can Cui
# @File    : eval_multi_cell_seg.py
# @Software: PyCharm
# @Comment:
import kfb.kfbslide as kfbslide
import torch
import cv2
from parse_embolus import read_region_kfb
import numpy as np
import torch.nn as nn
import os, sys
import openslide
import torchvision.transforms as transforms
from Slide.openslide_func import openSlide as di_openSlide
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_root)
from PIL import Image
from Core.papSmear.multi_cls_cell_seg import cal_ki67_np,cal_ki67_np_region
import math
import custom_transforms as tr
global label_dict,color_dict

def walk_dir(data_dir, file_types):
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:
                if f.endswith(this_type):
                    path_list.append(os.path.join(dirpath, f))
                    break
    return path_list

from pydaily import filesystem


def sort_edge(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def RGBAtoBGR(image_rgba):
    r, g, b, a = cv2.split(image_rgba)
    image_bgr = cv2.merge([r, g, b])
    return image_bgr



def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

def get_level_dim_dict(slide_path):
    level_dim_dict = {}
    slide= di_openSlide(slide_path).slide
    dims = slide.level_dimensions
    downsamples = slide.level_downsamples
    for i in range(len(downsamples)):
        level_dim_dict[i] = (dims[i], downsamples[i])
    return level_dim_dict
def transform_val(sample,size):
    composed_transforms = transforms.Compose([
       # tr.FixedResize(size=size),
        # tr.FixScaleCrop(crop_size=self.args.crop_size),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    return composed_transforms(sample)

def generate_region_mask(image, net, patch_size=512, overlap=512, resize=512):
    # split into 16 patches of size 250x250
    h, w = image.shape[0], image.shape[1]
    h_overlap = overlap
    w_overlap = overlap

    mask=np.zeros((h,w))
    for x in range(0, h - patch_size + 1, patch_size - h_overlap):
        for y in range(0, w - patch_size + 1, patch_size - w_overlap):
            patch = image[x:x + patch_size, y:y + patch_size, :]
            patch = patch[:, :, ::-1]
            patch_imgs=Image.fromarray(patch)

            # patch_imgs = patch_imgs * (2. / 255) - 1.
            with torch.no_grad():
                patch_imgs = transform_val(patch_imgs, resize)
                data_variable = patch_imgs.unsqueeze(0)
                if net.parameters().__next__().is_cuda:
                    data_variable = data_variable.cuda(net.parameters().__next__().get_device())
                    result = net(data_variable)
                    #result = transforms.Resize([patch_size, patch_size])(result)
                    result = torch.softmax(result, dim=1)[0,:,:,:].cpu().numpy()
                    result=result.transpose((1,2,0))
                    region_ind = np.argmax(result, axis=-1)
                    try:
                        mask[x:x + patch_size, y:y + patch_size]=region_ind
                    except:
                        r_h,r_w=region_ind.shape[0],region_ind.shape[1]
                        mask[x:x + r_h, y:y + r_w]=region_ind
    map=np.where(mask==0,1,0)#backgroud is 0
    suppress_map=1-map #now, forground is 1
    suppress_map = suppress_map[:, :, np.newaxis] * 255
    _, RedThresh = cv2.threshold(suppress_map.astype(np.uint8), 160, 255, cv2.THRESH_BINARY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (32, 32))
    eroded = cv2.erode(RedThresh, kernel1)  # 腐蚀图像
    dilated = cv2.dilate(eroded, kernel2)# 膨胀图像
    binary=255-dilated #now we can surpress the forground
    map = binary
    map=map.astype(np.uint8)
    # map=cv2.resize(map,ori_size)
    return map

def split_kfb(slides_dir, R_net, save_path):
    slide_path, slide_name = os.path.split(slides_dir)
    kfb_name, kfb_ext = os.path.splitext(slide_name)
    result_save_sub_dir = save_path
    os.makedirs(result_save_sub_dir, exist_ok=True)
    level=4
    if not os.path.exists(os.path.join(result_save_sub_dir, "summary_{}.csv".format(kfb_name))):
        level_dim_dict = get_level_dim_dict(slides_dir)
        scale = level_dim_dict[level][1]
        dim = level_dim_dict[level][0]
        if 'ndpi' in slide_name or 'mrxs' in slide_name or 'sdpc' in slide_name:
            slide = openslide.OpenSlide(slides_dir)
            thumbnail = np.array(
                slide.read_region(location=(0, 0), level=level, size=dim))
            thumbnail = thumbnail[:, :, :3].astype(np.uint8)
        else:
            slide = kfbslide.open_kfbslide(slides_dir)
            thumbnail = read_region_kfb(slide, location=(0, 0), level=level, size=dim)
        # thumbnail = slide.read_region(location=(0, 0), level=level, size=dim)
        thumbnail = np.array(thumbnail)
        thumbnail = thumbnail.astype(np.uint8)
        result_masks = generate_region_mask(thumbnail, R_net, patch_size=2048, overlap=512,
                                            resize=0)

        contours, _ = cv2.findContours(result_masks, cv2.RETR_LIST, 2)
        mask_save=thumbnail.copy()
        mask_save=cv2.cvtColor(mask_save,cv2.COLOR_RGB2BGR)
        mask_save = cv2.drawContours(mask_save, contours, -1, (0,0,255), 4)
        cv2.imwrite(os.path.join(result_save_sub_dir, kfb_name+'region_results.jpg'), mask_save)





def compare_with_annotations(R_net,save_path, is_display=True):
    result_save_dir = save_path
    os.makedirs(result_save_dir, exist_ok=True)
    ndpi_list = filesystem.find_ext_files(test_dir, ".ndpi")
    kfb_list = filesystem.find_ext_files(test_dir, '.kfb')
    mrxs_list = filesystem.find_ext_files(test_dir, '.mrxs')

    sdpc_list = filesystem.find_ext_files(test_dir, '.sdpc')
    for slide_path in kfb_list:

        #if 'HE' not in image_name and 'NC' not in image_name and 'V-' not in image_name:
        print("evaluating image {}".format(slide_path))
        split_kfb(slide_path,R_net, save_path)

    for slide_path in sdpc_list:
        # image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        #if 'HE' not in image_name and 'NC' not in image_name and 'V-' not in image_name:
        print("evaluating image {}".format(slide_path))


        split_kfb(slide_path,R_net, save_path)

    for slide_path in ndpi_list:
        image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        if 'HE' not in image_name and 'NC' not in image_name:
            print("evaluating image {}".format(slide_path))
            split_kfb(slide_path,R_net, save_path)

    for slide_path in mrxs_list:
        print("evaluating image {}".format(slide_path))

        split_kfb(slide_path,R_net, save_path)


def run_test(test_dir,R_weight_path,save_path):
    from deeplab.deeplab import DeepLab as R_detnet
    R_net = R_detnet(num_classes=5,
                 backbone='resnet',
                 output_stride=16,
                 sync_bn=True,
                 freeze_bn=False)

    R_net = nn.DataParallel(R_net)
    if torch.cuda.is_available():
        R_net.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        print('You are using GPU')
    else:
        print('You are using CPU')
    R_best_checkpoint = torch.load(R_weight_path)
    R_net.load_state_dict(R_best_checkpoint['state_dict'])
    R_net.eval()
    compare_with_annotations(R_net=R_net,is_display=True,save_path=save_path)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    R_weight_path = '../weights/R_net.pth.tar'
    test_dir = "/media/deepin/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/测试用切片/九院/免疫组化无1546"
    save_path='/media/deepin/DeepInformatic_dataset/lihansheng/HER2/0503/0504'
    os.makedirs(save_path,exist_ok=True)
    run_test(test_dir, R_weight_path=R_weight_path,save_path=save_path)

    # run_test(test_dir, 3,
    #          weight_path= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'pytorch_unet',
    #                                    "Model", "unet_mix_1_2", "weights_epoch_125_1.5490121394395828.pth" ))
