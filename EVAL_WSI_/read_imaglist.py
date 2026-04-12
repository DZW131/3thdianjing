import numpy as np
from imageio import imsave
import openslide
import os, json
import cv2
import kfb.kfbslide as kfbslide
from EVAL_WSI_1GPU.WSI_data_sample_feiai import deduplicate_paths
from Slide_0.openslide_func import openSlide as di_openSlide
from Slide.dispatch import openSlide
from parse_embolus import read_region_kfb
import random
from tqdm import tqdm

if __name__ == '__main__':
    # get_properties(slide_path)
    save_dir='/home/songlinru/Project/quanceng_traindata'
    img_folder = '/home/songlinru/data/WSI/河南省妇幼保健院_202402_妊娠期甲减胎膜识别_曾宪旭/'
    # class映射的值必须从1开始，且为连续的整数
    label_dict = {
                  "成熟绒毛": 1,
                  "中间型绒毛": 2,
                  "无血管绒毛/核碎裂": 3,
                  "梗死/纤维素沉积/血肿/血栓": 4,
                  "合体结节": 5,
                  "绒毛发育不良": 6,
                  "绒毛血管增生":7
                  }

    label_color = {
        '成熟绒毛': (0, 255, 0),
        '中间型绒毛': (255, 0, 0),
        '无血管绒毛/核碎裂': (128, 128, 0),
        "梗死/纤维素沉积/血肿/血栓": (0, 0, 255),
        '合体结节': (255, 64, 128),
        '绒毛发育不良': (0, 128, 128),
        '绒毛血管增生':(128, 0, 128)
        # '羊膜':(128, 64, 64)
    }

    label_num = 7 + 1
    randomrate = 0.00001
    if not os.path.exists(os.path.join(save_dir, 'ImageSets')):
        os.makedirs(os.path.join(save_dir, 'JPEGImages'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'SegmentationClass'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'ImageSets', 'Segmentation'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'VisWSI'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'VisPatch'), exist_ok=True)
    # mpp=0.25, level=5; mpp=0.5, level=4;
    level = 3
    index = 0
    mpp_dic = {"mpp=0.25": 0, "mpp=0.50": 0, "other": 0}

    from pydaily import filesystem
    # ext_list = ['.kfb', '.mrxs', '.ndpi', '.sdpc']
    ext_list = ['.svs','.sdpc']
    image_name_list = []
    for ext in ext_list:
        imglist = filesystem.find_ext_files(img_folder, ext)
        imglist = deduplicate_paths(imglist)
        image_name_list.extend([os.path.splitext(os.path.basename(i))[0] for i in imglist])
    json_list = filesystem.find_ext_files(img_folder, '.json')
    json_name_list = [os.path.splitext(os.path.basename(i))[0] for i in json_list]
    # print(image_name_list)
    # print(len(image_name_list))
    # print("json_name_list")
    # print(json_name_list)
    res = set(json_name_list) - set(image_name_list)
    print(res)
    print(len(res))