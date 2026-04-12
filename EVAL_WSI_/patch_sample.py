#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 16:13
# @Author  : Can Cui
# @File    : eval_multi_cell_seg.py
# @Software: PyCharm
# @Comment:
import kfb.kfbslide as kfbslide
from options import Options
import torch
import cv2
from parse_embolus import read_region_kfb
import numpy as np
import torch.nn as nn
import os, sys
import openslide
import pandas as pd
import pickle
import torchvision.transforms as transforms
from Slide.openslide_func import openSlide as di_openSlide
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_root)
from PIL import Image
from Core.papSmear.multi_cls_cell_seg import cal_ki67_np,cal_ki67_np_region
import math
import custom_transforms as tr
global label_dict,color_dict
def Hex_to_RGB(hex):
    r = int(hex[1:3],16)
    g = int(hex[3:5],16)
    b = int(hex[5:7], 16)
    rgb = (r,g,b)
    return rgb
label_dict = {
        '微弱的不完整膜阳性肿瘤细胞': 0,
        '弱-中等的完整细胞膜阳性肿瘤细胞': 1,
        '阴性肿瘤细胞': 2,
        '纤维细胞': 3,
        '淋巴细胞': 4,
        '难以区分的非肿瘤细胞': 5,
        '组织细胞': 6,
        '强度的完整细胞膜阳性肿瘤细胞': 7,
        '中-强度的不完整细胞膜阳性肿瘤细胞': 8

    }
Reverse_label_dicr={ind:name for ind,name in enumerate(label_dict.keys())}

color_card_hex={'阴性肿瘤细胞':'#00FF00',
                        '微弱的不完整膜阳性肿瘤细胞':'#FF3399',
                        '强度的完整细胞膜阳性肿瘤细胞':'#FF0033',
                        '弱-中等的完整细胞膜阳性肿瘤细胞':'#FF6633',
                        '中-强度的不完整细胞膜阳性肿瘤细胞':'#4051B5',
                        '淋巴细胞':'#660066',
                        '纤维细胞':'#FFFF00',
                        '血管内皮细胞':'#8DA1D5',
                        '组织细胞':'#0033F',
                        '脂肪细胞':'#80DEFF',
                        '难以区分的非肿瘤细胞':'#AAEA63',
                        '导管内癌阳性肿瘤细胞':'#FF59C2',
                        '导管内癌阴性肿瘤细胞':'#B7AD79',
}
color_card={name:Hex_to_RGB(color_card_hex[name]) for name in color_card_hex.keys() }
color_dict={str(i):color_card[name] for i,name in enumerate(label_dict.keys())}
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

        radius = 6
        thickness =6

        for i in range(num_center):
            if str(label[i]) in color_dict:
                img = cv2.line(img,(center[i,0]-10,center[i,1]),(center[i,0]+10,center[i,1]),color_dict[str(label[i])],thickness)
                img = cv2.line(img, (center[i, 0] , center[i, 1]- 10), (center[i, 0] , center[i, 1]+ 10),
                         color_dict[str(label[i])], thickness)
                # img = cv2.circle(img, (center[i, 0], center[i, 1]), radius, color_dict[str(label[i])], thickness)

        return img
    else:
        return img


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


def count_summary(result_summary,result_save_sub_dir,kfb_name):
    total_sum={}

    total_sum['细胞总数']=np.sum(np.array(result_summary['细胞总数']),dtype=np.uint16)
    for key in label_dict.keys():
        total_sum[key]=np.sum(np.array(result_summary[key]),dtype=np.uint16)
    out_put=open(os.path.join(result_save_sub_dir,'%s.pkl'%kfb_name),'wb')
    pickle.dump(total_sum,out_put)
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
        tr.FixedResize(size=size),
        # tr.FixScaleCrop(crop_size=self.args.crop_size),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    return composed_transforms(sample)
def generate_result_resize_deeplab(image, net, patch_size=512, overlap=128, batch_size=4):
    # image=cv2.resize(image,(patch_size,patch_size))
    h, w = image.shape[0], image.shape[1]
    image=image[:,:,::-1]
    this_batch = Image.fromarray(image)
    this_batch=transform_val(this_batch,patch_size)
    data_variable = this_batch.unsqueeze(0)
    data_variable = data_variable.cuda(net.parameters().__next__().get_device())
    result = net(data_variable)
    result=transforms.Resize((h,w))(result)
    result = torch.softmax(result,dim=1).squeeze()
    result=result.permute(1,2,0)
    result=result.detach().cpu().numpy()
   # result_masks=result_masks[:,:,0:-1]
    return result
def generate_region_mask(image, net, patch_size=512, overlap=512, ori_size=None):
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
                this_batch = transform_val(patch_imgs, patch_size)
                data_variable = this_batch.unsqueeze(0)
                if net.parameters().__next__().is_cuda:
                    data_variable = data_variable.cuda(net.parameters().__next__().get_device())
                    result = net(data_variable)
                    result = torch.softmax(result, dim=1)[0,:,:,:].cpu().numpy()
                    result=result.transpose((1,2,0))
                    region_ind = np.argmax(result, axis=-1)
                    try:
                        mask[x:x + patch_size, y:y + patch_size]=region_ind
                    except:
                        r_h,r_w=region_ind.shape[0],region_ind.shape[1]
                        mask[x:x + r_h, y:y + r_w]=region_ind
    map=np.where(mask==0,1,0)#background is 1
    suppress_map = map[:, :, np.newaxis] * 255
    _, RedThresh = cv2.threshold(suppress_map.astype(np.uint8), 160, 255, cv2.THRESH_BINARY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(RedThresh, kernel1)  # 膨胀图像
    eroded = cv2.erode(dilated, kernel2)  # 腐蚀图像
    map = map * (eroded// 255)
    map=map.astype(np.uint8)
    # map=cv2.resize(map,ori_size)
    return map
def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut)
    output_img = np.uint8(output_img + 0.5)  # 这句一定要加上
    return output_img
def auto_threshold(image, image_ori, type=0):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if type == 1:
        img[np.where(img == 0)] = 255
    img = cv2.medianBlur(img, 5)

    th2_ori = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th2 = 255 - th2_ori
    kernel_2 = np.ones((10, 10), np.uint8)
    th2 = cv2.dilate(th2, kernel_2)
    kernel = np.ones((30, 30), np.uint8)  # 设置小
    th2 = cv2.erode(th2, kernel)  # 做一个连tong
    th2 = cv2.dilate(th2, kernel)
    contours = sort_edge(th2)
    area_list = []
    for roi_list in contours:
        area = cv2.contourArea(roi_list)
        area_list.append(area)
    area_list = np.array(area_list, dtype=int)
    area_max = np.max(area_list)
    array = np.zeros((th2.shape[0], th2.shape[1], 3))
    new_contours = []
    for i in range(area_list.size):
        if area_list[i] / area_max >= 0.01:
            cv2.drawContours(array, contours, i, (0, 0, 255), -1)
            new_contours.append(contours[i])
    x = np.array(0.6 * image_ori + 0.4 * array, dtype=int)
    return x, new_contours
def gamma_main(img, result_save_sub_dir, img_name):
    img_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = gamma(img_ori, 0.00000005, 4.0)
    if 'mrxs' in img_name:
        out, new_contours = auto_threshold(out, img_ori, type=1)
        img_name = img_name.replace('mrxs', 'png')

    else:
        out, new_contours = auto_threshold(out, img_ori)
        img_name = img_name.replace('kfb', 'png')
        img_name = img_name.replace('ndpi', 'png')

    # cv2.imwrite(os.path.join(result_save_sub_dir, img_name), out)
    return new_contours
def split_kfb(slides_dir,result_summary, crop_len, C_net, R_net, is_display):
    slide_path, slide_name = os.path.split(slides_dir)
    # degree=slide_name.split('  ')[1]
    # if not (('0' in degree) or ('1' in degree)): return 0
    kfb_name, kfb_ext = os.path.splitext(slide_name)
    result_save_sub_dir = os.path.join(opt.test['WSI_save_dir'],kfb_name)
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
        h,w=thumbnail.shape[0],thumbnail.shape[1]
        for l in range(level, 8):

            cv2.imwrite(os.path.join(result_save_sub_dir, 'level' + str(l)+'WSI.jpg'), cv2.resize(thumbnail,(w//(l-3),h//(l-3))))







def compare_with_annotations( C_net,R_net, is_display=True):
    result_save_dir = opt.test['WSI_save_dir']
    os.makedirs(result_save_dir, exist_ok=True)
    ndpi_list = filesystem.find_ext_files(test_dir, ".ndpi")
    kfb_list = filesystem.find_ext_files(test_dir, '.kfb')
    mrxs_list = filesystem.find_ext_files(test_dir, '.mrxs')

    sdpc_list = filesystem.find_ext_files(test_dir, '.sdpc')
    for slide_path in kfb_list:
        annotation_count_dict = {}
        result_summary = {}
        for key in label_dict.keys():
            annotation_count_dict[key] = 0
            result_summary[key] = []
        annotation_count_dict['细胞总数'] = 0
        result_summary['slide name'] = []
        result_summary['细胞总数'] = []
        #if 'HE' not in image_name and 'NC' not in image_name and 'V-' not in image_name:
        print("evaluating image {}".format(slide_path))
        split_kfb(slide_path, result_summary,1024, C_net,R_net, is_display)

    for slide_path in sdpc_list:
        annotation_count_dict = {}
        result_summary = {}
        for key in label_dict.keys():
            annotation_count_dict[key] = 0
            result_summary[key] = []
        annotation_count_dict['细胞总数'] = 0
        result_summary['slide name'] = []
        result_summary['细胞总数'] = []
        # image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        #if 'HE' not in image_name and 'NC' not in image_name and 'V-' not in image_name:
        print("evaluating image {}".format(slide_path))


        split_kfb(slide_path, result_summary,1024, C_net,R_net, is_display)

    for slide_path in ndpi_list:
        annotation_count_dict = {}
        test_count_dict = {}
        result_summary = {}
        for key in label_dict.keys():
            annotation_count_dict[key] = 0
            test_count_dict[key] = 0
            result_summary[key] = []
        annotation_count_dict['细胞总数'] = 0
        result_summary['slide name'] = []
        result_summary['细胞总数'] = []
        test_count_dict['细胞总数'] = 0
        image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        if 'HE' not in image_name and 'NC' not in image_name:
            print("evaluating image {}".format(slide_path))
            split_kfb(slide_path, 1024, C_net,R_net, test_count_dict, is_display, result_save_dir)

    for slide_path in mrxs_list:
        annotation_count_dict = {}
        result_summary = {}
        for key in label_dict.keys():
            annotation_count_dict[key] = 0
            result_summary[key] = []
        annotation_count_dict['细胞总数'] = 0
        result_summary['slide name'] = []
        result_summary['细胞总数'] = []
        # image_name = str(os.path.split(slide_path)[-1].split('.')[0]) + ".png"

        # if 'HE' not in image_name and 'NC' not in image_name and 'V-' not in image_name:
        print("evaluating image {}".format(slide_path))

        split_kfb(slide_path, result_summary, 1024, C_net,R_net, is_display)


def count_test_summary(image, C_net, cell_count_dict,a_mask):
    center_coords, labels = cal_ki67_np(image, C_net,a_mask=a_mask)
    total_count = 0
    reverse_dict = Reverse_label_dicr
    # print(reverse_dict)
    if len(labels) != 0:
        for i in range(np.max(labels) + 1):
            this_num = np.sum(labels == i)
            cell_count_dict[reverse_dict[i]] = this_num
            total_count += this_num
        cell_count_dict["细胞总数"] = total_count
        # cell_count_dict["PD-L1指数"] = \
        #     round(cell_count_dict["阳性肿瘤细胞"] / (cell_count_dict["阳性肿瘤细胞"] + cell_count_dict["阴性肿瘤细胞"] + 1e-10), 3)
    else:
        cell_count_dict["细胞总数"] = 0
        # cell_count_dict["PD-L1指数"] = 0
    return cell_count_dict, center_coords, labels


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_test(test_dir, C_weight_path,R_weight_path):
    if opt.model['C_name'] == 'FullNet':
        from FullNet import FullNet as detnet
        C_net = detnet(opt.model['in_c'], 10, n_layers=opt.model['n_layers'],
                     growth_rate=opt.model['growth_rate'], drop_rate=opt.model['drop_rate'],
                     dilations=opt.model['dilations'], is_hybrid=opt.model['is_hybrid'],
                     compress_ratio=opt.model['compress_ratio'], layer_type=opt.model['layer_type'])
    elif opt.model['C_name']== 'Unet':
        from attention_unet import U_Net as detnet
        C_net = detnet(opt.model['in_c'], 10)
    elif opt.model['C_name'] == 'Att_Unet':
        from attention_unet import AttU_Net as detnet
        C_net = detnet(opt.model['in_c'], 10)
    elif opt.model['name'] == 'Resnet_Unet':
        from attention_unet import Resnet_Unet
        C_net = Resnet_Unet(opt.model['in_c'], opt.model['out_c'], resnet_pretrain=False)
    elif opt.model['C_name'] == 'FCN_pooling':
        from FullNet import FCN_pooling as detnet
        C_net = detnet(opt.model['in_c'], 10, n_layers=opt.model['n_layers'],
                     growth_rate=opt.model['growth_rate'], drop_rate=opt.model['drop_rate'],
                     dilations=opt.model['dilations'], compress_ratio=opt.model['compress_ratio'],
                     layer_type=opt.model['layer_type'])
    from deeplab.deeplab import DeepLab as R_detnet
    R_net = R_detnet(num_classes=5,
                 backbone='resnet',
                 output_stride=16,
                 sync_bn=True,
                 freeze_bn=False)
    C_net = nn.DataParallel(C_net)
    R_net = nn.DataParallel(R_net)
    if torch.cuda.is_available():
        C_net.cuda()
        R_net.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        print('You are using GPU')
    else:
        print('You are using CPU')

    C_best_checkpoint = torch.load(C_weight_path)
    C_net.load_state_dict(C_best_checkpoint['state_dict'])
    C_net = C_net.module
    C_net.eval()

    R_best_checkpoint = torch.load(R_weight_path)
    R_net.load_state_dict(R_best_checkpoint['state_dict'])
    R_net.eval()
    compare_with_annotations( C_net=C_net,R_net=R_net,is_display=True)


if __name__ == "__main__":
    import os

    global opt
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    C_weight_path='../weights/C_net.pth.tar'
    R_weight_path = '../weights/model_best0418.pth.tar'
    test_dir = "/media/deepin/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/测试用切片/上肿总切片/上肿不含阳性对照"
    run_test(test_dir, C_weight_path=C_weight_path,R_weight_path=R_weight_path)

    # run_test(test_dir, 3,
    #          weight_path= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'pytorch_unet',
    #                                    "Model", "unet_mix_1_2", "weights_epoch_125_1.5490121394395828.pth" ))
