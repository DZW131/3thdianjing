import numpy as np
from imageio import imsave
import openslide
import os, json

# print(type(slide))

def get_contours(slide_path,ext='.kfb'):
    dir_name = os.path.dirname(slide_path)
    index_file = slide_path.replace(ext,'.json')
    roi_contours = []
    roi_labels = []

    try:
        f=open(index_file,'r')
        info_dict = json.load(f)
    except:
        try:
            f=open(index_file,'r', encoding='gbk')
            info_dict = json.load(f)
        except:
            try:
                f = open(index_file, 'r', encoding='GBK')
                info_dict = json.load(f)
            except:
                try:
                    f=open(index_file,'r', encoding='utf-8')
                    info_dict = json.load(f)
                except:
                    raise Exception("encoding error")

    roilist = info_dict['roilist']
    print('{} roi regions are labeled in '.format(len(roilist)))
    for i, roi_dict in enumerate(roilist):
        path = roi_dict["path"]
        try:
            remark_name = roi_dict["remark"]
        except:
            continue
        try:
            label_count_dict[remark_name]+=1
        except:
            continue



def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist

if __name__ == '__main__':
    # get_properties(slide_path)
    kfb_img_folder='/media/deepin/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/上肿'
    mrxs_img_folder = '/media/deepin/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/九院-IHC 3D'
    label_count_dict = {"筛状": 0, '粉刺型': 0, '实性型': 0, '淋巴组织': 0, '脂肪': 0, '筛状型': 0,'淋巴细胞':0}
    # label_dict = {"筛状": 1, '粉刺型': 2, '实性型': 3,'筛状型': 1}

    from pydaily import filesystem


    mrxs_imglist = filesystem.find_ext_files(mrxs_img_folder, ".mrxs")
    json_list = filesystem.find_ext_files(mrxs_img_folder, ".json")
    for ind, slide_path in enumerate(mrxs_imglist):
        json_name = slide_path.replace('.mrxs', '.json')
        if json_name not in json_list:
            continue
        else:

            get_contours(slide_path,ext='.mrxs')

    kfb_imglist =  filesystem.find_ext_files(kfb_img_folder, ".kfb")
    json_list=filesystem.find_ext_files(kfb_img_folder, ".json")
    for ind,slide_path in enumerate(kfb_imglist):
        json_name=slide_path.replace('.kfb','.json')
        if json_name not in json_list:
            continue
        else:
            get_contours(slide_path)

    print(label_count_dict)


