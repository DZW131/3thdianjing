import os
from options import Options
import pickle
import numpy as np
import csv
import pandas as pd
import shutil
label_file=''
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
piclke_root_dir='/media/deepin/DeepInformatic_dataset/lihansheng/HER2/Exp_v2/jiuyuan_new'
result_save_dir='/media/deepin/DeepInformatic_dataset/lihansheng/HER2/Exp_v2/jiuyuan_new_filtered '
os.makedirs(result_save_dir,exist_ok=True)
global opt
opt = Options(isTrain=False)
opt.parse()
opt.save_options()
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
def cal_score(a_wsi_result):
    total_num=0
    for key in a_wsi_result:
        if key=='细胞总数':
            total_num=a_wsi_result['细胞总数']
    cell_count1 = float(a_wsi_result['微弱的不完整膜阳性肿瘤细胞'])
    cell_count2 = float(a_wsi_result['弱-中等的完整细胞膜阳性肿瘤细胞'])
    cell_count3 = float(a_wsi_result['中-强度的不完整细胞膜阳性肿瘤细胞'])
    cell_count4 = float(a_wsi_result['强度的完整细胞膜阳性肿瘤细胞'])
    cell_count5 = cell_count2 + cell_count4  # 完整细胞膜阳性肿瘤细胞
    cell_count6 = cell_count1 + cell_count2 + cell_count3 + cell_count4  # 阳性肿瘤细胞
    cell_count7 = float(a_wsi_result['阴性肿瘤细胞'])
    cell_count8 = cell_count6 + cell_count7
    if (cell_count4/cell_count8)>0.1:
        return 3
    elif (cell_count5/cell_count8)>0.1 and (cell_count4/cell_count8)<0.1:
        return 2
    elif ((cell_count1+cell_count3)/cell_count8)>0.05 and (cell_count5/cell_count8)<0.1:
        return 1
    elif  ((cell_count1+cell_count3)/cell_count8)<0.1:
        return 0
    elif (cell_count6/cell_count8)<0.01:
        return 0
    else:
        return -1

def read_csv(csv_path):
    summary={}
    f = csv.reader(open(csv_path, 'r'))
    csv_sub_path, slide_name = os.path.split(csv_path)
    kfb_name, kfb_ext = os.path.splitext(slide_name)
    save_path=os.path.join(result_save_dir,kfb_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path,exist_ok=True)
    kfb_name = kfb_name.replace('summary_', '')
    cur_line = 0
    keys=None
    copy_cout=0
    for i in f:
        if copy_cout>2:
            continue
        if cur_line==0:
            for key in i:
                if key=='slide name':
                    summary[key]=kfb_name
                else:
                    summary[key]=0
            keys=list(summary.keys())
            cur_line += 1
        else:
            name=None
            count=0
            for ind,value in enumerate(i):
                if ind==0:
                    name=value
                else:
                    if '肿瘤细胞' in keys[ind]:
                        count+=int(value.split('[')[1].split(']')[0])
                    else:
                        continue
            if count>100:
                shutil.copy(os.path.join(csv_sub_path,name),os.path.join(save_path,name))
                copy_cout+=1





def main(result_save_dir):
    Pkl_list = getFileList(piclke_root_dir, [], 'csv')
    result_summary = {}
    for key in label_dict.keys():
        result_summary[key] = []
    #result_summary['细胞总数'] = []
    result_summary['slide name'] = []
    result_summary['predict'] = []
    for a_wsi_dir in Pkl_list:
        if 'her2' in a_wsi_dir:continue
        wsi_name=os.path.basename(a_wsi_dir).split('.csv')[0]
        result_summary['slide name'].append(wsi_name)
        read_csv(a_wsi_dir)
    #     her_2_score=cal_score(a_wsi_result)
    #     result_summary['predict'].append(her_2_score)
    #     for key in a_wsi_result.keys():
    #         if key=='细胞总数':
    #            continue
    #         if key=='slide name':
    #            continue
    #
    #         result_summary[key].append(a_wsi_result[key])
    # df = pd.DataFrame(result_summary)
    # columns = ['slide name','predict','微弱的不完整膜阳性肿瘤细胞', '弱-中等的完整细胞膜阳性肿瘤细胞', '阴性肿瘤细胞', '纤维细胞', '淋巴细胞', '难以区分的非肿瘤细胞', '组织细胞',
    #            '强度的完整细胞膜阳性肿瘤细胞', '中-强度的不完整细胞膜阳性肿瘤细胞']
    # df.to_csv(os.path.join(result_save_dir, "summary_{}.csv".format('her2')),
    #           columns=columns, index=False)

main(result_save_dir)
