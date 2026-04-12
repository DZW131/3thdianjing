import os
from options import Options
import pickle
import numpy as np
import pandas as pd
ano_txt='/media/deepin/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/HER2_result.txt'
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
piclke_root_dir='/media/deepin/DeepInformatic_dataset/lihansheng/HER2/Exp_v2/jiuyuan'
result_save_dir='/media/deepin/DeepInformatic_dataset/lihansheng/HER2/Exp_v2/jiuyuan'
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
            continue
        else:
            total_num+=a_wsi_result[key]
    cell_count1 = float(a_wsi_result['微弱的不完整膜阳性肿瘤细胞'])
    cell_count2 = float(a_wsi_result['弱-中等的完整细胞膜阳性肿瘤细胞'])
    cell_count3 = float(a_wsi_result['中-强度的不完整细胞膜阳性肿瘤细胞'])
    cell_count4 = float(a_wsi_result['强度的完整细胞膜阳性肿瘤细胞'])
    cell_count5 = cell_count2 + cell_count4  # 完整细胞膜阳性肿瘤细胞
    cell_count6 = cell_count1 + cell_count2 + cell_count3 + cell_count4  # 阳性肿瘤细胞
    cell_count7 = float(a_wsi_result['阴性肿瘤细胞'])
    cell_count8 = cell_count6 + cell_count7
    if ((cell_count6 / total_num) >= 0.1):
        if ((cell_count4 / total_num) >= 0.1):
            return 3
        elif ((cell_count5 / total_num) >= 0.1):
            return 2
        elif ((cell_count4 / total_num) <= 0.1):
            return 2
    elif ((cell_count1+cell_count3)/total_num>0.1) and ((cell_count5/total_num)<0.1):
        return 1
    elif (((cell_count1+cell_count3)/total_num)<=0.1) or ((cell_count6/total_num)<=0.1):
        return 0
    else:
        return -1
def main(result_save_dir):
    Pkl_list = getFileList(piclke_root_dir, [], 'pkl')
    result_summary = {}
    for key in label_dict.keys():
        result_summary[key] = []
    #result_summary['细胞总数'] = []
    result_summary['slide name'] = []
    result_summary['label'] = []
    result_summary['predict'] = []
    her_2_score = 0
    for a_wsi_dir in Pkl_list:

        wsi_name=os.path.basename(a_wsi_dir).split('.pkl')[0]
        result_summary['slide name'].append(wsi_name)
        pkl_file=open(a_wsi_dir,'rb')
        a_wsi_result=pickle.load(pkl_file)
        her_2_score=cal_score(a_wsi_result)
        label=-1
        with open(ano_txt,'r') as f:
            for line in f.readlines():
                ID=line.strip().split('\t')[0]
                if ID in wsi_name:
                    try:
                        label=line.strip().split('\t')[3]
                    except:
                        label=-1
        result_summary['label'].append(label)
        result_summary['predict'].append(her_2_score)
        for key in a_wsi_result.keys():
            if key=='细胞总数':
               continue
            else:
                result_summary[key].append(a_wsi_result[key])
    df = pd.DataFrame(result_summary)
    columns = ['slide name', 'label','predict','微弱的不完整膜阳性肿瘤细胞', '弱-中等的完整细胞膜阳性肿瘤细胞', '阴性肿瘤细胞', '纤维细胞', '淋巴细胞', '难以区分的非肿瘤细胞', '组织细胞',
               '强度的完整细胞膜阳性肿瘤细胞', '中-强度的不完整细胞膜阳性肿瘤细胞']
    df.to_csv(os.path.join(result_save_dir, "summary_{}.csv".format('her2')),
              columns=columns, index=False)

main(result_save_dir)