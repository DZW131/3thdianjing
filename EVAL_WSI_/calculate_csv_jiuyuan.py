import os
from options import Options
import pickle
import numpy as np
import csv
import pandas as pd

label_file = ''
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
piclke_root_dir = '/media/deepin/DeepInformatic_dataset/lihansheng/HER2/0317/hangzhong'
result_save_dir = '/media/deepin/DeepInformatic_dataset/lihansheng/HER2/Exp_v2/0322_result'
os.makedirs(result_save_dir,exist_ok=True)
name='shangzhong_2'
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
    total_num = 0

    cell_count1 = float(a_wsi_result['微弱的不完整膜阳性肿瘤细胞'])
    cell_count2 = float(a_wsi_result['弱-中等的完整细胞膜阳性肿瘤细胞'])
    cell_count3 = float(a_wsi_result['中-强度的不完整细胞膜阳性肿瘤细胞'])
    cell_count4 = float(a_wsi_result['强度的完整细胞膜阳性肿瘤细胞'])
    cell_count5 = cell_count2 + cell_count4  # 完整细胞膜阳性肿瘤细胞
    cell_count6 = cell_count1 + cell_count2 + cell_count3 + cell_count4  # 阳性肿瘤细胞
    cell_count7 = float(a_wsi_result['阴性肿瘤细胞'])
    cell_count8 = cell_count6 + cell_count7
    if (cell_count4 / cell_count8) >= 0.1:
        return 3,'强度的完整细胞膜阳性肿瘤细胞占比%.4f'%(cell_count4 / cell_count8)
    elif (cell_count5 / cell_count8) >= 0.1:
        return 2,'完整细胞膜阳性肿瘤细胞%.4f'%(cell_count5 / cell_count8)
    elif (cell_count4 / cell_count8) >= 0.005:
        return 2,'强度的完整细胞膜阳性肿瘤细胞占比%.4f'%(cell_count4 / cell_count8)
    if (cell_count2 / cell_count8) > 0.01:
        return 1,'弱-中等的完整细胞膜阳性肿瘤细胞占比%.4f,包含较多完整染色细胞'%(cell_count4 / cell_count8)
    elif ((cell_count1 + cell_count3) / cell_count8) >= 0.1 :
        return 1,'不完整膜阳性肿瘤细胞占比%.4f,包含较多染色细胞'%((cell_count1 + cell_count3) / cell_count8)
    elif ((cell_count1 + cell_count3) / cell_count8) < 0.1:
        return 0,'阴性肿瘤细胞占比%.4f'%(cell_count7 / cell_count8)
    elif (cell_count6 / cell_count8) < 0.01:
        return 0,'阴性肿瘤细胞占比%.4f'%(cell_count7 / cell_count8)
    else:
        return -1

def cal_score_reverse(a_wsi_result):
    total_num = 0
    for key in a_wsi_result:
        if key == '细胞总数':
            total_num = a_wsi_result['细胞总数']
    cell_count1 = float(a_wsi_result['微弱的不完整膜阳性肿瘤细胞'])
    cell_count2 = float(a_wsi_result['弱-中等的完整细胞膜阳性肿瘤细胞'])
    cell_count3 = float(a_wsi_result['中-强度的不完整细胞膜阳性肿瘤细胞'])
    cell_count4 = float(a_wsi_result['强度的完整细胞膜阳性肿瘤细胞'])
    cell_count5 = cell_count2 + cell_count4  # 完整细胞膜阳性肿瘤细胞
    cell_count6 = cell_count1 + cell_count2 + cell_count3 + cell_count4  # 阳性肿瘤细胞
    cell_count7 = float(a_wsi_result['阴性肿瘤细胞'])
    cell_count8 = cell_count6 + cell_count7

    if (cell_count6 / cell_count8) < 0.01:
        return 0,'阳性肿瘤细胞占比%.4f'%(cell_count6 / cell_count8)
    elif ((cell_count1+cell_count3) / cell_count8) <=0.1:
        return 0,'不完整膜阳性肿瘤细胞占比%.4f,包含较shao染色细胞'%((cell_count1 + cell_count3) / cell_count8)

    if ((cell_count2 / cell_count8) >= 0.01) and ((cell_count2 / cell_count8) <=0.1) :
        return 1, '弱-中等的完整细胞膜阳性肿瘤细胞占比%.4f,包含较多完整染色细胞' % (cell_count4 / cell_count8)

    elif (((cell_count1 + cell_count3) / cell_count8) >= 0.1) and ((cell_count5/cell_count8)<0.1):
        return 1, '不完整膜阳性肿瘤细胞占比%.4f,包含较多染色细胞' % ((cell_count1 + cell_count3) / cell_count8)

    if ((cell_count5 / cell_count8) >= 0.1) and ((cell_count4/cell_count8)<0.1):
        return 2,'完整细胞膜阳性肿瘤细胞%.4f'%(cell_count5 / cell_count8)
    elif ((cell_count4 / cell_count8) >= 0.005) and ((cell_count4 / cell_count8) < 0.1):
        return 2,'强度的完整细胞膜阳性肿瘤细胞占比%.4f'%(cell_count4 / cell_count8)

    if (cell_count4 / cell_count8) >= 0.1:
        return 3, '强度的完整细胞膜阳性肿瘤细胞占比%.4f' % (cell_count4 / cell_count8)

    return -1,'Hard Example'
def read_csv(csv_path):
    summary = {}
    f = csv.reader(open(csv_path, 'r'))
    csv_sub_path, slide_name = os.path.split(csv_path)
    kfb_name, kfb_ext = os.path.splitext(slide_name)
    kfb_name = kfb_name.replace('summary_', '')
    cur_line = 0
    keys = None
    for i in f:
        if cur_line == 0:
            for key in i:
                if key == 'slide name':
                    summary[key] = kfb_name
                else:
                    summary[key] = 0
            keys = list(summary.keys())
            cur_line += 1
        else:
            for ind, value in enumerate(i):
                if ind == 0:
                    continue
                else:
                    summary[keys[ind]] += int(value.split('[')[1].split(']')[0])
    return summary


def main(result_save_dir):
    Pkl_list = getFileList(piclke_root_dir, [], 'csv')
    result_summary = {}
    for key in label_dict.keys():
        result_summary[key] = []
    # result_summary['细胞总数'] = []
    result_summary['slide name'] = []
    result_summary['label'] = []
    result_summary['predict'] = []
    result_summary['原因']=[]
    for a_wsi_dir in Pkl_list:
        if 'her2' in a_wsi_dir: continue
        wsi_name = os.path.basename(a_wsi_dir).split('.csv')[0]
        result_summary['slide name'].append(wsi_name)
        a_wsi_result = read_csv(a_wsi_dir)
        her_2_score1,reson1 = cal_score(a_wsi_result)
        her_2_score2,reson2=cal_score_reverse(a_wsi_result)
        if her_2_score1==her_2_score2:
            her_2_score=her_2_score2
            reson = reson1
        elif her_2_score1>her_2_score2:
            her_2_score=her_2_score1
            reson=reson1
        else:
            her_2_score = her_2_score2
            reson = reson2
        label = wsi_name.split(' ')[-1]
        if '+' in label:
            label = label.replace('+', '')
        result_summary['label'].append(label)
        result_summary['predict'].append(her_2_score)
        result_summary['原因'].append(reson)
        for key in a_wsi_result.keys():
            if key == '细胞总数':
                continue
            if key == 'slide name':
                continue

            result_summary[key].append(a_wsi_result[key])
    df = pd.DataFrame(result_summary)
    columns = ['slide name', 'label', 'predict', '微弱的不完整膜阳性肿瘤细胞', '弱-中等的完整细胞膜阳性肿瘤细胞', '阴性肿瘤细胞', '纤维细胞', '淋巴细胞',
               '难以区分的非肿瘤细胞', '组织细胞',
               '强度的完整细胞膜阳性肿瘤细胞', '中-强度的不完整细胞膜阳性肿瘤细胞','原因']
    df.to_csv(os.path.join(result_save_dir, "{}.csv".format(name)),
              columns=columns, index=False,encoding='gbk')


main(result_save_dir)
