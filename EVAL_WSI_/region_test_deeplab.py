import numpy as np
from imageio import imsave
import openslide
import os, json
import cv2
import kfb.kfbslide as kfbslide
from Slide.openslide_func import openSlide as di_openSlide
from parse_embolus import read_region_kfb
import torchvision.transforms as transforms
import sys
from PIL import Image
from torch.autograd import Variable
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_root)

def get_level_dim_dict(slide_path):
    level_dim_dict = {}
    slide= di_openSlide(slide_path).slide
    dims = slide.level_dimensions
    downsamples = slide.level_downsamples
    for i in range(len(downsamples)):
        level_dim_dict[i] = (dims[i], downsamples[i])
    return level_dim_dict



def generate_region_mask(image, net, patch_size=2048, overlap=512, batch_size=4,image_name=None,save_dir=None):
    # split into 16 patches of size 250x250
    h, w = image.shape[0], image.shape[1]
    h_overlap = overlap
    w_overlap = overlap
    label_color = {"筛状": (255, 0, 0), '粉刺型': (0, 255, 0), '实性型': (0, 0, 255), '淋巴组织': (128, 128, 0),
                   '脂肪': (0, 128, 128)}
    mask=np.ones((h,w))*6
    for x in range(0, h - patch_size + 1, patch_size - h_overlap):
        for y in range(0, w - patch_size + 1, patch_size - w_overlap):
            patch = image[x:x + patch_size, y:y + patch_size, :]
            patch_imgs=Image.fromarray(patch)

            # patch_imgs = patch_imgs * (2. / 255) - 1.
            with torch.no_grad():
                data_variable = transforms.ToTensor()(patch_imgs)
                Norm_ = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                data_variable = Norm_(data_variable)
                data_variable=data_variable.unsqueeze(0)
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


    image_and_label = 0.6 * image
    for i in range(1,5):
        cur_label = np.where(mask==i,1,0)
        cur_label = cur_label[:, :, np.newaxis]
        cur_label = np.repeat(cur_label, 3, -1)
        cur_color = list(label_color.values())[i]
        label_region_color = cur_label * cur_color
        label_region_color=np.asarray(label_region_color,np.uint8)
        image_and_label = image_and_label + 0.4 * label_region_color
    if not image_name==None:
        cv2.imwrite(os.path.join(save_dir,image_name.split('.')[0]+'.png'),image[:,:,::-1])
        cv2.imwrite(os.path.join(save_dir,image_name.split('.')[0]+'_predict'+'.png'),image_and_label[:,:,::-1])



def test_region(slide_path,level,save_dir,net,predix):
    level_dim_dict = get_level_dim_dict(slide_path)
    scale = level_dim_dict[level][1]
    dim = level_dim_dict[level][0]
    if os.path.exists (os.path.join(save_dir,os.path.basename(slide_path)).replace(predix,'.png')):
        return 0
    if 'ndpi' in slide_path or 'mrxs' in slide_path or 'sdpc' in slide_path:
        slide = openslide.OpenSlide(slide_path)
        thumbnail = np.array(
            slide.read_region(location=(0, 0), level = level, size = dim))
        thumbnail = thumbnail[:, :, :3].astype(np.uint8)
    else:
        slide = kfbslide.open_kfbslide(slide_path)
        thumbnail = read_region_kfb(slide,location=(0, 0), level = level, size = dim)
    # thumbnail = slide.read_region(location=(0, 0), level=level, size=dim)
    thumbnail = np.array(thumbnail)
    thumbnail = thumbnail.astype(np.uint8)#thumbnail[:, :, ::-1].astype(np.uint8)
    result_masks=generate_region_mask(thumbnail,net,patch_size=512,overlap=0,image_name=os.path.basename(slide_path),save_dir=save_dir)
    slide.close()
def check_overlap(bboxes, bbox):
    overlap = False
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    # print('bbox', bbox)
    for cur_bbox in bboxes:
        x_left_bound, x_right_bound, y_top_bound, y_bottom_bound = cur_bbox[0], cur_bbox[2], cur_bbox[1], cur_bbox[3]
        if (x1 <= x_right_bound and x1 >= x_left_bound) or (x2 <= x_right_bound and x2 >= x_left_bound):
            if (y1 <= y_bottom_bound and y1 >= y_top_bound) or (y2 <= y_bottom_bound and y2 >= y_top_bound):
                overlap = True
                return overlap
    return overlap

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
    save_dir='../results1206'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    kfb_img_folder='/media/deepin/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/上肿不含阳性对照'
    mrxs_img_folder = '/media/deepin/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/九院-IHC 3D'
    weight_path='/media/deepin/Research/Research/diyingjia/Her2/main/region_exp/region_1008/deeplab/model_best.pth.tar'
    label_dict = {"筛状": 1, '粉刺型': 2, '实性型': 3, '筛状型': 1}
    level=4
    import torch.nn as nn
    import torch
    from deeplab.deeplab import DeepLab as detnet
    from pydaily import filesystem

    best_checkpoint = torch.load(weight_path)
    net = detnet(num_classes=4,
                backbone='resnet',
                output_stride=16,
                sync_bn=True,
                freeze_bn=False)
    net=nn.DataParallel(net)
    if torch.cuda.is_available():
        net.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        print('You are using GPU')
    else:
        print('You are using CPU')
    print("=> loading trained model")

    net.load_state_dict(best_checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(best_checkpoint['epoch']))
    net.eval()
    print('reload detection net weights from {}'.format(weight_path))
    if mrxs_img_folder!=None:
        mrxs_imglist = filesystem.find_ext_files(mrxs_img_folder, ".mrxs")[0:20]
        json_list = filesystem.find_ext_files(mrxs_img_folder, ".json")
        for ind, slide_path in enumerate(mrxs_imglist):
            json_name = slide_path.replace('.mrxs', '.json')
            if json_name  not in json_list:
                test_region(slide_path, level=level, save_dir=save_dir, net=net,predix='.mrxs')
            else:
                continue

    if kfb_img_folder != None:
        kfb_imglist =  filesystem.find_ext_files(kfb_img_folder, ".kfb")[0:20]
        json_list=filesystem.find_ext_files(kfb_img_folder, ".json")
        for ind,slide_path in enumerate(kfb_imglist):
            json_name=slide_path.replace('.kfb','.json')
            if json_name  not in json_list:
                test_region(slide_path, level=level, save_dir=save_dir, net=net,predix='.kfb')

            else:
                continue




