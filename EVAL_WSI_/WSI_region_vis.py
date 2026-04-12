import numpy as np
from imageio import imsave
import openslide
import os, json
import cv2
import kfb.kfbslide as kfbslide
from Slide.openslide_func import openSlide as di_openSlide
from parse_embolus import read_region_kfb
import random
import deepdish as dd
# slide = openslide.OpenSlide(slide_path)
# print(type(slide))
def get_properties(slide_path):
    with openslide.OpenSlide(slide_path) as slide:
        ## level_count: 这张图片有几个级别的分辨率, 0 表示最高分辨率, 最低分辨率
        print('level_count', slide.level_count, '\n')
        ## dimensions: level 为 0 时的 (width, height), 也就是最高分辨率的情况下 slide 的宽和高（元组）
        print('dimensions', slide.dimensions, '\n')
        ## level_dimensions: 每个 level 的 (width, height)
        print('level_dimensions', slide.level_dimensions, '\n')
        ## level_downsamples: 每个 level 下采样的倍数, 相对于 level 0, 即 level_dimension[k] = dimensions / level_downsamples[k]
        print('level_downsamples', slide.level_downsamples, '\n')
        ## associated_images: 也是 metadata, 不过 dict 的值都是一张 pil 图片.
        print('associated_images', slide.associated_images, '\n')
        ## whole-slide 的 metadata, 是一个类似 dict 的对象, 其值都是 字符串
        proper = slide.properties
        print(proper['mirax.DATAFILE.FILE_0'])

def get_level_dim_dict(slide_path):
    level_dim_dict = {}
    slide= di_openSlide(slide_path).slide
    dims = slide.level_dimensions
    downsamples = slide.level_downsamples
    for i in range(len(downsamples)):
        level_dim_dict[i] = (dims[i], downsamples[i])
    return level_dim_dict

def read_region(slide_path):
    ## 其中 location 是读取区域的左上角在 level 0 中的坐标，level 表示我们要读取的是第几个 level 的图片，
    # size 是 (width, height), 返回的是 PIL.Image.
    # 注意：不管 level 是不是 0，location 的定位都是根据 level 0 来的，而 size 是在不同 level 上选取的。
    with openslide.OpenSlide(slide_path) as slide:
        region = slide.read_region((47712, 94343), 5, (512, 512))
        region = np.array(region)
        imsave('/data2/Caijt/WSI_ROI/region.png', region)

def get_contours(slide_path,level,ext='.kfb'):
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
            remark=label_dict[remark_name]
        except:
            continue
        x_coords_list = path["x"]
        y_coords_list = path["y"]
        corrds = list(zip(x_coords_list, y_coords_list))
        roi_contours.append(corrds)
        roi_labels.append(remark)
    f.close()
    return roi_contours, roi_labels


def best_level_downsample(slide_path):
    with openslide.OpenSlide(slide_path) as slide:
        for i in range(10, 500, 25):
            print("对于下采样%d倍，其最好的level是%d" % (i, slide.get_best_level_for_downsample(i)))

def get_thumbnail(slide_path, size = (1920, 1920)):
    ## size 是缩略图的 (width, height)，注意到，这个缩略图是保持比例的，
    ## 所以其会将 width 和 height 中最大的那个达到指定的值，另一个等比例缩放。
    with openslide.OpenSlide(slide_path) as slide:
        thumbnail = slide.get_thumbnail(size)
        thumbnail = np.array(thumbnail)
        imsave('/data2/Caijt/WSI_ROI/thumbnail.png', thumbnail)
def split_patches(image, mask,save_dir,label_count_dict,save_ind):
    """ split large image into small patches """
    seg_imgs = []
    # split into 16 patches of size 250x250
    h, w = image.shape[0], image.shape[1]
    patch_size = 512
    h_overlap = 150
    w_overlap = 150
    f_index=0
    label_color = {"筛状": (255, 0, 0), '粉刺型': (0, 255, 0), '实性型': (0, 0, 255), '淋巴组织': (128, 128, 0),
                   '脂肪': (0, 128, 128)}

    #show the labels
    cv2.imwrite('origion.jpg', image)
    image_and_label = 0.9 * image
    mask[:, :, 0] = 255 - np.sum(mask[:, :, 1::], axis=2)
    for i in range(3):
        cur_label = mask[:, :, i]
        cur_label = cur_label[:, :, np.newaxis]//255
        cur_label = np.repeat(cur_label, 3, -1)
        cur_color = list(label_color.values())[i]
        label_region_color = cur_label * cur_color
        image_and_label = image_and_label + 0.1 * label_region_color
    cv2.imwrite('label.jpg', image_and_label)
    f_train=open(os.path.join(save_dir, 'ImageSets','Segmentation','train.txt'),'a')
    f_val = open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'val.txt'), 'a')
    f_test = open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'test.txt'), 'a')
    #split data
    for x in range(0, h-patch_size+1, patch_size-h_overlap):
        for y in range(0, w-patch_size+1, patch_size-w_overlap):
                patch = image[x:x+patch_size, y:y+patch_size, :]
                if np.sum(patch)<20:continue
                patch_label=mask[x:x+patch_size, y:y+patch_size, :]
                if np.sum(patch_label[:,:,1::])<2000:
                    continue
                patch_label_class=np.argmax(patch_label,axis=-1)
                keys=list(label_dict.keys())
                for cls_ind in list(np.unique(patch_label_class)):
                    if cls_ind==0:continue
                    else:
                        print('extract %s in %d WSI'%(keys[cls_ind-1],save_ind))
                cv2.imwrite(os.path.join(save_dir, 'JPEGImages', "%s_%d.jpg" % (save_ind, f_index)), patch)
                cv2.imwrite(os.path.join(save_dir, 'SegmentationClass', "%s_%d.png" % (save_ind, f_index)), patch_label_class)
                if f_index%4==0:
                    f_val.write("%s_%d\n" % (save_ind, f_index))
                    f_test.write("%s_%d\n" % (save_ind, f_index))

                else:
                    f_train.write("%s_%d\n" % (save_ind, f_index))
                f_index+=1
    f_train.close()
    f_test.close()
    f_val.close()
def vis_contour(slide_path, roi_contours, roi_labels, level_dim_dict, alpha = 1000000, level = 5):
    base_name = os.path.basename(slide_path).split('.')[0]
    dim = level_dim_dict[0][0]
    scale = level_dim_dict[level][1]
    for i, roi_contour in enumerate(roi_contours):
        roi_contour = np.array(roi_contour)
        roi_label = roi_labels[i]
        # print('countour_shape', roi_contour.shape)
        left_coord = np.maximum(min(roi_contour[:, 0]), 0)
        top_coord = np.maximum(min(roi_contour[:, 1]), 0)
        # print('start_point', left_coord, top_coord)
        right_coord = np.minimum(max(roi_contour[:, 0]), dim[0])
        bottom_coord = np.minimum(max(roi_contour[:, 1]), dim[1])
        height = bottom_coord - top_coord
        width = right_coord - left_coord
        # print('height, width', height, width)
        try:
            with openslide.OpenSlide(slide_path) as slide:
                region = slide.read_region((left_coord, top_coord), level, (int(width/scale), int(height/scale)))
                resized_x_coords = list(map(int, ((roi_contour[:, 0] - left_coord) / scale).tolist()))
                resized_y_coords = list(map(int, ((roi_contour[:, 1] - top_coord) / scale).tolist()))
                coords = np.concatenate((np.expand_dims(np.expand_dims(np.array(resized_x_coords), axis= -1), axis = -1),
                                         np.expand_dims(np.expand_dims(np.array(resized_y_coords), axis= -1), axis = -1)),
                                        axis=-1)
                # print('coords', coords)
                region = np.array(region)[:, :, :3]
                region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
                # print('region_shape', region.shape)
                # region = np.int8(np.array(region))
                # cv2.drawContours(region, coords, -1, (0, 255, 0), 5)
                area = cv2.contourArea(coords)
                if area < 70000:
                    print('< 7 * 1e4, invalid scale')
                    sample_num = 0
                elif area < 1000000:
                    sample_num = 5 * np.log2(10 * area / alpha)
                    print('7 * 1e4 ~ 1e6')
                elif area < 10000000:
                    sample_num = 50 * np.log2(area / alpha)
                    print('1e6 ~ 1e7')
                else:
                    sample_num = 10 * (1+np.log2(area/(alpha*10)))
                    print(' > 1e7')
                print('area, sample_num', area, sample_num)
                if sample_num > 0:
                    bboxes = sample_patches(region, sample_num, coords, roi_label, base_name, i)
                    for bbox in bboxes:
                        cv2.rectangle(region, (bbox[0], bbox[1]),(bbox[2], bbox[3]), (255, 0, 0), 5)

                    cv2.imwrite('/data2/Caijt/WSI_ROI/vis_contour_v4/vis_contour_{}_{}.png'.format(base_name, i), region)
            # pdb.set_trace()
        except:
            pass

def vis_anno(slide_path,roi_contours, roi_labels,level,save_dir,index):
    level_dim_dict = get_level_dim_dict(slide_path)
    scale = level_dim_dict[level][1]
    dim = level_dim_dict[level][0]
    _, slide_name = os.path.split(slide_path)
    # degree=slide_name.split('  ')[1]
    # if not (('0' in degree) or ('1' in degree)): return 0
    kfb_name, kfb_ext = os.path.splitext(slide_name)
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
    thumbnail = thumbnail[:, :, ::-1].astype(np.uint8)


    label_color=[(255, 0, 0),(255, 0, 0),(255, 0, 0),(255, 0, 0),(0, 0, 255),]
    cur_mask_forshow = thumbnail.copy()
    for i, roi_contour in enumerate(roi_contours):
        coords_np_list = []
        # print(roi_contour)
        # print('--------------------------------------------------------------')
        x_coord = np.array([int(contour[0]/ 2**level) for contour in roi_contour])
        y_coord = np.array([int(contour[1]/2**level) for contour in roi_contour])
        # print(x_coord)
        coords_np = np.concatenate((x_coord[:, np.newaxis],y_coord[:, np.newaxis]), axis= 1)
        # print(coords_np.shape)
        coords_np_list.append(coords_np)
        label = roi_labels[i]
        color_map=label_color[label]
        cur_mask_forshow = cv2.drawContours(cur_mask_forshow, coords_np_list, -1, color_map, 2)
    cv2.imwrite(os.path.join(save_dir,kfb_name+'.png'), cur_mask_forshow)


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

def sample_patches(region, sample_num, contour, roi_label, base_name, i):
    bboxes = []
    size_w, size_h = 256, 256
    margin = 30
    num = 0
    sample_times = 0
    height, width = region.shape[0], region.shape[1]
    print('selection')
    # print('height, width', height, width)
    while num < sample_num and sample_times < 100000:
        sample_times += 1
        start_x = random.randrange(0, width)
        start_y = random.randrange(0, height)
        start_point = (start_x, start_y)
        top_left = start_point
        top_right = (start_x + size_w, start_y)
        bottom_left = (start_x, start_y + size_h)
        bottom_right = (start_x + size_w, start_y + size_h)
        signed_dist0 = cv2.pointPolygonTest(contour, top_left, True)
        signed_dist1 = cv2.pointPolygonTest(contour, top_right, True)
        signed_dist2 = cv2.pointPolygonTest(contour, bottom_left, True)
        signed_dist3 = cv2.pointPolygonTest(contour, bottom_right, True)
        if (signed_dist0 > margin) and (signed_dist1 > margin) and \
                (signed_dist2 > margin) and (signed_dist3 > margin):
            x1, y1, x2, y2= top_left[0], top_left[1], bottom_right[0], bottom_right[1]
            bbox = (x1, y1, x2, y2)
            print('here')
            if not check_overlap(bboxes, bbox):
                bboxes.append(bbox)
                cv2.imwrite('/data2/Caijt/WSI_ROI/patches_v4/{}/vis_patches_{}_{}_{}.png'.format(roi_label, base_name, i, num),
                            region[y1:y2, x1:x2])
                num += 1
                print('congratulations!!!')
    return bboxes
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
    save_dir='/media/deepin/DeepInformatic_dataset/lihansheng/HER2/labeled_region'
    kfb_img_folder=None#'/media/deepin/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/上肿'
    mrxs_img_folder = '/media/deepin/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/九院-IHC 3D'
    # label_dict = {"筛状": 0, '粉刺型': 1, '实性型': 2, '淋巴组织': 3, '脂肪': 4, '筛状型': 0,'淋巴细胞':3}
    label_dict = {"筛状": 1, '粉刺型': 2, '实性型': 3, '筛状型': 1,'正常导管':4}
    os.makedirs(save_dir,exist_ok=True)
    level=4
    index=0
    from pydaily import filesystem
    mrxs_imglist = filesystem.find_ext_files(mrxs_img_folder, ".mrxs")
    json_list = filesystem.find_ext_files(mrxs_img_folder, ".json")
    for ind, slide_path in enumerate(mrxs_imglist):
        json_name = slide_path.replace('.mrxs', '.json')
        if json_name not in json_list:
            continue
        else:
            level_dim_dict = get_level_dim_dict(slide_path)
            roi_contours, roi_labels = get_contours(slide_path, level,ext='.mrxs')
            vis_anno(slide_path, roi_contours, roi_labels, level=level, save_dir=save_dir, index=index)
            index += 1

    kfb_imglist =  filesystem.find_ext_files(kfb_img_folder, ".kfb")
    json_list=filesystem.find_ext_files(kfb_img_folder, ".json")
    for ind,slide_path in enumerate(kfb_imglist):
        json_name=slide_path.replace('.kfb','.json')
        if json_name not in json_list:
            continue
        else:
            level_dim_dict = get_level_dim_dict(slide_path)
            roi_contours, roi_labels= get_contours(slide_path,level)
            vis_anno(slide_path,roi_contours, roi_labels,level=level,save_dir=save_dir,index=index)
            index += 1



