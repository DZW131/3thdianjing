import numpy as np
from imageio import imsave
import openslide
import os, json
import cv2
import kfb.kfbslide as kfbslide
from Slide_0.openslide_func import openSlide as di_openSlide
from parse_embolus import read_region_kfb
import random
from tqdm import tqdm
import deepdish as dd
# slide = openslide.OpenSlide(slide_path)
# print(type(slide))
from Slide.dispatch import openSlide

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

def get_contours(slide_path, ext='.kfb'):
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

    roilist = info_dict['annotation']
    print('{} roi regions are labeled in '.format(len(roilist)))
    for i, roi_dict in enumerate(roilist):
        path = roi_dict['position']
        try:
            remark_name = roi_dict["label"][0]
        except:
            continue
        try:
            remark=label_dict[remark_name]
        except:
            continue
        x_coords_list = path["x"]
        y_coords_list = path["y"]
        corrds = list(zip(x_coords_list, y_coords_list))

        corrds_np = np.array(corrds, dtype=np.int32)

        roi_contours.append(corrds_np)
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

def get_keys(d, value):
    List = [k for k,v in d.items() if v == value]
    if len(List) == 0:
        List = ["背景"]
    return List

def split_patches(slide, mask, resize_ratio, save_dir, save_ind, label_count_dict):
    """ split large image into small patches """
    seg_imgs = []
    # split into 16 patches of size 250x250

    vis_resize_ratio = slide.mpp/vis_mpp
    vis_h, vis_w = int(slide.height*vis_resize_ratio), int(slide.width*vis_resize_ratio)

    thumbnail = np.zeros((int(slide.height*resize_ratio), int(slide.width*resize_ratio), 3), dtype=np.uint8)
    thumbnail[:, :, :] = 255
    interval = 10000
    fist_size = [0, 0]
    for read_y in range(0, slide.height, interval):
        for read_x in range(0, slide.width, interval):
            read_w = interval if slide.width-read_x > interval else slide.width-read_x
            read_h = interval if slide.height-read_y > interval else slide.height-read_y
            temp_img = slide.read(location=[read_x, read_y], size=[read_w, read_h], scale=1 / resize_ratio)

            # fortrain_interval_w = int(read_w * resize_ratio)
            # fortrain_interval_h = int(read_h * resize_ratio)
            # vis_x = int(fortrain_x * resize_ratio)
            # vis_y = int(fortrain_y * resize_ratio)

            # 获取读到的第一个滑块的尺寸
            if read_x == 0 and read_y == 0:
                fist_size[0] = temp_img.shape[1]
                fist_size[1] = temp_img.shape[0]

            # 根据读取出来的patch来滑块，而不是直接滑块，防止滑块中间有断层
            fortrain_interval_w = temp_img.shape[1]
            fortrain_interval_h = temp_img.shape[0]
            fortrain_x = int(int(read_x * resize_ratio)/fist_size[1] + 0.5) * fist_size[1]
            fortrain_y = int(int(read_y * resize_ratio)/fist_size[0] + 0.5) * fist_size[0]

            # 防止滑块超出画布
            cur_patch_h = min(thumbnail.shape[0], fortrain_y+fortrain_interval_h) - fortrain_y
            cur_patch_w = min(thumbnail.shape[1], fortrain_x+fortrain_interval_w) - fortrain_x
            thumbnail[fortrain_y:fortrain_y+cur_patch_h, fortrain_x:fortrain_x+cur_patch_w, :] = temp_img[:cur_patch_h, :cur_patch_w, :]

            vis_image = cv2.resize(thumbnail, dsize=(vis_w, vis_h))
            cv2.imwrite(os.path.join(save_dir, 'VisWSI', f"{slide_name}_patch.jpg"), vis_image[:, :, ::-1])

    # image = slide.read(location=[0, 0], size=None, scale=1/resize_ratio)
    h, w = mask.shape[0], mask.shape[1]

    #show the labels
    temp_mask = 0.4 * (1-mask[:, :, 0]/255)
    temp_mask[temp_mask == 0] = 1
    temp_mask = np.repeat(temp_mask[:, :, np.newaxis], 3, axis=-1)
    image_and_label = temp_mask * thumbnail
    for i in range(label_num):
        if i==0: continue
        cur_label = mask[:, :, i]
        cur_label = cur_label[:, :, np.newaxis]//255
        cur_label = np.repeat(cur_label, 3, -1)
        cur_color = list(label_color.values())[i-1]
        label_region_color = cur_label * cur_color
        image_and_label = image_and_label + 0.6 * label_region_color
    # cv2.imwrite('label.jpg', image_and_label)
    vis_image = cv2.resize(thumbnail, dsize=(vis_w, vis_h))
    image_and_label = cv2.resize(image_and_label, dsize=(vis_w, vis_h))
    vis_mask = cv2.resize(mask, dsize=(vis_w, vis_h))
    cv2.imwrite(os.path.join(save_dir, 'VisWSI', f"{slide_name}.jpg"), vis_image[:, :, ::-1])
    cv2.imwrite(os.path.join(save_dir, 'VisWSI', f"{slide_name}_label.jpg"), image_and_label[:, :, ::-1])
    cv2.imwrite(os.path.join(save_dir, 'VisWSI', f"{slide_name}_mask.jpg"), 255-vis_mask[:, :, 0])



    f_train=open(os.path.join(save_dir, 'ImageSets','Segmentation','train.txt'),'a')
    f_val = open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'val.txt'), 'a')
    f_test = open(os.path.join(save_dir, 'ImageSets', 'Segmentation', 'test.txt'), 'a')

    patch_size = 512
    h_overlap = 150
    w_overlap = 150
    f_index=0
    #split data
    for x in range(0, h-patch_size+1, patch_size-h_overlap):
        for y in range(0, w-patch_size+1, patch_size-w_overlap):
                patch = thumbnail[x:x+patch_size, y:y+patch_size, :]
                if np.sum(patch)<20:continue
                patch_label=mask[x:x+patch_size, y:y+patch_size, :]
                if np.sum(patch_label[:,:,1::]) < 2000 and random.random() >= randomrate: #随机保留负样本
                    continue
                patch_label_class=np.argmax(patch_label, axis=-1)
                # keys=list(label_dict.keys())
                for cls_ind in list(np.unique(patch_label_class)):
                    if cls_ind==0:continue
                    else:
                        # print('extract %s in %d WSI'%(keys[cls_ind-1],save_ind))
                        # print('extract %s in %d WSI' % (get_keys(label_dict, cls_ind)[0], save_ind))
                        pass

                cv2.imwrite(os.path.join(save_dir, 'JPEGImages', "%s_%d.jpg" % (save_ind, f_index)), patch[:, :, ::-1])
                cv2.imwrite(os.path.join(save_dir, 'SegmentationClass', "%s_%d.png" % (save_ind, f_index)), patch_label_class)
                # print("patch_label_class.max():", patch_label_class.max())

                # show the labels
                temp_mask = 0.4 * (1 - patch_label[:, :, 0] / 255)
                temp_mask[temp_mask == 0] = 1
                temp_mask = np.repeat(temp_mask[:, :, np.newaxis], 3, axis=-1)
                image_and_label = temp_mask * patch
                for i in range(label_num):
                    if i == 0: continue
                    cur_label = patch_label[:, :, i]
                    cur_label = cur_label[:, :, np.newaxis] // 255
                    cur_label = np.repeat(cur_label, 3, -1)
                    cur_color = list(label_color.values())[i - 1]
                    label_region_color = cur_label * cur_color
                    image_and_label = image_and_label + 0.6 * label_region_color
                cv2.imwrite(os.path.join(save_dir, 'VisPatch', "%s_%d.jpg" % (save_ind, f_index)), image_and_label[:, :, ::-1])

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

def vis_anno(slide_path,roi_contours, roi_labels, save_dir,index):
    slide = openSlide(slide_path)
    resize_ratio = slide.mpp/target_mpp

    h, w = int(slide.height*resize_ratio), int(slide.width*resize_ratio)
    mask_fortrain=np.zeros((h, w, label_num), dtype=np.int16)
    coords_np_list = []
    for i, roi_contour in enumerate(roi_contours):

        x_coord = np.array([int(contour[0] * resize_ratio) for contour in roi_contour])
        y_coord = np.array([int(contour[1] * resize_ratio) for contour in roi_contour])

        coords_np = np.concatenate((x_coord[:, np.newaxis],y_coord[:, np.newaxis]), axis=1)
        coords_np_list.append(coords_np)

        label = roi_labels[i]
        coords_np = [coords_np, ]

        cur_mask_fortrain = mask_fortrain[:, :, label].copy()
        cur_mask_fortrain = cv2.drawContours(cur_mask_fortrain, coords_np, -1, (255, 255, 255), cv2.FILLED)
        mask_fortrain[:, :, label] = cur_mask_fortrain

    mask_fortrain[:, :, 0] = 255 - np.clip(np.sum(mask_fortrain[:, :, 1:], axis=2), a_min=0, a_max=255)
    mask_fortrain = mask_fortrain.astype(np.uint8)
    # cv2.imwrite("./111.jpg", mask_fortrain[:, :, 1])
    split_patches(slide, mask_fortrain, resize_ratio, save_dir=save_dir, save_ind=index, label_count_dict=label_dict)

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


def deduplicate_paths(file_paths):
    file_dict = {}  # 用于存储文件名及其对应的路径列表的字典

    # 构建文件名及其对应的路径列表的字典
    for path in file_paths:
        file_name = path.split('/')[-1]  # 获取文件名
        if file_name in file_dict:
            file_dict[file_name].append(path)
        else:
            file_dict[file_name] = [path]

    # 提取唯一路径，生成新的列表
    unique_paths = []
    for file_name, paths in file_dict.items():
        unique_paths.append(paths[0])  # 只保留一个路径

    return unique_paths



if __name__ == '__main__':
    # get_properties(slide_path)
    # save_dir = '../data/Her2Data_4class_deeplab'
    # save_dir = '../data/qidai'
    save_dir = '../data/taimo'
    # img_folder = '/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/region_data/测试用切片(加入训练，新加了标注)/'
    # img_folder = '/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/region_data/训练集/'
    # img_folder = '/home/zhouzhihao/pathological_data/beijingertongyiyuan/20240229_region/'
    # img_folder = '/home/zhouzhihao/pathological_data/zhengdasan_qidai/qidai/20240331/'
    img_folder = '/home/zhouzhihao/pathological_data/zhengdasan_qidai/taimo/20240524/'
    # class映射的值必须从1开始，且为连续的整数
    label_dict = {
        "羊膜": 1,
        '绒毛膜': 2,
        '蜕膜': 3,
        '增生羊膜上皮': 4,
        '含铁血黄素': 5,
    }
    label_color = {
        "羊膜": (255, 0, 0),
        '绒毛膜': (0, 0, 255),
        '蜕膜': (255, 128, 0),
        '增生羊膜上皮': (102, 255, 102),
        '含铁血黄素': (255, 255, 103),
    }

    label_num = 5 + 1
    randomrate = 0.00001
    if not os.path.exists(os.path.join(save_dir, 'ImageSets')):
        os.makedirs(os.path.join(save_dir, 'JPEGImages'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'SegmentationClass'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'ImageSets', 'Segmentation'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'VisWSI'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'VisPatch'), exist_ok=True)
    # mpp=0.25, level=5; mpp=0.5, level=4;
    # level = 5
    index = 0

    # target_mpp = 1.0    # 5X
    # target_mpp = 2.0    # 2.5X
    target_mpp = 4.0    # 1X
    vis_mpp = 4.0

    from pydaily import filesystem
    # ext_list = ['.kfb', '.mrxs', '.ndpi', '.sdpc', '.svs']
    ext_list = ['.sdpc']
    for ext in ext_list:
        imglist = filesystem.find_ext_files(img_folder, ext)
        imglist = deduplicate_paths(imglist)
        image_name_list = [os.path.splitext(os.path.basename(i))[0] for i in imglist]
        json_list = filesystem.find_ext_files(img_folder, '.json')
        json_name_list = [os.path.splitext(os.path.basename(os.path.splitext(os.path.basename(i))[0]))[0] for i in json_list]
        res = set(json_name_list) - set(image_name_list)
        print(res)
        for ind,slide_path in tqdm(enumerate(imglist)):
            # json_name = slide_path.replace(ext, '.json')
            # json_name = slide_path + ".json"
            json_name = os.path.splitext(os.path.basename(slide_path))[0]
            slide_name = os.path.basename(slide_path)
            if json_name not in json_name_list:
                continue
            else:
                # level_dim_dict = get_level_dim_dict(slide_path)
                json_path = json_list[json_name_list.index(json_name)]
                roi_contours, roi_labels= get_contours(json_path)
                if (len(roi_contours) == 0):
                    print(f"无标注： {slide_name}")
                    continue

                vis_anno(slide_path, roi_contours, roi_labels, save_dir=save_dir, index=index)
                index += 1



