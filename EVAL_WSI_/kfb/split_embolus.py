import kfbslide
import os
import scipy.misc as misc
import numpy as np
from PIL import Image
from parse_embolus import read_region_kfb
from parse_embolus import parse_annotation
# from extract_det_samples import convert_cnts_to_polygons
from pydaily import filesystem
from shapely.geometry import Polygon


def convert_cnts_to_polygons(cnts_dict, crop_len):
    poly_list = []
    for key in cnts_dict:
        cur_cnt = cnts_dict[key]
        x_arr, y_arr = cur_cnt[0], cur_cnt[1]
        min_x, max_x = np.min(x_arr), np.max(x_arr)
        min_y, max_y = np.min(y_arr), np.max(y_arr)
        cnt_width = max_x - min_x + 1
        cnt_height = max_y - min_y + 1
        if cnt_width >= crop_len or cnt_height >= crop_len:
            continue

        cur_poly = Polygon([(int(x), int(y)) for x, y in zip(x_arr, y_arr)])
        poly_list.append(cur_poly)

    return poly_list


def split_embolus(slide_dir, save_dir):
    slide_path, slide_name = os.path.split(slide_dir)
    kfb_name, kfb_ext = os.path.splitext(slide_name)
    json_name = kfb_name + '.json'
    # get information of slide
    slide = kfbslide.open_kfbslide(slide_dir)
    width, height = slide.level_dimensions[0]
    TILE_SIZE = 256
    width = width - width % TILE_SIZE - 1
    height = height - height % TILE_SIZE - 1
    # get information of regions
    annotation_path = os.path.join(slide_path, json_name)
    annotation_dict = parse_annotation(slide_dir, annotation_path)
    if annotation_dict['img_name'] != kfb_name:
        print("Annotation not match")
        return 0
    assert annotation_dict['img_name'] == kfb_name, "Annotation not match"

    cnts_dict = annotation_dict['regions']  # get information of regions
    poly_list = convert_cnts_to_polygons(cnts_dict, 100000000)
    index = 0
    for ind, cur_poly in enumerate(poly_list):
        (minx, miny, maxx, maxy) = cur_poly.bounds
        if maxx > width or maxy > height:
            continue
        region_size = (int(maxx - minx), int(maxy - miny))
        region_start = (int(minx), int(miny))
        cur_region = read_region_kfb(slide, region_start, region_size)
        img_name = kfb_name + '-' + str(index) + ".png"
        try:
            misc.imsave(os.path.join(save_dir, img_name), cur_region)
        except OverflowError as e:
            pass
            continue
        print(img_name)
        index = index + 1


def processing_batch(slides_dir, save_dir):
    kfb_list = filesystem.find_ext_files(slides_dir, "kfb")
    for ind, cur_kfb in enumerate(kfb_list):
        print("Processing {:3d}/{:3d}".format(ind + 1, len(kfb_list)))
        slide_fullname = os.path.basename(cur_kfb)
        slide_name, slide_ext = os.path.splitext(slide_fullname)
        annotation_path = os.path.join(slides_dir, slide_name + ".json")  # get the path of json
        if not os.path.exists(annotation_path):
            continue
        slide_path = os.path.join(slides_dir, cur_kfb)
        print(slide_path)
        split_embolus(slide_path, save_dir)


if __name__ == '__main__':
    # slide_dir = '09-17583 CA1_2019-01-08 13_00_27.kfb'
    # save_dir = '/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/上肿/patch'
    slides_dir = '/home/kd/测试数据集0511/已上传04'
    save_dir = '/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/数据集/0511新测试数据集/已上传04'
    # split_embolus(slide_dir, save_dir)
    processing_batch(slides_dir, save_dir)
