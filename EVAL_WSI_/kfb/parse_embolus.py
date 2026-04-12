# -*- coding: utf-8 -*-

import os, sys
# Add kfb support
FileAbsPath = os.path.abspath(__file__)
PreprocessPath = os.path.dirname(FileAbsPath)
sys.path.append(os.path.join(PreprocessPath, 'kfb'))
import Code与文档.Embolus.preprocess.kfb.kfbslide as kfbslide

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import math, glob, json
import deepdish as dd
from scipy import misc


def parse_annotation(slide_path, json_path):
    assert os.path.exists(slide_path), "slide file not exist"
    assert os.path.exists(json_path), "Json file not exist"
    slide = kfbslide.open_kfbslide(slide_path)
    slide_width, slide_height = slide.level_dimensions[0]

    annotate_dict = {}
    with open(json_path) as json_file:
        annotation = json.load(json_file)
        # Set image name
        annotate_dict['img_name'] = str(annotation['img_name'])
        regions = annotation['regions']
        region_dict = {}
        for region_id in regions.keys():
            region_name = 'r' + str(region_id)
            cur_points = regions[region_id]['points']
            num_points = len(cur_points)
            points_coors = np.zeros((2, num_points), dtype=np.int32)
            for ind, ipoint in enumerate(cur_points):
                points_coors[0, ind] = int(round(ipoint['x']))
                points_coors[1, ind] = int(round(ipoint['y']))

            # Check if contours inside the slide
            x_min, x_max = np.min(points_coors[0, :]), np.max(points_coors[0, :])
            y_min, y_max = np.min(points_coors[1, :]), np.max(points_coors[1, :])
            if x_min < 0 or y_min < 0 or x_max >= slide_width or y_max > slide_height:
                continue
            region_dict[region_name] = points_coors
        annotate_dict['regions'] = region_dict
    return annotate_dict



def read_region_kfb(kfb_slide, location, size, level=0, TILE_SIZE=256):
    start_x, start_y = location[0], location[1]
    region_w, region_h = size[0], size[1]
    max_w = start_x + region_w
    max_h = start_y + region_h

    slide_width, slide_height = kfb_slide.level_dimensions[level]
    #Imgsize = kfb_slide.shape
    #slide_width = Imgsize[1]
    #slide_height = Imgsize[0]
    slide_truncate_width = slide_width - slide_width % TILE_SIZE-TILE_SIZE
    slide_truncate_height = slide_height - slide_height % TILE_SIZE-TILE_SIZE

    assert max_w < slide_truncate_width and max_h < slide_truncate_height, "Cell near the boundary"

    rx_start, ry_start = int(start_x / TILE_SIZE), int(start_y / TILE_SIZE)
    rx_end, ry_end = int(max_w / TILE_SIZE) + 1, int(max_h / TILE_SIZE) + 1

    cur_region_img = np.zeros(((ry_end-ry_start)*TILE_SIZE,
                                (rx_end-rx_start)*TILE_SIZE, 3), dtype=np.uint8)
    # read region one by one
    for rx in range(rx_start, rx_end): # traverse through x
        for ry in range(ry_start, ry_end): # traverse though y
            cur_region = kfb_slide.read_region((rx*TILE_SIZE, ry*TILE_SIZE), level=0)
            #cur_region=kfb_slide[(ry-ry_start)*TILE_SIZE:(ry-ry_start+1)*TILE_SIZE,(rx-rx_start)*TILE_SIZE:(rx-rx_start+1)*TILE_SIZE,:]
            #cur_region = kfb_slide[(ry - ry_start) * TILE_SIZE:(ry - ry_start + 1) * TILE_SIZE,
                         #(rx - rx_start) * TILE_SIZE:(rx - rx_start + 1) * TILE_SIZE, :]
            buf = BytesIO(cur_region)
            cur_img = np.asanyarray(Image.open(buf))
            #cur_img=cur_region
            cur_region_img[(ry-ry_start)*TILE_SIZE:(ry-ry_start+1)*TILE_SIZE,
                            (rx-rx_start)*TILE_SIZE:(rx-rx_start+1)*TILE_SIZE, :] = cur_img
    region_start_x = start_x % TILE_SIZE
    region_start_y = start_y % TILE_SIZE

    cur_region_img = cur_region_img[region_start_y: region_start_y+region_h,
                                  region_start_x: region_start_x+region_w, :]
    return cur_region_img
