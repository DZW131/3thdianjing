# -*- coding: utf-8 -*-

import os, sys
# Add kfb support
FileAbsPath = os.path.abspath(__file__)
PreprocessPath = os.path.dirname(FileAbsPath)
sys.path.append(os.path.join(PreprocessPath, 'kfb'))
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

def read_region_kfb(kfb_slide, location, size, level=0, TILE_SIZE=256):
    start_x, start_y = location[0], location[1]
    region_w, region_h = size[0], size[1]
    max_w = start_x + region_w
    max_h = start_y + region_h
    # if 'ndpi' in slides_dir:
    #     slide_width, slide_height = kfb_slide.level_dimensions[level]
    # else:
    slide_width, slide_height = kfb_slide.level_dimensions[level]
    slide_truncate_width = slide_width - slide_width % TILE_SIZE
    slide_truncate_height = slide_height - slide_height % TILE_SIZE

    # assert max_w < slide_truncate_width and max_h < slide_truncate_height, "Cell near the boundary"

    rx_start, ry_start = int(start_x / TILE_SIZE), int(start_y / TILE_SIZE)
    rx_end, ry_end = int(max_w / TILE_SIZE) , int(max_h / TILE_SIZE)

    cur_region_img = np.zeros(((ry_end-ry_start)*TILE_SIZE,
                                (rx_end-rx_start)*TILE_SIZE, 3), dtype=np.uint8)
    # read region one by one
    for rx in range(rx_start, rx_end): # traverse through x
        for ry in range(ry_start, ry_end): # traverse though y
            cur_region = kfb_slide.read_region((rx*TILE_SIZE, ry*TILE_SIZE), level=level)
            buf = BytesIO(cur_region)
            cur_img = np.asanyarray(Image.open(buf))
            try:
                cur_region_img[(ry-ry_start)*TILE_SIZE:(ry-ry_start+1)*TILE_SIZE,
                            (rx-rx_start)*TILE_SIZE:(rx-rx_start+1)*TILE_SIZE, :] = cur_img
            except:
                cur_img=np.zeros((TILE_SIZE,TILE_SIZE,3))
                cur_region_img[(ry - ry_start) * TILE_SIZE:(ry - ry_start + 1) * TILE_SIZE,
                (rx - rx_start) * TILE_SIZE:(rx - rx_start + 1) * TILE_SIZE, :] = cur_img
            # cv2.imwrite('../region_test_data/level_%d_%d_%d.jpg'%(level,rx,ry),cur_img[:,:,::-1])
    region_start_x = start_x % TILE_SIZE
    region_start_y = start_y % TILE_SIZE

    cur_region_img = cur_region_img[region_start_y: region_start_y+region_h,
                                  region_start_x: region_start_x+region_w, :]
    return cur_region_img
