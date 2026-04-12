#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 23:06
# @Author  : Can Cui
# @File    : utils.py
# @Software: PyCharm
# @Comment:
import os, json
from numba import jit
import numba
if numba.__version__.startswith('0.5'):
    from numba.typed import List
    ls = List
else:
    ls = list
from shapely.geometry import box, Polygon




def map_results(center_coords_ls, label_ls, prob_ls):
    # print('center_coords_ls, label_ls', center_coords_ls, label_ls)
    if len(center_coords_ls) > 0:
        center_coords_all = np.array(center_coords_ls).astype(int)
        labels_all = np.array(label_ls).astype(int)
        prob_all = np.array(prob_ls).astype(float)
        # Remap labels
        # map_dict = {0:1, 1:0, 2:3, 3:2}
        # map_dict = {0:1, 1:0, 2:3, 3:2,  4:3, 5:3, 6:3, 7:3, 8:3, }
        # map_dict = {0:3, 1:1, 2:2, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
        # labels_all = np.vectorize(map_dict.get)(labels_all)
    else:
        center_coords_all, labels_all,prob_all = np.empty((0,2), dtype=int),np.empty((0,), dtype=int),np.empty((0,), dtype=int)
    return center_coords_all, labels_all, prob_all


def dump_results(slide_path, result):
    result_root = os.path.dirname(slide_path)
    os.makedirs(result_root, exist_ok=True)
    with open(os.path.join(str(result_root), 'result_file.json'), 'w', encoding="utf-8") as result_file:
        json.dump(result, result_file)


def delete_prev_json(result_root, json_name):
    if os.path.exists(os.path.join(result_root, json_name)):
        os.remove(os.path.join(result_root, json_name))

def split_patches_and_label(slide, patch_w, patch_h, contours=[], roilist_np=[]):
    width, height = slide.width, slide.height
    all_region_info_dict = {}
    valid_region_info_dict = {}
    valid_region_info_dict_with_label = {}
    num_patch = 0
    for x in range(0, width, patch_w):
        for y in range(0, height, patch_h):
            box(x, y, x+patch_w, y+patch_h)
            p = (x,y)
            all_region_info_dict[num_patch] = p
            num_patch+=1
    if len(contours)>0:
        for contour in contours:
            x_coords = list(contour[:,0,0])
            y_coords = list(contour[:,0,1])
            poly = Polygon(list(zip(x_coords, y_coords)))
            for idx, (x,y) in all_region_info_dict.items():
                roi = box(x, y, x+patch_w, y+patch_h)
                if poly.intersects(roi):
                    valid_region_info_dict[idx] = (x,y)
    else:
        valid_region_info_dict = all_region_info_dict
    for index in range(roilist_np.shape[0]):
        try:
            x_coords = list(roilist_np[index, 0, :])
            y_coords = list(roilist_np[index, 1, :])
        except Exception as e:
            # 可能包含多边形标注
            x_coords = list(roilist_np[index, 0])
            y_coords = list(roilist_np[index, 1])
        poly = Polygon(list(zip(x_coords, y_coords)))
        for idx, (x, y) in valid_region_info_dict.items():
            roi = box(x, y, x + patch_w, y + patch_h)
            if poly.intersects(roi):
                valid_region_info_dict_with_label[idx] = (x, y)

    if len(valid_region_info_dict_with_label) == 0:
        valid_region_info_dict_with_label[0] = (0, 0)
    return valid_region_info_dict_with_label

def split_patches(slide, patch_w, patch_h, contours=[]):
    width, height = slide.width, slide.height
    all_region_info_dict = {}
    valid_region_info_dict = {}
    num_patch = 0
    for x in range(0, width, patch_w):
        for y in range(0, height, patch_h):
            box(x, y, x+patch_w, y+patch_h)
            p = (x,y)
            all_region_info_dict[num_patch] = p
            num_patch+=1
    if len(contours)>0:
        for contour in contours:
            x_coords = list(contour[:,0,0])
            y_coords = list(contour[:,0,1])
            poly = Polygon(list(zip(x_coords, y_coords)))
            for idx, (x,y) in all_region_info_dict.items():
                roi = box(x, y, x+patch_w, y+patch_h)
                if poly.intersects(roi):
                    valid_region_info_dict[idx] = (x,y)
    else:
        valid_region_info_dict = all_region_info_dict
    if len(valid_region_info_dict) == 0:
        valid_region_info_dict[0] = (0, 0)
    return valid_region_info_dict

def split2groups(region_info_dict, comm_size):
    list_length = len(region_info_dict)
    regions_per_gpu = list_length // comm_size
    reserved_region_num = list_length % comm_size
    full_region_num = regions_per_gpu * comm_size
    if regions_per_gpu > 0:
        group = [dict(list(region_info_dict.items())[i:i + regions_per_gpu]) for i in
                range(0, full_region_num, regions_per_gpu)]
    else:
        group = [{} for _ in range(comm_size)]
        group[0] = region_info_dict
    if reserved_region_num == 0:
        return group
    else:
        for i_region in range(reserved_region_num):
            key, value = list(region_info_dict.items())[full_region_num + i_region]
            group[i_region][key] = value
        return group

@jit(nopython=True)
def is_inside_sm(polygon, point):
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1
    while ii<length:
        dy  = dy2
        dy2 = point[1] - polygon[jj][1]
        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):
            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]
                if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F: # point on line
                    return 2
            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                return 2
        ii = jj
        jj += 1
    #print 'intersections =', intersections
    return intersections & 1


@jit(nopython=True, parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean)
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon,points[i])
    return D

from numba import jit, njit
import numba
import numpy as np

@jit(nopython=True)
def pointinpolygon(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in numba.prange(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


@njit(parallel=True)
def parallelpointinpolygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean)
    for i in numba.prange(0, len(D)):
        D[i] = pointinpolygon(points[i,0], points[i,1], polygon)
    return D


def poly_make_valid(poly):
    """Convert a potentially invalid polygon to a valid one by eliminating
    self-crossing or self-touching parts.
    Args:
        poly (Polygon): A polygon needed to be converted.
    Returns:
        A valid polygon.
    """
    return poly if poly.is_valid else poly.buffer(0)


def filter_points(center_coords, labels, x_coords, y_coords):

    if center_coords.shape[0] > 0 and len(x_coords) > 0:
        try:
            pick_idx = parallelpointinpolygon(center_coords, List(zip(x_coords, y_coords)))
            center_coords = center_coords[pick_idx]
            labels = labels[pick_idx]
        except Exception as e:
            print(e)
    # else:
    #     center_coords, labels = np.empty((0, 2),dtype=np.int32), np.empty((0,), dtype=np.int32)
    return center_coords, labels

def filter_contours(contour_coords, contour_labels, x_coords, y_coords):

    filtered_contour_coords, filtered_contour_labels = [], []
    if len(contour_coords) > 0 and len(x_coords) > 0:
        poly_manual = poly_make_valid(Polygon(list(zip(x_coords, y_coords))))
        for contour, label in zip(contour_coords,contour_labels):
            poly_contour = poly_make_valid(Polygon(contour))
            intersection_poly = poly_manual.intersection(poly_contour)
            if intersection_poly.area>0:
                print(intersection_poly.area)
                filtered_contour_coords.append(list(map(list, intersection_poly.exterior.coords)))
                filtered_contour_labels.append(label)
    else:
        filtered_contour_coords, filtered_contour_labels = contour_coords, contour_labels
    return filtered_contour_coords, filtered_contour_labels



