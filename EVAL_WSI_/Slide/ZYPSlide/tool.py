#!/usr/bin/env python

import os
import math
import sys
import cv2
import numpy as np
from PIL import Image
from ctypes import *
from enum import IntEnum
from threading import Lock

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(base_dir)

from ..SlideBase import SlideBase

if sys.platform=='win32':
    cur_encoding='gbk'
    # 加载dll
    os.environ["PATH"] = os.path.join(os.path.abspath(os.path.dirname(__file__)),'win_libs') + ";" + os.environ["PATH"]
    if sys.version.startswith('3.8'):
        os.add_dll_directory(os.path.join(os.path.abspath(os.path.dirname(__file__)),'win_libs'))
    lib = windll.LoadLibrary(os.path.join(os.path.abspath(os.path.dirname(__file__)),'win_libs','ZYSlideLib_C_X64.dll'))
else:
    cur_encoding='utf-8'
    lib = cdll.LoadLibrary(os.path.join(os.path.abspath(os.path.dirname(__file__)),'linux_libs', 'libZYSlideLib_Linux.so'))

lock = Lock()

class SlideInfo(IntEnum):  # 自定义枚举类型
    ScanScale = 0
    SlideWidth = 1
    SlideHeight = 2
    MicrometersPerPixel = 3
    CreateDateTime = 4
    ScanTime = 5

    @classmethod
    def from_param(cls, obj):
        return obj.value

class ZYPSlide(SlideBase):
    def __init__(self, filename):
        self.slide = POINTER(c_uint64)()
        self.filename = filename
        self.bfilename= filename.encode(cur_encoding)
        lib.OpenSlide.restypes = c_bool
        lib.OpenSlide.argtypes = c_char_p, POINTER(POINTER(c_uint64))
        lib.GetSlideInfo.restypes = c_int
        lib.GetSlideInfo.argtypes = SlideInfo, POINTER(c_char_p), POINTER(c_uint64)
        lib.GetScaleImage.restypes = c_int
        lib.GetScaleImage.argtypes = c_float, c_int, c_int, c_int, c_int, POINTER(POINTER(c_ubyte)), POINTER(c_uint64)
        lib.OpenSlide(c_char_p(self.bfilename), byref(self.slide))
        scale, width, height = c_char_p(),c_char_p(), c_char_p()
        lib.GetSlideInfo(SlideInfo.ScanScale, byref(scale), self.slide)
        lib.GetSlideInfo(SlideInfo.SlideWidth, byref(width), self.slide)
        lib.GetSlideInfo(SlideInfo.SlideHeight, byref(height), self.slide)
        self.max_scale = float(scale.value)
        self.width = int(width.value)
        self.height = int(height.value)
        SlideBase.__init__(self)


    def read(self, location=[0, 0], size=None, scale=1, greyscale=False):
        '''
        :param location: (x, y) at level=0
        :param size: (width, height)
        :param scale: resize scale, scale>1 -> zoom out, scale<1 -> zoom in
        :param greyscale: if True, convert image to greyscale
        :return: a numpy image,  np_img.shape=[height, width, channel=1 or 3]
        '''

        if size is None:
            width, height = self.width, self.height
        else:
            width, height = size

        crop_start_x, crop_start_y = location
        crop_start_x = math.ceil(min(max(crop_start_x, 0), self.width))
        crop_start_y = math.ceil(min(max(crop_start_y, 0), self.height))
        crop_end_x = math.ceil(min(max(width + crop_start_x, 0), self.width))
        crop_end_y = math.ceil(min(max(height + crop_start_y, 0), self.height))

        crop_width = math.ceil((crop_end_x - crop_start_x) / 1)
        crop_height = math.ceil((crop_end_y - crop_start_y) / 1)
        if crop_height <= 0 or crop_width <= 0:
            return None

        img = POINTER(c_ubyte)()
        imgLen = lib.GetScaleImage(self.max_scale/scale, int(crop_start_x/scale), int(crop_start_y/scale)
                                   , math.floor(crop_width/scale), int(crop_height/scale), byref(img), self.slide)

        bits = np.ctypeslib.as_array(img, (imgLen,))
        crop_region = cv2.imdecode(bits, cv2.IMREAD_COLOR)
        lib.FreePtr(byref(img))
        del bits
        crop_region = crop_region[:, :, ::-1]
        return crop_region

    def saveLabel(self, path=None):
        charlabel = POINTER(c_ubyte)()
        iLabelSize = lib.GetLabel(byref(charlabel), self.slide)
        np_arr = np.ctypeslib.as_array(charlabel, (iLabelSize,))
        with open(path, 'wb') as f:
            f.write(np_arr)
        lib.FreePtr(byref(charlabel))

    def getThumbnail(self, size=500):
        scale = math.ceil(max(self.width, self.height)/size)
        thumbnail = self.read((0,0), (self.height, self.width), scale)
        return Image.fromarray(thumbnail)


        # thumbnail = POINTER(c_ubyte)()
        # data_size = lib.GetPreview(byref(thumbnail), self.slide)
        # bits = np.ctypeslib.as_array(thumbnail, (data_size,))
        # img = cv2.imdecode(bits, cv2.IMREAD_COLOR)
        # lib.FreePtr(byref(thumbnail))
        # if size:
        #     img = cv2.resize(img, dsize=(size, size))
        # img = Image.fromarray(img[:, :, ::-1])
        # return img

    @property
    def mpp(self):
        mpp = c_char_p()
        lib.GetSlideInfo(SlideInfo.MicrometersPerPixel, byref(mpp), self.slide)
        return float(mpp.value)

    def __del__(self):
        lib.CloseSlide(self.slide)
