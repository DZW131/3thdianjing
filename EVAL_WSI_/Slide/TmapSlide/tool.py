#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import math
import sys
import cv2
import numpy as np
from PIL import Image
from ctypes import *
from enum import Enum
from threading import Lock
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(base_dir)
from ..SlideBase import SlideBase

if sys.platform=='win32':
    cur_encoding='gbk'
    # 加载dll
    os.environ["PATH"] = os.path.join(os.path.abspath(os.path.dirname(__file__)),'win_libs') + ";" + os.environ["PATH"]
    if sys.version.startswith('3.8'):
        os.add_dll_directory(os.path.join(os.path.abspath(os.path.dirname(__file__)),'win_libs'))
    lib = windll.LoadLibrary(os.path.join(os.path.abspath(os.path.dirname(__file__)),'win_libs','iViewerSDK.dll'))
else:
    cur_encoding='utf-8'
    lib = cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)),'linux_libs','libiViewerSDK2.so'))

lock = Lock()

class MyEnum(Enum):  # 自定义枚举类型
    uImageThumbnail = 0
    uImageNavigate = 1
    uImageMacro = 2
    uImageLabel = 3
    uImageMacroLabel = 4
    uImageTile = 5
    uImageWhole = 6
    uImageAll = 7

    @classmethod
    def from_param(cls, obj):
        return obj.value


class ImgSize(Structure):  # 自定义结构体
    _pack_ = 1
    _fields_ = [
        ('imgsize', c_longlong),
        ('width', c_int),
        ('height', c_int),
        ('depth', c_int)
    ]

# 声明方法参数及返回数据类型
lib.OpenTmapFile.restype = c_void_p  # 打开切片
lib.CloseTmapFile.argtypes = (c_void_p,)  # 关闭切片
lib.CloseTmapFile.restype = c_bool
lib.GetImageInfoEx.argtypes = (c_void_p, MyEnum)  # 获取整体切片信息
lib.GetImageInfoEx.restype = ImgSize
lib.GetImageData.argtypes = (c_void_p, MyEnum, POINTER(c_ubyte), c_int)  # 获取切片数据
lib.GetImageData.restype = c_bool
lib.GetScanScale.argtypes = (c_void_p,)  # 最大倍率
lib.GetScanScale.restype = c_int
lib.GetPixelSize.argtypes = (c_void_p,)  # 分辨率
lib.GetPixelSize.restype = c_float
lib.GetLayerNum.argtypes = (c_void_p,)  # 最大层级
lib.GetLayerNum.restype = c_int
lib.GetImageSizeEx.argtypes = (c_void_p, c_int, c_int, c_int, c_int, c_float)
lib.GetImageSizeEx.restype = ImgSize
lib.GetCropImageDataEx.argtypes = (c_void_p, c_int, c_int, c_int, c_int, c_int, c_float, c_int)
lib.GetCropImageDataEx.restype = POINTER(c_ubyte)


class TmapSlide(SlideBase):
    def __init__(self, filename):
        self.filename = filename
        self.bfilename = self.filename.encode(cur_encoding)
        self.slide = lib.OpenTmapFile(self.bfilename, 1)
        self.width = lib.GetImageInfoEx(self.slide, MyEnum.uImageWhole).width
        self.height = lib.GetImageInfoEx(self.slide, MyEnum.uImageWhole).height
        self.max_scale = lib.GetScanScale(self.slide)  # 扫描最大倍率
        self.pixel_size = lib.GetPixelSize(self.slide)  # 分辨率
        self.layer_num=lib.GetLayerNum(self.slide)
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

        crop_level = math.floor(math.log2(scale))

        crop_level = max(0, min(crop_level, self.layer_num))
        level_ratio = 2 ** crop_level
        resize_ratio = level_ratio / scale

        crop_start_x, crop_start_y = location
        crop_start_x = math.ceil(min(max(crop_start_x, 0), self.width))
        crop_start_y = math.ceil(min(max(crop_start_y, 0), self.height))
        crop_end_x = math.ceil(min(max(width + crop_start_x, 0), self.width))
        crop_end_y = math.ceil(min(max(height + crop_start_y, 0), self.height))

        crop_width = math.ceil((crop_end_x - crop_start_x) / level_ratio)
        crop_height = math.ceil((crop_end_y - crop_start_y) / level_ratio)
        if crop_height == 0 or crop_width == 0:
            return None

        with lock:  #优纳五片机不支持多线程读图,120片机的可以,但是无法区分机型,也就先这样了
            crop_image_info = lib.GetImageSizeEx(self.slide, crop_start_x, crop_start_y, crop_end_x, crop_end_y,self.max_scale/math.pow(2,crop_level))

            crop_image_width = crop_image_info.width
            crop_image_height = crop_image_info.height
            crop_image_imgsize = crop_image_info.imgsize

            crop_image_data = lib.GetCropImageDataEx(self.slide, 1,crop_start_x, crop_start_y, crop_end_x, crop_end_y,self.max_scale/math.pow(2,crop_level),
                                                     crop_image_imgsize)

            bits = np.ctypeslib.as_array(crop_image_data, shape=(crop_image_imgsize,))
            bits = bits.reshape(crop_image_height, crop_image_width, 3)

            finalwidth=int(crop_width * resize_ratio)
            finalheight=int(crop_height * resize_ratio)

            if finalwidth==0 and finalheight==0:
                crop_region=cv2.resize(bits,(1,1))
            elif finalwidth==0 and finalheight!=0:
                crop_region=cv2.resize(bits,(1,finalheight))
            elif finalwidth!=0 and finalheight==0:
                crop_region=cv2.resize(bits,(finalwidth,1))
            else:
                crop_region=cv2.resize(bits,(finalwidth,finalheight))

            crop_region=crop_region[:,:,::-1]
            return crop_region

    def getTile(self, x, y, z):
        scale = math.pow(2, self.maxlvl - z)
        r = 1024 * scale
        r = int(r)
        tile = self.read([x * r, y * r], [r, r], scale, greyscale=False)
        return Image.fromarray(tile, mode='RGB')
        # return tile

    def saveLabel(self, path=None):
        label_info = lib.GetImageInfoEx(self.slide, MyEnum.uImageLabel)
        label_size = label_info.imgsize
        label_width = label_info.width
        label_height = label_info.height
        buffer = (c_ubyte * label_size)()
        lib.GetImageData(self.slide, MyEnum.uImageLabel, buffer, label_size)
        bits = np.ctypeslib.as_array(buffer, shape=(label_size,))
        bits = bits.reshape((label_height, label_width, 3))
        image = Image.fromarray(bits)
        image.save(path)

    def getThumbnail(self, *args, **kwargs):
        thumb_info = lib.GetImageInfoEx(self.slide, MyEnum.uImageNavigate)
        thumb_size = thumb_info.imgsize
        thumb_width = thumb_info.width
        thumb_height = thumb_info.height
        buffer = (c_ubyte * thumb_size)()
        lib.GetImageData(self.slide, MyEnum.uImageNavigate, buffer, thumb_size)
        bits = np.ctypeslib.as_array(buffer, shape=(thumb_size,))
        bits = bits.reshape((thumb_height, thumb_width, 3))
        bits=bits[:,:,::-1]
        return Image.fromarray(bits, mode='RGB')

    @property
    def mpp(self):
        mpp = self.pixel_size * (100 / self.max_scale) * 1000
        return mpp

    def __del__(self):
        lib.CloseTmapFile(self.slide)


if __name__ == "__main__":
    slide = TmapSlide(r"C:\Users\TB\Desktop\5片机.TMAP")
    print(slide.width,slide.height)
    print(slide.layer_num)
    print(slide.pixel_size)
    print(slide.max_scale)

