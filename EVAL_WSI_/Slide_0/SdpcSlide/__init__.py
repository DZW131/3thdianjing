#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os.path,math,sys

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR
base_dir=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(base_dir)

from EVAL_WSI_1GPU.Slide_0.SlideBase import SlideBase
import numpy as np
from pystruct import SqSdpcInfo
from ctypes import *
import cv2
from PIL import Image
from ctypes import *


os.environ["PATH"] = os.path.abspath( os.path.dirname(__file__) ) + ";" + os.environ["PATH"]
lib = cdll.LoadLibrary('DecodeSdpcDll.dll')

class SdpcSlide(SlideBase):

    def __init__(self,filename):

        self.filename = filename
        self.bfilename=self.filename.encode('utf-8')
        self.slide = SqSdpcInfo()
        lib.SqOpenSdpc(c_char_p(self.bfilename),byref(self.slide))
        self.width=self.slide.picHead.contents.srcWidth
        self.height=self.slide.picHead.contents.srcHeight
        SlideBase.__init__(self)


    def read(self, location=[0,0], size=None, scale=1, greyscale=False):
        '''
        :param location: (x, y) at level=0
        :param size: (width, height)
        :param scale: resize scale, scale>1 -> zoom out, scale<1 -> zoom in
        :param greyscale: if True, convert image to greyscale
        :return: a numpy image,  np_img.shape=[height, width, channel=1 or 3]
        '''
        if size == None:
            width, height = self.width, self.height
        else:
            width, height = size

        crop_start_x, crop_start_y = location
        crop_level = math.floor(math.log2(scale))
        crop_level = max(0, min(crop_level, self.slide.picHead.contents.hierarchy-1))

        level_ratio = 2 ** crop_level
        resize_ratio = level_ratio / scale

        # make sure the crop region is inside the slide
        crop_start_x, crop_start_y = min(max(crop_start_x, 0), self.width), min(max(crop_start_y, 0), self.height)
        crop_end_x = math.ceil(min(max(width + crop_start_x, 0), self.width))
        crop_end_y = math.ceil(min(max(height + crop_start_y, 0), self.height))

        crop_width = math.ceil((crop_end_x - crop_start_x) / level_ratio)
        crop_height = math.ceil((crop_end_y - crop_start_y) / level_ratio)

        if crop_height == 0 or crop_width == 0:
            return None
        layer = c_int(crop_level)
        rgb = pointer(pointer(c_ubyte()))
        # rgb = pointer(c_ubyte())
        w = c_int(crop_width)
        h = c_int(crop_height)
        xp = c_uint(int(crop_start_x/level_ratio))
        yp = c_uint(int(crop_start_y/level_ratio))


        lib.SqGetRoiRgbOfSpecifyLayer(byref(self.slide),rgb,w,h,xp,yp,layer)

        bits = np.ctypeslib.as_array(rgb.contents, shape=(crop_width * crop_height * 3,))
        bits = bits.reshape((crop_height, crop_width, 3))

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

        lib.SqFreeMemory(rgb.contents)

        crop_region = crop_region[:, :, ::-1]

        return crop_region

    def getTile(self, x, y, z):

        scale = math.pow(2, self.maxlvl - z)
        r = 1024 * scale
        r=int(r)
        tile = self.read([x * r, y * r], [r, r], scale, greyscale=False)
        return Image.fromarray(tile, mode='RGB')

    def saveLabel(self, path):

        size = self.slide.macrograph.contents.contents.jpegSize
        data = self.slide.macrograph.contents.contents.jpeg
        bits = np.ctypeslib.as_array(data, shape=(size,))
        with open(path, 'wb') as f:
            f.write(bits)


    @property
    def mpp(self):
        return self.slide.picHead.contents.ruler

    def __del__(self):

        lib.SqCloseSdpc(self.slide)



if __name__ == "__main__":
    slide = SdpcSlide(r"C:\Users\Admin\Desktop\\20191129_140040.sdpc")
    res=slide.getTile(1,1,14)
    # print(help(slide.info))
    # print(res)
    # for z in range(17):
    #     os.makedirs("test/{}".format(z), exist_ok= True)
    #     for x in range(100):
    #         for y in range(100):
    #             try:
    #                 img = slide.getTile(x,y,z)
    #                 img.save("test/{}/{}_{}.jpg".format(z,y,x))
    #             except:
    #                 pass
