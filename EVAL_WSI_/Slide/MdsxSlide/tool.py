#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os.path, math, sys
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from ..SlideBase import SlideBase
import numpy as np
from io import BytesIO
import cv2
from PIL import Image
from threading import Lock
from ctypes import *

if sys.platform=='win32':
    cur_encoding='gbk'
    os.environ["PATH"] = os.path.join(os.path.abspath(os.path.dirname(__file__)),'win_libs') + ";" + os.environ["PATH"]
    if sys.version.startswith('3.8'):
        os.add_dll_directory(os.path.join(os.path.abspath(os.path.dirname(__file__)),'win_libs'))
    lib = windll.LoadLibrary(os.path.join(os.path.abspath(os.path.dirname(__file__)),'win_libs','MDSFileParser.dll'))
else:
    cur_encoding='utf-8'
    lib = cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)),'linux_libs','libMDSParser.so'))

lock = Lock()

class MdsxSlide(SlideBase):

    def __init__(self, filename):

        self.filename = filename
        lib.MDS_open.restype = c_void_p
        self.slide = c_void_p(lib.MDS_open(c_char_p(self.filename.encode(cur_encoding))))
        width = pointer(c_int(0))
        height = pointer(c_int(0))
        lib.MDS_size(self.slide, width, height)
        self.width = width.contents.value
        self.height = height.contents.value
        self.layercount = lib.MDS_layerCount(self.slide)
        SlideBase.__init__(self)

    def read(self, location=[0, 0], size=None, scale=1, greyscale=False):
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
        crop_level = max(0,min(crop_level, self.layercount - 2))

        level_ratio = 2 ** crop_level
        resize_ratio = level_ratio / scale
        # resize_ratio = level_ratio / (2**(math.floor(math.log2(scale))-1))

        # make sure the crop region is inside the slide
        crop_start_x, crop_start_y = min(max(crop_start_x, 0), self.width), min(max(crop_start_y, 0), self.height)
        crop_end_x = math.ceil(min(max(width + crop_start_x, 0), self.width))
        crop_end_y = math.ceil(min(max(height + crop_start_y, 0), self.height))

        crop_width = math.floor((crop_end_x - crop_start_x) / level_ratio)
        crop_height = math.floor((crop_end_y - crop_start_y) / level_ratio)

        if crop_height == 0 or crop_width == 0:
            return None

        layer = c_int(crop_level)
        w = c_int(crop_width)
        h = c_int(crop_height)
        xp = c_int(int(crop_start_x / level_ratio))
        yp = c_int(int(crop_start_y / level_ratio))

        with lock:

            # 不指定下边两行的话 linux环境下会报错 Segmentation fault
            lib.MDS_previewImage.restype=c_int
            lib.MDS_previewImage.argtypes=[c_void_p,c_int,c_int,c_int,c_int,c_int,c_void_p]

            res = lib.MDS_previewImage(self.slide, layer, xp, yp, w, h, 0)

            if res > 0:
                img = pointer((c_ubyte * res)())
            if lib.MDS_previewImage(self.slide, layer, xp, yp, w, h, img):
                bits = np.ctypeslib.as_array(img, shape=(1,))

        buf = BytesIO(bits)
        cur_img = Image.open(buf)
        cur_img = cur_img.convert("RGB")
        cur_img = np.asanyarray(cur_img).astype(np.uint8)
        finalwidth = int(crop_width * resize_ratio)
        finalheight = int(crop_height * resize_ratio)

        if finalwidth == 0 and finalheight == 0:
            crop_region = cv2.resize(cur_img, (1, 1))
        elif finalwidth == 0 and finalheight != 0:
            crop_region = cv2.resize(cur_img, (1, finalheight))
        elif finalwidth != 0 and finalheight == 0:
            crop_region = cv2.resize(cur_img, (finalwidth, 1))
        else:
            crop_region = cv2.resize(cur_img, (finalwidth, finalheight))
        # crop_region = cv2.resize(cur_img, (int(crop_width * resize_ratio), int(crop_height * resize_ratio)))
        # crop_region = crop_region[:, :, ::-1]
        return crop_region

    def saveLabel(self, path):

        size = lib.MDS_labelJpeg(self.slide, 0)
        if size > 0:
            data = pointer((c_ubyte * size)())
        if lib.MDS_labelJpeg(self.slide, data) == size:
            # print(data)
            bits = np.ctypeslib.as_array(data, shape=(1,))
            # print(bits[0])
            with open(path, 'wb') as f:
                f.write(bits[0])

    @property
    def mpp(self):

        lib.MDS_scale.restype = c_double
        result = lib.MDS_scale(self.slide)
        return result

    def __del__(self):

        lib.MDS_close(pointer(self.slide))


if __name__ == "__main__":
    slide = MdsxSlide(r"C:\Users\Admin\Desktop\1.mdsx")
# res=slide.getTile(0,0,0)

# res.save('test.jpg')

# print(res)
# # print(slide.mpp)
# # print(res)
# for z in range(15):
#     os.makedirs("testmdsx/{}".format(z), exist_ok= True)
#     for x in range(100):
#         for y in range(100):
#             try:
#                 img = slide.getTile(x,y,z)
#                 img.save("testmdsx/{}/{}_{}.jpg".format(z,y,x))
#             except:
#                 pass

