#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os.path,math,sys
dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir)
from ..SlideBase import SlideBase
from . import bioformats as bf
from . import javabridge
from .bioformats import log4j
import cv2
import numpy as np
import json


if javabridge.get_env() is None:
    javabridge.start_vm(class_path=bf.JARS)
else:
    javabridge.attach()

class BioformatsSlide(SlideBase):
    def __init__(self,filename):
        log4j.basic_config()
        self.filename = filename
        md = bf.get_omexml_metadata(filename)
        ome = bf.OMEXML(md)
        self.width  = int(ome.image().Pixels.get_SizeX())
        self.height = int(ome.image().Pixels.get_SizeY())
        self._mpp   = ome.image().Pixels.get_PhysicalSizeX()
        self.label_series_idx = None
        self.macro_series_idx = None
        depth = 0
        img_list = {}
        num_img = ome.get_image_count()
        for i in range(num_img):
            img_name = ome.image(i).get_Name()
            img_size = (ome.image().Pixels.get_SizeX(), ome.image().Pixels.get_SizeY())
            img_list[img_name] = img_size
            if os.path.basename(filename) in img_name:
                depth+=1
            if 'label' in img_name:
                self.label_series_idx = i
            if 'macro' in img_name:
                self.macro_series_idx = i
        self.depth = depth-1
        self.img_list = img_list
        self.reader = bf.ImageReader(filename)

        SlideBase.__init__(self)


    def read(self, location=[0,0], size=None, scale=1.0, greyscale=False):

        if size == None:
            width, height = self.width, self.height
        else:
            width, height = size

        crop_start_x, crop_start_y = location
        crop_level = math.floor(math.log(scale,2))
        crop_level = min(self.depth, crop_level)
        level_downsamples = 2**crop_level
        resize_ratio = 2**crop_level/scale
        width, height = width//level_downsamples, height//level_downsamples

        # make sure the crop region is inside the slide
        crop_start_x = math.ceil(min(max(crop_start_x/level_downsamples, 0), self.width/level_downsamples))
        crop_start_y = math.ceil(min(max(crop_start_y/level_downsamples, 0), self.height/level_downsamples))
        crop_end_x = math.floor(min(max((width+crop_start_x), 0), self.width/level_downsamples))
        crop_end_y = math.floor(min(max((height+crop_start_y), 0), self.height/level_downsamples))

        crop_width = math.ceil((crop_end_x - crop_start_x))
        crop_height = math.ceil((crop_end_y - crop_start_y))

        if crop_height == 0 or crop_width == 0:
            return None

        crop_region = self.reader.read(series=crop_level,rescale=False,XYWH=(crop_start_x, crop_start_y, crop_width, crop_height))
        # crop_region = cv2.cvtColor(crop_region, cv2.COLOR_BGR2RGB)
        if greyscale:
            crop_region = 0.2989*crop_region[:,:,0] + 0.5870*crop_region[:,:,1] + 0.1140*crop_region[:,:,2]
            crop_region = crop_region[:,:,np.newaxis]

        crop_region = cv2.resize(crop_region, (math.ceil(crop_width*resize_ratio), math.ceil(crop_height*resize_ratio)))
        return crop_region


    def saveLabel(self,path):
        try:
            if self.label_series_idx is not None:
                label_img = self.reader.read(series=self.label_series_idx, rescale=False)
                label_img = (label_img-np.min(label_img))/(np.max(label_img)-np.min(label_img)) * 255
                cv2.imwrite(path, label_img)
        except:
            pass

    def macro(self,path):
        try:
            if self.macro_series_idx is not None:
                macro_img = self.reader.read(series=self.macro_series_idx, rescale=False)
                macro_img = (macro_img-np.min(macro_img))/(max(macro_img)-min(macro_img)) * 255
                cv2.imwrite(path, macro_img)
        except:
            pass

    @property
    def mpp(self):
        mpp = None
        try:
            if self._mpp is not None:
                return float(self._mpp)
            with open(os.path.join(os.path.dirname(self.filename), "index.json"), "r", encoding="utf-8") as f:
                slide_info = json.load(f)
                mpp = slide_info.get("mppx")
                return float(mpp)
        except:
            pass
        return mpp
