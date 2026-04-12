#!/usr/bin/env python
# -*- coding:utf-8 -*-

import configparser
import os,sys
import math
import cv2
import collections
import numpy as np
import PIL
from PIL import Image

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from ..SlideBase import SlideBase

class HdxSlide(SlideBase):

    def __init__(self, filename):
        self.filename = filename
        self.config = configparser.ConfigParser()
        self.config.read(os.path.splitext(filename)[0] + os.sep + 'Scan.txt')
        self.width = int(self.config.get('General', 'sceneWidthAfterProcessed'))
        self.height = int(self.config.get('General', 'sceneHeightAfterProcessed'))
        self.rowcount=int(self.config.get('General', 'RowCount'))
        self.columncount=int(self.config.get('General', 'ColumnCount'))
        self.tilewidth=int(self.config.get('General', 'ImageWidth'))
        self.tileheight=int(self.config.get('General', 'ImageHeight'))
        self.lablename=self.config.get('General', 'Preview')
        self.backgroundImagePosX=int(self.config.get('General', 'backgroundImagePosX'))
        self.backgroundImagePosY=int(self.config.get('General', 'backgroundImagePosY'))
        convert=lambda x:str(x+1001)[1:]
        self.picinfo=[]   #存放每个切片的坐标和名字信息
        res=collections.OrderedDict()
        for i in self.config.options('Images'):
            res[i]=self.config.get('Images',i)
        for i in range(self.columncount):
            for j in range(self.rowcount):
                tileinfo={}              #将每个切片的信息存储{x：xx，y：xx，filename：xx}
                tileinfo['filename']=convert(j)+'x'+convert(i)
                tileinfo['x']=int(res['col'+tileinfo['filename']])-self.backgroundImagePosX
                tileinfo['y']=int(res['row'+tileinfo['filename']])-self.backgroundImagePosY
                self.picinfo.append(tileinfo)
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

        crop_level = math.floor(math.log2(scale))

        # print('crop_level',crop_level
        if crop_level>2:
            width=int(width/scale)
            height=int(height/scale)
            y=int(location[1]/scale)
            yn=int(y+height)
            x=int(location[0]/scale)
            xn=int(x+width)
            finalpic=os.path.splitext(self.filename)[0] + os.sep + 'Thumbs'+os.sep+'Result-%s.jpg'%self.config.get('General','SlideID')
            img=cv2.imdecode(np.fromfile(finalpic,dtype=np.uint8),cv2.IMREAD_COLOR)
            finalpicheight=img.shape[0]
            resize_ratio=self.height/scale/finalpicheight
            resize_width=int(resize_ratio*img.shape[1])
            resize_height=int(resize_ratio*img.shape[0])
            x, y = min(max(x, 0), resize_width), min(max(y, 0), resize_height)
            xn = math.ceil(min(max(width + x, 0), resize_width))
            yn = math.ceil(min(max(height + y, 0), resize_height))
            if resize_width==0 and resize_height==0:
                crop_region=cv2.resize(img,(1,1))
            elif resize_width==0 and resize_height!=0:
                crop_region=cv2.resize(img,(1,resize_height))
            elif resize_width!=0 and resize_height==0:
                crop_region=cv2.resize(img,(resize_width,1))
            else:
                crop_region=cv2.resize(img,(resize_width,resize_height))
            crop_region=crop_region[y:yn,x:xn,:]
            crop_region = crop_region[:, :, ::-1]
            return crop_region

        else:
            x=location[0]          #在第0层算出要找的tile的坐标信息
            xn= min(x+width,self.width)
            y=location[1]
            yn=min(y+height,self.height)
            crop_region = Image.new("RGB",(int((xn-x)/scale),int((yn-y)/scale)))   #构造一个空的tile
            for k in range(len(self.picinfo)):  #循环遍历所有小的切片
                if self.picinfo[k]['x']+self.tilewidth<x or self.picinfo[k]['x']>xn or self.picinfo[k]['y']+self.tileheight<y or self.picinfo[k]['y']>yn:  #和要找的tile不相交就跳过
                    continue
                try:
                    jpg = Image.open(os.path.splitext(self.filename)[0] + os.sep+'Images\\IMG'+self.picinfo[k]['filename']+'.jpg')    #打开和要找的tile相交的小图
                    jpg = jpg.resize((int(self.tilewidth/scale), int(self.tileheight/scale)))
                    crop_region.paste(jpg,(int(int(self.picinfo[k]['x']-x)/scale),int(int(self.picinfo[k]['y'] - y)/scale)))       #将需要的信息画在crop_region上
                except:
                    pass
        return np.array(crop_region)



    def saveLabel(self, path):

        labelpath=os.path.splitext(self.filename)[0] + os.sep +self.lablename
        # print('labelpath',labelpath)
        with open(labelpath,'rb') as f1:
            with open(path,'wb') as f2:
                f2.write(f1.read())

    def getThumbnail(self,size):

        finalpic=os.path.splitext(self.filename)[0] + os.sep + 'Thumbs'+os.sep+'Result-%s.jpg'%self.config.get('General','SlideID')
        img=cv2.imdecode(np.fromfile(finalpic,dtype=np.uint8),cv2.IMREAD_COLOR)
        img=cv2.resize(img,(size,size))
        img=Image.fromarray(img[:,:,::-1])

        return img

    @property
    def mpp(self):

        return float(self.config.get('General', 'Size').split('mm')[0].split('x')[0])*1000/self.width


if __name__ == "__main__":
    # filename=r'D:\new\data\xjh\data\2020_05_11_14_54_39_62505\slices\17690\1905697.hdx'
    filename=r'Y:\hdx\data\2020_05_18_18_27_12_285092\slices\66580\1905073.hdx'
    # filename1=r'D:\new\data\xjh\data\2020_05_11_14_54_39_62505\slices\17690\1905697\Images\IMG004x013.jpg'
    # filename2=r'D:\new\data\xjh\data\2020_05_11_14_54_39_62505\slices\17690\1905697\Images\IMG004x014.jpg'
    slide = HdxSlide(filename)
    # slide.saveLabel(r'D:\new\data\company1\data\2020_05_12_10_20_25_330421\slices\86491\label.png')
    # # print(slide.width)
    res=slide.getTile(0,0,14)
    # print(res)
    res.show()

    # t=slide.getThumbnail(500)
    # t.show()

    # print(res)
    # # print(slide.mpp)
    # # print(res)
    # for z in range(12,13):
    #     os.makedirs("testhdx/{}".format(z), exist_ok= True)
    #     for x in range(5):
    #         for y in range(5):
    #             try:
    #                 img = slide.getTile(x,y,z)
    #                 img.save("testhdx/{}/{}_{}.jpg".format(z,y,x))
    #             except:
    #                 pass
