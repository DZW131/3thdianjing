#! /usr/bin/env python
# -*- coding:utf-8 -*-

from ctypes import *

class SqPicHead(Structure):  #sdpc文件头信息
    _pack_ = 1
    _fields_= [
        ('flag',c_ushort),
        ('version',c_ubyte*16),
        ('headSize',c_long),
        ('fileSize',c_ulonglong),
        ('macrograph',c_uint),
        ('personInfor',c_uint),
        ('hierarchy',c_uint),
        ('srcWidth',c_uint),
        ('srcHeight',c_uint),
        ('sliceWidth',c_uint),
        ('sliceHeight',c_uint),
        ('thumbnailWidth',c_uint),
        ('thumbnailHeight',c_uint),
        ('bpp',c_ubyte),
        ('quality',c_ubyte),
        ('colrSpace',c_int),  #枚举类型
        ('scale',c_float),
        ('ruler',c_double),
        ('rate', c_uint),
        ('extraOffset', c_size_t),
        ('tileOffset', c_size_t),
        ('headSpace', c_ubyte*252)
    ]

class SqPersonInfo(Structure):  #病人信息
    _pack_=1
    _fields_=[
        ('flag',c_ushort),
        ('inforSize',c_uint),
        ('pathologyID',c_ubyte*64),
        ('name',c_ubyte*64),
        ('sex',c_ubyte),
        ('age',c_ubyte),
        ('departments',c_ubyte*64),
        ('hospital',c_ubyte*64),
        ('submittedSamples',c_ubyte*1024),
        ('clinicalDiagnosis',c_ubyte*2048),
        ('pathologicalDiagnosis',c_ubyte*2048),
        ('reportDate',c_ubyte*64),
        ('attendingDoctor',c_ubyte*64),
        ('remark',c_ubyte*1024),
        ('nextOffset',c_size_t),
        ('reversed_1',c_uint),
        ('reversed_2',c_uint),
        ('reversed',c_ubyte*256),
    ]

class SqExtraInfo(Structure): #额外信息
    _pack_ = 1
    _fields_=[
        ('flag',c_short),
        ('inforSize',c_uint),
        ('nextOffset',c_size_t),
        ('model',c_ubyte*20),
        ('ccmGamma',c_float),
        ('ccmRgbRate',c_float*3),
        ('ccmHsvRate',c_float*3),
        ('ccm',c_float*9),
        ('timeConsuming',c_ubyte*32),
        ('scanTime',c_uint),
        ('stepTime',c_ushort*10),
        ('serial',c_ubyte*32),
        ('fusionLayer',c_ubyte),
        ('step',c_float),
        ('focusPoint',c_ushort),
        ('validFocusPoint',c_ushort),
        ('barCode',c_ubyte*128),
        ('cameraGamma',c_float),
        ('cameraExposure',c_float),
        ('cameraGain',c_float),
        ('reversed',c_ubyte*433),
    ]

class SqImageInfo(Structure): #宏观图信息/缩略图信息
    _pack_ = 1
    _fields_=[
        ('jpeg',POINTER(c_ubyte)),
        ('rgb',POINTER(c_ubyte)),
        ('width',c_int),
        ('height',c_int),
        ('channel',c_int),
        ('jpegSize',c_int),
    ]

class SqPicInfo(Structure):  #切片层级信息
    _pack_ = 1
    _fileds_ = [
        ('flag', c_ushort),
        ('infoSize', c_uint),
        ('layer', c_uint),
        ('sliceNum', c_uint),
        ('scliceNumX', c_uint),
        ('scliceNumY', c_uint),
        ('layerSize', c_size_t),
        ('nextLayerOffset', c_size_t),
        ('curScale', c_float),
        ('ruler', c_double),
        ('defaultX', c_uint),
        ('defaultY', c_uint),
        ('headSpace', c_ubyte * 64)
    ]

class SqSliceInfo(Structure):  #切片的位置和大小信息
    _pack_ = 1
    _fields_=[
        ('sliceOffset',POINTER(c_size_t)),
        ('sliceSize',POINTER(c_int))
    ]

class SqSdpcInfo(Structure):   #图片所有信息
    _pack_ = 1
    _fields_ = [
        ("fileName", c_char_p),
        ("picHead",POINTER(SqPicHead)),
        ("personInfo", POINTER(SqPersonInfo)),
        ("extra", POINTER(SqExtraInfo)),
        ("macrograph", POINTER(POINTER(SqImageInfo))),
        ("thumbnail", POINTER(SqImageInfo)),
        ("sliceLayerInfo", POINTER(POINTER(SqPicInfo))),
        ("sliceInfo", POINTER(POINTER(SqSliceInfo))),
]














