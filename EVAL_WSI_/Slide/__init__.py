#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
# from .MrxsSlide import MrxsSlide
from .SdpcSlide.tool import SdpcSlide
from .OtherSlide.tool import OtherSlide
from .MdsxSlide.tool import MdsxSlide
from .KfbSlide.tool import KfbSlide
from .HdxSlide.tool import HdxSlide
# from .ZySlide.tool import ZySlide
from .TmapSlide.tool import TmapSlide
# from .ZYPSlide.tool import ZYPSlide


# 公共方法，打开一个切片，直接返回合适的对象
def openSlide(filename):
    ext = os.path.splitext(filename)[1][1:].lower()

    if ext == 'kfb':#宁波江丰
        slide = KfbSlide(filename)
    elif ext == 'sdpc':#深圳生强
        slide = SdpcSlide(filename)
    elif ext == 'mdsx':#麦克奥迪
        slide = MdsxSlide(filename)
    elif ext == 'hdx': #海德星5片机
        slide= HdxSlide(filename)
    elif ext == 'zyp': #志盈
        slide = ZYPSlide(filename)
    elif ext == 'tmap': #优纳
        slide = TmapSlide(filename)
    else:  #openslide支持的切片格式
        slide = OtherSlide(filename)

    return slide

