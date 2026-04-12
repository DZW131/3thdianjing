#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
# from .MrxsSlide import MrxsSlide


# 公共方法，打开一个切片，直接返回合适的对象
import sys


def openSlide(filename):
    ext = os.path.splitext(filename)[1][1:].lower()

    if ext == 'kfb':#宁波江丰
        from .KfbSlide.tool import KfbSlide
        slide = KfbSlide(filename)
    elif ext == 'sdpc':#深圳生强
        from .SdpcSlide.tool import SdpcSlide
        slide = SdpcSlide(filename)
    elif ext == 'mdsx':#麦克奥迪
        from .MdsxSlide.tool import MdsxSlide
        slide = MdsxSlide(filename)
    elif ext == 'hdx': #海德星5片机
        from .HdxSlide.tool import HdxSlide
        slide= HdxSlide(filename)
    #志盈和优纳linux版本还有些问题，后期改完再统一
    elif ext == 'zyp' and sys.platform=='win32': #志盈
        from .ZYPSlide.tool import ZYPSlide
        slide = ZYPSlide(filename)
    elif ext == 'tmap' and sys.platform=='win32': #优纳
        from EVAL_WSI_1GPU.TmapSlide import TmapSlide
        slide = TmapSlide(filename)
    else:  #openslide支持的切片格式
        from .OtherSlide.tool import OtherSlide
        slide = OtherSlide(filename)

    return slide

