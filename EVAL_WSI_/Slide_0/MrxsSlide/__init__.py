import os.path,math,json
from SlideBase import SlideBase
import cv2
import numpy as np
from PIL import Image
from ctypes import *
import os, sys
from threading import Lock
del_lock=Lock()

os.environ["PATH"] = os.path.abspath(__file__ + "/../dll/") + ";" + os.environ["PATH"]
_mrxs_lib = windll.LoadLibrary("SlideAcWrapper.dll")




class _GB18030_p(object):
    @classmethod
    def from_param(cls, obj):
        if isinstance(obj, bytes):
            return obj
        elif isinstance(obj, str):
            return obj.encode('GB18030')
# # resolve and return an OpenSlide function with the specified properties
def _func(name, restype, argtypes):
    func = getattr(_mrxs_lib, name)
    func.argtypes = argtypes
    func.restype = restype
    return func

close_Slide = _func('CloseSlide', c_bool, [c_void_p])
open_Slide = _func('OpenSlide', c_void_p, [_GB18030_p])
get_Image = _func("GetImage", c_uint,[c_void_p, c_long, c_long, c_long, c_long, c_long, c_void_p])
get_IntSlideProperties = _func("GetIntSlideProperties", c_uint, [c_void_p, c_uint])
get_DoubleSlideProperties = _func("GetDoubleSlideProperties", c_double, [c_void_p, c_uint])
get_AssociatedImages = _func("GetAssociatedImages", c_uint, [c_void_p, c_long, c_void_p])
get_AssociatedImagesProperties = _func("GetAssociatedImagesProperties", c_bool,[c_void_p, c_long, POINTER(c_uint), POINTER(c_uint), POINTER(c_uint)])





class MrxsSlide(SlideBase):
    def __init__(self,filename):
        self.filename=filename[0:-5]
        self.slide = open_Slide(self.filename)
        #print("打開的文件為：{}，打開的slide指針為：{}".format( self.filename,self.slide))
        self.width=get_IntSlideProperties(self.slide,306)
        self.height=get_IntSlideProperties(self.slide,307)
        SlideBase.__init__(self)

    def __del__(self):
        with del_lock:
            #print("开始释放",self.slide)
            _bool=close_Slide(self.slide)
            if _bool is False:
                print(_bool,self.slide,"释放失败")
            else:
                pass
                #print(_bool,self.slide,"釋放成功")

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
        crop_level = max(0,min(crop_level, 9))
        level_ratio = 2**crop_level
        resize_ratio = level_ratio/scale

        # make sure the crop region is inside the slide
        crop_start_x, crop_start_y = min(max(crop_start_x, 0), self.width), min(max(crop_start_y, 0), self.height)
        crop_end_x = math.ceil(min(max(width+crop_start_x, 0), self.width))
        crop_end_y = math.ceil(min(max(height+crop_start_y, 0), self.height))

        rx = math.floor(crop_start_x / 256 / level_ratio) * 256
        ry = math.floor(crop_start_y / 256 / level_ratio) * 256

        start_x=rx >> 8
        start_y=ry >> 8
        end_x=math.floor((crop_end_x-1) / 256 / level_ratio)
        end_y= math.floor((crop_end_y-1) / 256 / level_ratio)
        crop_width, crop_height = (end_x - start_x + 1) << 8, (end_y - start_y + 1) << 8
        if crop_height == 0 or crop_width == 0:
            return None
        data_length,bits =self.read_region(self.slide,start_x,start_y,end_x,end_y,crop_width,crop_height,crop_level)
        s=int(data_length/crop_height)
        cut_len=s-3*crop_width
        crop_region =np.frombuffer(bits,np.uint8,data_length)
        if cut_len!= 0:
            print("cut_len:",cut_len)
            crop_region = np.reshape(crop_region, (crop_height, s))[:, :-cut_len]
        crop_region = crop_region.reshape(crop_height,crop_width, 3)
        crop_region = crop_region[ int(crop_start_y) // level_ratio - ry : int(crop_end_y) // level_ratio - ry, \
                      int(crop_start_x) // level_ratio - rx : int(crop_end_x) // level_ratio - rx,:]
        if greyscale:
            crop_region = 0.2989*crop_region[:,:,0] + 0.5870*crop_region[:,:,1] + 0.1140*crop_region[:,:,2]
            crop_region = crop_region[:,:,np.newaxis]

        h,w = crop_region.shape[0:2]
        resize_width=int(w*resize_ratio)
        resize_height=int(h*resize_ratio)
        if resize_width==0:
            resize_width=1
        if resize_height==0:
            resize_height=1
        crop_region = cv2.resize(crop_region, (resize_width, resize_height))

        return crop_region

    def saveLabel(self,path):
        try:
            s, ass_width, ass_height = self.get_associated_images_properties(self.slide, 100)
            ass_bits = self.get_associated_images(slide=self.slide, image_type=100, s=s, ass_height=ass_height)
            image = np.ctypeslib.as_array(ass_bits)
            cut_len = s - ass_width * 3
            if ass_height * ass_width * 3 != len(image):
                image = np.reshape(image, (ass_height, s))[:, :-cut_len]
            image = image.reshape(ass_height, ass_width, 3)
            image = self.rotate(image, 180) #因为保存的时候倒了，需要旋转一下
            cv2.imwrite(path,image)
        except:
            pass

    @property
    def mpp(self):
        mpp = None
        try:
            if get_DoubleSlideProperties(self.slide,308) is not None:
                return float(get_DoubleSlideProperties(self.slide,308))
            with open(os.path.join(os.path.dirname(self.filename), "index.json"), "r", encoding="utf-8") as f:
                slide_info = json.load(f)
                mpp = slide_info.get("mppx")
                if mpp is not None:
                    return float(mpp)
        except:
            pass
        return mpp

    def getThumbnail(self, size=500):
        '''
        :param size: thumbnail image size
        :return:  a thumbnail image
        '''
        s, ass_width, ass_height = self.get_associated_images_properties(self.slide, 102)
        ass_bits = self.get_associated_images(slide=self.slide, image_type=102, s=s, ass_height=ass_height)
        image = np.ctypeslib.as_array(ass_bits)
        # image = np.flipud(image)
        cut_len = s - ass_width * 3
        if ass_height * ass_width * 3 != len(image):
            image = np.reshape(image, (ass_height, s))[:, :-cut_len]
        image = image.reshape(ass_height, ass_width, 3)[:, :, ::-1]
        thumbnail_img = Image.fromarray(image, mode='RGB')
        if thumbnail_img:
            if thumbnail_img.mode == 'RGBA':
                thumbnail_img = thumbnail_img.convert('RGB')
        return thumbnail_img


    def rotate(self,image, angle, center=None, scale=1.0):
        # 获取图像尺寸
        (h, w) = image.shape[:2]
        # 若未指定旋转中心，则将图像中心设为旋转中心
        if center is None:
            center = (w / 2, h / 2)
        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        image = cv2.warpAffine(image, M, (w, h))
        # 返回旋转后的图像
        return image

    def get_associated_images(self,slide, image_type, s, ass_height):
        len = ass_height * s
        ass_bits = (len * c_uint8)()
        res = get_AssociatedImages(slide, image_type, pointer(ass_bits))
        if res is None:
            print("Fail to mrxsslide_associated_images")
        return ass_bits

    def get_associated_images_properties(self,slide, image_type):
        s = c_uint()
        ass_height = c_uint()
        ass_width = c_uint()
        res = get_AssociatedImagesProperties(slide, image_type, pointer(s), pointer(ass_width), pointer(ass_height))
        if res is None:
            print("Fail to mrxsslide_get_properties")
        return s.value, ass_width.value, ass_height.value

    def read_region(self,slide,x1, y1, x2, y2, w, h, level):
        len = 3*h*(w+10)
        bits = (len * c_uint8)()
        data_length = get_Image(slide, x1, y1, x2, y2, level, pointer(bits))
        if not data_length:
            print("Fail to mrxsslide_read_region")
        return data_length,  bits



