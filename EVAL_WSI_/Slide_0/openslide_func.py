import os.path

from EVAL_WSI_1GPU.Slide.SdpcSlide.tool import SdpcSlide
from .OtherSlide import OtherSlide
####from .KfbSlide import KfbSlide
from scipy import misc
from .LRUCacheDict import LRUCacheDict
from threading import Lock
import openslide
slides = LRUCacheDict()
_dict_lock = Lock()
from pydaily import filesystem


# 公共方法，打开一个切片，直接返回合适的对象
def openSlide(filename):
    ext = os.path.splitext(filename)[1][1:].lower()

    if filename in slides:
        return slides[filename]

    with _dict_lock:
        if filename in slides:
            return slides[filename]

        # print("新加载一张切片：" + filename)

        slide = None
        if ext == 'kfb':  # 宁波江丰
            slide = KfbSlide(filename)
        # if ext == 'ndpi':  # 宁波江丰
        #     slide = openslide.open_slide(filename)
        elif ext == 'sdpc':#深圳生强
            slide = SdpcSlide(filename)
        # elif ext == 'mdsx':#麦克奥迪
        #     slide = MdsxSlide(filename)
        # elif ext == 'czi':
        #     # 使用该格式需要装jdk numpy1.16 boto3 多线程需关掉
        #     from .BioformatsSlide import BioformatsSlide
        #     slide = BioformatsSlide(filename)
        # elif ext == 'mrxs':#3Dtech
        #     slide = MrxsSlide(filename)
        else:# open slide
             slide = OtherSlide(filename)

        slides[filename] = slide
        print("切片加载完成：" + filename)
        return slide


def getFiles(dir, suffix):  # 查找根目录，文件后缀
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename)  # =>文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename))  # =>吧一串字符串组合成路径
    return res


if __name__ == '__main__':

    test_dir = "/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/数据集/PDL1_Positive_cell_seg_detection/PDL1200切片/Slide"
    root_dir = '/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/数据集/PDL1_Positive_cell_seg_detection/PDL1200切片/region_level5'
    ndpi_list = filesystem.find_ext_files(test_dir, ".ndpi")
    kfb_list = filesystem.find_ext_files(test_dir, '.kfb')
    sdpc_list = filesystem.find_ext_files(test_dir, '.sdpc')
    for file in ndpi_list:
        # file='/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/数据集/PDL1_Positive_cell_seg_detection/PDL1200切片/PD-L1SP263原切片/SP263/sp263 协和腺癌/A030 PD-L1 V-.kfb'
        image_name = str(os.path.split(file)[-1].split('.')[0]) + ".png"
        if 'HE' not in image_name and 'NC' not in image_name:
            slide = openSlide(file)
            size = slide.slide.level_dimensions[0]
            #size = slide.level_dimensions[0]
            if size[0] > 2 ** 5 and size[1] > 2 **5:
                try:
                    cur_slide = slide.read(location=[0, 0], size=size, scale=32)
                    #cur_slide=slide.read(location=[0, 0], size=size,scale=16)
                    misc.imsave(os.path.join(root_dir, image_name), cur_slide)
                except:
                    print("Fail to read region")

        #print(1)

    for file in kfb_list:
        # file='/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/数据集/PDL1_Positive_cell_seg_detection/PDL1200切片/PD-L1SP263原切片/SP263/sp263 协和腺癌/A030 PD-L1 V-.kfb'
        image_name = str(os.path.split(file)[-1].split('.')[0]) + ".png"
        if 'HE' not in image_name and 'NC' not in image_name:
            slide = openSlide(file)
            size = slide.slide.level_dimensions[0]
            # size = slide.level_dimensions[0]
            if size[0] > 2 ** 5 and size[1] > 2 **5:
                try:
                    cur_slide = slide.read(location=[0, 0], size=size, scale=32)
                    # cur_slide=slide.read(location=[0, 0], size=size,scale=16)
                    misc.imsave(os.path.join(root_dir, image_name), cur_slide)
                except:
                    print("Fail to read region")

        #print(1)
    for file in sdpc_list:
        image_name = str(os.path.split(file)[-1].split('.')[0]) + ".png"
        if 'HE' not in image_name and 'NC' not in image_name:
            slide = openSlide(file)
            size = slide.slide.level_dimensions[0]
            # size = slide.level_dimensions[0]
            if size[0] > 2 ** 5 and size[1] > 2 ** 5:
                try:
                    cur_slide = slide.read(location=[0, 0], size=size, scale=32)
                    # cur_slide=slide.read(location=[0, 0], size=size,scale=16)
                    misc.imsave(os.path.join(root_dir, image_name), cur_slide)
                except:
                    print("Fail to read region")