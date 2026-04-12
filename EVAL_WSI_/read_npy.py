# import numpy as np
# data = np.load('/home/songlinru/Project/region_level_train/EVAL_WSI_1GPU/mean_std.npy')
# print(data)
import openslide

# 打开SVS切片文件
slide = openslide.OpenSlide('/home/songlinru/data/WSI/河南省妇幼保健院_2404_子宫诊刮项目/240628_第三批数据/化生/B2410383-9.2.svs')

# 获取切片的元数据
print(slide.properties)

# 获取切片的分辨率
mpp_x, mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_X), slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
print(f"X resolution: {mpp_x} μm/pixel")
print(f"Y resolution: {mpp_y} μm/pixel")

# 读取切片的某个区域
# level = 0 # 从最高分辨率级别开始读取
# region = slide.read_region((x, y), level, (w, h)) # 读取坐标(x, y)处，大小为(w, h)的区域

# 关闭切片文件
slide.close()