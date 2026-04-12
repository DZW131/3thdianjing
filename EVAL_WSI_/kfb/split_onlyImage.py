import kfbslide
import os
import scipy.misc as misc
from pydaily import filesystem
from parse_embolus import read_region_kfb
import cv2

def split_kfb(slides_dir, save_dir,crop_len):
    slide_path, slide_name = os.path.split(slides_dir)
    kfb_name, kfb_ext = os.path.splitext(slide_name)
    slide = kfbslide.open_kfbslide(slides_dir)
    #slide=cv2.imread(slides_dir)
    #slide_size=slide.shape
    #width=slide_size[0]
    #height=slide_size[1]
    width, height = slide.level_dimensions[0]
    TILE_SIZE = 512
    width = width - width % TILE_SIZE
    height = height - height % TILE_SIZE
    num_x = int(width / crop_len-1)
    num_y = int(height / crop_len-1)
    num_patch = 0
    for i in range(num_y):
        crop_start_y = i * (crop_len)
        crop_start_x = 0
        for j in range(num_x):
            region_size = (crop_len, crop_len)
            region_start = (crop_start_x, crop_start_y)
            cur_region = read_region_kfb(slide, region_start, region_size)
            #cur_region=slide[crop_start_x:crop_start_x+crop_len,crop_start_y:crop_start_y+crop_len,:]
            print(str(width)+'-'+str(height)+str(kfb_name) + '-region_start:' + str(region_start))
            img_name = kfb_name + '-' + str(i) + '-' + str(j) + '-' + str(num_patch) + ".png"
            cur_region=misc.imresize(cur_region,(512,512,3))
            #cur_region=misc.imresize(cur_region,[512,512,3])
            misc.imsave(os.path.join(save_dir, img_name), cur_region)
            crop_start_x = (j + 1) * crop_len
            num_patch = num_patch + 1


def processing_batch(slides_dir, save_dir, crop_len):
    kfb_list = filesystem.find_ext_files(slides_dir, "kfb")
    for ind, cur_kfb in enumerate(kfb_list):
        print("Processing {:3d}/{:3d}".format(ind+1, len(kfb_list)))
        slide_path = os.path.join(slides_dir, cur_kfb)
        print(slide_path)
        split_kfb(slide_path, save_dir, crop_len)


if __name__=='__main__':
    #slides_dir='/media/kd/DS/sz2/PDL1/'
    #save_dir='/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/research/PDL12HE/test/PDL1_small_test_224_new/'
    #processing_batch(slides_dir, save_dir, 256)
    slides_dir ='/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/research/Image-Classifier-using-SVM-master/dataset/dataset/F53573-D2-40.kfb'
    save_dir='/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/research/Image-Classifier-using-SVM-master/dataset/dataset/F53573-D2-40'
    split_kfb(slides_dir, save_dir,2048)