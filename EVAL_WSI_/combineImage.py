import os
from PIL import Image
from pydaily import filesystem
import cv2
import numpy as np
def judgePicture(filename,type):
    ext = filename.split('.')[1]
    if ext == type:
        return True
    else:
        return False


def slideWindow(img,image_name, save_dir, image_width,image_height,patch_size):
        img = np.array(img).astype(np.float)
        #height, width, depth = image.shape
        width=image_width
        height=image_height
        print(width, height)
        num_x = int(width / patch_size)
        num_y = int(height / patch_size)
        num_patch = 0
        for i in range(num_y):
            crop_start_y = i * (patch_size)
            crop_start_x = 0
            for j in range(num_x):
                region_start = (crop_start_x, crop_start_y)
                cropImg = img[crop_start_y:crop_start_y + patch_size, crop_start_x:crop_start_x + patch_size]
                print(str(image_name) + '-region_start:' + str(region_start))
                img_name = image_name + '-' + str(i) + '-' + str(j) + '-' + str(num_patch) + ".png"
                cv2.imwrite(save_dir + '/' + img_name, cropImg)
                crop_start_x = (j + 1) * patch_size
                num_patch = num_patch + 1



def merged(list_filt,patchs_path,save_dir,image_size,overlay_width,overlay_height,resize_number):
    #the size of unoverlay
    width=int((image_size-overlay_width)/resize_number)
    height=int((image_size-overlay_height)/resize_number)

    print(len(list_filt))
    # get number of rows and lines
    max_width = 0
    max_height = 0
    min_width = 1000000
    min_height = 1000000

    # list_filt = []
    # for x in list:
    #     list_filt.append(x)
    # print('1list_filt:' + str(len(list_filt)))

    for x in list_filt:#需要修改
        patch_name = os.path.splitext(x)

        index = patch_name[0].split('-')
        if int(index[len(index) - 3]) > max_width:
            max_width = int(index[len(index) - 3])
        if int(index[len(index) - 2]) > max_height:
            max_height = int(index[len(index) - 2])
        if int(index[len(index) - 3]) < min_width:
            min_width = int(index[len(index) - 3])
        if int(index[len(index) - 2]) < min_height:
            min_height = int(index[len(index) - 2])

    for i in range(len(list_filt) - 1):
        for j in range(len(list_filt) - i - 1):
            patch_name_j = os.path.splitext(list_filt[j])
            index_j = patch_name_j[0].split('-')
            patch_name_jp = os.path.splitext(list_filt[j + 1])
            index_jp = patch_name_jp[0].split('-')
            num1=index_j[len(index_j) - 1].split('_')[0]
            num2=index_jp[len(index_jp) - 1].split('_')[0]
            if int(num1) > int(num2):#需要修改啊
                temp = list_filt[j + 1]
                list_filt[j + 1] = list_filt[j]
                list_filt[j] = temp

    # for i in range(len(list_filt)):
    #     list_filt[i]=patchs_path+'/'+list_filt[i]

    row_max = max_height-min_height+1
    line_max = max_width-min_width+1
    print('row_max:'+str(row_max)+' line_max:'+str(line_max))

    image_width=width*(row_max-1)+int(image_size/resize_number)
    image_height=height*(line_max-1)+int(image_size/resize_number)
    toImage = Image.new('RGB', (image_width,image_height))
    print(image_width,image_height)
    print('2list_filt:'+str(len(list_filt)))
    num = 0
    for i in range(0, line_max):
        for j in range(0, row_max):
            pic_fole_head = Image.open(list_filt[num]).resize((int(image_size/resize_number),int(image_size/resize_number)))
            loc = (int(j * (width)),int(i * (height)))
            print(str(list_filt[num]) + " location:" + str(loc))
            toImage.paste(pic_fole_head, loc)
            num = num + 1
    print(num)
    print(toImage.size)
    save_name=os.path.split(list_filt[0])[1]
    save_name=os.path.splitext(save_name)[0]
    index_name=save_name.split('-')
    save_name=''
    for i in range(len(index_name)-3):
        if i==0:
            save_name = save_name + str(index_name[i])
        else:
            save_name = save_name + '-' + str(index_name[i])

    #slideWindow(toImage,save_name, pathchsave_dir,image_width,image_height,1024)
    toImage.save(save_dir + '/' + save_name + '.png')
    print(save_dir + '/' + save_name + '.png')

    # pathchsave_dir = '/media/kd/50e6b0a1-557f-451b-a4bc-34464a164acf/testA_patch'
    # #toImage=np.array(toImage).astype(np.float)
    # width = image_width
    # height = image_height
    # print(width, height)
    # patch_size=1024
    # num_x = int( width/ patch_size)
    # num_y = int(height / patch_size)
    # num_patch = 0
    # for i in range(num_y):
    #     crop_start_y = i * (patch_size)
    #     crop_start_x = 0
    #     for j in range(num_x):
    #         region_start = (crop_start_x, crop_start_y)
    #         box=(crop_start_x,crop_start_y,crop_start_x + patch_size,crop_start_y + patch_size)
    #         cropImg = toImage.crop(box)
    #         #cropImg=np.array(cropImg).astype(np.float)
    #         #[crop_start_y:crop_start_y + patch_size, crop_start_x:crop_start_x + patch_size]
    #         print(str(save_name) + '-region_start:' + str(region_start))
    #         img_name = save_name + '-' + str(i) + '-' + str(j) + '-' + str(num_patch) + ".png"
    #         img_name=pathchsave_dir + '/' + img_name
    #         cropImg.save(img_name)
    #         #cv2.imwrite(pathchsave_dir + '/' + img_name, cropImg)
    #         crop_start_x = (j + 1) * patch_size
    #         num_patch = num_patch + 1




def split_image(image_path):
    patch_name = os.path.splitext(image_path)
    index = patch_name[0].split('-')
    name=''
    num=0
    for x in index[0:(len(index)-3)]:#需要修改
        if num==0:
            name=name+x
        else:
            name=name+'-'+x
        num=num+1
    return name



def processingBatch(patchs_path,save_dir,image_size,overlay_width,overlay_height,resize_number):
    '''获取目录下不同病理图像的名字'''
    name_image=[]
    #list = os.listdir(patchs_path)
    list = filesystem.find_ext_files(patchs_path, "png")#need to change

    name_image.append(split_image(list[0]))
    for x in list:
        judge=True
        for y in name_image:
            if split_image(x)==y:
                judge=False
        if judge:
            name_image.append(split_image(x))

    for x in name_image:
        print(x)

    '''将属于同一病理图像的patch放到一个列表中'''
    patchsList_list=[]
    for x in name_image:
        patchsList_list.append([])
    for x in list:
        num=0
        for y in name_image:
            if split_image(x)==y:
                patchsList_list[num].append(x)
                break
            else:
                num=num+1

    num=1
    for x in patchsList_list:
        merged(x, patchs_path,save_dir,image_size,overlay_width,overlay_height,resize_number)

def combine(path,save,size):
    processingBatch(path, save,size, 0, 0, 1)


if __name__=='__main__':
    path = '/media/deepin/DeepInformatic_dataset/lihansheng/HER2/Exp_v2/shangzhong0120/14-00069E-2021-02-23_19_09_40'
    save = "/media/deepin/DeepInformatic_dataset/lihansheng/HER2/Exp_v2/shangzhong0120/combine"
    os.makedirs(save,exist_ok=True)
    processingBatch(path, save,1024, 0, 0, 1)



