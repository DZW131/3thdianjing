from pydaily import filesystem
import os
if __name__ == '__main__':
    # get_properties(slide_path)
    save_dir='../Her2Data'
    mrxs_img_folder = '/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/免疫组化her2切片/训练集/'
    from pydaily import filesystem
    mrxs_imglist = filesystem.find_ext_files(mrxs_img_folder, ".mrxs")
    json_list = filesystem.find_ext_files(mrxs_img_folder, ".json")
    with open(r'./slice_number.txt', 'w+') as f:
        for ind, slide_path in enumerate(mrxs_imglist):
            json_name = slide_path.replace('.mrxs', '.json')
            if json_name not in json_list:
                continue
            else:
                filepath, fullflname = os.path.split(json_name)
                fname, ext = os.path.splitext(fullflname)
                print(fname)
                f.write(fname+"\n")