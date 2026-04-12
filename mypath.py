import os


class Path(object):
    @staticmethod
    def _project_data_path(*parts):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)

    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'her2_region':
            # return 'Her2Data/'
            return '/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/code/region_level_train/data/Her2Data_4class_deeplab'
        elif dataset == 'feiai_region':
            # return '/home/zhouzhihao/pathological_data/beichaoyang_feiai/train_data_merge/'
            # return '/home/songlinru/feiai_train/'
            return '/home/songlinru/data/jiezhichang_data'
        elif dataset == 'beiertongbxr_region':
            return '/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/code/region_level_train/data/beiertongbxr'
        elif dataset == 'qidai_region':
            return '/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/code/region_level_train/data/qidai'
        elif dataset == 'taimo_region':
            return '/media/zhouzhihao/DeepInformatic_dataset/lihansheng/HER2/code/region_level_train/data/taimo'
        elif dataset == 'prostate_tls':
            return r'D:\msqtry\region_level_train\EVAL_WSI_1GPU\patch384_lap192_2000_level3'
        elif dataset == 'jijie':
            project_dataset = Path._project_data_path('data', 'jijie')
            if os.path.isdir(project_dataset):
                return project_dataset
            return r'D:\msqtry\muscle indicator\patch_pengzhang'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
