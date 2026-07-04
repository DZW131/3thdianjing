from copy import deepcopy


CLASS_NAMES = [
    "Background",
    "严重损伤线粒体",
    "中度损伤线粒体",
    "健康线粒体",
    "自噬线粒体",
    "肌浆网",
    "闰盘",
    "Z线样物质堆积",
    "脂滴",
    "糖原颗粒",
    "Z线",
    "T管",
    "M线",
    "心肌侧管",
]


TASK_PRESETS = {
    "mito": {
        "task_name": "mito",
        "task_display_name": "线粒体状态分割",
        "selected_classes": [1, 2, 3, 4],
        "metric_target_original_classes": [1, 2, 3, 4],
        "quantify_original_classes": [1, 2, 3, 4],
        "manifest_dir": "data/jijie/ImageSets/Segmentation/tasks/mito/default",
        "train_resize_mode": "random_crop",
        "eval_resize_mode": "none",
        "crop_size": 640,
        "base_size": 640,
        "batch_size": 4,
        "test_batch_size": 1,
        "epochs": 150,
        "lr": 0.007,
        "loss_type": "ce_dice",
        "selection_metric": "mean_positive_dice",
        "inference_mode": "sliding",
        "sliding_window_size": 640,
        "sliding_window_stride": 320,
        "checkname": "jijie_mito_v0_2",
    },
    "mito_sr": {
        "task_name": "mito_sr",
        "task_display_name": "线粒体-肌浆网关系",
        "selected_classes": [1, 2, 3, 4, 5],
        "metric_target_original_classes": [1, 2, 3, 4, 5],
        "quantify_original_classes": [1, 2, 3, 4, 5],
        "manifest_dir": "data/jijie/ImageSets/Segmentation/tasks/mito_sr/default",
        "train_resize_mode": "random_crop",
        "eval_resize_mode": "none",
        "crop_size": 768,
        "base_size": 768,
        "batch_size": 2,
        "test_batch_size": 1,
        "epochs": 160,
        "lr": 0.005,
        "loss_type": "ce_dice",
        "selection_metric": "mean_positive_dice",
        "inference_mode": "sliding",
        "sliding_window_size": 768,
        "sliding_window_stride": 384,
        "checkname": "jijie_mito_sr_v0_2",
    },
    "sarcomere": {
        "task_name": "sarcomere",
        "task_display_name": "肌节几何结构",
        "selected_classes": [10, 11, 12, 13],
        "metric_target_original_classes": [10, 12],
        "quantify_original_classes": [10, 11, 12, 13],
        "manifest_dir": "data/jijie/ImageSets/Segmentation/tasks/sarcomere/default",
        "train_resize_mode": "random_crop",
        "eval_resize_mode": "none",
        "crop_size": 1024,
        "base_size": 1024,
        "batch_size": 1,
        "test_batch_size": 1,
        "epochs": 180,
        "lr": 0.003,
        "loss_type": "ce_dice",
        "selection_metric": "mean_target_iou",
        "inference_mode": "sliding",
        "sliding_window_size": 1024,
        "sliding_window_stride": 512,
        "label_dilate_original_classes": [10, 12],
        "label_dilate_radius": 1,
        "checkname": "jijie_sarcomere_v0_2",
    },
}


def get_task_preset(task_name):
    if not task_name:
        return None
    preset = TASK_PRESETS.get(task_name)
    if preset is None:
        raise KeyError("Unknown jijie task preset: {}".format(task_name))
    return deepcopy(preset)
