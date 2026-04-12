from __future__ import division, print_function

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders import custom_transforms as tr
from mypath import Path


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

SPLIT_PROFILE_SUFFIX = {
    "annotated": "",
    "all_usable": "_all_usable",
}


def _normalize_selected_classes(selected_classes):
    if selected_classes is None:
        return None

    if isinstance(selected_classes, str):
        selected_classes = [item.strip() for item in selected_classes.split(",") if item.strip()]

    normalized = []
    for class_id in selected_classes:
        value = int(class_id)
        if value <= 0 or value >= len(CLASS_NAMES):
            raise ValueError("selected class ids must be in [1, {}], got {}".format(len(CLASS_NAMES) - 1, value))
        if value not in normalized:
            normalized.append(value)
    return normalized


class JijieSegmentation(Dataset):
    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(self, args, base_dir=Path.db_root_dir("jijie"), split="train"):
        super().__init__()
        self.args = args
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, "JPEGImages")
        self._cat_dir = os.path.join(self._base_dir, "SegmentationClass")
        self._splits_dir = os.path.join(self._base_dir, "ImageSets", "Segmentation")

        if isinstance(split, str):
            self.split = [split]
        else:
            self.split = sorted(split)

        self.split_profile = getattr(args, "split_profile", "annotated")
        if self.split_profile not in SPLIT_PROFILE_SUFFIX:
            raise ValueError(
                "Unsupported split profile '{}'. Expected one of {}.".format(
                    self.split_profile, sorted(SPLIT_PROFILE_SUFFIX)
                )
            )

        self.resize_mode = getattr(args, "resize_mode", "pad")
        self.selected_classes = _normalize_selected_classes(getattr(args, "selected_classes", None))
        self.class_mapping = None
        self.class_names = CLASS_NAMES
        self.NUM_CLASSES = len(CLASS_NAMES)

        if self.selected_classes is not None:
            self.class_mapping = {0: 0}
            for new_id, old_id in enumerate(self.selected_classes, 1):
                self.class_mapping[old_id] = new_id
            self.class_names = [CLASS_NAMES[0]] + [CLASS_NAMES[class_id] for class_id in self.selected_classes]
            self.NUM_CLASSES = len(self.class_names)

        self.im_ids = []
        self.images = []
        self.categories = []

        for split_name in self.split:
            split_file = self._resolve_split_file(split_name)
            with open(split_file, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()

            for image_id in lines:
                image_path = os.path.join(self._image_dir, image_id + ".jpg")
                mask_path = os.path.join(self._cat_dir, image_id + ".png")
                if not os.path.isfile(image_path):
                    raise FileNotFoundError("Missing image file: {}".format(image_path))
                if not os.path.isfile(mask_path):
                    raise FileNotFoundError("Missing mask file: {}".format(mask_path))
                self.im_ids.append(image_id)
                self.images.append(image_path)
                self.categories.append(mask_path)

        assert len(self.images) == len(self.categories)
        print(
            "Loaded jijie split={} profile={} size_policy={} samples={}".format(
                self.split,
                self.split_profile,
                self.resize_mode,
                len(self.images),
            )
        )

    def _resolve_split_file(self, split_name):
        suffix = SPLIT_PROFILE_SUFFIX[self.split_profile]
        split_file = os.path.join(self._splits_dir, "{}{}.txt".format(split_name, suffix))
        if not os.path.isfile(split_file):
            raise FileNotFoundError("Missing split file: {}".format(split_file))
        return split_file

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, target = self._make_img_gt_point_pair(index)
        sample = {"image": image, "label": target}

        if "train" in self.split:
            return self.transform_tr(sample)
        return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.categories[index])

        if self.class_mapping is not None:
            target = self._remap_classes(target)

        return image, target

    def _remap_classes(self, target_pil):
        target_array = np.array(target_pil)
        new_target = np.zeros_like(target_array)

        for old_class, new_class in self.class_mapping.items():
            new_target[target_array == old_class] = new_class

        selected_mask = np.zeros_like(target_array, dtype=bool)
        for old_class in self.class_mapping.keys():
            selected_mask |= target_array == old_class
        new_target[~selected_mask] = 0
        return Image.fromarray(new_target.astype(np.uint8))

    def _build_resize_transform(self, train):
        if self.resize_mode == "none":
            return None
        if self.resize_mode == "crop":
            if train:
                return tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size)
            return tr.FixScaleCrop(crop_size=self.args.crop_size)
        if self.resize_mode == "resize":
            return tr.FixedResize(self.args.crop_size)
        if self.resize_mode == "pad":
            return tr.ResizeLongestSideAndPad(self.args.crop_size)
        raise ValueError("Unsupported resize_mode '{}'".format(self.resize_mode))

    def transform_tr(self, sample):
        transforms_list = [tr.RandomHorizontalFlip()]
        resize_transform = self._build_resize_transform(train=True)
        if resize_transform is not None:
            transforms_list.append(resize_transform)
        transforms_list.extend(
            [
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return transforms.Compose(transforms_list)(sample)

    def transform_val(self, sample):
        transforms_list = []
        resize_transform = self._build_resize_transform(train=False)
        if resize_transform is not None:
            transforms_list.append(resize_transform)
        transforms_list.extend(
            [
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return transforms.Compose(transforms_list)(sample)

    def __str__(self):
        return "JijieSegmentation(split={}, profile={})".format(self.split, self.split_profile)


FeiaiSegmentation = JijieSegmentation
