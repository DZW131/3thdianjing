from __future__ import division, print_function

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders import custom_transforms as tr
from mypath import Path
from utils.jijie.sarcomere_prior import AddSideTubulePrior, AddTTubulePrior
from utils.jijie.tasks import CLASS_NAMES


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
    return normalized or None


class JijieSegmentation(Dataset):
    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(self, args, base_dir=Path.db_root_dir("jijie"), split="train"):
        super().__init__()
        self.args = args
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, "JPEGImages")
        self._cat_dir = os.path.join(self._base_dir, "SegmentationClass")
        self._default_splits_dir = os.path.join(self._base_dir, "ImageSets", "Segmentation")

        if isinstance(split, str):
            self.split = [split]
        else:
            self.split = sorted(split)

        self.split_profile = getattr(args, "split_profile", "annotated")
        self.manifest_dir = getattr(args, "manifest_dir", None)
        self.selected_classes = _normalize_selected_classes(getattr(args, "selected_classes", None))
        self.task_name = getattr(args, "task_name", None)

        self.class_mapping = None
        self.class_names = CLASS_NAMES
        self.NUM_CLASSES = len(CLASS_NAMES)

        if not os.path.isdir(self._image_dir):
            raise FileNotFoundError(
                "Missing jijie image directory: {}. "
                "The prepared dataset currently needs JPEGImages alongside SegmentationClass.".format(self._image_dir)
            )
        if not os.path.isdir(self._cat_dir):
            raise FileNotFoundError("Missing jijie mask directory: {}".format(self._cat_dir))

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
                lines = [line.strip() for line in handle.read().splitlines() if line.strip()]

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
            "[Data] jijie split={} task={} profile={} manifest={} train_resize={} eval_resize={} samples={}".format(
                ",".join(self.split),
                self.task_name or "default",
                self.split_profile,
                self.manifest_dir or "default",
                getattr(self.args, "train_resize_mode", getattr(self.args, "resize_mode", "pad")),
                getattr(self.args, "eval_resize_mode", getattr(self.args, "resize_mode", "pad")),
                len(self.images),
            )
        )

    def _resolve_split_file(self, split_name):
        if self.manifest_dir:
            split_file = os.path.join(self.manifest_dir, "{}.txt".format(split_name))
        else:
            suffix = SPLIT_PROFILE_SUFFIX.get(self.split_profile)
            if suffix is None:
                raise ValueError(
                    "Unsupported split profile '{}'. Expected one of {}.".format(
                        self.split_profile,
                        sorted(SPLIT_PROFILE_SUFFIX),
                    )
                )
            split_file = os.path.join(self._default_splits_dir, "{}{}.txt".format(split_name, suffix))

        if not os.path.isfile(split_file):
            raise FileNotFoundError("Missing split file: {}".format(split_file))
        return split_file

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, target = self._make_img_gt_point_pair(index)
        sample = {
            "image": image,
            "label": target,
            "sample_id": self.im_ids[index],
            "image_path": self.images[index],
            "mask_path": self.categories[index],
            "original_size": image.size[::-1],
            "task_name": self.task_name or "",
        }

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
        resize_mode = getattr(
            self.args,
            "train_resize_mode" if train else "eval_resize_mode",
            getattr(self.args, "resize_mode", "pad"),
        )
        crop_size = getattr(self.args, "crop_size", 512)

        if resize_mode == "none":
            return None
        if resize_mode == "crop":
            if train:
                return tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=crop_size)
            return tr.FixScaleCrop(crop_size=crop_size)
        if resize_mode == "random_crop":
            return tr.RandomCropPad(crop_size=crop_size)
        if resize_mode == "resize":
            return tr.FixedResize(crop_size)
        if resize_mode == "pad":
            return tr.ResizeLongestSideAndPad(crop_size)
        raise ValueError("Unsupported resize_mode '{}'".format(resize_mode))

    def _build_optional_label_dilation(self):
        class_ids = getattr(self.args, "label_dilate_class_ids", None)
        radius = getattr(self.args, "label_dilate_radius", 0)
        if not class_ids or radius <= 0:
            return None
        return tr.DilateMaskClasses(class_ids, radius)

    def _build_optional_t_tubule_prior(self):
        if not getattr(self.args, "enable_t_tubule_prior", False):
            return None
        return AddTTubulePrior(
            z_class_ids=getattr(self.args, "t_tubule_prior_z_class_ids", None),
            m_class_ids=getattr(self.args, "t_tubule_prior_m_class_ids", None),
            target_class_id=getattr(self.args, "t_tubule_prior_target_class_id", None),
            line_radius=getattr(self.args, "t_tubule_prior_line_radius", 1),
            min_distance=getattr(self.args, "t_tubule_prior_min_distance", 8),
            max_distance=getattr(self.args, "t_tubule_prior_max_distance", 384),
            min_component_area=getattr(self.args, "t_tubule_prior_min_component_area", 4),
            min_confidence=getattr(self.args, "t_tubule_prior_min_confidence", 0.25),
            exclude_non_background=getattr(self.args, "t_tubule_prior_exclude_non_background", True),
        )

    def _build_optional_side_tubule_prior(self):
        if not getattr(self.args, "enable_side_tubule_prior", False):
            return None
        return AddSideTubulePrior(
            z_class_ids=getattr(self.args, "side_tubule_prior_z_class_ids", None),
            m_class_ids=getattr(self.args, "side_tubule_prior_m_class_ids", None),
            target_class_id=getattr(self.args, "side_tubule_prior_target_class_id", None),
            line_radius=getattr(self.args, "side_tubule_prior_line_radius", 1),
            min_distance=getattr(self.args, "side_tubule_prior_min_distance", 8),
            max_distance=getattr(self.args, "side_tubule_prior_max_distance", 384),
            min_component_area=getattr(self.args, "side_tubule_prior_min_component_area", 4),
            min_confidence=getattr(self.args, "side_tubule_prior_min_confidence", 0.25),
            exclude_non_background=getattr(self.args, "side_tubule_prior_exclude_non_background", True),
        )

    def transform_tr(self, sample):
        transforms_list = [
            tr.RandomHorizontalFlip(),
        ]
        if getattr(self.args, "train_vertical_flip", False):
            transforms_list.append(tr.RandomVerticalFlip())
        if getattr(self.args, "train_rotate_degree", 0):
            transforms_list.append(tr.RandomRotate(self.args.train_rotate_degree))

        resize_transform = self._build_resize_transform(train=True)
        if resize_transform is not None:
            transforms_list.append(resize_transform)

        dilation_transform = self._build_optional_label_dilation()
        if dilation_transform is not None:
            transforms_list.append(dilation_transform)

        transforms_list.extend(
            [
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        prior_transform = self._build_optional_t_tubule_prior()
        if prior_transform is not None:
            transforms_list.append(prior_transform)
        side_prior_transform = self._build_optional_side_tubule_prior()
        if side_prior_transform is not None:
            transforms_list.append(side_prior_transform)
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
        return "JijieSegmentation(split={}, task={}, manifest={})".format(
            self.split,
            self.task_name or "default",
            self.manifest_dir or "default",
        )


FeiaiSegmentation = JijieSegmentation
