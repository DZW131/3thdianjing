import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps


def _clone_sample(sample, image, label):
    cloned = dict(sample)
    cloned["image"] = image
    cloned["label"] = label
    return cloned


class Normalize(object):
    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample["image"]).astype(np.float32)
        mask = np.array(sample["label"]).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return _clone_sample(sample, img, mask)


class ToTensor(object):
    def __call__(self, sample):
        img = np.array(sample["image"]).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(sample["label"]).astype(np.float32)
        return _clone_sample(
            sample,
            torch.from_numpy(img).float(),
            torch.from_numpy(mask).float(),
        )


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return _clone_sample(sample, img, mask)


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return _clone_sample(sample, img, mask)


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        return _clone_sample(sample, img, mask)


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample["image"]
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return _clone_sample(sample, img, sample["label"])


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        width, height = img.size
        if height > width:
            out_width = short_size
            out_height = int(1.0 * height * out_width / width)
        else:
            out_height = short_size
            out_width = int(1.0 * width * out_height / height)

        img = img.resize((out_width, out_height), Image.BILINEAR)
        mask = mask.resize((out_width, out_height), Image.NEAREST)

        if short_size < self.crop_size:
            pad_height = self.crop_size - out_height if out_height < self.crop_size else 0
            pad_width = self.crop_size - out_width if out_width < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, pad_width, pad_height), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_width, pad_height), fill=self.fill)

        width, height = img.size
        x1 = random.randint(0, width - self.crop_size)
        y1 = random.randint(0, height - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return _clone_sample(sample, img, mask)


class RandomCropPad(object):
    def __init__(self, crop_size, fill=0):
        self.crop_size = int(crop_size)
        self.fill = fill

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        width, height = img.size
        pad_width = max(0, self.crop_size - width)
        pad_height = max(0, self.crop_size - height)
        if pad_width or pad_height:
            img = ImageOps.expand(img, border=(0, 0, pad_width, pad_height), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_width, pad_height), fill=self.fill)

        width, height = img.size
        x1 = random.randint(0, max(0, width - self.crop_size))
        y1 = random.randint(0, max(0, height - self.crop_size))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return _clone_sample(sample, img, mask)


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        width, height = img.size
        if width > height:
            out_height = self.crop_size
            out_width = int(1.0 * width * out_height / height)
        else:
            out_width = self.crop_size
            out_height = int(1.0 * height * out_width / width)
        img = img.resize((out_width, out_height), Image.BILINEAR)
        mask = mask.resize((out_width, out_height), Image.NEAREST)

        width, height = img.size
        x1 = int(round((width - self.crop_size) / 2.0))
        y1 = int(round((height - self.crop_size) / 2.0))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return _clone_sample(sample, img, mask)


class ResizeLongestSideAndPad(object):
    def __init__(self, size, fill=0):
        self.size = int(size)
        self.fill = fill

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        width, height = img.size
        scale = float(self.size) / float(max(width, height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        img = img.resize((new_width, new_height), Image.BILINEAR)
        mask = mask.resize((new_width, new_height), Image.NEAREST)

        canvas_img = Image.new("RGB", (self.size, self.size), 0)
        canvas_mask = Image.new(mask.mode, (self.size, self.size), self.fill)

        offset_x = (self.size - new_width) // 2
        offset_y = (self.size - new_height) // 2
        canvas_img.paste(img, (offset_x, offset_y))
        canvas_mask.paste(mask, (offset_x, offset_y))

        return _clone_sample(sample, canvas_img, canvas_mask)


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return _clone_sample(sample, img, mask)


class DilateMaskClasses(object):
    def __init__(self, class_ids, radius):
        self.class_ids = [int(class_id) for class_id in class_ids or []]
        self.radius = int(radius)

    def __call__(self, sample):
        if not self.class_ids or self.radius <= 0:
            return sample

        from scipy import ndimage

        mask_array = np.array(sample["label"]).astype(np.uint8)
        dilated = mask_array.copy()
        structure = ndimage.generate_binary_structure(2, 1)

        for class_id in self.class_ids:
            class_mask = mask_array == class_id
            if not np.any(class_mask):
                continue
            expanded = ndimage.binary_dilation(class_mask, structure=structure, iterations=self.radius)
            dilated[np.logical_and(expanded, dilated == 0)] = class_id

        return _clone_sample(sample, sample["image"], Image.fromarray(dilated, mode="L"))
