import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.float64)

    def Pixel_Accuracy(self):
        total = self.confusion_matrix.sum()
        if total == 0:
            return 0.0
        acc = np.diag(self.confusion_matrix).sum() / total
        return float(acc)

    def valid_class_mask(self):
        return self.confusion_matrix.sum(axis=1) > 0

    def per_class_accuracy(self):
        denominator = self.confusion_matrix.sum(axis=1)
        return np.divide(
            np.diag(self.confusion_matrix),
            denominator,
            out=np.zeros(self.num_class, dtype=np.float64),
            where=denominator > 0,
        )

    def Pixel_Accuracy_Class(self):
        valid_mask = self.valid_class_mask()
        if not np.any(valid_mask):
            return 0.0
        acc = self.per_class_accuracy()[valid_mask].mean()
        return float(acc)

    def per_class_iou(self):
        intersection = np.diag(self.confusion_matrix)
        union = (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - intersection
        )
        return np.divide(
            intersection,
            union,
            out=np.zeros(self.num_class, dtype=np.float64),
            where=union > 0,
        )

    def Mean_Intersection_over_Union(self):
        class_iou = self.per_class_iou()
        valid_mask = self.valid_class_mask()
        if not np.any(valid_mask):
            return 0.0
        miou = class_iou[valid_mask].mean()
        return float(miou)

    def Frequency_Weighted_Intersection_over_Union(self):
        total = np.sum(self.confusion_matrix)
        if total == 0:
            return 0.0

        freq = np.sum(self.confusion_matrix, axis=1) / total
        iu = self.per_class_iou()
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return float(fwiou)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
