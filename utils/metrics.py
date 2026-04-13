import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.float64)

    def Pixel_Accuracy(self):
        total = self.confusion_matrix.sum()
        if total == 0:
            return 0.0
        return float(np.diag(self.confusion_matrix).sum() / total)

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
        return float(self.per_class_accuracy()[valid_mask].mean())

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

    def per_class_precision(self):
        true_positive = np.diag(self.confusion_matrix)
        predicted = np.sum(self.confusion_matrix, axis=0)
        return np.divide(
            true_positive,
            predicted,
            out=np.zeros(self.num_class, dtype=np.float64),
            where=predicted > 0,
        )

    def per_class_recall(self):
        true_positive = np.diag(self.confusion_matrix)
        positives = np.sum(self.confusion_matrix, axis=1)
        return np.divide(
            true_positive,
            positives,
            out=np.zeros(self.num_class, dtype=np.float64),
            where=positives > 0,
        )

    def per_class_dice(self):
        true_positive = np.diag(self.confusion_matrix)
        predicted = np.sum(self.confusion_matrix, axis=0)
        positives = np.sum(self.confusion_matrix, axis=1)
        denominator = predicted + positives
        return np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros(self.num_class, dtype=np.float64),
            where=denominator > 0,
        )

    def mean_over_classes(self, values, class_ids=None, include_background=False):
        values = np.asarray(values, dtype=np.float64)
        if class_ids is None:
            class_ids = list(range(self.num_class))
            if not include_background and self.num_class > 1:
                class_ids = class_ids[1:]
        class_ids = [int(class_id) for class_id in class_ids if 0 <= int(class_id) < self.num_class]
        if not class_ids:
            return 0.0
        valid_mask = self.valid_class_mask()
        selected = [class_id for class_id in class_ids if valid_mask[class_id]]
        if not selected:
            return 0.0
        return float(values[selected].mean())

    def Mean_Intersection_over_Union(self):
        return self.mean_over_classes(self.per_class_iou(), include_background=True)

    def Frequency_Weighted_Intersection_over_Union(self):
        total = np.sum(self.confusion_matrix)
        if total == 0:
            return 0.0
        freq = np.sum(self.confusion_matrix, axis=1) / total
        iu = self.per_class_iou()
        return float((freq[freq > 0] * iu[freq > 0]).sum())

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype(int) + pre_image[mask].astype(int)
        count = np.bincount(label, minlength=self.num_class ** 2)
        return count.reshape(self.num_class, self.num_class)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.float64)
