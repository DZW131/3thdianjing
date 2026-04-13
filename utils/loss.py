import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode="ce"):
        if mode == "ce":
            return self.CrossEntropyLoss
        if mode == "focal":
            return self.FocalLoss
        if mode == "dice":
            return self.DiceLoss
        if mode == "ce_dice":
            return self.CrossEntropyDiceLoss
        if mode == "focal_dice":
            return self.FocalDiceLoss
        if mode == "tversky_dice":
            return self.TverskyDiceLoss
        raise NotImplementedError("Unsupported loss mode '{}'".format(mode))

    def _cross_entropy(self, logit, target):
        reduction = "mean" if self.size_average else "sum"
        criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=reduction,
        )
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())
        if reduction == "sum" and self.batch_average:
            loss = loss / max(1, logit.size(0))
        return loss

    def CrossEntropyLoss(self, logit, target):
        return self._cross_entropy(logit, target)

    def FocalLoss(self, logit, target, gamma=2.0, alpha=0.5):
        ce_loss = self._cross_entropy(logit, target)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** gamma) * ce_loss
        if alpha is not None:
            focal = alpha * focal
        return focal

    def DiceLoss(self, logit, target, smooth=1.0):
        probabilities = F.softmax(logit, dim=1)
        target_long = target.long()
        valid_mask = (target_long != self.ignore_index).float()
        target_long = torch.where(target_long == self.ignore_index, torch.zeros_like(target_long), target_long)

        one_hot = F.one_hot(target_long, num_classes=logit.size(1)).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)

        probabilities = probabilities * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot, dims)
        denominator = torch.sum(probabilities + one_hot, dims)
        dice_score = (2.0 * intersection + smooth) / (denominator + smooth)

        if dice_score.numel() > 1:
            dice_score = dice_score[1:]
        return 1.0 - dice_score.mean()

    def TverskyLoss(self, logit, target, alpha=0.3, beta=0.7, smooth=1.0):
        probabilities = F.softmax(logit, dim=1)
        target_long = target.long()
        valid_mask = (target_long != self.ignore_index).float()
        target_long = torch.where(target_long == self.ignore_index, torch.zeros_like(target_long), target_long)

        one_hot = F.one_hot(target_long, num_classes=logit.size(1)).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)
        probabilities = probabilities * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        true_positive = torch.sum(probabilities * one_hot, dims)
        false_positive = torch.sum(probabilities * (1.0 - one_hot), dims)
        false_negative = torch.sum((1.0 - probabilities) * one_hot, dims)

        tversky = (true_positive + smooth) / (
            true_positive + alpha * false_positive + beta * false_negative + smooth
        )
        if tversky.numel() > 1:
            tversky = tversky[1:]
        return 1.0 - tversky.mean()

    def CrossEntropyDiceLoss(self, logit, target):
        return self.CrossEntropyLoss(logit, target) + self.DiceLoss(logit, target)

    def FocalDiceLoss(self, logit, target):
        return self.FocalLoss(logit, target) + self.DiceLoss(logit, target)

    def TverskyDiceLoss(self, logit, target):
        return self.TverskyLoss(logit, target) + self.DiceLoss(logit, target)


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=False)
    logits = torch.rand(2, 4, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32))
    print("ce", float(loss.CrossEntropyLoss(logits, target)))
    print("focal", float(loss.FocalLoss(logits, target)))
    print("dice", float(loss.DiceLoss(logits, target)))
    print("ce_dice", float(loss.CrossEntropyDiceLoss(logits, target)))
