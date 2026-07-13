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


class MetaSSLLabeledLoss(object):
    """Quadripartition-based supervised heterogeneous CE + Dice loss.

    This is the labeled-image part of MetaSSL. A second stochastic prediction is
    used only to estimate whether each manual label pixel looks reliable:

    UC: reference agrees with annotation and is confident
    US: reference agrees with annotation but is suspicious
    DC: reference disagrees with annotation and is confident
    DS: reference disagrees with annotation and is suspicious

    For labeled images the paper uses UC > US > DS > DC, because a confident
    disagreement may indicate noisy annotation. We keep a small minimum weight
    by default so early wrong predictions cannot fully silence a region.
    """

    def __init__(
        self,
        num_classes,
        class_weight=None,
        ignore_index=255,
        beta=3.0,
        delta_l=0.6,
        ema_alpha=0.99,
        initial_threshold=0.5,
        min_threshold=0.5,
        max_threshold=0.99,
        min_region_weight=0.05,
        protect_positive_labels=False,
        positive_label_min_weight=0.8,
        smooth=1.0,
    ):
        self.num_classes = int(num_classes)
        self.class_weight = class_weight
        self.ignore_index = int(ignore_index)
        self.beta = float(beta)
        self.delta_l = float(delta_l)
        self.ema_alpha = float(ema_alpha)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.min_region_weight = float(min_region_weight)
        self.protect_positive_labels = bool(protect_positive_labels)
        self.positive_label_min_weight = float(positive_label_min_weight)
        self.smooth = float(smooth)
        self.thresholds = torch.full((self.num_classes,), float(initial_threshold), dtype=torch.float32)

        self.region_weights = self._build_region_weights()

    def _phi(self, u):
        return float(torch.exp(torch.tensor(-(float(u) ** self.beta))).item())

    def _build_region_weights(self):
        w_uc = self._phi(0.0)
        w_us = self._phi(self.delta_l)
        w_ds = self._phi(2.0 * self.delta_l)
        w_dc = self._phi(3.0 * self.delta_l)
        floor = self.min_region_weight
        return {
            "uc": max(w_uc, floor),
            "us": max(w_us, floor),
            "ds": max(w_ds, floor),
            "dc": max(w_dc, floor),
        }

    def _ensure_device(self, device):
        if self.thresholds.device != device:
            self.thresholds = self.thresholds.to(device=device)
        if self.class_weight is not None and self.class_weight.device != device:
            self.class_weight = self.class_weight.to(device=device)

    @torch.no_grad()
    def _update_thresholds(self, ref_probs, ref_label, valid_mask):
        confidence = ref_probs.max(dim=1).values
        current = self.thresholds.clone()
        for class_id in range(self.num_classes):
            class_mask = (ref_label == class_id) & valid_mask
            if not torch.any(class_mask):
                continue
            observed = confidence[class_mask].mean()
            current[class_id] = self.ema_alpha * observed + (1.0 - self.ema_alpha) * current[class_id]
        self.thresholds = torch.clamp(current, min=self.min_threshold, max=self.max_threshold)

    def _partition_weight_map(self, ref_logit, target):
        ref_probs = F.softmax(ref_logit, dim=1)
        confidence, ref_label = torch.max(ref_probs, dim=1)
        target_long = target.long()
        valid_mask = target_long != self.ignore_index
        safe_target = torch.where(valid_mask, target_long, torch.zeros_like(target_long))

        self._update_thresholds(ref_probs.detach(), ref_label.detach(), valid_mask.detach())

        thresholds = self.thresholds[ref_label]
        confident = confidence > thresholds
        unanimous = ref_label == safe_target

        weights = torch.full_like(confidence, self.region_weights["ds"])
        weights = torch.where(unanimous & confident, torch.full_like(weights, self.region_weights["uc"]), weights)
        weights = torch.where(unanimous & ~confident, torch.full_like(weights, self.region_weights["us"]), weights)
        weights = torch.where(~unanimous & confident, torch.full_like(weights, self.region_weights["dc"]), weights)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        if self.protect_positive_labels:
            positive_disagreement = valid_mask & (safe_target > 0) & (~unanimous)
            positive_floor = torch.full_like(weights, self.positive_label_min_weight)
            weights = torch.where(positive_disagreement, torch.maximum(weights, positive_floor), weights)

        stats = {
            "metassl_uc_fraction": float(((unanimous & confident) & valid_mask).float().mean().detach().cpu()),
            "metassl_us_fraction": float(((unanimous & ~confident) & valid_mask).float().mean().detach().cpu()),
            "metassl_dc_fraction": float(((~unanimous & confident) & valid_mask).float().mean().detach().cpu()),
            "metassl_ds_fraction": float(((~unanimous & ~confident) & valid_mask).float().mean().detach().cpu()),
            "metassl_positive_disagreement_fraction": float(
                ((safe_target > 0) & (~unanimous) & valid_mask).float().mean().detach().cpu()
            ),
            "metassl_threshold_mean": float(self.thresholds.mean().detach().cpu()),
        }
        return weights, safe_target, valid_mask, stats

    def _weighted_cross_entropy(self, logit, target, weight_map):
        ce = F.cross_entropy(
            logit,
            target.long(),
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        normalizer = torch.clamp(weight_map.sum(), min=1.0)
        return (ce * weight_map).sum() / normalizer

    def _weighted_dice(self, logit, target, weight_map, valid_mask):
        probs = F.softmax(logit, dim=1)
        one_hot = F.one_hot(target.long(), num_classes=logit.size(1)).permute(0, 3, 1, 2).float()
        weights = weight_map.unsqueeze(1)
        valid = valid_mask.unsqueeze(1).float()

        probs = probs * valid
        one_hot = one_hot * valid
        weights = weights * valid

        dims = (0, 2, 3)
        intersection = torch.sum(weights * probs * one_hot, dims)
        denominator = torch.sum(weights * (probs + one_hot), dims)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if dice_score.numel() > 1:
            dice_score = dice_score[1:]
        return 1.0 - dice_score.mean()

    def __call__(self, logit, ref_logit, target):
        self._ensure_device(logit.device)
        weight_map, safe_target, valid_mask, stats = self._partition_weight_map(ref_logit, target)
        ce_loss = self._weighted_cross_entropy(logit, safe_target, weight_map)
        dice_loss = self._weighted_dice(logit, safe_target, weight_map, valid_mask)
        loss = ce_loss + dice_loss
        stats["metassl_ce_loss"] = float(ce_loss.detach().cpu())
        stats["metassl_dice_loss"] = float(dice_loss.detach().cpu())
        return loss, stats


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=False)
    logits = torch.rand(2, 4, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32))
    print("ce", float(loss.CrossEntropyLoss(logits, target)))
    print("focal", float(loss.FocalLoss(logits, target)))
    print("dice", float(loss.DiceLoss(logits, target)))
    print("ce_dice", float(loss.CrossEntropyDiceLoss(logits, target)))
