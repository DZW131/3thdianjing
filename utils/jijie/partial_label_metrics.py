import numpy as np
from scipy import ndimage


def _safe_divide(numerator, denominator, default=0.0):
    if denominator == 0:
        return default
    return float(numerator / denominator)


def _label_components(binary_mask):
    structure = ndimage.generate_binary_structure(2, 2)
    return ndimage.label(binary_mask.astype(bool), structure=structure)


def _class_name(class_names, class_id):
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return "Class_{}".format(class_id)


def compute_gt_overlap_rows(
    gt_mask,
    pred_mask,
    class_ids,
    class_names,
    sample_id,
    min_overlap_pixels=1,
):
    """Compute partial-label metrics by ignoring predicted components without GT overlap.

    This is intended for sparsely annotated electron microscopy images where doctors
    may annotate representative structures instead of exhaustively labeling every
    visible instance. For each class, a predicted connected component is evaluated
    only if it overlaps at least ``min_overlap_pixels`` same-class GT pixels.
    """

    gt_mask = np.asarray(gt_mask)
    pred_mask = np.asarray(pred_mask)
    rows = []

    for class_id in class_ids or []:
        class_id = int(class_id)
        gt_binary = gt_mask == class_id
        pred_binary = pred_mask == class_id
        gt_area_px = int(gt_binary.sum())
        total_pred_area_px = int(pred_binary.sum())

        pred_labeled, pred_component_count = _label_components(pred_binary)
        gt_labeled, gt_component_count = _label_components(gt_binary)

        kept_pred_mask = np.zeros_like(pred_binary, dtype=bool)
        kept_component_count = 0
        ignored_component_count = 0
        kept_pred_area_px = 0
        ignored_pred_area_px = 0

        for component_id in range(1, int(pred_component_count) + 1):
            component_mask = pred_labeled == component_id
            component_area = int(component_mask.sum())
            overlap_px = int(np.logical_and(component_mask, gt_binary).sum())
            if overlap_px >= min_overlap_pixels:
                kept_pred_mask |= component_mask
                kept_component_count += 1
                kept_pred_area_px += component_area
            else:
                ignored_component_count += 1
                ignored_pred_area_px += component_area

        intersection_px = int(np.logical_and(kept_pred_mask, gt_binary).sum())
        union_px = int(gt_area_px + kept_pred_area_px - intersection_px)
        matched_gt_components = 0
        if gt_component_count > 0 and intersection_px > 0:
            matched_gt_labels = np.unique(gt_labeled[np.logical_and(kept_pred_mask, gt_binary)])
            matched_gt_components = int(np.sum(matched_gt_labels > 0))

        dice_denominator = gt_area_px + kept_pred_area_px
        iou = _safe_divide(intersection_px, union_px, default=1.0 if dice_denominator == 0 else 0.0)
        dice = _safe_divide(2.0 * intersection_px, dice_denominator, default=1.0 if dice_denominator == 0 else 0.0)
        precision = _safe_divide(intersection_px, kept_pred_area_px, default=1.0 if gt_area_px == 0 else 0.0)
        recall = _safe_divide(intersection_px, gt_area_px, default=1.0 if kept_pred_area_px == 0 else 0.0)

        rows.append(
            {
                "sample_id": sample_id,
                "class_id": class_id,
                "class_name": _class_name(class_names, class_id),
                "gt_area_px": gt_area_px,
                "total_pred_area_px": total_pred_area_px,
                "kept_pred_area_px": int(kept_pred_area_px),
                "ignored_pred_area_px": int(ignored_pred_area_px),
                "intersection_px": intersection_px,
                "union_px": union_px,
                "iou": float(iou),
                "dice": float(dice),
                "precision": float(precision),
                "recall": float(recall),
                "gt_component_count": int(gt_component_count),
                "matched_gt_component_count": matched_gt_components,
                "total_pred_component_count": int(pred_component_count),
                "kept_pred_component_count": int(kept_component_count),
                "ignored_pred_component_count": int(ignored_component_count),
                "overlap_min_pixels": int(min_overlap_pixels),
            }
        )

    return rows


def aggregate_gt_overlap_rows(rows, class_ids, class_names, metric_target_class_ids=None):
    grouped = {}
    for row in rows:
        class_id = int(row["class_id"])
        grouped.setdefault(class_id, []).append(row)

    per_class_rows = []
    for class_id in class_ids or []:
        class_id = int(class_id)
        group = grouped.get(class_id, [])
        gt_area_px = int(sum(row.get("gt_area_px", 0) for row in group))
        kept_pred_area_px = int(sum(row.get("kept_pred_area_px", 0) for row in group))
        total_pred_area_px = int(sum(row.get("total_pred_area_px", 0) for row in group))
        ignored_pred_area_px = int(sum(row.get("ignored_pred_area_px", 0) for row in group))
        intersection_px = int(sum(row.get("intersection_px", 0) for row in group))
        union_px = int(gt_area_px + kept_pred_area_px - intersection_px)
        dice_denominator = gt_area_px + kept_pred_area_px

        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": _class_name(class_names, class_id),
                "sample_count": len(group),
                "gt_area_px": gt_area_px,
                "total_pred_area_px": total_pred_area_px,
                "kept_pred_area_px": kept_pred_area_px,
                "ignored_pred_area_px": ignored_pred_area_px,
                "intersection_px": intersection_px,
                "union_px": union_px,
                "iou": _safe_divide(intersection_px, union_px, default=1.0 if dice_denominator == 0 else 0.0),
                "dice": _safe_divide(
                    2.0 * intersection_px,
                    dice_denominator,
                    default=1.0 if dice_denominator == 0 else 0.0,
                ),
                "precision": _safe_divide(intersection_px, kept_pred_area_px, default=1.0 if gt_area_px == 0 else 0.0),
                "recall": _safe_divide(intersection_px, gt_area_px, default=1.0 if kept_pred_area_px == 0 else 0.0),
                "gt_component_count": int(sum(row.get("gt_component_count", 0) for row in group)),
                "matched_gt_component_count": int(sum(row.get("matched_gt_component_count", 0) for row in group)),
                "total_pred_component_count": int(sum(row.get("total_pred_component_count", 0) for row in group)),
                "kept_pred_component_count": int(sum(row.get("kept_pred_component_count", 0) for row in group)),
                "ignored_pred_component_count": int(sum(row.get("ignored_pred_component_count", 0) for row in group)),
            }
        )

    target_ids = metric_target_class_ids if metric_target_class_ids is not None else class_ids
    target_ids = {int(class_id) for class_id in target_ids or []}
    target_rows = [row for row in per_class_rows if int(row["class_id"]) in target_ids and row["gt_area_px"] > 0]
    summary = {
        "description": (
            "Predicted connected components are evaluated only when they overlap "
            "same-class GT. Non-overlapping predictions are reported as ignored, "
            "which is useful for partially annotated data but should not replace "
            "full-image metrics."
        ),
        "metric_target_class_ids": sorted(target_ids),
        "class_ids": [int(class_id) for class_id in class_ids or []],
        "class_count_with_gt": len(target_rows),
        "mean_iou": float(np.mean([row["iou"] for row in target_rows])) if target_rows else 0.0,
        "mean_dice": float(np.mean([row["dice"] for row in target_rows])) if target_rows else 0.0,
        "mean_precision": float(np.mean([row["precision"] for row in target_rows])) if target_rows else 0.0,
        "mean_recall": float(np.mean([row["recall"] for row in target_rows])) if target_rows else 0.0,
        "gt_area_px": int(sum(row["gt_area_px"] for row in target_rows)),
        "kept_pred_area_px": int(sum(row["kept_pred_area_px"] for row in target_rows)),
        "ignored_pred_area_px": int(sum(row["ignored_pred_area_px"] for row in target_rows)),
        "intersection_px": int(sum(row["intersection_px"] for row in target_rows)),
    }
    return per_class_rows, summary
