#!/usr/bin/env python3
"""Compare raw and postprocessed mitochondria masks against GT.

This utility is intentionally independent from model inference.  It is useful
when postprocessing masks were produced by an external experiment and we want a
repeatable metric table for reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


CLASS_NAMES = {
    0: "Background",
    1: "Normal mitochondria",
    2: "Mild injury mitochondria",
    3: "Moderate injury mitochondria",
    4: "Severe injury mitochondria",
}


def read_ids(path: Path) -> List[str]:
    ids: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            ids.append(item)
    return ids


def read_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64, copy=False)


def normalize_mask(arr: np.ndarray, num_classes: int) -> np.ndarray:
    out = arr.copy()
    out[(out < 0) | (out >= num_classes)] = 0
    return out


def update_confmat(confmat: np.ndarray, gt: np.ndarray, pred: np.ndarray, num_classes: int) -> None:
    valid = (gt >= 0) & (gt < num_classes)
    idx = num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
    confmat += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def class_metrics(confmat: np.ndarray, class_names: Dict[int, str]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    diag = np.diag(confmat).astype(np.float64)
    gt_sum = confmat.sum(axis=1).astype(np.float64)
    pred_sum = confmat.sum(axis=0).astype(np.float64)
    total = confmat.sum().astype(np.float64)
    union = gt_sum + pred_sum - diag

    iou = np.divide(diag, union, out=np.full_like(diag, np.nan, dtype=np.float64), where=union > 0)
    dice = np.divide(2.0 * diag, gt_sum + pred_sum, out=np.full_like(diag, np.nan, dtype=np.float64), where=(gt_sum + pred_sum) > 0)
    precision = np.divide(diag, pred_sum, out=np.full_like(diag, np.nan, dtype=np.float64), where=pred_sum > 0)
    recall = np.divide(diag, gt_sum, out=np.full_like(diag, np.nan, dtype=np.float64), where=gt_sum > 0)

    per_class: List[Dict[str, float]] = []
    for cid in range(confmat.shape[0]):
        per_class.append(
            {
                "class_id": cid,
                "class_name": class_names.get(cid, str(cid)),
                "iou": float(iou[cid]) if np.isfinite(iou[cid]) else None,
                "dice": float(dice[cid]) if np.isfinite(dice[cid]) else None,
                "precision": float(precision[cid]) if np.isfinite(precision[cid]) else None,
                "recall": float(recall[cid]) if np.isfinite(recall[cid]) else None,
                "gt_pixel_count": int(gt_sum[cid]),
                "pred_pixel_count": int(pred_sum[cid]),
            }
        )

    positive = np.arange(1, confmat.shape[0])
    summary = {
        "pixel_accuracy": float(diag.sum() / total) if total > 0 else None,
        "class_accuracy": float(np.nanmean(recall)) if recall.size else None,
        "mean_iou": float(np.nanmean(iou)) if iou.size else None,
        "mean_positive_iou": float(np.nanmean(iou[positive])) if positive.size else None,
        "mean_positive_dice": float(np.nanmean(dice[positive])) if positive.size else None,
    }
    return summary, per_class


def mixed_instance_stats(mask: np.ndarray, min_area: int) -> Dict[str, float]:
    mito = mask > 0
    labels, count = ndi.label(mito)
    mixed = 0
    kept = 0
    class_counts = np.zeros(5, dtype=np.int64)
    for label_id in range(1, count + 1):
        coords = labels == label_id
        area = int(coords.sum())
        if area < min_area:
            continue
        kept += 1
        cls = np.unique(mask[coords])
        cls = cls[cls > 0]
        if cls.size > 1:
            mixed += 1
        if cls.size == 1 and 0 <= cls[0] < class_counts.size:
            class_counts[int(cls[0])] += 1
    return {
        "instances": int(kept),
        "mixed_instances": int(mixed),
        "mixed_fraction": float(mixed / kept) if kept else 0.0,
        "single_class_instances": int(kept - mixed),
        "single_normal": int(class_counts[1]),
        "single_mild": int(class_counts[2]),
        "single_moderate": int(class_counts[3]),
        "single_severe": int(class_counts[4]),
    }


def mask_paths(mask_dir: Path, sample_id: str) -> Iterable[Path]:
    yield mask_dir / f"{sample_id}.png"
    yield mask_dir / f"{sample_id}.jpg"
    yield mask_dir / sample_id


def resolve_mask(mask_dir: Path, sample_id: str) -> Path:
    for path in mask_paths(mask_dir, sample_id):
        if path.exists():
            return path
    raise FileNotFoundError(f"No mask for {sample_id} under {mask_dir}")


def evaluate_variant(
    ids: List[str],
    gt_dir: Path,
    pred_dir: Path,
    num_classes: int,
    min_area: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, float], List[Dict[str, float]]]:
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)
    inst_totals: Dict[str, float] = {}
    per_image: List[Dict[str, float]] = []

    for sample_id in ids:
        gt = normalize_mask(read_mask(resolve_mask(gt_dir, sample_id)), num_classes)
        pred = normalize_mask(read_mask(resolve_mask(pred_dir, sample_id)), num_classes)
        if gt.shape != pred.shape:
            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred = np.array(pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST), dtype=np.int64)
        update_confmat(confmat, gt, pred, num_classes)

        inst = mixed_instance_stats(pred, min_area=min_area)
        for key, value in inst.items():
            inst_totals[key] = inst_totals.get(key, 0.0) + float(value)
        inst["sample_id"] = sample_id
        per_image.append(inst)

    summary, per_class = class_metrics(confmat, CLASS_NAMES)
    n = max(len(ids), 1)
    inst_summary = {f"{key}_mean": value / n for key, value in inst_totals.items()}
    summary.update(inst_summary)
    return summary, per_class, {"confusion_matrix": confmat.tolist()}, per_image


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--raw-pred-dir", required=True, type=Path)
    parser.add_argument("--post-pred-dir", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--min-instance-area", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = read_ids(args.split_file)
    raw_summary, raw_per_class, raw_extra, raw_per_image = evaluate_variant(
        ids, args.gt_dir, args.raw_pred_dir, args.num_classes, args.min_instance_area
    )
    post_summary, post_per_class, post_extra, post_per_image = evaluate_variant(
        ids, args.gt_dir, args.post_pred_dir, args.num_classes, args.min_instance_area
    )

    comparison = {
        "split_file": str(args.split_file),
        "num_images": len(ids),
        "raw": raw_summary,
        "postprocessed": post_summary,
        "delta_post_minus_raw": {
            key: (post_summary.get(key) - raw_summary.get(key))
            for key in post_summary
            if isinstance(post_summary.get(key), (int, float)) and isinstance(raw_summary.get(key), (int, float))
        },
        "raw_confusion_matrix": raw_extra["confusion_matrix"],
        "post_confusion_matrix": post_extra["confusion_matrix"],
    }
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows: List[Dict[str, object]] = []
    raw_by_id = {row["class_id"]: row for row in raw_per_class}
    post_by_id = {row["class_id"]: row for row in post_per_class}
    for cid in range(args.num_classes):
        raw = raw_by_id[cid]
        post = post_by_id[cid]
        row = {
            "class_id": cid,
            "class_name": CLASS_NAMES.get(cid, str(cid)),
            "raw_iou": raw["iou"],
            "post_iou": post["iou"],
            "delta_iou": None if raw["iou"] is None or post["iou"] is None else post["iou"] - raw["iou"],
            "raw_dice": raw["dice"],
            "post_dice": post["dice"],
            "delta_dice": None if raw["dice"] is None or post["dice"] is None else post["dice"] - raw["dice"],
            "raw_precision": raw["precision"],
            "post_precision": post["precision"],
            "raw_recall": raw["recall"],
            "post_recall": post["recall"],
            "gt_pixel_count": raw["gt_pixel_count"],
            "raw_pred_pixel_count": raw["pred_pixel_count"],
            "post_pred_pixel_count": post["pred_pixel_count"],
        }
        rows.append(row)
    write_csv(
        args.output_dir / "per_class_comparison.csv",
        rows,
        [
            "class_id",
            "class_name",
            "raw_iou",
            "post_iou",
            "delta_iou",
            "raw_dice",
            "post_dice",
            "delta_dice",
            "raw_precision",
            "post_precision",
            "raw_recall",
            "post_recall",
            "gt_pixel_count",
            "raw_pred_pixel_count",
            "post_pred_pixel_count",
        ],
    )

    per_image_rows: List[Dict[str, object]] = []
    post_by_sample = {row["sample_id"]: row for row in post_per_image}
    for raw in raw_per_image:
        sample_id = raw["sample_id"]
        post = post_by_sample[sample_id]
        per_image_rows.append(
            {
                "sample_id": sample_id,
                "raw_instances": raw["instances"],
                "post_instances": post["instances"],
                "raw_mixed_instances": raw["mixed_instances"],
                "post_mixed_instances": post["mixed_instances"],
                "raw_mixed_fraction": raw["mixed_fraction"],
                "post_mixed_fraction": post["mixed_fraction"],
            }
        )
    write_csv(
        args.output_dir / "per_image_instance_mixing.csv",
        per_image_rows,
        [
            "sample_id",
            "raw_instances",
            "post_instances",
            "raw_mixed_instances",
            "post_mixed_instances",
            "raw_mixed_fraction",
            "post_mixed_fraction",
        ],
    )

    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
