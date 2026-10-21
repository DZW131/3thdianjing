"""Create fold-wise and class-wise baseline versus fine-tune CV comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SUMMARY_KEYS = (
    "pixel_accuracy",
    "pixel_accuracy_class",
    "mean_iou",
    "mean_positive_iou",
    "mean_positive_dice",
)
OBJECT_KEYS = (
    "overlap_component_mean_iou",
    "overlap_component_mean_dice",
)
CLASS_KEYS = ("iou", "dice", "precision", "recall")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--finetune-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_metrics(root, pattern):
    result = {}
    for path in sorted(root.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold = path.parent.name.rsplit("_", 1)[-1]
        result[fold] = payload
    return result


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value(payload, key):
    if key in OBJECT_KEYS:
        return (payload.get("partial_label_overlap_summary") or {}).get(key.removeprefix("overlap_component_"))
    return payload.get(key)


def main():
    args = parse_args()
    baseline = load_metrics(args.baseline_root, "baseline_fold_*/metrics_summary.json")
    finetune = load_metrics(args.finetune_root, "*_fold_*/metrics_summary.json")
    folds = sorted(set(baseline) & set(finetune))
    if len(folds) != 5:
        raise RuntimeError("expected five comparable folds, found {}".format(folds))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    for fold in folds:
        row = {"fold": fold}
        for key in (*SUMMARY_KEYS, *OBJECT_KEYS):
            base = value(baseline[fold], key)
            fine = value(finetune[fold], key)
            row["baseline_" + key] = base
            row["finetune_" + key] = fine
            row["delta_" + key] = None if base is None or fine is None else float(fine) - float(base)
        fold_rows.append(row)
    write_csv(args.output_dir / "fold_metric_comparison.csv", fold_rows)

    class_rows = []
    for fold in folds:
        base_classes = {row["class_name"]: row for row in baseline[fold].get("per_class_metrics", [])}
        fine_classes = {row["class_name"]: row for row in finetune[fold].get("per_class_metrics", [])}
        for class_name in sorted(set(base_classes) & set(fine_classes)):
            if class_name == "Background":
                continue
            row = {"fold": fold, "class_name": class_name}
            for key in CLASS_KEYS:
                base = base_classes[class_name].get(key)
                fine = fine_classes[class_name].get(key)
                row["baseline_" + key] = base
                row["finetune_" + key] = fine
                row["delta_" + key] = None if base is None or fine is None else float(fine) - float(base)
            class_rows.append(row)
    write_csv(args.output_dir / "per_class_metric_comparison.csv", class_rows)

    macro_delta = {}
    for key in (*SUMMARY_KEYS, *OBJECT_KEYS):
        deltas = [row["delta_" + key] for row in fold_rows if row["delta_" + key] is not None]
        macro_delta[key] = {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas, ddof=0)),
        }
    report = {
        "comparison_definition": "same five case-held-out revised-data folds; positive metrics exclude ignored 255 pixels",
        "macro_delta_finetune_minus_baseline": macro_delta,
        "folds": folds,
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
