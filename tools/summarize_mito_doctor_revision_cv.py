"""Aggregate per-fold evaluation JSON files for the doctor-revision mito CV run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = (
    "pixel_accuracy",
    "pixel_accuracy_class",
    "mean_iou",
    "frequency_weighted_iou",
    "mean_positive_iou",
    "mean_positive_dice",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pattern",
        default="*_fold_*/metrics_summary.json",
        help="glob pattern relative to eval-root used to locate fold metric JSON files",
    )
    return parser.parse_args()


def mean_std(values):
    values = [float(value) for value in values]
    return {
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values, ddof=0)) if values else None,
        "n": len(values),
    }


def main():
    args = parse_args()
    metric_paths = sorted(args.eval_root.glob(args.pattern))
    if len(metric_paths) != 5:
        raise RuntimeError("expected five fold metrics under {}, found {}".format(args.eval_root, len(metric_paths)))

    rows = []
    for path in metric_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {"fold": path.parent.name.rsplit("_", 1)[-1], "metrics_path": str(path)}
        row.update({key: payload.get(key) for key in METRIC_KEYS})
        partial = payload.get("partial_label_overlap_summary") or {}
        row["overlap_component_mean_iou"] = partial.get("mean_iou")
        row["overlap_component_mean_dice"] = partial.get("mean_dice")
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "per_fold_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "evaluation_definition": "case-grouped five-fold CV on 52 doctor-revised partial-label slices; pixels with value 255 are excluded from pixel metrics",
        "folds": rows,
        "macro_mean_std": {
            key: mean_std([row[key] for row in rows if row.get(key) is not None])
            for key in fieldnames
            if key not in {"fold", "metrics_path"}
        },
    }
    (args.output_dir / "cross_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["macro_mean_std"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
