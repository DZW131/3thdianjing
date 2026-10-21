"""Build a 52-slice mito training set that keeps only non-conflicting supervision.

The two input roots must describe the same samples:

* ``current_root`` is reconstructed from the latest slice DB and distinguishes
  retained AI contours (source 1) from new physician marks (source 3).
* ``multisource_root`` contains the same rendered labels plus first-batch
  physician masks (source 2).

Latest physician edits always take precedence.  A historical physician label
and a retained AI label are used together only when their class agrees; an
AI-vs-historical class conflict is ignored instead of teaching contradictory
targets.  Pixels supervised by a single retained AI contour keep a lower loss
weight, whereas physician-only and agreement pixels are retained strongly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


IGNORE_INDEX = 255
VALID_CLASSES = (1, 2, 3, 4)
SOURCE_AI = 1
SOURCE_HISTORICAL_DOCTOR = 2
SOURCE_REVISED_DOCTOR = 3
SOURCE_DOCTOR_AI_AGREEMENT = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--multisource-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ai-weight", type=float, default=0.35)
    parser.add_argument("--historical-doctor-weight", type=float, default=0.75)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_png(path):
    return np.asarray(Image.open(path), dtype=np.uint8)


def write_png(values, path):
    Image.fromarray(values.astype(np.uint8), mode="L").save(path)


def weight_value(weight):
    if not 0.0 < weight <= 1.0:
        raise ValueError("weights must be in (0, 1]")
    return int(round(255.0 * weight))


def link_or_copytree(source, destination):
    try:
        destination.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, destination)


def load_sample_ids(manifest_root):
    train = manifest_root / "train.txt"
    if not train.is_file():
        raise FileNotFoundError(train)
    return [line.strip() for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_sample(current_mask, current_source, combined_mask, combined_source, ai_weight, historical_weight):
    if not (current_mask.shape == current_source.shape == combined_mask.shape == combined_source.shape):
        raise ValueError("mask/source shape mismatch")

    current_valid = np.isin(current_mask, VALID_CLASSES)
    historical = combined_source == SOURCE_HISTORICAL_DOCTOR
    if np.any(historical & ~np.isin(combined_mask, VALID_CLASSES)):
        raise ValueError("historical source map points to a non-mito class")

    output_mask = np.full(current_mask.shape, IGNORE_INDEX, dtype=np.uint8)
    output_weight = np.zeros(current_mask.shape, dtype=np.uint8)
    output_source = np.zeros(current_mask.shape, dtype=np.uint8)

    revised = current_source == SOURCE_REVISED_DOCTOR
    ai = current_source == SOURCE_AI
    if np.any(current_valid & ~(revised | ai)):
        raise ValueError("current source map contains a supervised pixel with an unknown source")

    current_only = current_valid & ~historical
    historical_only = historical & ~current_valid
    agreement = historical & current_valid & (combined_mask == current_mask)
    ai_historical_conflict = historical & ai & (combined_mask != current_mask)
    revised_conflict = historical & revised & (combined_mask != current_mask)

    # Latest physician edits are the highest-confidence authority.
    keep_revised = revised
    output_mask[keep_revised] = current_mask[keep_revised]
    output_weight[keep_revised] = 255
    output_source[keep_revised] = SOURCE_REVISED_DOCTOR

    # Historical labels reinforce retained AI contours only when they agree.
    output_mask[agreement] = current_mask[agreement]
    output_weight[agreement] = 255
    output_source[agreement] = SOURCE_DOCTOR_AI_AGREEMENT

    # Accepted AI predictions without a historical contradiction remain weak.
    ai_only = current_only & ai
    output_mask[ai_only] = current_mask[ai_only]
    output_weight[ai_only] = ai_weight
    output_source[ai_only] = SOURCE_AI

    # Trusted old physician labels contribute where no current contour exists.
    output_mask[historical_only] = combined_mask[historical_only]
    output_weight[historical_only] = historical_weight
    output_source[historical_only] = SOURCE_HISTORICAL_DOCTOR

    # ``ai_historical_conflict`` deliberately stays ignored.  Revised conflicts
    # were retained above and are counted for audit only.
    audit = {
        "current_only_ai_px": int(ai_only.sum()),
        "current_only_revised_px": int((current_only & revised).sum()),
        "historical_only_px": int(historical_only.sum()),
        "doctor_ai_agreement_px": int(agreement.sum()),
        "ai_historical_conflict_ignored_px": int(ai_historical_conflict.sum()),
        "revised_doctor_priority_conflict_px": int(revised_conflict.sum()),
        "kept_pixels": int(np.count_nonzero(output_mask != IGNORE_INDEX)),
    }
    return output_mask, output_weight, output_source, audit


def main():
    args = parse_args()
    if args.output_root.exists():
        if not args.force:
            raise RuntimeError("output already exists: {}".format(args.output_root))
        shutil.rmtree(args.output_root)

    current_masks = args.current_root / "SegmentationClass"
    current_sources = args.current_root / "SupervisionSource"
    combined_masks = args.multisource_root / "SegmentationClass"
    combined_sources = args.multisource_root / "SupervisionSource"
    manifests = args.current_root / "ImageSets" / "Segmentation" / "tasks" / "mito"
    for path in (current_masks, current_sources, combined_masks, combined_sources, manifests):
        if not path.is_dir():
            raise FileNotFoundError(path)

    output_masks = args.output_root / "SegmentationClass"
    output_weights = args.output_root / "SupervisionWeight"
    output_sources = args.output_root / "SupervisionSource"
    output_metadata = args.output_root / "metadata"
    for path in (output_masks, output_weights, output_sources, output_metadata):
        path.mkdir(parents=True, exist_ok=True)
    link_or_copytree(args.current_root / "JPEGImages", args.output_root / "JPEGImages")
    shutil.copytree(args.current_root / "ImageSets", args.output_root / "ImageSets")

    sample_ids = load_sample_ids(manifests)
    rows = []
    totals = Counter()
    for sample_id in sample_ids:
        paths = [
            current_masks / (sample_id + ".png"),
            current_sources / (sample_id + ".png"),
            combined_masks / (sample_id + ".png"),
            combined_sources / (sample_id + ".png"),
        ]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError("missing mask/source for {}".format(sample_id))
        mask, weight, source, audit = build_sample(
            read_png(paths[0]),
            read_png(paths[1]),
            read_png(paths[2]),
            read_png(paths[3]),
            weight_value(args.ai_weight),
            weight_value(args.historical_doctor_weight),
        )
        write_png(mask, output_masks / (sample_id + ".png"))
        write_png(weight, output_weights / (sample_id + ".png"))
        write_png(source, output_sources / (sample_id + ".png"))
        rows.append({"sample_id": sample_id, **audit})
        totals.update(audit)

    with (output_metadata / "consensus_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "samples": len(rows),
        "policy": {
            "latest_doctor": "keep at weight 1.0",
            "historical_and_ai_same_class": "keep at weight 1.0",
            "historical_only": args.historical_doctor_weight,
            "ai_only": args.ai_weight,
            "historical_vs_ai_class_conflict": "ignore",
        },
        "pixel_audit": dict(totals),
    }
    (output_metadata / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
