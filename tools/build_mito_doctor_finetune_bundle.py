"""Compose original mito training data with doctor-revised partial labels.

The output uses links when possible, so the baseline dataset is never changed
or duplicated.  Revised training samples can be repeated in ``train.txt`` to
make their supervised pixels appear often enough during low-learning-rate
fine-tuning, while validation and test remain source-batch holdouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--base-manifest-dir", type=Path, required=True)
    parser.add_argument("--base-metadata-csv", type=Path, required=True)
    parser.add_argument("--doctor-root", type=Path, required=True)
    parser.add_argument("--doctor-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--doctor-train-repeat", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_ids(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def link_or_copy(source, destination):
    if destination.exists() or destination.is_symlink():
        return
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def add_sample(root, sample_id, image_dir, mask_dir):
    source_image = root / "JPEGImages" / (sample_id + ".jpg")
    source_mask = root / "SegmentationClass" / (sample_id + ".png")
    if not source_image.is_file() or not source_mask.is_file():
        raise FileNotFoundError("Missing image or mask for {} under {}".format(sample_id, root))
    link_or_copy(source_image, image_dir / source_image.name)
    link_or_copy(source_mask, mask_dir / source_mask.name)


def source_key(batch_name, slice_name):
    return batch_name.rsplit("_", 1)[0], slice_name.rsplit("_", 1)[0]


def load_base_ids(metadata_csv):
    with metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row["timestamp_dir"], row["sample_dir"]): row["prepared_sample_id"]
            for row in csv.DictReader(handle)
            if row.get("timestamp_dir") and row.get("sample_dir")
        }


def load_doctor_records(doctor_root):
    with (doctor_root / "metadata" / "samples.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build(args):
    for path in (args.base_root, args.base_manifest_dir, args.base_metadata_csv, args.doctor_root, args.doctor_manifest_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.doctor_train_repeat < 1:
        raise ValueError("--doctor-train-repeat must be positive")
    if args.output_root.exists():
        if not args.force:
            raise RuntimeError("Output already exists: {}".format(args.output_root))
        shutil.rmtree(args.output_root)

    image_dir = args.output_root / "JPEGImages"
    mask_dir = args.output_root / "SegmentationClass"
    manifest_dir = args.output_root / "ImageSets" / "Segmentation" / "tasks" / "mito"
    metadata_dir = args.output_root / "metadata"
    for path in (image_dir, mask_dir, manifest_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    base_train = read_ids(args.base_manifest_dir / "train.txt")
    base_by_source = load_base_ids(args.base_metadata_csv)
    doctor_records = load_doctor_records(args.doctor_root)
    doctor_ids = [row["sample_id"] for row in doctor_records]
    replaced_base_ids = {
        base_by_source[key]
        for key in (source_key(row["batch"], row["slice"]) for row in doctor_records)
        if key in base_by_source and base_by_source[key] in base_train
    }
    retained_base_train = [sample_id for sample_id in base_train if sample_id not in replaced_base_ids]

    for sample_id in retained_base_train:
        add_sample(args.base_root, sample_id, image_dir, mask_dir)
    for sample_id in doctor_ids:
        add_sample(args.doctor_root, sample_id, image_dir, mask_dir)

    train_ids = retained_base_train + doctor_ids * args.doctor_train_repeat
    splits = {"train": train_ids, "val": [], "test": []}
    for split, sample_ids in splits.items():
        (manifest_dir / (split + ".txt")).write_text("\n".join(sample_ids) + "\n", encoding="utf-8")

    unique_ids = set().union(*[set(items) for items in splits.values()])
    summary = {
        "base_root": str(args.base_root.resolve()),
        "doctor_root": str(args.doctor_root.resolve()),
        "base_train_samples": len(base_train),
        "replaced_base_train_samples": len(replaced_base_ids),
        "retained_base_train_samples": len(retained_base_train),
        "doctor_revised_samples": len(doctor_ids),
        "doctor_train_repeat": args.doctor_train_repeat,
        "effective_train_samples": len(train_ids),
        "doctor_validation_samples": 0,
        "doctor_test_samples": 0,
        "unique_linked_samples": len(unique_ids),
    }
    (metadata_dir / "bundle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    build(parse_args())
