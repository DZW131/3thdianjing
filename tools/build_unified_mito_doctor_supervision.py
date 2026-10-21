"""Build a leakage-safe, provenance-weighted mito fine-tuning dataset.

The original doctor-labelled data are useful but incomplete: a zero-valued
pixel means "not annotated", not "true background".  This builder therefore
keeps only mito-positive pixels from the original masks and assigns them full
supervision weight.  Revised slices supersede their original counterparts and
retain their per-pixel doctor/AI provenance weights.

Five folds are grouped by patient case.  A held-out revised case is excluded
from *every* training source, including the historical doctor annotation set.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


MITO_CLASS_IDS = (1, 2, 3, 4)
IGNORE_INDEX = 255


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True, help="historical Jijie data root")
    parser.add_argument("--revised-root", type=Path, required=True, help="prepared 52-slice revised data root")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=None,
        help="historical supervised-positive manifest; defaults to the mito all_supervised_positive list",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_ids(path: Path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_ids(path: Path, values):
    values = sorted(values)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def copy_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_partial_doctor_mask(source: Path, destination: Path, weight_destination: Path):
    values = np.asarray(Image.open(source), dtype=np.uint8)
    positive = np.isin(values, MITO_CLASS_IDS)
    mask = np.full(values.shape, IGNORE_INDEX, dtype=np.uint8)
    mask[positive] = values[positive]
    weight = np.zeros(values.shape, dtype=np.uint8)
    weight[positive] = 255
    Image.fromarray(mask, mode="L").save(destination)
    Image.fromarray(weight, mode="L").save(weight_destination)
    return int(positive.sum())


def copy_revised_label(source_mask: Path, source_weight: Path, mask_destination: Path, weight_destination: Path):
    mask = np.asarray(Image.open(source_mask), dtype=np.uint8)
    weight = np.asarray(Image.open(source_weight), dtype=np.uint8)
    if mask.shape != weight.shape:
        raise ValueError("revised mask and weight shape differ: {}".format(source_mask))
    invalid = (mask == IGNORE_INDEX) & (weight != 0)
    if np.any(invalid):
        raise ValueError("ignored revised pixels must have weight 0: {}".format(source_mask))
    Image.fromarray(mask, mode="L").save(mask_destination)
    Image.fromarray(weight, mode="L").save(weight_destination)
    return int((mask != IGNORE_INDEX).sum())


def case_folds(revised_rows, fold_count):
    groups = defaultdict(list)
    for row in revised_rows:
        groups[row["case_id"]].append(row["output_id"])
    if len(groups) < fold_count:
        raise ValueError("need at least {} distinct revised cases, found {}".format(fold_count, len(groups)))

    folds = [[] for _ in range(fold_count)]
    sizes = [0] * fold_count
    for case_id in sorted(groups, key=lambda item: (-len(groups[item]), item)):
        index = min(range(fold_count), key=lambda item: (sizes[item], len(folds[item]), item))
        folds[index].append(case_id)
        sizes[index] += len(groups[case_id])
    return folds


def main():
    args = parse_args()
    base_root = args.base_root.resolve()
    revised_root = args.revised_root.resolve()
    output_root = args.output_root.resolve()
    base_manifest = args.base_manifest or (
        base_root / "ImageSets" / "Segmentation" / "tasks" / "mito" / "default" / "all_supervised_positive.txt"
    )

    for path in (base_root / "JPEGImages", base_root / "SegmentationClass", revised_root / "JPEGImages", revised_root / "SegmentationClass", revised_root / "SupervisionWeight", base_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_root.exists():
        if not args.force:
            raise RuntimeError("output already exists: {}".format(output_root))
        shutil.rmtree(output_root)

    images_dir = output_root / "JPEGImages"
    masks_dir = output_root / "SegmentationClass"
    weights_dir = output_root / "SupervisionWeight"
    manifest_root = output_root / "ImageSets" / "Segmentation" / "tasks" / "mito"
    metadata_dir = output_root / "metadata"
    for path in (images_dir, masks_dir, weights_dir, manifest_root, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    base_metadata = {
        row["prepared_sample_id"]: row
        for row in read_csv_rows(base_root / "metadata" / "samples.csv")
    }
    revised_metadata = read_csv_rows(revised_root / "metadata" / "samples.csv")
    revised_by_tif = {row["tif_name"].lower(): row for row in revised_metadata}
    if len(revised_by_tif) != len(revised_metadata):
        raise ValueError("revised TIF names are not unique")

    records = []
    source_counts = Counter()
    original_ids = read_ids(base_manifest)
    for source_id in original_ids:
        metadata = base_metadata.get(source_id)
        if metadata is None:
            raise KeyError("base metadata missing {}".format(source_id))
        # Newer doctor revision replaces the same historical image completely.
        if metadata["tif_name"].lower() in revised_by_tif:
            continue

        output_id = "base__{}".format(source_id)
        image_source = base_root / "JPEGImages" / (source_id + ".jpg")
        mask_source = base_root / "SegmentationClass" / (source_id + ".png")
        copy_file(image_source, images_dir / (output_id + ".jpg"))
        valid_pixels = build_partial_doctor_mask(
            mask_source,
            masks_dir / (output_id + ".png"),
            weights_dir / (output_id + ".png"),
        )
        records.append(
            {
                "output_id": output_id,
                "source": "historical_doctor",
                "case_id": str(metadata["group_id"]),
                "tif_name": metadata["tif_name"],
                "original_sample_id": source_id,
                "revised_sample_id": "",
                "valid_pixels": valid_pixels,
            }
        )
        source_counts["historical_doctor"] += 1

    revised_records = []
    for metadata in revised_metadata:
        revised_id = metadata["sample_id"]
        output_id = "revised__{}".format(revised_id)
        copy_file(revised_root / "JPEGImages" / (revised_id + ".jpg"), images_dir / (output_id + ".jpg"))
        valid_pixels = copy_revised_label(
            revised_root / "SegmentationClass" / (revised_id + ".png"),
            revised_root / "SupervisionWeight" / (revised_id + ".png"),
            masks_dir / (output_id + ".png"),
            weights_dir / (output_id + ".png"),
        )
        record = {
            "output_id": output_id,
                "source": "doctor_revised_composite",
            "case_id": str(metadata["case_id"]),
            "tif_name": metadata["tif_name"],
            "original_sample_id": "",
            "revised_sample_id": revised_id,
            "valid_pixels": valid_pixels,
        }
        records.append(record)
        revised_records.append(record)
        source_counts["doctor_revised_composite"] += 1

    output_ids = {row["output_id"] for row in records}
    if len(output_ids) != len(records):
        raise ValueError("duplicate output sample ids")

    folds = case_folds(revised_records, args.cv_folds)
    by_case = defaultdict(list)
    for row in records:
        by_case[row["case_id"]].append(row["output_id"])
    revised_by_case = defaultdict(list)
    for row in revised_records:
        revised_by_case[row["case_id"]].append(row["output_id"])

    fold_rows = []
    all_ids = [row["output_id"] for row in records]
    for fold_index, held_out_cases in enumerate(folds):
        held_out = set(held_out_cases)
        train_ids = [row["output_id"] for row in records if row["case_id"] not in held_out]
        val_ids = [sample_id for case_id in held_out_cases for sample_id in revised_by_case[case_id]]
        leak_ids = set(train_ids).intersection(val_ids)
        if leak_ids:
            raise AssertionError("train/validation overlap in fold {}: {}".format(fold_index, sorted(leak_ids)[:3]))
        fold_dir = manifest_root / "folds" / "fold_{:02d}".format(fold_index)
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_ids(fold_dir / "train.txt", train_ids)
        write_ids(fold_dir / "val.txt", val_ids)
        write_ids(fold_dir / "test.txt", val_ids)
        summary = {
            "fold": fold_index,
            "held_out_case_ids": held_out_cases,
            "train_samples": len(train_ids),
            "validation_revised_samples": len(val_ids),
            "train_source_counts": dict(Counter(row["source"] for row in records if row["case_id"] not in held_out)),
        }
        (fold_dir / "fold_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        fold_rows.append(summary)

    write_ids(manifest_root / "train.txt", all_ids)
    write_ids(manifest_root / "val.txt", [])
    write_ids(manifest_root / "test.txt", [])
    with (metadata_dir / "samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "historical_base_root": str(base_root),
        "revised_root": str(revised_root),
        "historical_manifest": str(base_manifest),
        "total_samples": len(records),
        "source_counts": dict(source_counts),
        "trust_policy": {
            "historical_doctor": "mitochondria annotations only; unlabeled/background pixels are ignored; loss weight 1.0",
            "doctor_revised_composite": "physician edits weight 1.0; retained AI contours are auxiliary and use the revised dataset provenance weight",
        },
        "grouped_cv": fold_rows,
    }
    (metadata_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
