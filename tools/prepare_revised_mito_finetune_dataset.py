"""Prepare an isolated partial-label mito fine-tuning set from revised TIF + DB.

The platform database retains pre-annotation contours as well as physician
edits.  AI marks are rendered first and physician marks second, making doctor
edits authoritative.  Pixels outside reviewed contours are 255 (ignore), not
synthetic background labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IGNORE_INDEX = 255
MITO_NAME_TO_ID = {
    "严重损伤线粒体": 1,
    "中度损伤线粒体": 2,
    "健康线粒体": 3,
    "自噬线粒体": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--historical-base-root",
        type=Path,
        default=None,
        help="optional first-batch Jijie root; its mito annotations are fused as trusted labels for the same revised images",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-raw", action="store_true")
    parser.add_argument("--include-ai-marks", action="store_true")
    parser.add_argument(
        "--ai-supervision-weight",
        type=float,
        default=0.35,
        help="loss weight assigned to retained historical AI contours (0, 1]",
    )
    parser.add_argument(
        "--doctor-supervision-weight",
        type=float,
        default=1.0,
        help="loss weight assigned to physician-drawn or physician-edited contours (0, 1]",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--cv-folds", type=int, default=5, help="number of case-grouped cross-validation folds")
    return parser.parse_args()


def parse_group_ids(value):
    if value is None:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return [int(token) for token in re.findall(r"\d+", str(value))]
    if isinstance(parsed, list):
        return [int(item) for item in parsed if str(item).isdigit()]
    return [parsed] if isinstance(parsed, int) else []


def parse_points(value, width, height):
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    xs, ys = payload.get("x") or [], payload.get("y") or []
    if len(xs) != len(ys):
        return []
    points = []
    for x, y in zip(xs, ys):
        try:
            points.append((max(0, min(width - 1, round(float(x)))), max(0, min(height - 1, round(float(y))))))
        except (TypeError, ValueError):
            return []
    return points


def table_exists(connection, name):
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def iter_revised_samples(root):
    for batch_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        slices_dir = batch_dir / "slices"
        if not slices_dir.is_dir():
            continue
        for sample_dir in sorted(path for path in slices_dir.iterdir() if path.is_dir() and path.name.endswith("_1")):
            tifs = sorted([*sample_dir.glob("*.tif"), *sample_dir.glob("*.tiff")])
            yield batch_dir, sample_dir, tifs, sample_dir / "slice.db"


def read_marks(db_path, width, height, include_ai_marks):
    connection = sqlite3.connect(db_path)
    try:
        groups = {
            int(row[0]): {"name": row[1] or "", "is_ai": int(row[2] or 0)}
            for row in connection.execute("SELECT id, groupName, isAi FROM MarkGroup")
        }
        marks = []
        for table in ("Mark_label_None", "Mark_human"):
            if not table_exists(connection, table):
                continue
            rows = connection.execute(f"SELECT id, position, groupId, aiStatus FROM {table} ORDER BY id")
            for row_id, position, raw_group_id, ai_status in rows:
                class_id, is_ai = None, False
                for group_id in parse_group_ids(raw_group_id):
                    group = groups.get(group_id, {})
                    candidate = MITO_NAME_TO_ID.get(group.get("name"))
                    if candidate is not None:
                        class_id = candidate
                        is_ai = bool(group.get("is_ai")) or ai_status == 1
                        break
                if class_id is None or (is_ai and not include_ai_marks):
                    continue
                points = parse_points(position, width, height)
                if points:
                    marks.append({"row_id": row_id, "class_id": class_id, "is_ai": is_ai, "points": points})
        return marks
    finally:
        connection.close()


def _draw_mark(draw, points, value):
    if len(points) >= 3:
        draw.polygon(points, fill=value, outline=value)
    elif len(points) == 2:
        draw.line(points, fill=value, width=1)
    else:
        draw.point(points[0], fill=value)


def build_historical_mask_lookup(base_root):
    if base_root is None:
        return {}
    metadata_path = base_root / "metadata" / "samples.csv"
    image_dir = base_root / "JPEGImages"
    mask_dir = base_root / "SegmentationClass"
    for path in (metadata_path, image_dir, mask_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    lookup = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            tif_name = (row.get("tif_name") or "").lower()
            sample_id = row.get("prepared_sample_id")
            if not tif_name or not sample_id:
                continue
            mask_path = mask_dir / (sample_id + ".png")
            if mask_path.is_file():
                if tif_name in lookup:
                    raise ValueError("duplicate historical TIF name: {}".format(tif_name))
                lookup[tif_name] = mask_path
    return lookup


def render_mask(
    image_path,
    db_path,
    include_ai_marks,
    ai_supervision_weight,
    doctor_supervision_weight,
    historical_mask_path=None,
):
    with Image.open(image_path) as image:
        width, height = image.size
    mask = Image.new("L", (width, height), IGNORE_INDEX)
    supervision_weight = Image.new("L", (width, height), 0)
    supervision_source = Image.new("L", (width, height), 0)
    marks = read_marks(db_path, width, height, include_ai_marks)
    mask_draw = ImageDraw.Draw(mask)
    weight_draw = ImageDraw.Draw(supervision_weight)
    source_draw = ImageDraw.Draw(supervision_source)
    counts = Counter()
    for mark in sorted((item for item in marks if item["is_ai"]), key=lambda item: item["row_id"]):
        points = mark["points"]
        _draw_mark(mask_draw, points, mark["class_id"])
        _draw_mark(weight_draw, points, int(round(255.0 * ai_supervision_weight)))
        _draw_mark(source_draw, points, 1)
        counts[("ai", mark["class_id"])] += 1

    if historical_mask_path is not None:
        historical = np.asarray(Image.open(historical_mask_path), dtype=np.uint8)
        if historical.shape != (height, width):
            raise ValueError("historical mask size mismatch: {}".format(historical_mask_path))
        historical_selected = np.isin(historical, list(MITO_NAME_TO_ID.values()))
        mask_array = np.asarray(mask, dtype=np.uint8).copy()
        weight_array = np.asarray(supervision_weight, dtype=np.uint8).copy()
        source_array = np.asarray(supervision_source, dtype=np.uint8).copy()
        mask_array[historical_selected] = historical[historical_selected]
        weight_array[historical_selected] = int(round(255.0 * doctor_supervision_weight))
        source_array[historical_selected] = 2
        mask = Image.fromarray(mask_array, mode="L")
        supervision_weight = Image.fromarray(weight_array, mode="L")
        supervision_source = Image.fromarray(source_array, mode="L")
        mask_draw = ImageDraw.Draw(mask)
        weight_draw = ImageDraw.Draw(supervision_weight)
        source_draw = ImageDraw.Draw(supervision_source)

    for mark in sorted((item for item in marks if not item["is_ai"]), key=lambda item: item["row_id"]):
        points = mark["points"]
        _draw_mark(mask_draw, points, mark["class_id"])
        _draw_mark(weight_draw, points, int(round(255.0 * doctor_supervision_weight)))
        _draw_mark(source_draw, points, 3)
        counts[("doctor", mark["class_id"])] += 1

    values = np.asarray(mask, dtype=np.uint8)
    weight_values = np.asarray(supervision_weight, dtype=np.uint8)
    source_values = np.asarray(supervision_source, dtype=np.uint8)
    pixels = {class_id: int((values == class_id).sum()) for class_id in range(1, 5)}
    source_pixels = {
        "ai": int((source_values == 1).sum()),
        "historical_doctor": int((source_values == 2).sum()),
        "revised_doctor": int((source_values == 3).sum()),
    }
    return mask, supervision_weight, supervision_source, counts, pixels, source_pixels


def infer_case_id(tif_name):
    """Return the six-digit source case identifier embedded in the TIF name."""
    stem = Path(tif_name).stem
    for pattern in (r"(?:^|[_-])s[-_]?([0-9]{6})", r"(?:^|[_-])e([0-9]{6})"):
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    raise ValueError("cannot infer six-digit case id from TIF name: {}".format(tif_name))


def write_manifest(path, sample_ids):
    values = sorted(sample_ids)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def build_grouped_cv_manifests(manifests, metadata, rows, n_folds):
    """Create deterministic case-held-out folds without duplicating image data."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"]].append(row["sample_id"])
    if len(groups) < n_folds:
        raise ValueError("need at least {} cases for grouped CV, found {}".format(n_folds, len(groups)))

    # Largest-group-first balancing keeps the unavoidable 17-slice case isolated.
    fold_groups = [[] for _ in range(n_folds)]
    fold_sizes = [0 for _ in range(n_folds)]
    for case_id in sorted(groups, key=lambda key: (-len(groups[key]), key)):
        fold_index = min(range(n_folds), key=lambda index: (fold_sizes[index], len(fold_groups[index]), index))
        fold_groups[fold_index].append(case_id)
        fold_sizes[fold_index] += len(groups[case_id])

    fold_rows = []
    all_ids = {row["sample_id"] for row in rows}
    folds_root = manifests / "folds"
    for fold_index, held_out_cases in enumerate(fold_groups):
        val_ids = sorted(sample_id for case_id in held_out_cases for sample_id in groups[case_id])
        train_ids = sorted(all_ids.difference(val_ids))
        fold_dir = folds_root / "fold_{:02d}".format(fold_index)
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(fold_dir / "train.txt", train_ids)
        write_manifest(fold_dir / "val.txt", val_ids)
        # The held-out fold is also exposed as test for one-command evaluation.
        write_manifest(fold_dir / "test.txt", val_ids)
        fold_summary = {
            "fold": fold_index,
            "held_out_case_ids": held_out_cases,
            "train_samples": len(train_ids),
            "validation_samples": len(val_ids),
        }
        (fold_dir / "fold_summary.json").write_text(
            json.dumps(fold_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for case_id in held_out_cases:
            fold_rows.append(
                {
                    "fold": fold_index,
                    "case_id": case_id,
                    "case_samples": len(groups[case_id]),
                    "role": "validation_test",
                }
            )

    with (metadata / "cv_fold_assignment.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "case_id", "case_samples", "role"])
        writer.writeheader()
        writer.writerows(fold_rows)
    return [
        {
            "fold": fold_index,
            "held_out_case_ids": fold_groups[fold_index],
            "held_out_samples": fold_sizes[fold_index],
            "train_samples": len(rows) - fold_sizes[fold_index],
        }
        for fold_index in range(n_folds)
    ]


def prepare(args):
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)
    if args.output_root.exists():
        if not args.force:
            raise RuntimeError("Output already exists: {}".format(args.output_root))
        shutil.rmtree(args.output_root)
    if not 0.0 < args.ai_supervision_weight <= 1.0:
        raise ValueError("--ai-supervision-weight must be in (0, 1]")
    if not 0.0 < args.doctor_supervision_weight <= 1.0:
        raise ValueError("--doctor-supervision-weight must be in (0, 1]")

    images = args.output_root / "JPEGImages"
    masks = args.output_root / "SegmentationClass"
    weights = args.output_root / "SupervisionWeight"
    sources = args.output_root / "SupervisionSource"
    manifests = args.output_root / "ImageSets" / "Segmentation" / "tasks" / "mito"
    metadata = args.output_root / "metadata"
    for path in (images, masks, weights, sources, manifests, metadata):
        path.mkdir(parents=True, exist_ok=True)

    historical_lookup = build_historical_mask_lookup(
        args.historical_base_root.resolve() if args.historical_base_root is not None else None
    )

    rows, invalid = [], []
    total_marks, total_pixels, total_source_pixels = Counter(), Counter(), Counter()
    for batch_dir, sample_dir, tifs, db_path in iter_revised_samples(args.input_root):
        if len(tifs) != 1 or not db_path.is_file():
            invalid.append({"batch": batch_dir.name, "slice": sample_dir.name, "reason": "expected one tif and slice.db"})
            continue
        tif_path = tifs[0]
        sample_id = "doctor_{}_{}".format(batch_dir.name, sample_dir.name)
        historical_mask_path = historical_lookup.get(tif_path.name.lower())
        if args.historical_base_root is not None and historical_mask_path is None:
            raise FileNotFoundError("no historical doctor mask found for revised TIF: {}".format(tif_path.name))
        mask, supervision_weight, supervision_source, mark_counts, pixels, source_pixels = render_mask(
            tif_path,
            db_path,
            args.include_ai_marks,
            args.ai_supervision_weight,
            args.doctor_supervision_weight,
            historical_mask_path=historical_mask_path,
        )
        with Image.open(tif_path) as image:
            image.convert("RGB").save(images / (sample_id + ".jpg"), quality=args.jpeg_quality)
        mask.save(masks / (sample_id + ".png"))
        supervision_weight.save(weights / (sample_id + ".png"))
        supervision_source.save(sources / (sample_id + ".png"))
        if args.copy_raw:
            raw_dir = args.output_root / "raw_selected" / batch_dir.name / "slices" / sample_dir.name
            raw_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tif_path, raw_dir / tif_path.name)
            shutil.copy2(db_path, raw_dir / "slice.db")
        row = {
            "sample_id": sample_id,
            "split": "all_revised",
            "batch": batch_dir.name,
            "slice": sample_dir.name,
            "tif_name": tif_path.name,
            "case_id": infer_case_id(tif_path.name),
            "source_dir": str(sample_dir),
            "valid_pixels": int(np.count_nonzero(np.asarray(mask) != IGNORE_INDEX)),
            "historical_doctor_supervision_pixels": source_pixels["historical_doctor"],
            "revised_doctor_supervision_pixels": source_pixels["revised_doctor"],
            "ai_supervision_pixels": source_pixels["ai"],
        }
        for class_id in range(1, 5):
            row["pixels_class_{}".format(class_id)] = pixels[class_id]
            row["doctor_marks_class_{}".format(class_id)] = mark_counts[("doctor", class_id)]
            row["ai_marks_class_{}".format(class_id)] = mark_counts[("ai", class_id)]
        rows.append(row)
        total_marks.update(mark_counts)
        total_pixels.update(pixels)
        total_source_pixels.update(source_pixels)

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    cv_folds = build_grouped_cv_manifests(manifests, metadata, rows, args.cv_folds)
    # The top-level manifest deliberately has all 52 inputs in train.  Actual
    # experiments must target one of folds/fold_XX to retain a held-out case.
    write_manifest(manifests / "train.txt", [row["sample_id"] for row in rows])
    write_manifest(manifests / "val.txt", [])
    write_manifest(manifests / "test.txt", [])
    fieldnames = list(rows[0]) if rows else ["sample_id"]
    with (metadata / "samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (metadata / "invalid_samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch", "slice", "reason"])
        writer.writeheader()
        writer.writerows(invalid)
    summary = {
        "input_root": str(args.input_root.resolve()),
        "historical_base_root": str(args.historical_base_root.resolve()) if args.historical_base_root is not None else None,
        "selected_revised_slices": len(rows),
        "invalid_slices": len(invalid),
        "include_ai_marks": bool(args.include_ai_marks),
        "ignore_index": IGNORE_INDEX,
        "supervision_weight_encoding": {
            "doctor": args.doctor_supervision_weight,
            "ai": args.ai_supervision_weight,
            "storage": "uint8 image in SupervisionWeight, decoded to [0, 1] during training",
        },
        "case_counts": {case_id: len([row for row in rows if row["case_id"] == case_id]) for case_id in sorted({row["case_id"] for row in rows})},
        "cross_validation": {"folds": args.cv_folds, "assignments": cv_folds},
        "mark_counts": {"{}_class_{}".format(source, class_id): total_marks[(source, class_id)] for source in ("doctor", "ai") for class_id in range(1, 5)},
        "supervised_pixel_counts": {str(class_id): total_pixels[class_id] for class_id in range(1, 5)},
        "supervision_source_pixel_counts": dict(total_source_pixels),
    }
    (metadata / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare(parse_args())
