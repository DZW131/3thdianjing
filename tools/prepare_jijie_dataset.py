import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
from collections import Counter, defaultdict

from PIL import Image, ImageDraw


CLASS_NAMES = [
    "Background",
    "严重损伤线粒体",
    "中度损伤线粒体",
    "健康线粒体",
    "自噬线粒体",
    "肌浆网",
    "闰盘",
    "Z线样物质堆积",
    "脂滴",
    "糖原颗粒",
    "Z线",
    "T管",
    "M线",
    "心肌侧管",
]

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES) if idx > 0}

STATUS_ANNOTATED = "annotated_nonzero"
STATUS_ZERO_MARK = "groups_but_zero_marks"
STATUS_TEMPLATE_ONLY = "template_only_no_mark_tables"
STATUS_INVALID = "invalid"

USABLE_STATUSES = {STATUS_ANNOTATED, STATUS_ZERO_MARK}
DEFAULT_SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
STATUS_PRIORITY = {
    STATUS_ANNOTATED: 3,
    STATUS_ZERO_MARK: 2,
    STATUS_TEMPLATE_ONLY: 1,
    STATUS_INVALID: 0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the jijie segmentation dataset from exported tif + slice.db folders."
    )
    parser.add_argument(
        "--input-root",
        default=r"F:\fenge_datasets",
        help="Root containing cases_export_* folders.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jijie"),
        help="Prepared dataset output root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed used for deterministic group split tie-breaking.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove output root before generating the dataset.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used when saving converted images.",
    )
    return parser.parse_args()


def ensure_empty_dir(path, force=False):
    if os.path.isdir(path):
        if not force and os.listdir(path):
            raise RuntimeError(f"Output directory already exists and is not empty: {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def sanitize_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def normalize_label_name(name):
    return (name or "").strip()


def extract_group_id(stem, timestamp_dir):
    base = stem.strip()
    base = base.split(" - ")[0].strip()

    match = re.match(r"^s-(\d+)-", base, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.match(r"^(\d+)-", base)
    if match:
        return match.group(1)

    match = re.match(r"^s-(\d+)$", base, re.IGNORECASE)
    if match:
        digits = match.group(1)
        if len(digits) > 4:
            return digits[:-4]

    match = re.match(r"^[Ee]?(\d+)$", base, re.IGNORECASE)
    if match:
        digits = match.group(1)
        if len(digits) > 6:
            return digits[:-2]

    return f"TS::{timestamp_dir}"


def parse_group_ids(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, int):
        return [raw_value]

    text = str(raw_value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [int(x) for x in re.findall(r"\d+", text)]

    if isinstance(parsed, list):
        result = []
        for item in parsed:
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, str) and item.isdigit():
                result.append(int(item))
        return result
    if isinstance(parsed, int):
        return [parsed]
    if isinstance(parsed, str) and parsed.isdigit():
        return [int(parsed)]
    return []


def parse_position(raw_position):
    if not raw_position:
        return []
    try:
        parsed = json.loads(raw_position)
    except json.JSONDecodeError:
        return []

    xs = parsed.get("x") or []
    ys = parsed.get("y") or []
    if len(xs) != len(ys):
        return []

    points = []
    for x, y in zip(xs, ys):
        try:
            points.append((float(x), float(y)))
        except (TypeError, ValueError):
            return []
    return points


def copy_db_snapshot(source_path, temp_path):
    shutil.copy2(source_path, temp_path)
    return sqlite3.connect(temp_path)


def inspect_sample(sample_dir, temp_db_path):
    tif_files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(".tif")])
    db_files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(".db")])

    if len(tif_files) != 1 or len(db_files) != 1:
        return {
            "status": STATUS_INVALID,
            "reason": f"expected 1 tif and 1 db, got tif={len(tif_files)} db={len(db_files)}",
            "tif_name": tif_files[0] if tif_files else "",
            "db_name": db_files[0] if db_files else "",
            "group_names": [],
            "classes_present": [],
            "mark_count": 0,
        }

    tif_name = tif_files[0]
    db_name = db_files[0]
    db_path = os.path.join(sample_dir, db_name)

    conn = copy_db_snapshot(db_path, temp_db_path)
    cur = conn.cursor()

    table_names = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    has_mark_label = "Mark_label_None" in table_names
    has_mark_human = "Mark_human" in table_names

    group_rows = []
    if "MarkGroup" in table_names:
        group_rows = cur.execute("SELECT id, groupName FROM MarkGroup").fetchall()
    group_name_by_id = {row[0]: normalize_label_name(row[1]) for row in group_rows if normalize_label_name(row[1])}

    mark_tables = [name for name in ("Mark_label_None", "Mark_human") if name in table_names]
    classes_present = set()
    mark_count = 0

    for table_name in mark_tables:
        query = f'SELECT groupId FROM "{table_name}"'
        for (raw_group_id,) in cur.execute(query).fetchall():
            group_ids = parse_group_ids(raw_group_id)
            resolved_name = None
            for group_id in group_ids:
                candidate = group_name_by_id.get(group_id)
                if candidate in CLASS_NAME_TO_ID:
                    resolved_name = candidate
                    break
            if resolved_name:
                classes_present.add(resolved_name)
            mark_count += 1

    conn.close()

    if not has_mark_label and not has_mark_human:
        status = STATUS_TEMPLATE_ONLY
    elif group_name_by_id and mark_count > 0:
        status = STATUS_ANNOTATED
    elif group_name_by_id:
        status = STATUS_ZERO_MARK
    else:
        status = STATUS_TEMPLATE_ONLY

    return {
        "status": status,
        "reason": "",
        "tif_name": tif_name,
        "db_name": db_name,
        "group_names": sorted(group_name_by_id.values()),
        "classes_present": sorted(classes_present, key=lambda name: CLASS_NAME_TO_ID[name]),
        "mark_count": mark_count,
    }


def sha1_file(path):
    sha1 = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def scan_samples(input_root, temp_db_path):
    scanned = []
    for export_name in sorted(os.listdir(input_root)):
        export_path = os.path.join(input_root, export_name)
        slice_root = os.path.join(export_path, "slice_files")
        if not os.path.isdir(slice_root):
            continue

        for timestamp_dir in sorted(os.listdir(slice_root)):
            timestamp_path = os.path.join(slice_root, timestamp_dir)
            if not os.path.isdir(timestamp_path):
                continue

            for sample_dir_name in sorted(os.listdir(timestamp_path)):
                sample_dir = os.path.join(timestamp_path, sample_dir_name)
                if not os.path.isdir(sample_dir):
                    continue

                inspection = inspect_sample(sample_dir, temp_db_path)
                tif_name = inspection["tif_name"]
                tif_path = os.path.join(sample_dir, tif_name) if tif_name else ""
                db_path = os.path.join(sample_dir, inspection["db_name"]) if inspection["db_name"] else ""
                stem = os.path.splitext(tif_name)[0] if tif_name else sample_dir_name

                record = {
                    "export_batch": export_name,
                    "timestamp_dir": timestamp_dir,
                    "sample_dir": sample_dir_name,
                    "sample_dir_path": sample_dir,
                    "tif_name": tif_name,
                    "tif_path": tif_path,
                    "db_path": db_path,
                    "stem": stem,
                    "group_id": extract_group_id(stem, timestamp_dir),
                    "status": inspection["status"],
                    "reason": inspection["reason"],
                    "group_names": inspection["group_names"],
                    "classes_present": inspection["classes_present"],
                    "mark_count": inspection["mark_count"],
                    "image_hash": sha1_file(tif_path) if tif_path else "",
                }
                scanned.append(record)
    return scanned


def dedupe_records(records):
    by_hash = defaultdict(list)
    invalid_records = []
    for record in records:
        if record["status"] == STATUS_INVALID or not record["image_hash"]:
            invalid_records.append(record)
        else:
            by_hash[record["image_hash"]].append(record)

    canonical_records = []
    duplicate_rows = []

    for image_hash, items in sorted(by_hash.items()):
        items.sort(
            key=lambda item: (
                STATUS_PRIORITY[item["status"]],
                item["mark_count"],
                item["export_batch"],
                item["timestamp_dir"],
                item["sample_dir"],
            ),
            reverse=True,
        )
        keep = dict(items[0])
        keep["duplicate_count"] = len(items)
        canonical_records.append(keep)

        for dropped in items[1:]:
            duplicate_rows.append(
                {
                    "image_hash": image_hash,
                    "kept_sample": build_sample_id(keep),
                    "kept_status": keep["status"],
                    "kept_mark_count": keep["mark_count"],
                    "dropped_sample": build_sample_id(dropped),
                    "dropped_status": dropped["status"],
                    "dropped_mark_count": dropped["mark_count"],
                }
            )

    return canonical_records, duplicate_rows, invalid_records


def build_sample_id(record):
    group_key = sanitize_name(record["group_id"])
    timestamp_key = sanitize_name(record["timestamp_dir"])
    sample_key = sanitize_name(record["sample_dir"])
    return f"{group_key}__{timestamp_key}__{sample_key}"


def assign_group_splits(records, seed):
    usable_records = [record for record in records if record["status"] in USABLE_STATUSES]
    groups = defaultdict(list)
    for record in usable_records:
        groups[record["group_id"]].append(record)

    group_stats = []
    for group_id, items in groups.items():
        annotated_count = sum(1 for item in items if item["status"] == STATUS_ANNOTATED)
        total_count = len(items)
        class_set = set()
        for item in items:
            class_set.update(item["classes_present"])
        group_stats.append(
            {
                "group_id": group_id,
                "items": items,
                "annotated_count": annotated_count,
                "total_count": total_count,
                "class_set": class_set,
            }
        )

    rng = random.Random(seed)
    group_stats.sort(
        key=lambda stat: (-stat["annotated_count"], -stat["total_count"], stat["group_id"])
    )

    split_names = list(DEFAULT_SPLIT_RATIOS.keys())
    total_annotated = sum(stat["annotated_count"] for stat in group_stats)
    total_usable = sum(stat["total_count"] for stat in group_stats)
    target_annotated = {
        split: total_annotated * ratio for split, ratio in DEFAULT_SPLIT_RATIOS.items()
    }
    target_usable = {
        split: total_usable * ratio for split, ratio in DEFAULT_SPLIT_RATIOS.items()
    }

    assignments = {}
    split_annotated = Counter()
    split_usable = Counter()
    split_classes = {split: set() for split in split_names}
    split_group_count = Counter()

    seeded = []
    for split_name, stat in zip(split_names, group_stats[: len(split_names)]):
        assignments[stat["group_id"]] = split_name
        split_annotated[split_name] += stat["annotated_count"]
        split_usable[split_name] += stat["total_count"]
        split_classes[split_name].update(stat["class_set"])
        split_group_count[split_name] += 1
        seeded.append(stat["group_id"])

    for stat in group_stats[len(seeded) :]:
        candidates = list(split_names)
        rng.shuffle(candidates)

        def score(split_name):
            next_annotated = split_annotated[split_name] + stat["annotated_count"]
            next_usable = split_usable[split_name] + stat["total_count"]
            annotated_gap = (
                (next_annotated - target_annotated[split_name]) / max(1.0, target_annotated[split_name])
            ) ** 2
            usable_gap = (
                (next_usable - target_usable[split_name]) / max(1.0, target_usable[split_name])
            ) ** 2
            coverage_bonus = 0.0
            if split_name in {"val", "test"}:
                new_classes = len(stat["class_set"] - split_classes[split_name])
                coverage_bonus = 0.03 * new_classes
            group_penalty = 0.002 * split_group_count[split_name]
            return annotated_gap * 3.0 + usable_gap + group_penalty - coverage_bonus

        chosen_split = min(candidates, key=score)
        assignments[stat["group_id"]] = chosen_split
        split_annotated[chosen_split] += stat["annotated_count"]
        split_usable[chosen_split] += stat["total_count"]
        split_classes[chosen_split].update(stat["class_set"])
        split_group_count[chosen_split] += 1

    return assignments


def render_mask(record, temp_db_path, mask_path):
    image = Image.open(record["tif_path"])
    width, height = image.size
    mask = Image.new("L", (width, height), 0)

    if record["status"] == STATUS_ZERO_MARK:
        mask.save(mask_path)
        return []

    conn = copy_db_snapshot(record["db_path"], temp_db_path)
    cur = conn.cursor()
    table_names = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    group_rows = cur.execute("SELECT id, groupName FROM MarkGroup").fetchall() if "MarkGroup" in table_names else []
    group_name_by_id = {row[0]: normalize_label_name(row[1]) for row in group_rows if normalize_label_name(row[1])}

    draw = ImageDraw.Draw(mask)
    unknown_labels = []
    for table_name in ("Mark_label_None", "Mark_human"):
        if table_name not in table_names:
            continue

        query = f'SELECT id, position, groupId FROM "{table_name}" ORDER BY id'
        for _, raw_position, raw_group_id in cur.execute(query).fetchall():
            group_ids = parse_group_ids(raw_group_id)
            class_id = None
            label_name = None
            for group_id in group_ids:
                label_name = group_name_by_id.get(group_id)
                if label_name in CLASS_NAME_TO_ID:
                    class_id = CLASS_NAME_TO_ID[label_name]
                    break

            if class_id is None:
                if label_name:
                    unknown_labels.append(label_name)
                continue

            points = parse_position(raw_position)
            if not points:
                continue

            clipped = []
            for x, y in points:
                xi = max(0, min(width - 1, int(round(x))))
                yi = max(0, min(height - 1, int(round(y))))
                clipped.append((xi, yi))

            if len(clipped) >= 3:
                draw.polygon(clipped, fill=class_id, outline=class_id)
            elif len(clipped) == 2:
                draw.line(clipped, fill=class_id, width=1)
            elif len(clipped) == 1:
                draw.point(clipped[0], fill=class_id)

    conn.close()
    mask.save(mask_path)
    return sorted(set(unknown_labels))


def save_split_files(output_root, records):
    split_dir = os.path.join(output_root, "ImageSets", "Segmentation")
    os.makedirs(split_dir, exist_ok=True)

    default_split_rows = defaultdict(list)
    all_usable_split_rows = defaultdict(list)
    zero_mark_rows = defaultdict(list)

    for record in records:
        if record["status"] not in USABLE_STATUSES:
            continue
        sample_id = record["prepared_sample_id"]
        split = record["split"]

        all_usable_split_rows[split].append(sample_id)
        if record["status"] == STATUS_ANNOTATED:
            default_split_rows[split].append(sample_id)
        if record["status"] == STATUS_ZERO_MARK:
            zero_mark_rows[split].append(sample_id)

    for split_name in DEFAULT_SPLIT_RATIOS:
        write_list_file(os.path.join(split_dir, f"{split_name}.txt"), default_split_rows[split_name])
        write_list_file(
            os.path.join(split_dir, f"{split_name}_all_usable.txt"), all_usable_split_rows[split_name]
        )
        write_list_file(
            os.path.join(split_dir, f"{split_name}_zero_mark_candidates.txt"),
            zero_mark_rows[split_name],
        )


def write_list_file(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_metadata(output_root, canonical_records, duplicates, invalid_records, unknown_labels, summary):
    metadata_dir = os.path.join(output_root, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    sample_rows = []
    for record in canonical_records:
        sample_rows.append(
            {
                "prepared_sample_id": record.get("prepared_sample_id", ""),
                "split": record.get("split", ""),
                "status": record["status"],
                "group_id": record["group_id"],
                "export_batch": record["export_batch"],
                "timestamp_dir": record["timestamp_dir"],
                "sample_dir": record["sample_dir"],
                "tif_name": record["tif_name"],
                "image_hash": record["image_hash"],
                "mark_count": record["mark_count"],
                "duplicate_count": record.get("duplicate_count", 1),
                "classes_present": "|".join(record["classes_present"]),
            }
        )

    duplicate_rows = duplicates or []
    excluded_rows = []
    for record in invalid_records:
        excluded_rows.append(
            {
                "status": record["status"],
                "reason": record["reason"],
                "export_batch": record["export_batch"],
                "timestamp_dir": record["timestamp_dir"],
                "sample_dir": record["sample_dir"],
                "tif_name": record["tif_name"],
                "db_path": record["db_path"],
            }
        )

    write_csv(
        os.path.join(metadata_dir, "samples.csv"),
        sample_rows,
        [
            "prepared_sample_id",
            "split",
            "status",
            "group_id",
            "export_batch",
            "timestamp_dir",
            "sample_dir",
            "tif_name",
            "image_hash",
            "mark_count",
            "duplicate_count",
            "classes_present",
        ],
    )
    write_csv(
        os.path.join(metadata_dir, "duplicates_removed.csv"),
        duplicate_rows,
        [
            "image_hash",
            "kept_sample",
            "kept_status",
            "kept_mark_count",
            "dropped_sample",
            "dropped_status",
            "dropped_mark_count",
        ],
    )
    write_csv(
        os.path.join(metadata_dir, "excluded_invalid_samples.csv"),
        excluded_rows,
        ["status", "reason", "export_batch", "timestamp_dir", "sample_dir", "tif_name", "db_path"],
    )

    with open(os.path.join(metadata_dir, "unknown_labels.json"), "w", encoding="utf-8") as handle:
        json.dump(sorted(unknown_labels), handle, ensure_ascii=False, indent=2)
    with open(os.path.join(metadata_dir, "class_mapping.json"), "w", encoding="utf-8") as handle:
        json.dump({name: idx for idx, name in enumerate(CLASS_NAMES)}, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(metadata_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def prepare_dataset(args):
    output_root = os.path.abspath(args.output_root)
    images_dir = os.path.join(output_root, "JPEGImages")
    masks_dir = os.path.join(output_root, "SegmentationClass")
    temp_dir = os.path.join(output_root, "_tmp")

    ensure_empty_dir(output_root, force=args.force)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    temp_db_path = os.path.join(temp_dir, "working_slice.db")

    print("Scanning raw samples...")
    scanned_records = scan_samples(args.input_root, temp_db_path)
    canonical_records, duplicate_rows, invalid_records = dedupe_records(scanned_records)

    split_assignments = assign_group_splits(canonical_records, args.seed)
    unknown_labels = set()

    split_counter = Counter()
    status_counter = Counter()
    split_status_counter = defaultdict(Counter)
    split_class_presence = defaultdict(set)

    print("Converting canonical usable samples...")
    for record in canonical_records:
        if record["status"] not in USABLE_STATUSES:
            continue

        sample_id = build_sample_id(record)
        record["prepared_sample_id"] = sample_id
        record["split"] = split_assignments[record["group_id"]]

        image_out_path = os.path.join(images_dir, f"{sample_id}.jpg")
        mask_out_path = os.path.join(masks_dir, f"{sample_id}.png")

        with Image.open(record["tif_path"]) as image:
            rgb_image = image.convert("RGB")
            rgb_image.save(image_out_path, format="JPEG", quality=args.jpeg_quality)

        labels = render_mask(record, temp_db_path, mask_out_path)
        unknown_labels.update(labels)

        split_counter[record["split"]] += 1
        status_counter[record["status"]] += 1
        split_status_counter[record["split"]][record["status"]] += 1
        split_class_presence[record["split"]].update(record["classes_present"])

    for record in canonical_records:
        if "prepared_sample_id" not in record:
            record["prepared_sample_id"] = ""
            record["split"] = ""

    save_split_files(output_root, canonical_records)

    summary = {
        "input_root": os.path.abspath(args.input_root),
        "output_root": output_root,
        "scan_summary": {
            "scanned_total": len(scanned_records),
            "canonical_total": len(canonical_records),
            "usable_total": sum(1 for record in canonical_records if record["status"] in USABLE_STATUSES),
            "annotated_total": sum(1 for record in canonical_records if record["status"] == STATUS_ANNOTATED),
            "zero_mark_total": sum(1 for record in canonical_records if record["status"] == STATUS_ZERO_MARK),
            "template_only_total": sum(
                1 for record in canonical_records if record["status"] == STATUS_TEMPLATE_ONLY
            ),
            "invalid_total": len(invalid_records),
            "duplicates_removed": len(duplicate_rows),
            "group_count": len({record["group_id"] for record in canonical_records if record["status"] in USABLE_STATUSES}),
        },
        "split_summary_all_usable": {
            split: {
                "total": split_counter[split],
                "annotated": split_status_counter[split][STATUS_ANNOTATED],
                "zero_mark_candidates": split_status_counter[split][STATUS_ZERO_MARK],
                "classes_present": sorted(
                    split_class_presence[split], key=lambda name: CLASS_NAME_TO_ID.get(name, 999)
                ),
            }
            for split in DEFAULT_SPLIT_RATIOS
        },
        "default_training_split_summary": {
            split: split_status_counter[split][STATUS_ANNOTATED] for split in DEFAULT_SPLIT_RATIOS
        },
        "class_names": CLASS_NAMES,
    }

    save_metadata(output_root, canonical_records, duplicate_rows, invalid_records, unknown_labels, summary)
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    prepare_dataset(args)


if __name__ == "__main__":
    main()
