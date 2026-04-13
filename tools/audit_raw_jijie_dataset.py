import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict

from PIL import Image


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

TASK_SPECS = {
    "mito": {
        "label_ids": [1, 2, 3, 4],
        "description": "线粒体状态分割",
    },
    "mito_sr": {
        "label_ids": [1, 2, 3, 4, 5],
        "description": "线粒体-肌浆网关系",
    },
    "sarcomere": {
        "label_ids": [10, 12],
        "description": "肌节几何结构",
    },
}

RAW_STATUS_ANNOTATED = "annotated_nonzero"
RAW_STATUS_ZERO_MARK = "groups_but_zero_marks"
RAW_STATUS_TEMPLATE_ONLY = "template_only_no_mark_tables"
RAW_STATUS_MARKED_UNKNOWN = "marked_unknown_only"
RAW_STATUS_INVALID = "invalid"

TASK_STATUS_SUPERVISED_POSITIVE = "supervised_positive"
TASK_STATUS_SUPERVISED_NEGATIVE = "supervised_negative_exhaustive"
TASK_STATUS_PARTIAL = "partial_labeled"
TASK_STATUS_INFERENCE_ONLY = "inference_only"
TASK_STATUS_DUPLICATE = "duplicate_dropped"
TASK_STATUS_INVALID = "invalid"

RAW_STATUS_PRIORITY = {
    RAW_STATUS_ANNOTATED: 4,
    RAW_STATUS_ZERO_MARK: 3,
    RAW_STATUS_TEMPLATE_ONLY: 2,
    RAW_STATUS_MARKED_UNKNOWN: 1,
    RAW_STATUS_INVALID: 0,
}

MARK_TABLES = ("Mark_label_None", "Mark_human", "Mark_label_custom")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit the raw jijie tif+db exports and build task-aware sample registries."
    )
    parser.add_argument(
        "--input-root",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "fenge_datasets"),
        help="Root containing cases_export_* folders.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jijie_audit"),
        help="Output directory for audit artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove output directory before writing audit artifacts.",
    )
    return parser.parse_args()


def ensure_empty_dir(path, force=False):
    if os.path.isdir(path):
        if not force and os.listdir(path):
            raise RuntimeError("Output directory already exists and is not empty: {}".format(path))
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

    return "TS::{}".format(timestamp_dir)


def build_sample_uid(record):
    return "{}__{}__{}".format(
        sanitize_name(record["group_id"]),
        sanitize_name(record["timestamp_dir"]),
        sanitize_name(record["sample_dir"]),
    )


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


def sha1_file(path):
    sha1 = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def normalize_tiff_tag_value(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return "<bytes:{}>".format(len(value))
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    return str(value)


def inspect_image(tif_path):
    with Image.open(tif_path) as image:
        tags = getattr(image, "tag_v2", {})
        return {
            "width": image.size[0],
            "height": image.size[1],
            "image_mode": image.mode,
            "image_format": image.format,
            "tiff_tag_282": normalize_tiff_tag_value(tags.get(282)),
            "tiff_tag_283": normalize_tiff_tag_value(tags.get(283)),
            "tiff_tag_296": normalize_tiff_tag_value(tags.get(296)),
            "vendor_tag_65024_present": int(65024 in tags),
            "vendor_tag_65025_present": int(65025 in tags),
            "vendor_tag_65027_present": int(65027 in tags),
            "indicated_magnification": "",
            "scale_bar_pixels": "",
            "um_per_pixel": "",
            "physical_scale_available": 0,
            "scale_parse_status": "not_parsed",
        }


def query_table_columns(cursor, table_name):
    return [row[1] for row in cursor.execute("PRAGMA table_info({})".format(table_name)).fetchall()]


def inspect_db(db_path):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp_file.close()
    shutil.copy2(db_path, tmp_file.name)

    try:
        conn = sqlite3.connect(tmp_file.name)
        cur = conn.cursor()
        table_names = sorted(row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
        table_name_set = set(table_names)

        group_rows = cur.execute("SELECT id, groupName FROM MarkGroup").fetchall() if "MarkGroup" in table_name_set else []
        group_name_by_id = {row[0]: normalize_label_name(row[1]) for row in group_rows if normalize_label_name(row[1])}

        mark_table_counts = {}
        mark_table_parseable_counts = {}
        known_labels_present = set()
        unknown_labels_present = set()
        known_mark_count = 0
        unknown_mark_count = 0
        parseable_mark_count = 0

        for table_name in MARK_TABLES:
            if table_name not in table_name_set:
                mark_table_counts[table_name] = 0
                mark_table_parseable_counts[table_name] = 0
                continue

            columns = set(query_table_columns(cur, table_name))
            if not {"position", "groupId"}.issubset(columns):
                mark_table_counts[table_name] = 0
                mark_table_parseable_counts[table_name] = 0
                continue

            rows = cur.execute('SELECT position, groupId FROM "{}"'.format(table_name)).fetchall()
            mark_table_counts[table_name] = len(rows)
            table_parseable = 0

            for raw_position, raw_group_id in rows:
                points = parse_position(raw_position)
                if points:
                    table_parseable += 1
                    parseable_mark_count += 1

                group_ids = parse_group_ids(raw_group_id)
                resolved_names = [group_name_by_id[group_id] for group_id in group_ids if group_id in group_name_by_id]
                known_resolved = [name for name in resolved_names if name in CLASS_NAME_TO_ID]
                unknown_resolved = [name for name in resolved_names if name not in CLASS_NAME_TO_ID]

                if known_resolved:
                    known_mark_count += 1
                    known_labels_present.update(known_resolved)
                elif unknown_resolved:
                    unknown_mark_count += 1
                    unknown_labels_present.update(unknown_resolved)

            mark_table_parseable_counts[table_name] = table_parseable

        screenshot_count = cur.execute("SELECT COUNT(*) FROM ScreenShot").fetchone()[0] if "ScreenShot" in table_name_set else 0
        roi_task_status_count = (
            cur.execute("SELECT COUNT(*) FROM markROITaskStatus").fetchone()[0]
            if "markROITaskStatus" in table_name_set
            else 0
        )
        conn.close()

        has_group_templates = bool(group_name_by_id)
        total_mark_count = sum(mark_table_counts.values())

        if not table_names:
            raw_status = RAW_STATUS_INVALID
            exclusion_reason = "empty_db_no_tables"
        elif known_mark_count > 0 and parseable_mark_count > 0:
            raw_status = RAW_STATUS_ANNOTATED
            exclusion_reason = ""
        elif total_mark_count > 0 and known_mark_count == 0:
            raw_status = RAW_STATUS_MARKED_UNKNOWN
            exclusion_reason = "mark_tables_present_but_no_known_labels_resolved"
        elif has_group_templates:
            raw_status = RAW_STATUS_ZERO_MARK
            exclusion_reason = ""
        else:
            raw_status = RAW_STATUS_TEMPLATE_ONLY
            exclusion_reason = "no_group_templates_and_no_usable_marks"

        return {
            "table_names": table_names,
            "group_names": sorted(group_name_by_id.values()),
            "known_labels_present": sorted(known_labels_present, key=lambda name: CLASS_NAME_TO_ID[name]),
            "unknown_labels_present": sorted(unknown_labels_present),
            "mark_count_total": total_mark_count,
            "mark_count_known": known_mark_count,
            "mark_count_unknown": unknown_mark_count,
            "mark_count_parseable": parseable_mark_count,
            "mark_count_Mark_label_None": mark_table_counts["Mark_label_None"],
            "mark_count_Mark_human": mark_table_counts["Mark_human"],
            "mark_count_Mark_label_custom": mark_table_counts["Mark_label_custom"],
            "parseable_Mark_label_None": mark_table_parseable_counts["Mark_label_None"],
            "parseable_Mark_human": mark_table_parseable_counts["Mark_human"],
            "parseable_Mark_label_custom": mark_table_parseable_counts["Mark_label_custom"],
            "has_markgroup": int("MarkGroup" in table_name_set),
            "has_mark_label_none": int("Mark_label_None" in table_name_set),
            "has_mark_human": int("Mark_human" in table_name_set),
            "has_mark_label_custom": int("Mark_label_custom" in table_name_set),
            "screenshot_row_count": screenshot_count,
            "roi_task_status_row_count": roi_task_status_count,
            "raw_status": raw_status,
            "exclusion_reason": exclusion_reason,
        }
    finally:
        os.unlink(tmp_file.name)


def infer_annotation_scope(record):
    if record["raw_status"] == RAW_STATUS_ANNOTATED:
        return "renderable_known_marks_present"
    if record["raw_status"] == RAW_STATUS_ZERO_MARK:
        return "group_templates_present_but_zero_marks_exhaustiveness_unknown"
    if record["raw_status"] == RAW_STATUS_TEMPLATE_ONLY:
        return "template_only_or_nonstandard_export_requires_review"
    if record["raw_status"] == RAW_STATUS_MARKED_UNKNOWN:
        return "marks_present_but_unknown_labels_require_parser_extension"
    return "invalid_or_incomplete_sample"


def inspect_sample(sample_dir, export_batch, timestamp_dir, sample_dir_name):
    tif_files = sorted([name for name in os.listdir(sample_dir) if name.lower().endswith(".tif")])
    db_files = sorted([name for name in os.listdir(sample_dir) if name.lower().endswith(".db")])
    tif_name = tif_files[0] if tif_files else ""
    db_name = db_files[0] if db_files else ""
    tif_path = os.path.join(sample_dir, tif_name) if tif_name else ""
    db_path = os.path.join(sample_dir, db_name) if db_name else ""
    stem = os.path.splitext(tif_name)[0] if tif_name else sample_dir_name
    group_id = extract_group_id(stem, timestamp_dir)

    base_record = {
        "export_batch": export_batch,
        "timestamp_dir": timestamp_dir,
        "sample_dir": sample_dir_name,
        "sample_dir_path": sample_dir,
        "tif_name": tif_name,
        "db_name": db_name,
        "source_tif_path": tif_path,
        "source_db_path": db_path,
        "stem": stem,
        "group_id": group_id,
        "sample_uid": "",
        "image_hash": "",
        "duplicate_count": 1,
        "duplicate_rank": 0,
        "canonical_sample_uid": "",
        "is_canonical": 0,
        "table_names": [],
        "group_names": [],
        "known_labels_present": [],
        "unknown_labels_present": [],
        "annotation_scope": "",
        "raw_status": RAW_STATUS_INVALID,
        "exclusion_reason": "",
        "mark_count_total": 0,
        "mark_count_known": 0,
        "mark_count_unknown": 0,
        "mark_count_parseable": 0,
        "mark_count_Mark_label_None": 0,
        "mark_count_Mark_human": 0,
        "mark_count_Mark_label_custom": 0,
        "parseable_Mark_label_None": 0,
        "parseable_Mark_human": 0,
        "parseable_Mark_label_custom": 0,
        "has_markgroup": 0,
        "has_mark_label_none": 0,
        "has_mark_human": 0,
        "has_mark_label_custom": 0,
        "screenshot_row_count": 0,
        "roi_task_status_row_count": 0,
        "width": "",
        "height": "",
        "image_mode": "",
        "image_format": "",
        "tiff_tag_282": "",
        "tiff_tag_283": "",
        "tiff_tag_296": "",
        "vendor_tag_65024_present": 0,
        "vendor_tag_65025_present": 0,
        "vendor_tag_65027_present": 0,
        "indicated_magnification": "",
        "scale_bar_pixels": "",
        "um_per_pixel": "",
        "physical_scale_available": 0,
        "scale_parse_status": "not_parsed",
    }

    if len(tif_files) != 1 or len(db_files) != 1:
        base_record["raw_status"] = RAW_STATUS_INVALID
        base_record["exclusion_reason"] = "expected_1_tif_and_1_db_got_tif_{}_db_{}".format(len(tif_files), len(db_files))
        base_record["sample_uid"] = build_sample_uid(base_record)
        return base_record

    image_info = inspect_image(tif_path)
    db_info = inspect_db(db_path)

    base_record.update(image_info)
    base_record.update(db_info)
    base_record["image_hash"] = sha1_file(tif_path)
    base_record["sample_uid"] = build_sample_uid(base_record)
    base_record["annotation_scope"] = infer_annotation_scope(base_record)
    return base_record


def scan_samples(input_root):
    records = []
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
                records.append(inspect_sample(sample_dir, export_name, timestamp_dir, sample_dir_name))
    return records


def dedupe_records(records):
    by_hash = defaultdict(list)
    no_hash_records = []

    for record in records:
        if record["image_hash"]:
            by_hash[record["image_hash"]].append(record)
        else:
            no_hash_records.append(record)

    duplicate_rows = []

    for image_hash, items in by_hash.items():
        items.sort(
            key=lambda item: (
                RAW_STATUS_PRIORITY[item["raw_status"]],
                item["mark_count_known"],
                item["mark_count_total"],
                item["vendor_tag_65027_present"],
                item["export_batch"],
                item["timestamp_dir"],
                item["sample_dir"],
            ),
            reverse=True,
        )

        canonical = items[0]
        for rank, item in enumerate(items, start=1):
            item["duplicate_count"] = len(items)
            item["duplicate_rank"] = rank
            item["canonical_sample_uid"] = canonical["sample_uid"]
            item["is_canonical"] = int(rank == 1)

            if rank > 1:
                duplicate_rows.append(
                    {
                        "image_hash": image_hash,
                        "canonical_sample_uid": canonical["sample_uid"],
                        "canonical_raw_status": canonical["raw_status"],
                        "canonical_mark_count_known": canonical["mark_count_known"],
                        "dropped_sample_uid": item["sample_uid"],
                        "dropped_raw_status": item["raw_status"],
                        "dropped_mark_count_known": item["mark_count_known"],
                    }
                )

    for record in no_hash_records:
        record["duplicate_count"] = 1
        record["duplicate_rank"] = 1
        record["canonical_sample_uid"] = record["sample_uid"]
        record["is_canonical"] = 1

    return records, duplicate_rows


def classify_task_row(record, task_name, task_spec):
    task_label_ids = task_spec["label_ids"]
    task_label_names = [CLASS_NAMES[label_id] for label_id in task_label_ids]
    present_label_names = [name for name in record["known_labels_present"] if CLASS_NAME_TO_ID.get(name) in task_label_ids]
    present_label_ids = [CLASS_NAME_TO_ID[name] for name in present_label_names]

    has_mito_label = int(any(label_id in {1, 2, 3, 4} for label_id in present_label_ids))
    has_sr_label = int(5 in present_label_ids)
    has_z_line = int(10 in present_label_ids)
    has_m_line = int(12 in present_label_ids)
    supports_quant_metrics = False
    usability_status = TASK_STATUS_INFERENCE_ONLY
    usable_train = 0
    usable_eval = 0
    usable_quantify = 0
    usable_quantify_physical = 0
    needs_manual_review = 0
    review_reason = ""
    notes = ""

    if not record["is_canonical"]:
        usability_status = TASK_STATUS_DUPLICATE
        notes = "重复样本，已由规范副本替代。"
    elif record["raw_status"] == RAW_STATUS_INVALID:
        usability_status = TASK_STATUS_INVALID
        notes = "样本文件不完整或数据库不可用。"
    elif record["raw_status"] == RAW_STATUS_MARKED_UNKNOWN:
        usability_status = TASK_STATUS_INFERENCE_ONLY
        needs_manual_review = 1
        review_reason = "存在标注，但标签无法映射到当前项目类别，需要扩展解析器。"
    elif record["raw_status"] in {RAW_STATUS_ZERO_MARK, RAW_STATUS_TEMPLATE_ONLY}:
        usability_status = TASK_STATUS_INFERENCE_ONLY
        needs_manual_review = 1
        review_reason = "当前无法确认是否为该任务的穷尽性负样本。"
    elif present_label_ids:
        if task_name == "mito":
            supports_quant_metrics = True
            usability_status = TASK_STATUS_SUPERVISED_POSITIVE
            usable_train = 1
            usable_eval = 1
            usable_quantify = 1
        elif task_name == "mito_sr":
            supports_quant_metrics = bool(has_mito_label and has_sr_label)
            usability_status = TASK_STATUS_SUPERVISED_POSITIVE
            usable_train = 1
            usable_eval = 1
            usable_quantify = int(supports_quant_metrics)
            if not supports_quant_metrics:
                notes = "可用于分割训练，但不满足线粒体-肌浆网关系量化条件。"
        elif task_name == "sarcomere":
            supports_quant_metrics = bool(has_z_line and has_m_line)
            usable_train = 1
            usable_eval = 1
            usable_quantify = int(supports_quant_metrics)
            if supports_quant_metrics:
                usability_status = TASK_STATUS_SUPERVISED_POSITIVE
            else:
                usability_status = TASK_STATUS_PARTIAL
                notes = "存在 Z 线或 M 线标注，但未形成完整几何量化条件。"
    else:
        usability_status = TASK_STATUS_PARTIAL
        needs_manual_review = 1
        review_reason = "样本存在其他任务标注，但当前任务无正标注，负样本完备性未知。"
        notes = "不可直接当作该任务负样本。"

    usable_quantify_physical = int(usable_quantify and record["physical_scale_available"])

    return {
        "sample_uid": record["sample_uid"],
        "canonical_sample_uid": record["canonical_sample_uid"],
        "is_canonical": record["is_canonical"],
        "group_id": record["group_id"],
        "raw_status": record["raw_status"],
        "task_name": task_name,
        "task_description": task_spec["description"],
        "task_label_ids": ",".join(str(label_id) for label_id in task_label_ids),
        "task_label_names": "|".join(task_label_names),
        "task_positive_label_ids": ",".join(str(label_id) for label_id in present_label_ids),
        "task_positive_label_names": "|".join(present_label_names),
        "task_positive_label_count": len(present_label_ids),
        "has_mito_label": has_mito_label,
        "has_sr_label": has_sr_label,
        "has_z_line": has_z_line,
        "has_m_line": has_m_line,
        "supports_quant_metrics": int(supports_quant_metrics),
        "usability_status": usability_status,
        "usable_train": usable_train,
        "usable_eval": usable_eval,
        "usable_quantify": usable_quantify,
        "physical_scale_available": record["physical_scale_available"],
        "usable_quantify_physical": usable_quantify_physical,
        "needs_manual_review": needs_manual_review,
        "manual_review_reason": review_reason,
        "notes": notes,
    }


def build_task_registry(records):
    rows = []
    for record in records:
        for task_name, task_spec in TASK_SPECS.items():
            rows.append(classify_task_row(record, task_name, task_spec))
    return rows


def build_manual_review_queue(records, task_rows):
    reasons_by_sample = defaultdict(set)

    for record in records:
        if not record["is_canonical"]:
            continue
        if record["raw_status"] == RAW_STATUS_ZERO_MARK:
            reasons_by_sample[record["sample_uid"]].add("zero_mark_需要确认是否为穷尽性负样本")
        if record["raw_status"] == RAW_STATUS_TEMPLATE_ONLY:
            reasons_by_sample[record["sample_uid"]].add("template_only_需要确认导出格式或标注是否缺失")
        if record["raw_status"] == RAW_STATUS_MARKED_UNKNOWN:
            reasons_by_sample[record["sample_uid"]].add("存在未知标签_需要扩展标签映射")

    for row in task_rows:
        if row["is_canonical"] and row["needs_manual_review"]:
            reasons_by_sample[row["sample_uid"]].add("{}: {}".format(row["task_name"], row["manual_review_reason"]))

    rows = []
    record_by_uid = {record["sample_uid"]: record for record in records}
    for sample_uid, reasons in sorted(reasons_by_sample.items()):
        record = record_by_uid[sample_uid]
        rows.append(
            {
                "sample_uid": sample_uid,
                "group_id": record["group_id"],
                "raw_status": record["raw_status"],
                "source_tif_path": record["source_tif_path"],
                "source_db_path": record["source_db_path"],
                "reasons": " | ".join(sorted(reasons)),
            }
        )
    return rows


def build_scale_metadata(records):
    rows = []
    for record in records:
        if not record["is_canonical"]:
            continue
        note = ""
        if record["vendor_tag_65027_present"]:
            note = "TIFF 中存在厂商元数据，但当前脚本未自动解码为最终 um_per_pixel。"
        rows.append(
            {
                "sample_uid": record["sample_uid"],
                "group_id": record["group_id"],
                "source_tif_path": record["source_tif_path"],
                "width": record["width"],
                "height": record["height"],
                "image_mode": record["image_mode"],
                "image_format": record["image_format"],
                "tiff_tag_282": record["tiff_tag_282"],
                "tiff_tag_283": record["tiff_tag_283"],
                "tiff_tag_296": record["tiff_tag_296"],
                "vendor_tag_65024_present": record["vendor_tag_65024_present"],
                "vendor_tag_65025_present": record["vendor_tag_65025_present"],
                "vendor_tag_65027_present": record["vendor_tag_65027_present"],
                "indicated_magnification": record["indicated_magnification"],
                "scale_bar_pixels": record["scale_bar_pixels"],
                "um_per_pixel": record["um_per_pixel"],
                "physical_scale_available": record["physical_scale_available"],
                "scale_parse_status": record["scale_parse_status"],
                "note": note,
            }
        )
    return rows


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_outputs(output_root, records, task_rows, duplicate_rows, review_rows, scale_rows):
    raw_fieldnames = [
        "sample_uid",
        "canonical_sample_uid",
        "is_canonical",
        "duplicate_count",
        "duplicate_rank",
        "group_id",
        "export_batch",
        "timestamp_dir",
        "sample_dir",
        "source_tif_path",
        "source_db_path",
        "image_hash",
        "width",
        "height",
        "image_mode",
        "image_format",
        "tiff_tag_282",
        "tiff_tag_283",
        "tiff_tag_296",
        "vendor_tag_65024_present",
        "vendor_tag_65025_present",
        "vendor_tag_65027_present",
        "indicated_magnification",
        "scale_bar_pixels",
        "um_per_pixel",
        "physical_scale_available",
        "scale_parse_status",
        "table_names",
        "group_names",
        "known_labels_present",
        "unknown_labels_present",
        "annotation_scope",
        "raw_status",
        "exclusion_reason",
        "mark_count_total",
        "mark_count_known",
        "mark_count_unknown",
        "mark_count_parseable",
        "mark_count_Mark_label_None",
        "mark_count_Mark_human",
        "mark_count_Mark_label_custom",
        "parseable_Mark_label_None",
        "parseable_Mark_human",
        "parseable_Mark_label_custom",
        "has_markgroup",
        "has_mark_label_none",
        "has_mark_human",
        "has_mark_label_custom",
        "screenshot_row_count",
        "roi_task_status_row_count",
    ]

    raw_rows = []
    for record in records:
        row = dict(record)
        row["table_names"] = "|".join(record["table_names"])
        row["group_names"] = "|".join(record["group_names"])
        row["known_labels_present"] = "|".join(record["known_labels_present"])
        row["unknown_labels_present"] = "|".join(record["unknown_labels_present"])
        raw_rows.append(row)

    task_fieldnames = [
        "sample_uid",
        "canonical_sample_uid",
        "is_canonical",
        "group_id",
        "raw_status",
        "task_name",
        "task_description",
        "task_label_ids",
        "task_label_names",
        "task_positive_label_ids",
        "task_positive_label_names",
        "task_positive_label_count",
        "has_mito_label",
        "has_sr_label",
        "has_z_line",
        "has_m_line",
        "supports_quant_metrics",
        "usability_status",
        "usable_train",
        "usable_eval",
        "usable_quantify",
        "physical_scale_available",
        "usable_quantify_physical",
        "needs_manual_review",
        "manual_review_reason",
        "notes",
    ]

    duplicate_fieldnames = [
        "image_hash",
        "canonical_sample_uid",
        "canonical_raw_status",
        "canonical_mark_count_known",
        "dropped_sample_uid",
        "dropped_raw_status",
        "dropped_mark_count_known",
    ]

    review_fieldnames = [
        "sample_uid",
        "group_id",
        "raw_status",
        "source_tif_path",
        "source_db_path",
        "reasons",
    ]

    scale_fieldnames = [
        "sample_uid",
        "group_id",
        "source_tif_path",
        "width",
        "height",
        "image_mode",
        "image_format",
        "tiff_tag_282",
        "tiff_tag_283",
        "tiff_tag_296",
        "vendor_tag_65024_present",
        "vendor_tag_65025_present",
        "vendor_tag_65027_present",
        "indicated_magnification",
        "scale_bar_pixels",
        "um_per_pixel",
        "physical_scale_available",
        "scale_parse_status",
        "note",
    ]

    write_csv(os.path.join(output_root, "raw_registry.csv"), raw_rows, raw_fieldnames)
    write_csv(os.path.join(output_root, "task_registry.csv"), task_rows, task_fieldnames)
    write_csv(os.path.join(output_root, "duplicate_resolution.csv"), duplicate_rows, duplicate_fieldnames)
    write_csv(os.path.join(output_root, "manual_review_queue.csv"), review_rows, review_fieldnames)
    write_csv(os.path.join(output_root, "scale_metadata.csv"), scale_rows, scale_fieldnames)

    unknown_labels = sorted({label for record in records for label in record["unknown_labels_present"]})
    with open(os.path.join(output_root, "unknown_labels.json"), "w", encoding="utf-8") as handle:
        json.dump(unknown_labels, handle, ensure_ascii=False, indent=2)


def build_summary(input_root, output_root, records, task_rows, duplicate_rows, review_rows):
    canonical_records = [record for record in records if record["is_canonical"]]
    raw_status_counts = Counter(record["raw_status"] for record in records)
    canonical_status_counts = Counter(record["raw_status"] for record in canonical_records)
    task_status_counts = defaultdict(Counter)

    for row in task_rows:
        if row["is_canonical"]:
            task_status_counts[row["task_name"]][row["usability_status"]] += 1

    return {
        "input_root": os.path.abspath(input_root),
        "output_root": os.path.abspath(output_root),
        "scan_summary": {
            "scanned_samples": len(records),
            "canonical_samples": len(canonical_records),
            "duplicate_rows": len(duplicate_rows),
            "manual_review_rows": len(review_rows),
        },
        "raw_status_counts_all_rows": dict(raw_status_counts),
        "raw_status_counts_canonical_only": dict(canonical_status_counts),
        "task_status_counts_canonical_only": {
            task_name: dict(counter) for task_name, counter in task_status_counts.items()
        },
    }


def main():
    args = parse_args()
    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)

    ensure_empty_dir(output_root, force=args.force)
    records = scan_samples(input_root)
    records, duplicate_rows = dedupe_records(records)
    task_rows = build_task_registry(records)
    review_rows = build_manual_review_queue(records, task_rows)
    scale_rows = build_scale_metadata(records)
    save_outputs(output_root, records, task_rows, duplicate_rows, review_rows, scale_rows)

    summary = build_summary(input_root, output_root, records, task_rows, duplicate_rows, review_rows)
    with open(os.path.join(output_root, "audit_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
