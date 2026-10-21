"""Export model pre-annotations as platform-ready TIF + slice.db folders.

The clinical annotation platform stores one original TIF and one ``slice.db``
per slice.  This tool pairs post-processed mitochondria masks and
mitochondria/SR masks with the corresponding raw TIF export.  It copies the
original database only as a schema/template source, clears its old marks by
default, and writes editable AI marks into an independent output tree.

The input raw export is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from export_prediction_masks_to_db import iter_components, iter_mito_instances, split_mito_instances


AI_CLASS_INFO = {
    1: ("严重损伤线粒体", "#FF0000"),
    2: ("中度损伤线粒体", "#3DC005"),
    3: ("健康线粒体", "#01F8E7"),
    4: ("自噬线粒体", "#00FFB7"),
    5: ("肌浆网", "#E1FF00"),
}
AI_CLASS_IDS = tuple(AI_CLASS_INFO)
MARK_TABLES = ("Mark_label_None", "Mark_human")
RELATION_TABLES = ("MarkToTile_label_None", "MarkToTile_human")
OPTIONAL_CLEAR_TABLES = ("ChangeRecord", "Pdl1sCount", "ScreenShot", "markROITaskStatus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair raw TIF exports with model masks and create platform-ready AI slice.db files."
    )
    parser.add_argument(
        "--raw-slice-root",
        type=Path,
        required=True,
        help="Directory containing <timestamp>/<slice_id>/<tif + slice.db> folders.",
    )
    parser.add_argument("--mito-mask-dir", type=Path, required=True)
    parser.add_argument("--sr-mask-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-mito-area", type=int, default=30)
    parser.add_argument(
        "--mito-instance-mode",
        choices=("components", "watershed"),
        default="watershed",
        help=(
            "How editable mitochondria are formed. 'watershed' preserves the existing "
            "post-processed semantic mask but separates touching mitochondria before DB export."
        ),
    )
    parser.add_argument(
        "--mito-watershed-min-distance",
        type=int,
        default=12,
        help="Minimum distance in pixels between watershed seeds for mitochondrial instance export.",
    )
    parser.add_argument("--min-sr-area", type=int, default=20)
    parser.add_argument("--simplify-epsilon", type=float, default=1.5)
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Export only this prediction sample ID. Repeat the option for multiple smoke-test samples.",
    )
    parser.add_argument(
        "--keep-doctor-marks",
        action="store_true",
        help="Keep source doctor marks and add AI groups prefixed with 'AI_'. Default exports AI marks only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory. Existing per-sample folders are still replaced.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_tif(sample_dir: Path) -> Path | None:
    candidates = sorted(
        path for path in sample_dir.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    return candidates[0] if len(candidates) == 1 else None


def parse_prediction_id(sample_id: str) -> tuple[str, str] | None:
    parts = sample_id.split("__")
    if len(parts) != 3:
        return None
    return parts[1], parts[2]


def collect_prediction_ids(mito_mask_dir: Path, sr_mask_dir: Path) -> tuple[set[str], set[str], set[str]]:
    mito_ids = {path.stem for path in mito_mask_dir.glob("*.png")}
    sr_ids = {path.stem for path in sr_mask_dir.glob("*.png")}
    return mito_ids & sr_ids, mito_ids - sr_ids, sr_ids - mito_ids


def collect_raw_samples(raw_slice_root: Path) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    samples: dict[tuple[str, str], dict] = {}
    issues: list[dict] = []
    for timestamp_dir in sorted(path for path in raw_slice_root.iterdir() if path.is_dir()):
        for sample_dir in sorted(path for path in timestamp_dir.iterdir() if path.is_dir()):
            tif_path = find_tif(sample_dir)
            db_path = sample_dir / "slice.db"
            key = (timestamp_dir.name, sample_dir.name)
            if tif_path is None or not db_path.is_file():
                issues.append(
                    {
                        "timestamp_dir": timestamp_dir.name,
                        "sample_dir": sample_dir.name,
                        "reason": "missing_or_ambiguous_tif" if tif_path is None else "missing_slice_db",
                        "source_dir": str(sample_dir),
                    }
                )
                continue
            if key in samples:
                issues.append(
                    {
                        "timestamp_dir": timestamp_dir.name,
                        "sample_dir": sample_dir.name,
                        "reason": "duplicate_timestamp_sample_key",
                        "source_dir": str(sample_dir),
                    }
                )
                continue
            samples[key] = {"sample_dir": sample_dir, "tif_path": tif_path, "db_path": db_path}
    return samples, issues


def table_names(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def clear_or_prepare_db(db_path: Path, keep_doctor_marks: bool) -> dict[int, int]:
    connection = sqlite3.connect(db_path)
    try:
        tables = table_names(connection)
        required = {"MarkGroup", "Mark_label_None"}
        missing = required - tables
        if missing:
            raise RuntimeError(f"template DB misses required tables: {sorted(missing)}")

        if not keep_doctor_marks:
            for table_name in (*MARK_TABLES, *RELATION_TABLES, *OPTIONAL_CLEAR_TABLES):
                if table_name in tables:
                    connection.execute(f'DELETE FROM "{table_name}"')
            connection.execute('DELETE FROM "MarkGroup"')

        now = int(time.time())
        group_ids: dict[int, int] = {}
        for class_id, (class_name, color) in AI_CLASS_INFO.items():
            group_name = f"AI_{class_name}" if keep_doctor_marks else class_name
            cursor = connection.execute(
                """
                INSERT INTO MarkGroup (
                    groupName, shape, color, opTime, isTemplate, isSelected,
                    selectable, editable, isAi, parentId, templateId, defaultColor,
                    isEmpty, isShow, isImport, createTime, diagnosisType
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group_name,
                    "",
                    color,
                    float(now),
                    0,
                    0,
                    0,
                    0,
                    1,
                    None,
                    None,
                    color,
                    0,
                    1,
                    0,
                    now,
                    None,
                ),
            )
            group_ids[class_id] = int(cursor.lastrowid)
        connection.commit()
        return group_ids
    finally:
        connection.close()


def insert_mark(
    connection: sqlite3.Connection,
    mark_id: int,
    group_id: int,
    color: str,
    points: list[tuple[float, float]],
) -> None:
    available_columns = {row[1] for row in connection.execute("PRAGMA table_info(Mark_label_None)")}
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    values = {
        "id": int(mark_id),
        "position": json.dumps({"x": xs, "y": ys}, ensure_ascii=False),
        "method": "freepen",
        "isExport": 0,
        "remark": "AI prediction; please review and correct.",
        "aiResult": "{}",
        "editable": 1,
        "strokeColor": color,
        "fillColor": None,
        "markType": 1,
        "diagnosis": '{"operators": ["AI"]}',
        "radius": None,
        "createTime": float(int(time.time() * 1000)),
        "groupId": json.dumps([int(group_id)]),
        "areaId": None,
        "dashed": 0,
        "doctorDiagnosis": "null",
        "markId": None,
        "is_export_to_pis": 0,
        "aiStatus": 1,
        "regionType": "AI",
    }
    columns = [column for column in values if column in available_columns]
    placeholders = ", ".join("?" for _ in columns)
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    connection.execute(
        f'INSERT INTO "Mark_label_None" ({quoted_columns}) VALUES ({placeholders})',
        [values[column] for column in columns],
    )


def write_ai_marks(
    db_path: Path,
    mito_mask: np.ndarray,
    sr_mask: np.ndarray,
    group_ids: dict[int, int],
    args: argparse.Namespace,
) -> dict[int, int]:
    connection = sqlite3.connect(db_path)
    class_counts: Counter[int] = Counter()
    try:
        max_id = connection.execute('SELECT COALESCE(MAX(id), 0) FROM "Mark_label_None"').fetchone()[0]
        next_mark_id = int(max_id) + 1
        if args.mito_instance_mode == "watershed":
            for class_id, _, points in iter_mito_instances(
                mito_mask,
                min_area=args.min_mito_area,
                min_distance=args.mito_watershed_min_distance,
                simplify_epsilon=args.simplify_epsilon,
                max_points=args.max_points,
            ):
                _, color = AI_CLASS_INFO[class_id]
                insert_mark(connection, next_mark_id, group_ids[class_id], color, points)
                class_counts[class_id] += 1
                next_mark_id += 1
        else:
            for class_id in (1, 2, 3, 4):
                _, color = AI_CLASS_INFO[class_id]
                for _, points in iter_components(
                    mito_mask,
                    class_id,
                    min_area=args.min_mito_area,
                    simplify_epsilon=args.simplify_epsilon,
                    max_points=args.max_points,
                ):
                    insert_mark(connection, next_mark_id, group_ids[class_id], color, points)
                    class_counts[class_id] += 1
                    next_mark_id += 1

        _, sr_color = AI_CLASS_INFO[5]
        for _, points in iter_components(
            sr_mask,
            5,
            min_area=args.min_sr_area,
            simplify_epsilon=args.simplify_epsilon,
            max_points=args.max_points,
        ):
            insert_mark(connection, next_mark_id, group_ids[5], sr_color, points)
            class_counts[5] += 1
            next_mark_id += 1
        connection.commit()
        return {class_id: int(class_counts[class_id]) for class_id in AI_CLASS_IDS}
    finally:
        connection.close()


def write_readme(output_root: Path) -> None:
    (output_root / "README_AI预标注说明.txt").write_text(
        "本目录为 AI 预标注导出，适用于原始 TIF + slice.db 标注平台。\n\n"
        "每个切片目录保留原始 TIF 文件名，并含独立的 slice.db。\n"
        "slice.db 仅包含 AI 线粒体损伤分类和肌浆网预标注；未修改医生原始导出。\n"
        "类别：严重损伤线粒体、中度损伤线粒体、健康线粒体、自噬线粒体、肌浆网。\n"
        "请以 manifest.csv 核对导入数量，以 unmatched_*.csv 查看未导出的样本。\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    raw_root = args.raw_slice_root.resolve()
    output_root = args.output_root.resolve()
    mito_mask_dir = args.mito_mask_dir.resolve()
    sr_mask_dir = args.sr_mask_dir.resolve()

    for required in (raw_root, mito_mask_dir, sr_mask_dir):
        if not required.is_dir():
            raise FileNotFoundError(required)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    slice_output_root = output_root / "slice_files"
    slice_output_root.mkdir(parents=True, exist_ok=True)

    prediction_ids, mito_only_ids, sr_only_ids = collect_prediction_ids(mito_mask_dir, sr_mask_dir)
    raw_samples, raw_issues = collect_raw_samples(raw_root)
    prediction_by_raw_key: dict[tuple[str, str], str] = {}
    invalid_prediction_ids: list[dict] = []
    for sample_id in sorted(prediction_ids):
        raw_key = parse_prediction_id(sample_id)
        if raw_key is None:
            invalid_prediction_ids.append({"sample_id": sample_id, "reason": "unexpected_sample_id_format"})
            continue
        if raw_key in prediction_by_raw_key:
            invalid_prediction_ids.append({"sample_id": sample_id, "reason": "duplicate_prediction_raw_key"})
            continue
        prediction_by_raw_key[raw_key] = sample_id

    rows: list[dict] = []
    unmatched_predictions: list[dict] = list(invalid_prediction_ids)
    matched_keys = set(raw_samples) & set(prediction_by_raw_key)
    for raw_key, sample_id in sorted(prediction_by_raw_key.items()):
        if raw_key not in raw_samples:
            unmatched_predictions.append(
                {"sample_id": sample_id, "reason": "raw_tif_slice_db_pair_not_found", "raw_key": "/".join(raw_key)}
            )

    selected_keys = set(matched_keys)
    if args.sample_id:
        requested_ids = set(args.sample_id)
        selected_keys = {key for key in matched_keys if prediction_by_raw_key[key] in requested_ids}
        missing_requested = requested_ids - {prediction_by_raw_key[key] for key in selected_keys}
        for sample_id in sorted(missing_requested):
            unmatched_predictions.append({"sample_id": sample_id, "reason": "requested_sample_not_matchable"})

    for raw_key in sorted(selected_keys):
        sample_id = prediction_by_raw_key[raw_key]
        raw = raw_samples[raw_key]
        mito_path = mito_mask_dir / f"{sample_id}.png"
        sr_path = sr_mask_dir / f"{sample_id}.png"
        mito_mask = np.asarray(Image.open(mito_path), dtype=np.uint8)
        sr_mask = np.asarray(Image.open(sr_path), dtype=np.uint8)
        with Image.open(raw["tif_path"]) as image:
            raw_size = image.size
        expected_shape = (raw_size[1], raw_size[0])
        if mito_mask.shape != expected_shape or sr_mask.shape != expected_shape:
            rows.append(
                {
                    "sample_id": sample_id,
                    "status": "size_mismatch_skipped",
                    "timestamp_dir": raw_key[0],
                    "sample_dir": raw_key[1],
                    "raw_tif": str(raw["tif_path"]),
                    "raw_size": f"{raw_size[0]}x{raw_size[1]}",
                    "mito_mask_shape": "x".join(map(str, mito_mask.shape)),
                    "sr_mask_shape": "x".join(map(str, sr_mask.shape)),
                }
            )
            continue

        destination_dir = slice_output_root / raw_key[0] / raw_key[1]
        if destination_dir.exists():
            if not args.overwrite:
                raise RuntimeError(f"destination exists: {destination_dir}")
            shutil.rmtree(destination_dir)
        destination_dir.mkdir(parents=True)
        destination_tif = destination_dir / raw["tif_path"].name
        destination_db = destination_dir / "slice.db"
        shutil.copy2(raw["tif_path"], destination_tif)
        shutil.copy2(raw["db_path"], destination_db)
        group_ids = clear_or_prepare_db(destination_db, keep_doctor_marks=args.keep_doctor_marks)
        class_counts = write_ai_marks(destination_db, mito_mask, sr_mask, group_ids, args)
        rows.append(
            {
                "sample_id": sample_id,
                "status": "ok",
                "timestamp_dir": raw_key[0],
                "sample_dir": raw_key[1],
                "raw_tif": str(raw["tif_path"]),
                "output_tif": str(destination_tif.relative_to(output_root)),
                "output_db": str(destination_db.relative_to(output_root)),
                "raw_size": f"{raw_size[0]}x{raw_size[1]}",
                "total_ai_marks": sum(class_counts.values()),
                **{f"class_{class_id}_marks": class_counts[class_id] for class_id in AI_CLASS_IDS},
            }
        )
        if len(rows) % 25 == 0:
            print(f"[Export] {len(rows)}/{len(selected_keys)} processed")

    raw_unmatched_rows = [
        {
            "timestamp_dir": key[0],
            "sample_dir": key[1],
            "reason": "no_prediction_mask_pair",
            "raw_tif": str(raw_samples[key]["tif_path"]),
        }
        for key in sorted(set(raw_samples) - set(prediction_by_raw_key))
    ]
    raw_unmatched_rows.extend(raw_issues)
    for sample_id in sorted(mito_only_ids):
        unmatched_predictions.append({"sample_id": sample_id, "reason": "missing_mito_sr_mask"})
    for sample_id in sorted(sr_only_ids):
        unmatched_predictions.append({"sample_id": sample_id, "reason": "missing_mito_mask"})

    fieldnames = [
        "sample_id",
        "status",
        "timestamp_dir",
        "sample_dir",
        "raw_tif",
        "output_tif",
        "output_db",
        "raw_size",
        "mito_mask_shape",
        "sr_mask_shape",
        "total_ai_marks",
        *(f"class_{class_id}_marks" for class_id in AI_CLASS_IDS),
    ]
    write_csv(output_root / "manifest.csv", rows, fieldnames)
    write_csv(
        output_root / "unmatched_raw.csv",
        raw_unmatched_rows,
        ["timestamp_dir", "sample_dir", "reason", "raw_tif", "source_dir"],
    )
    write_csv(
        output_root / "unmatched_predictions.csv",
        unmatched_predictions,
        ["sample_id", "reason", "raw_key"],
    )
    summary = {
        "input_raw_slice_root": str(raw_root),
        "input_mito_mask_dir": str(mito_mask_dir),
        "input_sr_mask_dir": str(sr_mask_dir),
        "output_root": str(output_root),
        "keep_doctor_marks": bool(args.keep_doctor_marks),
        "mito_instance_mode": args.mito_instance_mode,
        "mito_watershed_min_distance": int(args.mito_watershed_min_distance),
        "prediction_pairs": len(prediction_ids),
        "raw_tif_db_pairs": len(raw_samples),
        "matched_pairs": len(matched_keys),
        "selected_pairs": len(selected_keys),
        "exported_ok": sum(row["status"] == "ok" for row in rows),
        "skipped_size_mismatch": sum(row["status"] == "size_mismatch_skipped" for row in rows),
        "unmatched_raw": len(raw_unmatched_rows),
        "unmatched_predictions": len(unmatched_predictions),
        "total_ai_marks": sum(int(row.get("total_ai_marks", 0) or 0) for row in rows),
        "classes": {str(key): value[0] for key, value in AI_CLASS_INFO.items()},
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
