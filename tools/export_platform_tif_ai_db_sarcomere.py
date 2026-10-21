"""Export baseline sarcomere predictions as editable platform TIF + slice.db pairs.

This is intentionally independent from the mitochondrial export pipeline.  It
preserves original TIF files and writes four editable groups: Z-line, T-tubule,
M-line, and side tubule.
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

from export_prediction_masks_to_db import iter_components


CLASS_INFO = {
    10: ("Z线", "#006D77"),
    11: ("T管", "#F59E0B"),
    12: ("M线", "#3A86FF"),
    13: ("心肌侧管", "#6B4C9A"),
}
CLASS_IDS = tuple(CLASS_INFO)
MARK_TABLES = ("Mark_label_None", "Mark_human")
RELATION_TABLES = ("MarkToTile_label_None", "MarkToTile_human")
OPTIONAL_CLEAR_TABLES = ("ChangeRecord", "Pdl1sCount", "ScreenShot", "markROITaskStatus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create platform-ready sarcomere TIF + slice.db folders.")
    parser.add_argument("--raw-slice-root", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-structure-area", type=int, default=10)
    parser.add_argument("--simplify-epsilon", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=384)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--keep-doctor-marks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_tif(sample_dir: Path) -> Path | None:
    matches = sorted(
        path for path in sample_dir.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    return matches[0] if len(matches) == 1 else None


def prediction_key(sample_id: str) -> tuple[str, str] | None:
    parts = sample_id.split("__")
    return (parts[1], parts[2]) if len(parts) == 3 else None


def collect_raw_pairs(raw_root: Path) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    pairs: dict[tuple[str, str], dict] = {}
    issues: list[dict] = []
    for timestamp_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
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
                    }
                )
                continue
            pairs[key] = {"tif_path": tif_path, "db_path": db_path}
    return pairs, issues


def prepare_database(db_path: Path, keep_doctor_marks: bool) -> dict[int, int]:
    connection = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"MarkGroup", "Mark_label_None"}
        if required - tables:
            raise RuntimeError(f"template DB misses required tables: {sorted(required - tables)}")
        if not keep_doctor_marks:
            for table_name in (*MARK_TABLES, *RELATION_TABLES, *OPTIONAL_CLEAR_TABLES):
                if table_name in tables:
                    connection.execute(f'DELETE FROM "{table_name}"')
            connection.execute('DELETE FROM "MarkGroup"')

        group_ids: dict[int, int] = {}
        now = int(time.time())
        for class_id, (class_name, color) in CLASS_INFO.items():
            cursor = connection.execute(
                """
                INSERT INTO MarkGroup (
                    groupName, shape, color, opTime, isTemplate, isSelected,
                    selectable, editable, isAi, parentId, templateId, defaultColor,
                    isEmpty, isShow, isImport, createTime, diagnosisType
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"AI_{class_name}" if keep_doctor_marks else class_name,
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
    columns_available = {row[1] for row in connection.execute("PRAGMA table_info(Mark_label_None)")}
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
    columns = [column for column in values if column in columns_available]
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    connection.execute(
        f'INSERT INTO "Mark_label_None" ({quoted_columns}) VALUES ({placeholders})',
        [values[column] for column in columns],
    )


def write_marks(db_path: Path, mask: np.ndarray, group_ids: dict[int, int], args: argparse.Namespace) -> dict[int, int]:
    connection = sqlite3.connect(db_path)
    counts: Counter[int] = Counter()
    try:
        next_mark_id = int(connection.execute('SELECT COALESCE(MAX(id), 0) FROM "Mark_label_None"').fetchone()[0]) + 1
        for class_id, (_, color) in CLASS_INFO.items():
            for _, points in iter_components(
                mask,
                class_id,
                min_area=args.min_structure_area,
                simplify_epsilon=args.simplify_epsilon,
                max_points=args.max_points,
            ):
                insert_mark(connection, next_mark_id, group_ids[class_id], color, points)
                counts[class_id] += 1
                next_mark_id += 1
        connection.commit()
        return {class_id: int(counts[class_id]) for class_id in CLASS_IDS}
    finally:
        connection.close()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_slice_root.resolve()
    mask_dir = args.mask_dir.resolve()
    output_root = args.output_root.resolve()
    if not raw_root.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("raw slice root and mask directory must both exist")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    slice_root = output_root / "slice_files"
    slice_root.mkdir(parents=True, exist_ok=True)

    raw_pairs, raw_issues = collect_raw_pairs(raw_root)
    predictions: dict[tuple[str, str], str] = {}
    unmatched_predictions: list[dict] = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        key = prediction_key(mask_path.stem)
        if key is None:
            unmatched_predictions.append({"sample_id": mask_path.stem, "reason": "unexpected_sample_id_format"})
        elif key in predictions:
            unmatched_predictions.append({"sample_id": mask_path.stem, "reason": "duplicate_prediction_raw_key"})
        else:
            predictions[key] = mask_path.stem

    candidate_keys = set(raw_pairs) & set(predictions)
    if args.sample_id:
        selected_ids = set(args.sample_id)
        selected_keys = {key for key in candidate_keys if predictions[key] in selected_ids}
        for sample_id in sorted(selected_ids - {predictions[key] for key in selected_keys}):
            unmatched_predictions.append({"sample_id": sample_id, "reason": "requested_sample_not_matchable"})
    else:
        selected_keys = candidate_keys

    rows: list[dict] = []
    for index, key in enumerate(sorted(selected_keys), start=1):
        sample_id = predictions[key]
        raw = raw_pairs[key]
        mask = np.asarray(Image.open(mask_dir / f"{sample_id}.png"), dtype=np.uint8)
        with Image.open(raw["tif_path"]) as image:
            expected_shape = (image.height, image.width)
        if mask.shape != expected_shape:
            rows.append(
                {
                    "sample_id": sample_id,
                    "status": "size_mismatch_skipped",
                    "timestamp_dir": key[0],
                    "sample_dir": key[1],
                    "mask_shape": "x".join(map(str, mask.shape)),
                    "raw_size": f"{expected_shape[1]}x{expected_shape[0]}",
                }
            )
            continue

        destination = slice_root / key[0] / key[1]
        if destination.exists():
            if not args.overwrite:
                raise RuntimeError(f"destination exists: {destination}")
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        tif_out = destination / raw["tif_path"].name
        db_out = destination / "slice.db"
        shutil.copy2(raw["tif_path"], tif_out)
        shutil.copy2(raw["db_path"], db_out)
        group_ids = prepare_database(db_out, args.keep_doctor_marks)
        counts = write_marks(db_out, mask, group_ids, args)
        rows.append(
            {
                "sample_id": sample_id,
                "status": "ok",
                "timestamp_dir": key[0],
                "sample_dir": key[1],
                "output_tif": str(tif_out.relative_to(output_root)),
                "output_db": str(db_out.relative_to(output_root)),
                "raw_size": f"{expected_shape[1]}x{expected_shape[0]}",
                "total_ai_marks": sum(counts.values()),
                **{f"class_{class_id}_marks": counts[class_id] for class_id in CLASS_IDS},
            }
        )
        if index % 25 == 0:
            print(f"[Export] {index}/{len(selected_keys)} processed")

    for key, sample_id in sorted(predictions.items()):
        if key not in raw_pairs:
            unmatched_predictions.append({"sample_id": sample_id, "reason": "raw_tif_slice_db_pair_not_found"})
    unmatched_raw = [
        {"timestamp_dir": key[0], "sample_dir": key[1], "reason": "no_prediction_mask"}
        for key in sorted(set(raw_pairs) - set(predictions))
    ]
    unmatched_raw.extend(raw_issues)

    fields = [
        "sample_id", "status", "timestamp_dir", "sample_dir", "output_tif", "output_db", "raw_size", "mask_shape",
        "total_ai_marks", *(f"class_{class_id}_marks" for class_id in CLASS_IDS),
    ]
    write_csv(output_root / "manifest.csv", rows, fields)
    write_csv(output_root / "unmatched_raw.csv", unmatched_raw, ["timestamp_dir", "sample_dir", "reason"])
    write_csv(output_root / "unmatched_predictions.csv", unmatched_predictions, ["sample_id", "reason"])
    summary = {
        "model_task": "sarcomere_baseline",
        "raw_tif_db_pairs": len(raw_pairs),
        "prediction_masks": len(predictions),
        "matched_pairs": len(candidate_keys),
        "selected_pairs": len(selected_keys),
        "exported_ok": sum(row["status"] == "ok" for row in rows),
        "skipped_size_mismatch": sum(row["status"] == "size_mismatch_skipped" for row in rows),
        "total_ai_marks": sum(int(row.get("total_ai_marks", 0) or 0) for row in rows),
        "classes": {str(class_id): name for class_id, (name, _) in CLASS_INFO.items()},
        "min_structure_area_px": int(args.min_structure_area),
        "simplify_epsilon_px": float(args.simplify_epsilon),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "README_AI预标注说明.txt").write_text(
        "本目录为肌节结构 AI 预标注导出，适用于原始 TIF + slice.db 标注平台。\n\n"
        "每个切片目录包含未修改的原始 TIF 与独立 slice.db。\n"
        "类别：Z线、T管、M线、心肌侧管。所有 AI 对象均可逐个编辑。\n"
        "此版本使用 baseline final200 肌节模型，不包含结构先验或标签膨胀实验。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
