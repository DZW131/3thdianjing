"""Validate paired TIF + slice.db platform exports before delivery."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path


def class_totals(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        f"class_{class_id}_marks": sum(int(row.get(f"class_{class_id}_marks") or 0) for row in rows)
        for class_id in range(1, 6)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a platform TIF + slice.db export.")
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--compare-manifest", type=Path)
    args = parser.parse_args()

    export_root = args.export_root.resolve()
    manifest_rows = list(csv.DictReader((export_root / "manifest.csv").open(encoding="utf-8-sig")))
    db_paths = sorted((export_root / "slice_files").rglob("slice.db"))
    tif_paths = sorted(
        path
        for path in (export_root / "slice_files").rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    db_total = 0
    missing_tables: list[str] = []
    for db_path in db_paths:
        connection = sqlite3.connect(db_path)
        try:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "Mark_label_None" not in tables:
                missing_tables.append(str(db_path))
                continue
            db_total += int(connection.execute('SELECT COUNT(*) FROM "Mark_label_None"').fetchone()[0])
        finally:
            connection.close()

    manifest_total = sum(int(row.get("total_ai_marks") or 0) for row in manifest_rows)
    report = {
        "manifest_samples": len(manifest_rows),
        "tif_files": len(tif_paths),
        "slice_db_files": len(db_paths),
        "manifest_total_ai_marks": manifest_total,
        "sqlite_total_ai_marks": db_total,
        "mark_counts_match": manifest_total == db_total,
        "missing_mark_table_databases": len(missing_tables),
        "class_totals": class_totals(manifest_rows),
    }
    if args.compare_manifest:
        old_rows = list(csv.DictReader(args.compare_manifest.open(encoding="utf-8-sig")))
        old_totals = class_totals(old_rows)
        report["previous_class_totals"] = old_totals
        report["class_total_delta"] = {
            key: report["class_totals"][key] - old_totals[key] for key in report["class_totals"]
        }
    print(report)
    if (
        report["manifest_samples"] != report["tif_files"]
        or report["manifest_samples"] != report["slice_db_files"]
        or not report["mark_counts_match"]
        or missing_tables
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
