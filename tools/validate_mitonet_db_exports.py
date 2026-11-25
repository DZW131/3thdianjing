"""Validate editable MitoNet DB copies against their source manifest and counts."""
import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_variant(root, mode, samples):
    variant = root / mode
    manifest = rows(variant / "manifest.csv")
    if len(manifest) != len(samples):
        raise ValueError(f"{mode}: manifest count mismatch")
    by_id = {row["sample_id"]: row for row in manifest}
    bad = []
    total_marks = 0
    for sample in samples:
        sample_id = sample["sample_id"]
        record = by_id[sample_id]
        target = variant / Path(record["output_db"])
        if not target.is_file():
            bad.append((sample_id, "missing_db"))
            continue
        connection = sqlite3.connect(target)
        try:
            count = int(connection.execute('SELECT COUNT(*) FROM "Mark_label_None"').fetchone()[0])
            editable, noneditable = connection.execute(
                'SELECT SUM(CASE WHEN editable=1 THEN 1 ELSE 0 END), '
                'SUM(CASE WHEN editable!=1 OR editable IS NULL THEN 1 ELSE 0 END) '
                'FROM "Mark_label_None"').fetchone()
            if count != int(record["added_marks"]):
                bad.append((sample_id, f"mark_count:{count}!={record['added_marks']}"))
            if int(editable or 0) != count or int(noneditable or 0) != 0:
                bad.append((sample_id, "marks_not_all_editable"))
            for position, in connection.execute('SELECT position FROM "Mark_label_None"'):
                shape = json.loads(position)
                xs, ys = shape.get("x", []), shape.get("y", [])
                if len(xs) < 4 or len(xs) != len(ys):
                    bad.append((sample_id, "invalid_polygon"))
                    break
                if (min(xs) < 0 or min(ys) < 0 or max(xs) > int(sample["width"])
                        or max(ys) > int(sample["height"])):
                    bad.append((sample_id, "polygon_outside_image"))
                    break
        finally:
            connection.close()
        total_marks += int(record["added_marks"])
    return dict(mode=mode, samples=len(manifest), db_files=len(list((variant / "slice_files").rglob("slice.db"))),
                total_marks=total_marks, failures=bad)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = rows(args.samples)
    source_failures = []
    for sample in samples:
        source = args.raw_root / Path(sample["source_db"].replace("\\", "/"))
        if sha256(source) != sample["db_sha256"]:
            source_failures.append(sample["sample_id"])
    result = dict(source_db_hashes_verified=len(samples) - len(source_failures),
                  source_db_hash_failures=source_failures,
                  variants=[validate_variant(args.exports, mode, samples)
                            for mode in ("instances", "classified")])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if source_failures or any(variant["failures"] or variant["db_files"] != len(samples)
                              for variant in result["variants"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
