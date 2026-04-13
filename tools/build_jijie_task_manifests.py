import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED_ROOT = PROJECT_ROOT / "data" / "jijie"
DEFAULT_TASK_REGISTRY = PROJECT_ROOT / "data" / "jijie_audit" / "task_registry.csv"
DEFAULT_OUTPUT_ROOT = DEFAULT_PREPARED_ROOT / "ImageSets" / "Segmentation" / "tasks"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path, sample_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for sample_id in sample_ids:
            handle.write(sample_id + "\n")


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sort_sample_ids(rows):
    return sorted({row["canonical_sample_uid"] for row in rows})


def build_sample_index(prepared_root):
    samples_csv = prepared_root / "metadata" / "samples.csv"
    sample_rows = read_csv(samples_csv)
    return {row["prepared_sample_id"]: row for row in sample_rows}


def build_default_task_manifests(task_rows, sample_index, output_dir):
    split_rows = defaultdict(list)
    for row in task_rows:
        sample_row = sample_index.get(row["canonical_sample_uid"])
        if sample_row is None:
            continue
        row = dict(row)
        row["split"] = sample_row["split"]
        split_rows[sample_row["split"]].append(row)

    train_rows = [row for row in split_rows["train"] if row["usable_train"] == "1"]
    val_rows = [row for row in split_rows["val"] if row["usable_eval"] == "1"]
    test_rows = [row for row in split_rows["test"] if row["usable_eval"] == "1"]

    positive_rows = [row for row in task_rows if row["usability_status"] == "supervised_positive"]
    partial_rows = [row for row in task_rows if row["usability_status"] == "partial_labeled"]
    manual_rows = [row for row in task_rows if row["needs_manual_review"] == "1"]

    write_manifest(output_dir / "train.txt", sort_sample_ids(train_rows))
    write_manifest(output_dir / "val.txt", sort_sample_ids(val_rows))
    write_manifest(output_dir / "test.txt", sort_sample_ids(test_rows))
    write_manifest(output_dir / "train_positive_only.txt", sort_sample_ids([row for row in train_rows if row["usability_status"] == "supervised_positive"]))
    write_manifest(output_dir / "train_partial_labeled.txt", sort_sample_ids([row for row in train_rows if row["usability_status"] == "partial_labeled"]))
    write_manifest(output_dir / "manual_review.txt", sort_sample_ids(manual_rows))
    write_manifest(output_dir / "all_supervised_positive.txt", sort_sample_ids(positive_rows))
    write_manifest(output_dir / "all_partial_labeled.txt", sort_sample_ids(partial_rows))

    summary = {
        "task_name": task_rows[0]["task_name"] if task_rows else "",
        "train_count": len(sort_sample_ids(train_rows)),
        "val_count": len(sort_sample_ids(val_rows)),
        "test_count": len(sort_sample_ids(test_rows)),
        "positive_total": len(sort_sample_ids(positive_rows)),
        "partial_total": len(sort_sample_ids(partial_rows)),
        "manual_review_total": len(sort_sample_ids(manual_rows)),
    }
    (output_dir / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_group_folds(task_rows, sample_index, output_dir, num_folds):
    eligible_rows = []
    for row in task_rows:
        sample_row = sample_index.get(row["canonical_sample_uid"])
        if sample_row is None:
            continue
        split = sample_row["split"]
        if split not in {"train", "val"}:
            continue
        if row["usable_train"] != "1":
            continue
        merged = dict(row)
        merged["split"] = split
        eligible_rows.append(merged)

    groups = defaultdict(list)
    for row in eligible_rows:
        groups[row["group_id"]].append(row)

    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    fold_groups = [[] for _ in range(num_folds)]
    fold_sizes = [0 for _ in range(num_folds)]

    for group_id, rows in ordered_groups:
        target_fold = min(range(num_folds), key=lambda idx: (fold_sizes[idx], idx))
        fold_groups[target_fold].append(group_id)
        fold_sizes[target_fold] += len(rows)

    fold_summaries = []
    for fold_index in range(num_folds):
        val_group_ids = set(fold_groups[fold_index])
        fold_dir = output_dir / "folds" / "fold_{:02d}".format(fold_index)
        train_ids = sort_sample_ids([row for row in eligible_rows if row["group_id"] not in val_group_ids])
        val_ids = sort_sample_ids([row for row in eligible_rows if row["group_id"] in val_group_ids and row["usable_eval"] == "1"])

        write_manifest(fold_dir / "train.txt", train_ids)
        write_manifest(fold_dir / "val.txt", val_ids)

        summary = {
            "fold_index": fold_index,
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "val_group_ids": sorted(val_group_ids),
        }
        (fold_dir / "fold_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        fold_summaries.append(summary)

    return fold_summaries


def main():
    parser = argparse.ArgumentParser(description="Build jijie task-specific manifests from prepared metadata and audit outputs.")
    parser.add_argument("--prepared-root", type=str, default=str(DEFAULT_PREPARED_ROOT))
    parser.add_argument("--task-registry", type=str, default=str(DEFAULT_TASK_REGISTRY))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--num-folds", type=int, default=5, help="number of group-wise folds to generate for train/val pools")
    args = parser.parse_args()

    prepared_root = Path(args.prepared_root)
    task_registry_path = Path(args.task_registry)
    output_root = Path(args.output_root)

    sample_index = build_sample_index(prepared_root)
    task_rows = [
        row
        for row in read_csv(task_registry_path)
        if row["is_canonical"] == "1"
    ]

    task_buckets = defaultdict(list)
    for row in task_rows:
        task_buckets[row["task_name"]].append(row)

    overall_summary = {}
    for task_name, rows in sorted(task_buckets.items()):
        default_dir = output_root / task_name / "default"
        default_summary = build_default_task_manifests(rows, sample_index, default_dir)
        fold_summary = []
        if args.num_folds > 1:
            fold_summary = build_group_folds(rows, sample_index, output_root / task_name, args.num_folds)
        overall_summary[task_name] = {
            "default": default_summary,
            "folds": fold_summary,
        }

    (output_root / "task_manifest_summary.json").write_text(
        json.dumps(overall_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(overall_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
