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


def merge_rows_with_sample_index(task_rows, sample_index):
    merged_rows = []
    for row in task_rows:
        sample_row = sample_index.get(row["canonical_sample_uid"])
        if sample_row is None:
            continue
        merged = dict(row)
        merged["split"] = sample_row["split"]
        merged_rows.append(merged)
    return merged_rows


def build_default_task_manifests(task_rows, sample_index, output_dir):
    split_rows = defaultdict(list)
    for row in merge_rows_with_sample_index(task_rows, sample_index):
        split_rows[row["split"]].append(row)

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
    for row in merge_rows_with_sample_index(task_rows, sample_index):
        if row["split"] not in {"train", "val"}:
            continue
        if row["usable_train"] != "1":
            continue
        eligible_rows.append(row)

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


def _group_sizes(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    return groups


def _choose_train_groups_exact(groups, target_count):
    ordered = sorted(
        ((group_id, len(rows)) for group_id, rows in groups.items()),
        key=lambda item: (item[0]),
    )
    solutions = {0: []}
    total_groups = len(groups)
    total_samples = sum(len(rows) for rows in groups.values())

    def ideal_group_count(sample_count):
        if total_samples <= 0:
            return 0
        return round(total_groups * float(sample_count) / float(total_samples))

    def is_better_solution(candidate, current, sample_count):
        if current is None:
            return True
        candidate_gap = abs(len(candidate) - ideal_group_count(sample_count))
        current_gap = abs(len(current) - ideal_group_count(sample_count))
        if candidate_gap != current_gap:
            return candidate_gap < current_gap
        if len(candidate) != len(current):
            return len(candidate) < len(current)
        return tuple(candidate) < tuple(current)

    for group_id, sample_count in ordered:
        updates = {}
        for total, chosen_groups in list(solutions.items()):
            next_total = total + sample_count
            if next_total > target_count:
                continue
            candidate = chosen_groups + [group_id]
            current = solutions.get(next_total)
            staged = updates.get(next_total)
            best_existing = current
            if staged is not None and is_better_solution(staged, best_existing, next_total):
                best_existing = staged
            if is_better_solution(candidate, best_existing, next_total):
                updates[next_total] = candidate
        solutions.update(updates)

    if target_count in solutions:
        return solutions[target_count], True, target_count

    best_total = max(solutions)
    return solutions[best_total], False, best_total


def _sort_group_assignment_rows(rows):
    def sort_key(row):
        split_rank = {"train": 0, "test": 1, "excluded": 2}
        return (split_rank.get(row["assigned_split"], 9), row["group_id"])

    return sorted(rows, key=sort_key)


def build_final_task_manifest(task_rows, sample_index, output_dir, target_train_count):
    merged_rows = merge_rows_with_sample_index(task_rows, sample_index)
    eligible_train_rows = [row for row in merged_rows if row["usable_train"] == "1"]
    eligible_eval_rows = [row for row in merged_rows if row["usable_eval"] == "1"]
    manual_rows = [row for row in merged_rows if row["needs_manual_review"] == "1"]

    train_groups = _group_sizes(eligible_train_rows)
    chosen_train_groups, exact_hit, achieved_train_count = _choose_train_groups_exact(train_groups, target_train_count)
    chosen_train_groups = set(chosen_train_groups)

    train_rows = [row for row in eligible_train_rows if row["group_id"] in chosen_train_groups]
    test_rows = [row for row in eligible_eval_rows if row["group_id"] not in chosen_train_groups]
    excluded_rows = [
        row
        for row in merged_rows
        if row["group_id"] not in chosen_train_groups and row["usable_eval"] != "1"
    ]

    train_ids = sort_sample_ids(train_rows)
    test_ids = sort_sample_ids(test_rows)
    excluded_ids = sort_sample_ids(excluded_rows)

    write_manifest(output_dir / "train.txt", train_ids)
    write_manifest(output_dir / "val.txt", [])
    write_manifest(output_dir / "test.txt", test_ids)
    write_manifest(output_dir / "train_positive_only.txt", sort_sample_ids([row for row in train_rows if row["usability_status"] == "supervised_positive"]))
    write_manifest(output_dir / "train_partial_labeled.txt", sort_sample_ids([row for row in train_rows if row["usability_status"] == "partial_labeled"]))
    write_manifest(output_dir / "test_positive_only.txt", sort_sample_ids([row for row in test_rows if row["usability_status"] == "supervised_positive"]))
    write_manifest(output_dir / "test_partial_labeled.txt", sort_sample_ids([row for row in test_rows if row["usability_status"] == "partial_labeled"]))
    write_manifest(output_dir / "test_excluded_not_eval_safe.txt", excluded_ids)
    write_manifest(output_dir / "manual_review.txt", sort_sample_ids(manual_rows))

    group_assignment_rows = []
    for group_id, rows in sorted(train_groups.items(), key=lambda item: item[0]):
        assigned_split = "train" if group_id in chosen_train_groups else "test"
        eval_safe_count = len([row for row in rows if row["usable_eval"] == "1"])
        group_assignment_rows.append(
            {
                "group_id": group_id,
                "assigned_split": assigned_split,
                "usable_train_count": len(rows),
                "usable_eval_count": eval_safe_count,
                "sample_ids": "|".join(sort_sample_ids(rows)),
            }
        )

    excluded_group_rows = defaultdict(list)
    for row in excluded_rows:
        excluded_group_rows[row["group_id"]].append(row)
    for group_id, rows in excluded_group_rows.items():
        if group_id in train_groups:
            continue
        group_assignment_rows.append(
            {
                "group_id": group_id,
                "assigned_split": "excluded",
                "usable_train_count": len([row for row in rows if row["usable_train"] == "1"]),
                "usable_eval_count": len([row for row in rows if row["usable_eval"] == "1"]),
                "sample_ids": "|".join(sort_sample_ids(rows)),
            }
        )

    write_csv(
        output_dir / "group_assignment.csv",
        _sort_group_assignment_rows(group_assignment_rows),
        ["group_id", "assigned_split", "usable_train_count", "usable_eval_count", "sample_ids"],
    )

    summary = {
        "task_name": task_rows[0]["task_name"] if task_rows else "",
        "split_name": output_dir.name,
        "target_train_count": int(target_train_count),
        "exact_target_achieved": bool(exact_hit),
        "train_count": len(train_ids),
        "val_count": 0,
        "test_count": len(test_ids),
        "test_excluded_not_eval_safe_count": len(excluded_ids),
        "eligible_train_total": len(sort_sample_ids(eligible_train_rows)),
        "eligible_eval_total": len(sort_sample_ids(eligible_eval_rows)),
        "train_group_count": len(chosen_train_groups),
        "test_group_count": len(sorted({row["group_id"] for row in test_rows})),
        "train_group_ids": sorted(chosen_train_groups),
        "test_group_ids": sorted({row["group_id"] for row in test_rows}),
        "manual_review_total": len(sort_sample_ids(manual_rows)),
        "achieved_train_count": int(achieved_train_count),
    }
    (output_dir / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Build jijie task-specific manifests from prepared metadata and audit outputs.")
    parser.add_argument("--prepared-root", type=str, default=str(DEFAULT_PREPARED_ROOT))
    parser.add_argument("--task-registry", type=str, default=str(DEFAULT_TASK_REGISTRY))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--num-folds", type=int, default=5, help="number of group-wise folds to generate for train/val pools")
    parser.add_argument("--final-train-count", type=int, default=200, help="target number of train samples for the final group-wise holdout split")
    parser.add_argument("--final-manifest-name", type=str, default="final_200", help="directory name for the final holdout split")
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
        final_dir = output_root / task_name / args.final_manifest_name
        final_summary = build_final_task_manifest(rows, sample_index, final_dir, args.final_train_count)
        fold_summary = []
        if args.num_folds > 1:
            fold_summary = build_group_folds(rows, sample_index, output_root / task_name, args.num_folds)
        overall_summary[task_name] = {
            "default": default_summary,
            args.final_manifest_name: final_summary,
            "folds": fold_summary,
        }

    (output_root / "task_manifest_summary.json").write_text(
        json.dumps(overall_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(overall_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
