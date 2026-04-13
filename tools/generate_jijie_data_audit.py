import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "data" / "jijie" / "metadata"

SAMPLES_CSV = METADATA_DIR / "samples.csv"
DUPLICATES_CSV = METADATA_DIR / "duplicates_removed.csv"
INVALID_CSV = METADATA_DIR / "excluded_invalid_samples.csv"

AUDIT_CSV = METADATA_DIR / "data_usability_audit.csv"
GROUP_AUDIT_CSV = METADATA_DIR / "data_usability_audit_by_group.csv"
SUMMARY_JSON = METADATA_DIR / "data_usability_audit_summary.json"
SUMMARY_MD = METADATA_DIR / "data_usability_audit.md"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def split_classes(raw_value):
    return [item for item in (raw_value or "").split("|") if item]


def join_status_counts(counter):
    return "; ".join(f"{status}={counter[status]}" for status in sorted(counter))


def build_group_stats(sample_rows):
    group_status_counts = defaultdict(Counter)
    group_class_counts = defaultdict(Counter)

    for row in sample_rows:
        group_id = row["group_id"]
        group_status_counts[group_id][row["status"]] += 1
        for class_name in split_classes(row.get("classes_present", "")):
            group_class_counts[group_id][class_name] += 1

    return group_status_counts, group_class_counts


def classify_canonical_row(row, group_status_counts):
    status = row["status"]
    group_id = row["group_id"]
    group_profile = group_status_counts[group_id]
    duplicate_count = safe_int(row.get("duplicate_count"), 1)

    review_flags = []
    review_priority = "normal"

    if duplicate_count > 1:
        review_flags.append("has_deduplicated_siblings")

    if group_profile["template_only_no_mark_tables"] > 0:
        review_flags.append("group_contains_template_only_samples")

    if group_profile["annotated_nonzero"] > 0 and group_profile["groups_but_zero_marks"] > 0:
        review_flags.append("group_mixes_positive_and_zero_mark_samples")

    if status == "annotated_nonzero":
        audit_bucket = "positive_supervision"
        recommended_usage = "Use for positive segmentation supervision and task-specific quantification."
        immediate_action = "keep_in_default_training"
        reason_summary = "Non-zero marks were found in the source DB and converted into a usable mask."
        if "group_contains_template_only_samples" in review_flags:
            review_priority = "medium"
    elif status == "groups_but_zero_marks":
        audit_bucket = "negative_candidate"
        recommended_usage = "Use as a negative-control or false-positive-control candidate; do not treat as a positive mask."
        immediate_action = "keep_in_all_usable_only"
        reason_summary = "The sample had grouping metadata but no non-zero mark rows in the parsed source tables."
        if group_profile["annotated_nonzero"] > 0:
            review_priority = "medium"
    elif status == "template_only_no_mark_tables":
        audit_bucket = "manual_review"
        recommended_usage = "Do not use for supervised training until the raw export structure is rechecked."
        immediate_action = "review_raw_export"
        reason_summary = "No supported mark tables were found in the source DB snapshot."
        review_priority = "high"
    else:
        audit_bucket = "unknown"
        recommended_usage = "Review manually."
        immediate_action = "review_manually"
        reason_summary = f"Unhandled sample status: {status}"
        review_priority = "high"

    return {
        "record_origin": "canonical_sample",
        "sample_id": row.get("prepared_sample_id") or f'{row["group_id"]}__{row["timestamp_dir"]}__{row["sample_dir"]}',
        "group_id": group_id,
        "split": row.get("split", ""),
        "source_status": status,
        "audit_bucket": audit_bucket,
        "recommended_usage": recommended_usage,
        "immediate_action": immediate_action,
        "review_priority": review_priority,
        "reason_summary": reason_summary,
        "mark_count": safe_int(row.get("mark_count")),
        "duplicate_count": duplicate_count,
        "group_status_profile": join_status_counts(group_profile),
        "classes_present": row.get("classes_present", ""),
        "export_batch": row.get("export_batch", ""),
        "timestamp_dir": row.get("timestamp_dir", ""),
        "sample_dir": row.get("sample_dir", ""),
        "tif_name": row.get("tif_name", ""),
        "image_hash": row.get("image_hash", ""),
        "review_flags": "; ".join(review_flags),
    }


def classify_duplicate_row(row):
    return {
        "record_origin": "dropped_duplicate",
        "sample_id": row.get("dropped_sample", ""),
        "group_id": row.get("dropped_sample", "").split("__")[0],
        "split": "",
        "source_status": row.get("dropped_status", ""),
        "audit_bucket": "dropped_duplicate",
        "recommended_usage": "Do not use this dropped copy; keep only the canonical image hash representative.",
        "immediate_action": "exclude_keep_canonical",
        "review_priority": "info",
        "reason_summary": (
            "This sample shared an identical image hash with another sample and was removed by deduplication."
        ),
        "mark_count": safe_int(row.get("dropped_mark_count")),
        "duplicate_count": 0,
        "group_status_profile": "",
        "classes_present": "",
        "export_batch": "",
        "timestamp_dir": "",
        "sample_dir": "",
        "tif_name": "",
        "image_hash": row.get("image_hash", ""),
        "review_flags": f'kept_sample={row.get("kept_sample", "")}; kept_status={row.get("kept_status", "")}',
    }


def classify_invalid_row(row):
    sample_id = f'{row.get("timestamp_dir", "")}__{row.get("sample_dir", "")}'
    return {
        "record_origin": "invalid_source",
        "sample_id": sample_id,
        "group_id": "",
        "split": "",
        "source_status": row.get("status", ""),
        "audit_bucket": "invalid_source",
        "recommended_usage": "Recover the missing source file pair if possible; otherwise exclude.",
        "immediate_action": "recover_or_exclude",
        "review_priority": "critical",
        "reason_summary": row.get("reason", ""),
        "mark_count": 0,
        "duplicate_count": 0,
        "group_status_profile": "",
        "classes_present": "",
        "export_batch": row.get("export_batch", ""),
        "timestamp_dir": row.get("timestamp_dir", ""),
        "sample_dir": row.get("sample_dir", ""),
        "tif_name": row.get("tif_name", ""),
        "image_hash": "",
        "review_flags": row.get("db_path", ""),
    }


def build_group_audit_rows(sample_rows, group_status_counts, group_class_counts):
    rows = []
    for group_id in sorted(group_status_counts):
        status_counts = group_status_counts[group_id]
        class_counts = group_class_counts[group_id]
        annotated = status_counts["annotated_nonzero"]
        zero_mark = status_counts["groups_but_zero_marks"]
        template_only = status_counts["template_only_no_mark_tables"]
        total = sum(status_counts.values())

        if template_only > 0:
            review_priority = "high"
            primary_action = "review_template_only_exports"
        elif annotated > 0 and zero_mark > 0:
            review_priority = "medium"
            primary_action = "spot_check_zero_mark_samples"
        elif annotated > 0:
            review_priority = "normal"
            primary_action = "use_positive_pool"
        else:
            review_priority = "medium"
            primary_action = "use_negative_pool_after_spot_check"

        rows.append(
            {
                "group_id": group_id,
                "total_samples": total,
                "annotated_nonzero": annotated,
                "groups_but_zero_marks": zero_mark,
                "template_only_no_mark_tables": template_only,
                "dominant_audit_signal": primary_action,
                "review_priority": review_priority,
                "status_profile": join_status_counts(status_counts),
                "classes_present": "|".join(class_name for class_name, _ in class_counts.most_common()),
            }
        )

    priority_rank = {"critical": 0, "high": 1, "medium": 2, "normal": 3, "info": 4}
    rows.sort(key=lambda row: (priority_rank.get(row["review_priority"], 99), -row["total_samples"], row["group_id"]))
    return rows


def build_summary_payload(audit_rows, group_audit_rows):
    bucket_counts = Counter(row["audit_bucket"] for row in audit_rows)
    origin_counts = Counter(row["record_origin"] for row in audit_rows)
    priority_counts = Counter(row["review_priority"] for row in audit_rows)
    action_counts = Counter(row["immediate_action"] for row in audit_rows)

    top_review_groups = [
        {
            "group_id": row["group_id"],
            "review_priority": row["review_priority"],
            "total_samples": row["total_samples"],
            "status_profile": row["status_profile"],
            "dominant_audit_signal": row["dominant_audit_signal"],
        }
        for row in group_audit_rows[:10]
    ]

    return {
        "audit_row_count": len(audit_rows),
        "group_audit_row_count": len(group_audit_rows),
        "bucket_counts": dict(bucket_counts),
        "record_origin_counts": dict(origin_counts),
        "review_priority_counts": dict(priority_counts),
        "immediate_action_counts": dict(action_counts),
        "top_review_groups": top_review_groups,
    }


def write_markdown_summary(path, summary, group_audit_rows):
    lines = [
        "# Jijie Data Usability Audit",
        "",
        "## Bucket Counts",
        "",
        "| Audit Bucket | Count |",
        "| --- | ---: |",
    ]
    for bucket, count in sorted(summary["bucket_counts"].items()):
        lines.append(f"| {bucket} | {count} |")

    lines.extend(
        [
            "",
            "## Immediate Actions",
            "",
            "| Action | Count |",
            "| --- | ---: |",
        ]
    )
    for action, count in sorted(summary["immediate_action_counts"].items()):
        lines.append(f"| {action} | {count} |")

    lines.extend(
        [
            "",
            "## Priority Review Groups",
            "",
            "| Group ID | Priority | Total | Status Profile | Action |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )

    for row in group_audit_rows[:15]:
        lines.append(
            f'| {row["group_id"]} | {row["review_priority"]} | {row["total_samples"]} | '
            f'{row["status_profile"]} | {row["dominant_audit_signal"]} |'
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `positive_supervision`: safe default pool for positive segmentation supervision.",
            "- `negative_candidate`: not a bad sample; usable as a zero-mask or false-positive-control sample.",
            "- `manual_review`: needs raw export/db structure review before being trusted.",
            "- `dropped_duplicate`: excluded only because another identical image hash was kept.",
            "- `invalid_source`: truly incomplete source sample and should be recovered or excluded.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sample_rows = read_csv(SAMPLES_CSV)
    duplicate_rows = read_csv(DUPLICATES_CSV)
    invalid_rows = read_csv(INVALID_CSV)

    group_status_counts, group_class_counts = build_group_stats(sample_rows)

    audit_rows = [classify_canonical_row(row, group_status_counts) for row in sample_rows]
    audit_rows.extend(classify_duplicate_row(row) for row in duplicate_rows)
    audit_rows.extend(classify_invalid_row(row) for row in invalid_rows)

    priority_rank = {"critical": 0, "high": 1, "medium": 2, "normal": 3, "info": 4}
    audit_rows.sort(
        key=lambda row: (
            priority_rank.get(row["review_priority"], 99),
            row["audit_bucket"],
            row["group_id"],
            row["sample_id"],
        )
    )

    group_audit_rows = build_group_audit_rows(sample_rows, group_status_counts, group_class_counts)
    summary = build_summary_payload(audit_rows, group_audit_rows)

    write_csv(
        AUDIT_CSV,
        audit_rows,
        [
            "record_origin",
            "sample_id",
            "group_id",
            "split",
            "source_status",
            "audit_bucket",
            "recommended_usage",
            "immediate_action",
            "review_priority",
            "reason_summary",
            "mark_count",
            "duplicate_count",
            "group_status_profile",
            "classes_present",
            "export_batch",
            "timestamp_dir",
            "sample_dir",
            "tif_name",
            "image_hash",
            "review_flags",
        ],
    )

    write_csv(
        GROUP_AUDIT_CSV,
        group_audit_rows,
        [
            "group_id",
            "total_samples",
            "annotated_nonzero",
            "groups_but_zero_marks",
            "template_only_no_mark_tables",
            "dominant_audit_signal",
            "review_priority",
            "status_profile",
            "classes_present",
        ],
    )

    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_summary(SUMMARY_MD, summary, group_audit_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {AUDIT_CSV}")
    print(f"Wrote: {GROUP_AUDIT_CSV}")
    print(f"Wrote: {SUMMARY_JSON}")
    print(f"Wrote: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
