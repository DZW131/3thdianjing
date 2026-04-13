import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "data" / "jijie" / "metadata"

AUDIT_CSV = METADATA_DIR / "data_usability_audit.csv"
SAMPLES_CSV = METADATA_DIR / "samples.csv"
SUMMARY_JSON = METADATA_DIR / "summary.json"

CHECKLIST_CSV = METADATA_DIR / "manual_review_checklist.csv"
CHECKLIST_MD = METADATA_DIR / "manual_review_checklist.md"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sample_sort_key(row):
    return (row.get("timestamp_dir", ""), row.get("sample_dir", ""), row.get("tif_name", ""))


def build_group_samples(sample_rows):
    group_samples = defaultdict(list)
    for row in sample_rows:
        group_samples[row["group_id"]].append(row)
    for group_id in group_samples:
        group_samples[group_id].sort(key=sample_sort_key)
    return group_samples


def format_peer_list(rows, max_items=3):
    parts = []
    for row in rows[:max_items]:
        parts.append(
            f'{row.get("tif_name", "")} [{row.get("prepared_sample_id", "")}] mark_count={row.get("mark_count", "")}'
        )
    return " | ".join(parts)


def build_group_status_profile(rows):
    counts = Counter(row["status"] for row in rows)
    return "; ".join(f"{status}={counts[status]}" for status in sorted(counts))


def likely_scenario(group_rows):
    counts = Counter(row["status"] for row in group_rows)
    if counts["annotated_nonzero"] > 0 and counts["groups_but_zero_marks"] > 0:
        return "同组同时存在正样本和零标样本，更像导出版本不一致、部分patch未标，或该样本使用了脚本未覆盖的数据库表结构。"
    if counts["annotated_nonzero"] > 0:
        return "同组存在可正常解析的正样本，这批manual_review更像未带出标注表、导出为空壳库，或采用了非标准标注表。"
    return "当前组内没有可解析正样本，更像空模板导出或数据库结构与当前脚本明显不一致。"


def classify_after_manual_review_guidance():
    return (
        "若发现替代标注表且有坐标/轮廓行 -> 改为 positive_supervision；"
        "若确认没有任何标注对象 -> 改为 negative_candidate；"
        "若db损坏、缺失或目录异常 -> 改为 invalid_source；"
        "若仍不能判断 -> 维持 manual_review。"
    )


def make_raw_sample_dir(input_root, row):
    return str(
        Path(input_root)
        / row["export_batch"]
        / "slice_files"
        / row["timestamp_dir"]
        / row["sample_dir"]
    )


def make_checklist_rows(audit_rows, sample_rows, input_root):
    group_samples = build_group_samples(sample_rows)
    manual_rows = [row for row in audit_rows if row["audit_bucket"] == "manual_review"]
    manual_rows.sort(key=lambda row: (row["group_id"], row["timestamp_dir"], row["sample_dir"]))

    checklist_rows = []
    for row in manual_rows:
        group_id = row["group_id"]
        group_rows = group_samples[group_id]
        positive_peers = [item for item in group_rows if item["status"] == "annotated_nonzero"]
        zero_mark_peers = [item for item in group_rows if item["status"] == "groups_but_zero_marks"]
        template_peers = [item for item in group_rows if item["status"] == "template_only_no_mark_tables"]

        raw_sample_dir = make_raw_sample_dir(input_root, row)
        checklist_rows.append(
            {
                "sample_id": row["sample_id"],
                "group_id": group_id,
                "export_batch": row["export_batch"],
                "timestamp_dir": row["timestamp_dir"],
                "sample_dir": row["sample_dir"],
                "tif_name": row["tif_name"],
                "raw_sample_dir": raw_sample_dir,
                "db_file_hint": "打开该目录中唯一的 .db 文件",
                "current_audit_bucket": row["audit_bucket"],
                "why_flagged_now": row["reason_summary"],
                "same_group_status_profile": build_group_status_profile(group_rows),
                "same_group_positive_examples": format_peer_list(positive_peers),
                "same_group_zero_mark_examples": format_peer_list(zero_mark_peers),
                "same_group_manual_review_count": len(template_peers),
                "likely_scenario": likely_scenario(group_rows),
                "check_1_folder_complete": "",
                "check_2_db_opens_normally": "",
                "check_3_supported_tables_found": "",
                "check_4_other_annotation_tables_found": "",
                "check_5_markgroup_present": "",
                "check_6_any_annotation_rows_found": "",
                "check_7_viewer_shows_drawn_annotations": "",
                "check_8_compare_with_same_group_positive_peer": "",
                "decision_rule": classify_after_manual_review_guidance(),
                "final_decision": "",
                "next_action": "",
                "reviewer_notes": "",
            }
        )
    return checklist_rows


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, checklist_rows):
    group_counts = Counter(row["group_id"] for row in checklist_rows)
    lines = [
        "# Manual Review Checklist for 31 Jijie Samples",
        "",
        "## What \"database table structure differs\" means",
        "",
        "这里不是 Windows 注册表，而是 sqlite 数据库里的表结构。",
        "当前解析脚本只认识 `Mark_label_None` 和 `Mark_human` 这两类标注表。",
        "如果某次导出把轮廓、ROI 或标注点放到了别的表名或别的字段里，脚本就会漏读，样本会被暂时归到 `manual_review`。",
        "",
        "## How to decide after opening the raw folder",
        "",
        "1. 进入 `raw_sample_dir`，确认目录里确实有原图和唯一的 `.db` 文件。",
        "2. 打开 `.db` 查看表名：先找 `Mark_label_None` / `Mark_human`；再找名字里带 `Mark`、`Label`、`ROI`、`Polygon`、`Contour` 的表。",
        "3. 如果存在替代表且里面有坐标/轮廓/对象记录，这张样本应转为 `positive_supervision`，后续需要扩展解析脚本。",
        "4. 如果数据库能正常打开，但没有任何标注对象记录，这张样本可转为 `negative_candidate`。",
        "5. 如果数据库损坏、缺文件、或目录内容异常，这张样本转为 `invalid_source`。",
        "6. 如果仍无法判断，保留 `manual_review` 并把表名或截图记到 `reviewer_notes`。",
        "",
        "## Group Distribution",
        "",
        "| Group ID | Manual Review Count |",
        "| --- | ---: |",
    ]

    for group_id, count in sorted(group_counts.items()):
        lines.append(f"| {group_id} | {count} |")

    lines.extend(
        [
            "",
            "## Column Guide",
            "",
            "- `same_group_positive_examples`: 同组里已经能正常解析的正样本，可用来对照。",
            "- `check_4_other_annotation_tables_found`: 如果不是标准表，但有别的候选标注表，在这里记录 `yes + 表名`。",
            "- `check_6_any_annotation_rows_found`: 只要找到对象行、点集、轮廓、polygon 记录，就填 `yes`。",
            "- `final_decision`: 建议只填 `positive_supervision` / `negative_candidate` / `invalid_source` / `manual_review`。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main():
    audit_rows = read_csv(AUDIT_CSV)
    sample_rows = read_csv(SAMPLES_CSV)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    input_root = summary.get("input_root", "")

    checklist_rows = make_checklist_rows(audit_rows, sample_rows, input_root)

    write_csv(
        CHECKLIST_CSV,
        checklist_rows,
        [
            "sample_id",
            "group_id",
            "export_batch",
            "timestamp_dir",
            "sample_dir",
            "tif_name",
            "raw_sample_dir",
            "db_file_hint",
            "current_audit_bucket",
            "why_flagged_now",
            "same_group_status_profile",
            "same_group_positive_examples",
            "same_group_zero_mark_examples",
            "same_group_manual_review_count",
            "likely_scenario",
            "check_1_folder_complete",
            "check_2_db_opens_normally",
            "check_3_supported_tables_found",
            "check_4_other_annotation_tables_found",
            "check_5_markgroup_present",
            "check_6_any_annotation_rows_found",
            "check_7_viewer_shows_drawn_annotations",
            "check_8_compare_with_same_group_positive_peer",
            "decision_rule",
            "final_decision",
            "next_action",
            "reviewer_notes",
        ],
    )

    write_markdown(CHECKLIST_MD, checklist_rows)

    print(
        json.dumps(
            {
                "manual_review_count": len(checklist_rows),
                "groups": sorted({row["group_id"] for row in checklist_rows}),
                "checklist_csv": str(CHECKLIST_CSV),
                "checklist_md": str(CHECKLIST_MD),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
