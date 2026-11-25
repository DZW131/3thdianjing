"""Write MitoNet instance polygons into independent editable platform slice.db files."""
import argparse
import csv
import json
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
from skimage.measure import regionprops
from export_prediction_masks_to_db import CREATE_TABLE_SQL


CLASS_INFO = {
    1: ("严重损伤线粒体", "#FF0000"),
    2: ("中度损伤线粒体", "#3DC005"),
    3: ("健康线粒体", "#00A2E8"),
    4: ("自噬线粒体", "#A349A4"),
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def insert_dynamic(connection, table, values):
    available = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
    values = {key: value for key, value in values.items() if key in available}
    columns = list(values)
    quoted = ", ".join(f'"{key}"' for key in columns)
    placeholders = ", ".join("?" for _ in columns)
    cursor = connection.execute(f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})',
                                [values[key] for key in columns])
    return int(cursor.lastrowid)


def make_group(connection, name, color):
    return insert_dynamic(connection, "MarkGroup", dict(
        groupName=name, shape="", color=color, opTime=0, isTemplate=0, isSelected=0,
        selectable=1, editable=1, isAi=1, parentId=None, templateId=None,
        defaultColor=color, isEmpty=0, isShow=1, isImport=0, createTime=int(time.time()),
        diagnosisType=None))


def insert_polygon(connection, mark_id, group_id, color, points, remark):
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    insert_dynamic(connection, "Mark_label_None", dict(
        id=mark_id, position=json.dumps({"x": xs, "y": ys}, ensure_ascii=False),
        method="freepen", isExport=0, remark=remark, aiResult="{}", editable=1,
        strokeColor=color, fillColor=None, markType=1,
        diagnosis=json.dumps({"operators": ["AI"]}), radius=None, createTime=int(time.time() * 1000),
        groupId=json.dumps([group_id]), areaId=None, dashed=0,
        doctorDiagnosis="null", markId=None, is_export_to_pis=0,
        aiStatus=1, regionType="AI"))


def contours_for_instance(labels, region, min_area, epsilon, max_points):
    instance_id = int(region.label)
    y0, x0, y1, x1 = region.bbox
    crop = labels[y0:y1, x0:x1] == instance_id
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        crop.astype(np.uint8), connectivity=8
    )
    output = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = np.ascontiguousarray((components == component_id).astype(np.uint8))
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(contour) > max_points:
            contour = contour[np.linspace(0, len(contour) - 1, max_points, dtype=int)]
        if len(contour) >= 3:
            points = [(int(x) + x0, int(y) + y0) for x, y in contour]
            points.append(points[0])
            output.append((area, points))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--classification-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("instances", "classified", "both"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--simplify-epsilon", type=float, default=1.5)
    parser.add_argument("--max-points", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    if args.mode in {"classified", "both"} and not args.classification_csv:
        parser.error("--classification-csv is required for classified mode")

    samples = read_csv(args.samples)
    if args.limit:
        samples = samples[:args.limit]
    class_rows = read_csv(args.classification_csv) if args.classification_csv else []
    class_by_sample = {}
    for row in class_rows:
        class_by_sample.setdefault(row["sample_id"], {})[int(row["instance_id"])] = row

    args.output.mkdir(parents=True)
    modes = ("instances", "classified") if args.mode == "both" else (args.mode,)
    output_roots = {mode: args.output / mode if args.mode == "both" else args.output for mode in modes}
    manifests = {mode: [] for mode in modes}
    group_labels_by_mode = {
        "instances": {0: ("AI_线粒体实例（未分类）", "#00A2E8")},
        "classified": {class_id: (f"AI_{name}", color)
                       for class_id, (name, color) in CLASS_INFO.items()},
    }
    for root in output_roots.values():
        (root / "slice_files").mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples, start=1):
        source_rel = Path(sample["source_db"].replace("\\", "/"))
        source_db = args.raw_root / source_rel
        if not source_db.is_file():
            raise FileNotFoundError(source_db)
        labels = np.load(args.instances / f'{sample["sample_id"]}_instances.npz')["labels"]
        connections = {}
        next_ids = {}
        initial_counts = {}
        created = {mode: Counter() for mode in modes}
        group_ids = {}
        for mode in modes:
            root = output_roots[mode]
            destination = root / "slice_files" / source_rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_db, destination)
            connection = sqlite3.connect(destination)
            connections[mode] = connection
            for statement in CREATE_TABLE_SQL:
                connection.execute(statement)
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"MarkGroup", "Mark_label_None"}.issubset(tables):
                raise RuntimeError(f"Unsupported slice.db schema; tables={sorted(tables)}: {source_db}")
            group_labels = group_labels_by_mode[mode]
            group_ids[mode] = {class_id: make_group(connection, name, color)
                               for class_id, (name, color) in group_labels.items()}
            next_ids[mode] = int(connection.execute(
                'SELECT COALESCE(MAX(id),0) FROM "Mark_label_None"').fetchone()[0]) + 1
            initial_counts[mode] = next_ids[mode] - 1

        try:
            for region in regionprops(labels):
                instance_id = int(region.label)
                if region.area < args.min_area:
                    continue
                class_row = class_by_sample.get(sample["sample_id"], {}).get(instance_id)
                polygons = contours_for_instance(labels, region, args.min_area,
                                                 args.simplify_epsilon, args.max_points)
                for mode in modes:
                    if mode == "instances":
                        group_class = 0
                        remark = "MitoNet AI instance pre-annotation; unclassified; please review and edit."
                    else:
                        if class_row is None:
                            raise ValueError(f"Missing object class row: {sample['sample_id']} / {instance_id}")
                        group_class = int(class_row["mito_conditioned"])
                        uncertain = str(class_row["needs_review"]).lower() == "true"
                        status = "uncertain; please verify class" if uncertain else "AI class candidate; please review"
                        remark = f"MitoNet + DeepLab AI pre-annotation; {status}. instance_id={instance_id}"
                    _, color = group_labels_by_mode[mode][group_class]
                    connection = connections[mode]
                    for area, points in polygons:
                        insert_polygon(connection, next_ids[mode], group_ids[mode][group_class],
                                       color, points, f"{remark} area_px={area}")
                        next_ids[mode] += 1
                        created[mode][group_class] += 1

            for mode, connection in connections.items():
                connection.commit()
                final_count = int(connection.execute('SELECT COUNT(*) FROM "Mark_label_None"').fetchone()[0])
                if final_count - initial_counts[mode] != sum(created[mode].values()):
                    raise RuntimeError(f"slice.db mark count did not reconcile ({mode})")
        finally:
            for connection in connections.values():
                connection.close()

        for mode in modes:
            root = output_roots[mode]
            destination = root / "slice_files" / source_rel
            group_labels = group_labels_by_mode[mode]
            manifests[mode].append(dict(
                sample_id=sample["sample_id"], source_tif=sample["source_tif"],
                source_db=sample["source_db"], output_db=str(destination.relative_to(root)),
                predicted_instances=int(labels.max()), added_marks=sum(created[mode].values()),
                **{f"class_{key}_marks": int(created[mode][key]) for key in group_labels}))
        if index % 25 == 0:
            print(f"processed {index}/{len(samples)}", flush=True)

    for mode in modes:
        root = output_roots[mode]
        manifest = manifests[mode]
        group_labels = group_labels_by_mode[mode]
        with (root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        summary = dict(mode=mode, samples=len(manifest), source_tiffs_not_copied=True,
                       note="Each slice.db is an independent editable AI-mark copy; originals remain unchanged.",
                       class_groups={str(key): name for key, (name, _) in group_labels.items()},
                       added_marks=sum(row["added_marks"] for row in manifest),
                       min_area=args.min_area, simplify_epsilon=args.simplify_epsilon,
                       max_points=args.max_points)
        (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
