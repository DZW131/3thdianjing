import argparse
import json
import math
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    from skimage import feature, measure, segmentation
except ImportError:
    feature = None
    measure = None
    segmentation = None


CLASS_INFO = {
    1: ("严重损伤线粒体", "#FF0000"),
    2: ("中度损伤线粒体", "#3DC005"),
    3: ("健康线粒体", "#00A2E8"),
    4: ("自噬线粒体", "#A349A4"),
    5: ("肌浆网", "#FF7F27"),
}


CREATE_TABLE_SQL = [
    """
    CREATE TABLE IF NOT EXISTS ChangeRecord (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        markId INTEGER,
        content TEXT,
        tableName TEXT,
        opType TEXT,
        opName TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS MarkGroup (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        groupName TEXT,
        shape TEXT,
        color TEXT,
        opTime REAL,
        isTemplate INTEGER,
        isSelected INTEGER,
        selectable INTEGER,
        editable INTEGER,
        isAi INTEGER,
        parentId INTEGER,
        templateId INTEGER,
        defaultColor TEXT,
        isEmpty INT,
        isShow INT,
        isImport INT DEFAULT 0,
        createTime INT,
        diagnosisType TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS Mark_label_None (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        position JSON,
        method TEXT,
        isExport INTEGER,
        remark TEXT,
        aiResult JSON,
        editable INTEGER,
        strokeColor TEXT,
        fillColor TEXT,
        markType INTEGER,
        diagnosis JSON,
        radius FLOAT,
        createTime FLOAT,
        groupId JSON,
        areaId INTEGER,
        dashed INTEGER,
        doctorDiagnosis JSON,
        markId INTEGER,
        is_export_to_pis INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS Mark_human (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        position JSON,
        method TEXT,
        isExport INTEGER,
        remark TEXT,
        aiResult JSON,
        editable INTEGER,
        strokeColor TEXT,
        fillColor TEXT,
        markType INTEGER,
        diagnosis JSON,
        radius FLOAT,
        createTime FLOAT,
        groupId JSON,
        areaId INTEGER,
        dashed INTEGER,
        doctorDiagnosis JSON,
        markId INTEGER,
        is_export_to_pis INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS MarkToTile_label_None (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        markId INTEGER,
        tileId INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS MarkToTile_human (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        markId INTEGER,
        tileId INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS Pdl1sCount (
        tileId INTEGER PRIMARY KEY,
        posTumor INTEGER,
        negTumor INTEGER,
        posNorm INTEGER,
        negNorm INTEGER,
        total INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ScreenShot (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        filepath TEXT,
        scale TEXT,
        mark JSON,
        mark_id INTEGER,
        ai_type TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS markROITaskStatus (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT,
        mark_id INTEGER,
        rois JSON,
        ai_type TEXT
    )
    """,
]


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_mask(path):
    return np.asarray(Image.open(path), dtype=np.uint8)


def simplify_closed_polyline(points, epsilon):
    if len(points) <= 3 or epsilon <= 0:
        return points

    def perpendicular_distance(point, start, end):
        if np.allclose(start, end):
            return float(np.linalg.norm(point - start))
        vector = end - start
        offset = start - point
        cross = vector[0] * offset[1] - vector[1] * offset[0]
        return float(abs(cross) / np.linalg.norm(vector))

    def rdp(seq):
        if len(seq) < 3:
            return seq
        start, end = seq[0], seq[-1]
        distances = [perpendicular_distance(p, start, end) for p in seq[1:-1]]
        if not distances:
            return seq
        max_idx = int(np.argmax(distances)) + 1
        if distances[max_idx - 1] > epsilon:
            return rdp(seq[: max_idx + 1])[:-1] + rdp(seq[max_idx:])
        return [start, end]

    simplified = rdp([np.asarray(p, dtype=float) for p in points])
    return [(float(p[0]), float(p[1])) for p in simplified]


def component_to_contour(component, simplify_epsilon, max_points):
    if measure is not None:
        padded = np.pad(component.astype(np.uint8), pad_width=1, mode="constant", constant_values=0)
        contours = measure.find_contours(padded, 0.5)
        if not contours:
            return []
        contour = max(contours, key=len)
        points = []
        for row, col in contour:
            x = float(col - 1)
            y = float(row - 1)
            points.append((x, y))
    else:
        eroded = ndimage.binary_erosion(component, structure=np.ones((3, 3), dtype=np.uint8), border_value=0)
        boundary = component & ~eroded
        yy, xx = np.nonzero(boundary)
        if yy.size < 3:
            yy, xx = np.nonzero(component)
        if yy.size < 3:
            return []
        cx = float(xx.mean())
        cy = float(yy.mean())
        angles = np.arctan2(yy.astype(float) - cy, xx.astype(float) - cx)
        order = np.argsort(angles)
        points = [(float(xx[i]), float(yy[i])) for i in order]
    points = simplify_closed_polyline(points, simplify_epsilon)
    if len(points) > max_points:
        step = int(math.ceil(len(points) / float(max_points)))
        points = points[::step]
    if len(points) >= 3 and points[0] != points[-1]:
        points.append(points[0])
    return points


def iter_components(mask, class_id, min_area, simplify_epsilon, max_points):
    binary = mask == class_id
    labeled, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    for component_id in range(1, int(count) + 1):
        component = labeled == component_id
        area = int(component.sum())
        if area < min_area:
            continue
        points = component_to_contour(component, simplify_epsilon=simplify_epsilon, max_points=max_points)
        if len(points) >= 3:
            yield area, points


def split_mito_instances(
    postprocessed_mask,
    mito_classes=(1, 2, 3, 4),
    min_area=30,
    min_distance=12,
):
    """Split touching post-processed mitochondria into editable instances.

    The semantic classes in ``postprocessed_mask`` are not recalculated here.
    This routine only partitions the existing mitochondrial foreground with a
    distance-transform watershed, then assigns every instance the majority
    class already present in that post-processed mask.  It is therefore an
    export-time object separation step, not a replacement for model inference
    or mitochondrial post-processing.

    Returns
    -------
    class_mask : ndarray
        Existing post-processed classes restricted to retained instances.
    instance_labels : ndarray
        Consecutive instance IDs; zero is background.
    stats : dict
        Counts useful for export manifests and visual quality control.
    """
    if feature is None or segmentation is None:
        raise RuntimeError("scikit-image is required for watershed instance export")

    postprocessed_mask = np.asarray(postprocessed_mask)
    mitochondrial = np.isin(postprocessed_mask, mito_classes)
    # Four-connectivity prevents diagonal-only pixels becoming one editable object.
    source_labels, source_count = ndimage.label(
        mitochondrial, structure=ndimage.generate_binary_structure(2, 1)
    )
    source_slices = ndimage.find_objects(source_labels)
    class_mask = np.zeros_like(postprocessed_mask)
    instance_labels = np.zeros(postprocessed_mask.shape, dtype=np.int32)
    next_instance_id = 1
    split_source_components = 0
    discarded_instances = 0

    for source_id, component_slice in enumerate(source_slices, start=1):
        if component_slice is None:
            continue
        component = source_labels[component_slice] == source_id
        component_area = int(component.sum())
        if component_area < min_area:
            discarded_instances += 1
            continue

        distance = ndimage.distance_transform_edt(component)
        peaks = feature.peak_local_max(
            distance,
            labels=component,
            min_distance=max(1, int(min_distance)),
            exclude_border=False,
        )

        if len(peaks) <= 1:
            child_labels = component.astype(np.int32)
        else:
            markers = np.zeros(component.shape, dtype=np.int32)
            markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1, dtype=np.int32)
            child_labels = segmentation.watershed(-distance, markers, mask=component)
            split_source_components += 1

        source_class_mask = postprocessed_mask[component_slice]
        output_class_mask = class_mask[component_slice]
        output_instance_labels = instance_labels[component_slice]
        for child_id in range(1, int(child_labels.max()) + 1):
            child = child_labels == child_id
            child_area = int(child.sum())
            if child_area < min_area:
                discarded_instances += 1
                continue
            class_pixels = source_class_mask[child]
            votes = np.bincount(class_pixels.astype(np.intp), minlength=max(mito_classes) + 1)
            class_id = max(mito_classes, key=lambda candidate: (int(votes[candidate]), -candidate))
            output_class_mask[child] = class_id
            output_instance_labels[child] = next_instance_id
            next_instance_id += 1

    stats = {
        "source_components_4_connected": int(source_count),
        "watershed_split_source_components": int(split_source_components),
        "exported_instances": int(next_instance_id - 1),
        "discarded_small_components_or_children": int(discarded_instances),
    }
    return class_mask, instance_labels, stats


def iter_mito_instances(
    postprocessed_mask,
    min_area,
    min_distance,
    simplify_epsilon,
    max_points,
    mito_classes=(1, 2, 3, 4),
):
    """Yield one contour per watershed-separated post-processed mitochondrion."""
    class_mask, instance_labels, stats = split_mito_instances(
        postprocessed_mask,
        mito_classes=mito_classes,
        min_area=min_area,
        min_distance=min_distance,
    )
    for instance_id in range(1, int(instance_labels.max()) + 1):
        component = instance_labels == instance_id
        class_id = int(class_mask[component][0])
        points = component_to_contour(
            component,
            simplify_epsilon=simplify_epsilon,
            max_points=max_points,
        )
        if len(points) >= 3:
            yield class_id, int(component.sum()), points
    return stats


def create_empty_db(path):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    for sql in CREATE_TABLE_SQL:
        cur.execute(sql)
    conn.commit()
    return conn


def init_groups(conn, class_ids, start_group_id=1001):
    now = int(time.time())
    group_ids = {}
    cur = conn.cursor()
    for offset, class_id in enumerate(class_ids):
        group_id = start_group_id + offset
        group_name, color = CLASS_INFO[class_id]
        cur.execute(
            """
            INSERT INTO MarkGroup (
                id, groupName, shape, color, opTime, isTemplate, isSelected,
                selectable, editable, isAi, parentId, templateId, defaultColor,
                isEmpty, isShow, isImport, createTime, diagnosisType
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
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
        group_ids[class_id] = group_id
    conn.commit()
    return group_ids


def insert_mark(conn, mark_id, group_id, color, points):
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    position = json.dumps({"x": xs, "y": ys}, ensure_ascii=False)
    create_time_ms = float(int(time.time() * 1000))
    conn.execute(
        """
        INSERT INTO Mark_label_None (
            id, position, method, isExport, remark, aiResult, editable,
            strokeColor, fillColor, markType, diagnosis, radius, createTime,
            groupId, areaId, dashed, doctorDiagnosis, markId, is_export_to_pis
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(mark_id),
            position,
            "freepen",
            None,
            "AI prediction; please review",
            "{}",
            1,
            color,
            None,
            1,
            '{"operators": ["AI"]}',
            None,
            create_time_ms,
            json.dumps([int(group_id)]),
            None,
            None,
            "null",
            None,
            0,
        ),
    )


def export_one_db(sample_id, image_path, mito_mask_path, sr_mask_path, out_dir, args):
    sample_dir = ensure_dir(out_dir / sample_id)
    suffix = image_path.suffix.lower() if image_path.suffix else ".jpg"
    output_image = sample_dir / f"{sample_id}{suffix}"
    if not output_image.exists():
        shutil.copy2(image_path, output_image)

    mito_mask = read_mask(mito_mask_path) if mito_mask_path and mito_mask_path.is_file() else None
    sr_mask = read_mask(sr_mask_path) if sr_mask_path and sr_mask_path.is_file() else None
    if mito_mask is None and sr_mask is None:
        return {"sample_id": sample_id, "status": "missing_masks", "mark_count": 0}
    if mito_mask is not None and sr_mask is not None and mito_mask.shape != sr_mask.shape:
        sr_mask = np.asarray(Image.fromarray(sr_mask).resize((mito_mask.shape[1], mito_mask.shape[0]), Image.NEAREST), dtype=np.uint8)

    db_path = sample_dir / "slice.db"
    conn = create_empty_db(db_path)
    group_ids = init_groups(conn, [1, 2, 3, 4, 5])

    next_mark_id = int(time.time() * 1000000) % 900000000000000000 + 1000000000000000000
    rows = []
    class_mark_counts = {}
    for class_id in [1, 2, 3, 4]:
        if mito_mask is None:
            continue
        group_name, color = CLASS_INFO[class_id]
        count = 0
        for _, points in iter_components(
            mito_mask,
            class_id,
            min_area=args.min_mito_area,
            simplify_epsilon=args.simplify_epsilon,
            max_points=args.max_points,
        ):
            insert_mark(conn, next_mark_id, group_ids[class_id], color, points)
            next_mark_id += 1
            count += 1
        class_mark_counts[str(class_id)] = count

    if sr_mask is not None:
        group_name, color = CLASS_INFO[5]
        count = 0
        for _, points in iter_components(
            sr_mask,
            5,
            min_area=args.min_sr_area,
            simplify_epsilon=args.simplify_epsilon,
            max_points=args.max_points,
        ):
            insert_mark(conn, next_mark_id, group_ids[5], color, points)
            next_mark_id += 1
            count += 1
        class_mark_counts["5"] = count

    conn.commit()
    conn.close()
    mark_count = sum(class_mark_counts.values())
    return {
        "sample_id": sample_id,
        "status": "ok",
        "image": str(output_image.relative_to(out_dir)),
        "db": str(db_path.relative_to(out_dir)),
        "mark_count": mark_count,
        "class_mark_counts": json.dumps(class_mark_counts, ensure_ascii=False),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert mito + SR prediction masks into doctor-review slice.db files.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mito-mask-dir", type=Path, required=True)
    parser.add_argument("--sr-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-list", type=Path, default=None)
    parser.add_argument("--min-mito-area", type=int, default=30)
    parser.add_argument("--min-sr-area", type=int, default=20)
    parser.add_argument("--simplify-epsilon", type=float, default=1.5)
    parser.add_argument("--max-points", type=int, default=256)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    if args.sample_list:
        sample_ids = [line.strip() for line in args.sample_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        sample_ids = sorted(path.stem for path in args.image_dir.glob("*.jpg"))

    rows = []
    for index, sample_id in enumerate(sample_ids, start=1):
        image_path = args.image_dir / f"{sample_id}.jpg"
        if not image_path.is_file():
            image_path = args.image_dir / f"{sample_id}.tif"
        mito_mask_path = args.mito_mask_dir / f"{sample_id}.png"
        sr_mask_path = args.sr_mask_dir / f"{sample_id}.png"
        if not image_path.is_file():
            rows.append({"sample_id": sample_id, "status": "missing_image", "mark_count": 0, "class_mark_counts": "{}"})
            continue
        row = export_one_db(sample_id, image_path, mito_mask_path, sr_mask_path, args.output_dir, args)
        rows.append(row)
        if index % 20 == 0 or index == len(sample_ids):
            print(f"[DB] {index}/{len(sample_ids)} converted")

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        import csv

        fieldnames = ["sample_id", "status", "image", "db", "mark_count", "class_mark_counts"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print("DB export finished:", args.output_dir)


if __name__ == "__main__":
    main()
