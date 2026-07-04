import math

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist, pdist


def _label_components(binary_mask):
    structure = ndimage.generate_binary_structure(2, 2)
    return ndimage.label(binary_mask.astype(bool), structure=structure)


def _mask_boundary(component_mask):
    eroded = ndimage.binary_erosion(component_mask, structure=np.ones((3, 3), dtype=np.uint8), border_value=0)
    return np.logical_and(component_mask, np.logical_not(eroded))


def _sample_points(coords, max_points=256):
    if coords.shape[0] <= max_points:
        return coords
    indices = np.linspace(0, coords.shape[0] - 1, num=max_points, dtype=np.int64)
    return coords[indices]


def _feret_diameter(boundary_coords):
    if boundary_coords.shape[0] <= 1:
        return 0.0
    sampled = _sample_points(boundary_coords)
    return float(np.max(pdist(sampled.astype(np.float64))))


def _principal_axis_stats(coords):
    if coords.shape[0] <= 1:
        return 0.0, 0.0, 0.0

    points = coords[:, ::-1].astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    major_vector = eigenvectors[:, 0]
    minor_vector = eigenvectors[:, 1]

    major_projection = centered @ major_vector
    minor_projection = centered @ minor_vector
    major_length = float(major_projection.max() - major_projection.min() + 1.0)
    minor_length = float(minor_projection.max() - minor_projection.min() + 1.0)
    orientation_deg = float(math.degrees(math.atan2(major_vector[1], major_vector[0])))
    return major_length, minor_length, orientation_deg


def _component_stats(component_mask, class_id, class_name, sample_id, source, um_per_pixel=None):
    coords = np.column_stack(np.nonzero(component_mask))
    area_px = int(coords.shape[0])
    boundary = _mask_boundary(component_mask)
    boundary_coords = np.column_stack(np.nonzero(boundary))
    perimeter_px = float(boundary_coords.shape[0])

    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    height_px = int(max_row - min_row + 1)
    width_px = int(max_col - min_col + 1)
    feret_px = _feret_diameter(boundary_coords)
    circularity = float((4.0 * math.pi * area_px) / (perimeter_px ** 2)) if perimeter_px > 0 else 0.0
    bbox_aspect_ratio = float(max(width_px, height_px) / max(1, min(width_px, height_px)))
    major_axis_px, minor_axis_px, orientation_deg = _principal_axis_stats(coords)
    major_minor_aspect_ratio = float(major_axis_px / minor_axis_px) if minor_axis_px > 0 else 0.0
    form_factor = float((perimeter_px ** 2) / (4.0 * math.pi * area_px)) if area_px > 0 else 0.0

    record = {
        "sample_id": sample_id,
        "source": source,
        "class_id": int(class_id),
        "class_name": class_name,
        "area_px": area_px,
        "perimeter_px": perimeter_px,
        "bbox_width_px": width_px,
        "bbox_height_px": height_px,
        "feret_diameter_px": feret_px,
        "major_axis_length_px": major_axis_px,
        "minor_axis_length_px": minor_axis_px,
        "circularity": circularity,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "major_minor_aspect_ratio": major_minor_aspect_ratio,
        "orientation_deg": orientation_deg,
        "form_factor": form_factor,
    }

    if um_per_pixel:
        record["area_um2"] = float(area_px * (um_per_pixel ** 2))
        record["perimeter_um"] = float(perimeter_px * um_per_pixel)
        record["feret_diameter_um"] = float(feret_px * um_per_pixel)
        record["major_axis_length_um"] = float(major_axis_px * um_per_pixel)
        record["minor_axis_length_um"] = float(minor_axis_px * um_per_pixel)
    else:
        record["area_um2"] = None
        record["perimeter_um"] = None
        record["feret_diameter_um"] = None
        record["major_axis_length_um"] = None
        record["minor_axis_length_um"] = None

    return record, boundary_coords


def summarize_instances(mask, class_ids, class_names, sample_id, source, um_per_pixel=None):
    rows = []
    boundary_index = {}
    for class_id in class_ids or []:
        class_mask = mask == class_id
        labeled, component_count = _label_components(class_mask)
        boundary_index[class_id] = []
        for component_id in range(1, component_count + 1):
            component_mask = labeled == component_id
            row, boundary_coords = _component_stats(
                component_mask,
                class_id,
                class_names[class_id],
                sample_id,
                source,
                um_per_pixel=um_per_pixel,
            )
            row["component_index"] = component_id
            rows.append(row)
            boundary_index[class_id].append(boundary_coords)
    return rows, boundary_index


def _match_components(gt_binary, pred_binary, iou_threshold):
    gt_labeled, gt_count = _label_components(gt_binary)
    pred_labeled, pred_count = _label_components(pred_binary)
    if gt_count == 0 and pred_count == 0:
        return {
            "gt_count": 0,
            "pred_count": 0,
            "matched_count": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "count_abs_error": 0,
        }

    pair_keys = gt_labeled.astype(np.int64) * (pred_count + 1) + pred_labeled.astype(np.int64)
    pair_hist = np.bincount(pair_keys.ravel(), minlength=(gt_count + 1) * (pred_count + 1))

    gt_areas = np.bincount(gt_labeled.ravel(), minlength=gt_count + 1)
    pred_areas = np.bincount(pred_labeled.ravel(), minlength=pred_count + 1)

    candidates = []
    for gt_idx in range(1, gt_count + 1):
        for pred_idx in range(1, pred_count + 1):
            intersection = pair_hist[gt_idx * (pred_count + 1) + pred_idx]
            if intersection == 0:
                continue
            union = gt_areas[gt_idx] + pred_areas[pred_idx] - intersection
            iou = float(intersection / union) if union > 0 else 0.0
            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx))

    candidates.sort(reverse=True)
    matched_gt = set()
    matched_pred = set()
    matched_count = 0
    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matched_count += 1

    precision = float(matched_count / pred_count) if pred_count > 0 else 0.0
    recall = float(matched_count / gt_count) if gt_count > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if precision + recall > 0 else 0.0
    return {
        "gt_count": int(gt_count),
        "pred_count": int(pred_count),
        "matched_count": int(matched_count),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_abs_error": abs(int(pred_count) - int(gt_count)),
    }


def compute_detection_metrics(gt_mask, pred_mask, class_ids, class_names, sample_id, iou_threshold=0.1):
    rows = []
    for class_id in class_ids or []:
        metrics = _match_components(gt_mask == class_id, pred_mask == class_id, iou_threshold)
        metrics.update(
            {
                "sample_id": sample_id,
                "class_id": int(class_id),
                "class_name": class_names[class_id],
                "iou_threshold": float(iou_threshold),
            }
        )
        rows.append(metrics)
    return rows


def _aggregate_numeric_rows(rows, prefix):
    if not rows:
        return {
            "{}_count".format(prefix): 0,
        }

    area_values = np.asarray([row["area_px"] for row in rows], dtype=np.float64)
    perimeter_values = np.asarray([row["perimeter_px"] for row in rows], dtype=np.float64)
    feret_values = np.asarray([row["feret_diameter_px"] for row in rows], dtype=np.float64)
    major_values = np.asarray([row["major_axis_length_px"] for row in rows], dtype=np.float64)
    form_values = np.asarray([row["form_factor"] for row in rows], dtype=np.float64)
    return {
        "{}_count".format(prefix): int(len(rows)),
        "{}_area_px_sum".format(prefix): float(area_values.sum()),
        "{}_area_px_mean".format(prefix): float(area_values.mean()),
        "{}_area_px_median".format(prefix): float(np.median(area_values)),
        "{}_perimeter_px_mean".format(prefix): float(perimeter_values.mean()),
        "{}_feret_px_mean".format(prefix): float(feret_values.mean()),
        "{}_major_axis_px_mean".format(prefix): float(major_values.mean()),
        "{}_form_factor_mean".format(prefix): float(form_values.mean()),
    }


def _boundary_union(boundary_index, class_ids):
    coords = []
    for class_id in class_ids:
        for boundary_coords in boundary_index.get(class_id, []):
            if boundary_coords.size > 0:
                coords.append(boundary_coords)
    if not coords:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(coords, axis=0)


def compute_sr_distance_metrics(sr_rows, sr_boundary_index, mito_boundary_index, mito_class_ids, sr_class_id, um_per_pixel=None):
    mito_boundary_union = _boundary_union(mito_boundary_index, mito_class_ids)
    sr_boundaries = sr_boundary_index.get(sr_class_id, [])
    distances_px = []

    if mito_boundary_union.size == 0:
        return {
            "sr_distance_count": 0,
            "sr_distance_px_mean": None,
            "sr_distance_px_median": None,
            "sr_distance_um_mean": None,
            "sr_distance_um_median": None,
        }

    for boundary_coords in sr_boundaries:
        if boundary_coords.size == 0:
            continue
        sampled_sr = _sample_points(boundary_coords, max_points=256)
        sampled_mito = _sample_points(mito_boundary_union, max_points=512)
        distances = cdist(sampled_sr.astype(np.float64), sampled_mito.astype(np.float64))
        distances_px.append(float(np.min(distances)))

    if not distances_px:
        return {
            "sr_distance_count": 0,
            "sr_distance_px_mean": None,
            "sr_distance_px_median": None,
            "sr_distance_um_mean": None,
            "sr_distance_um_median": None,
        }

    distances_px = np.asarray(distances_px, dtype=np.float64)
    result = {
        "sr_distance_count": int(distances_px.shape[0]),
        "sr_distance_px_mean": float(distances_px.mean()),
        "sr_distance_px_median": float(np.median(distances_px)),
        "sr_distance_um_mean": None,
        "sr_distance_um_median": None,
    }
    if um_per_pixel:
        result["sr_distance_um_mean"] = float(distances_px.mean() * um_per_pixel)
        result["sr_distance_um_median"] = float(np.median(distances_px) * um_per_pixel)
    return result


def _mean_or_none(rows, key):
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize_class_rows(rows, class_ids, class_names, sample_id, source, image_area_px, um_per_pixel=None):
    class_rows = []
    total_count = len(rows)
    total_area = float(sum(row["area_px"] for row in rows))

    for class_id in class_ids or []:
        selected = [row for row in rows if row["class_id"] == class_id]
        area_sum = float(sum(row["area_px"] for row in selected))
        count = int(len(selected))
        row = {
            "sample_id": sample_id,
            "source": source,
            "class_id": int(class_id),
            "class_name": class_names[class_id],
            "component_count": count,
            "area_px_sum": area_sum,
            "content_percent": float((area_sum / image_area_px) * 100.0) if image_area_px else 0.0,
            "count_fraction": float(count / total_count) if total_count else 0.0,
            "area_fraction_within_quantified": float(area_sum / total_area) if total_area > 0 else 0.0,
            "area_px_mean": _mean_or_none(selected, "area_px"),
            "perimeter_px_mean": _mean_or_none(selected, "perimeter_px"),
            "feret_diameter_px_mean": _mean_or_none(selected, "feret_diameter_px"),
            "major_axis_length_px_mean": _mean_or_none(selected, "major_axis_length_px"),
            "minor_axis_length_px_mean": _mean_or_none(selected, "minor_axis_length_px"),
            "circularity_mean": _mean_or_none(selected, "circularity"),
            "form_factor_mean": _mean_or_none(selected, "form_factor"),
            "major_minor_aspect_ratio_mean": _mean_or_none(selected, "major_minor_aspect_ratio"),
        }
        if um_per_pixel:
            row["area_um2_sum"] = float(area_sum * (um_per_pixel ** 2))
            row["area_um2_mean"] = _mean_or_none(selected, "area_um2")
            row["perimeter_um_mean"] = _mean_or_none(selected, "perimeter_um")
            row["feret_diameter_um_mean"] = _mean_or_none(selected, "feret_diameter_um")
            row["major_axis_length_um_mean"] = _mean_or_none(selected, "major_axis_length_um")
            row["minor_axis_length_um_mean"] = _mean_or_none(selected, "minor_axis_length_um")
        else:
            row["area_um2_sum"] = None
            row["area_um2_mean"] = None
            row["perimeter_um_mean"] = None
            row["feret_diameter_um_mean"] = None
            row["major_axis_length_um_mean"] = None
            row["minor_axis_length_um_mean"] = None
        class_rows.append(row)

    return class_rows


def quantify_task_prediction(
    task_name,
    gt_mask,
    pred_mask,
    class_names,
    quantify_class_ids,
    sample_id,
    um_per_pixel=None,
    return_class_rows=False,
):
    image_summary = {
        "sample_id": sample_id,
        "task_name": task_name,
    }
    object_rows = []
    class_rows = []
    image_area_px = int(np.prod(gt_mask.shape))
    image_summary["image_area_px"] = image_area_px
    image_summary["image_area_um2"] = float(image_area_px * (um_per_pixel ** 2)) if um_per_pixel else None

    pred_rows, pred_boundary_index = summarize_instances(
        pred_mask, quantify_class_ids, class_names, sample_id, "pred", um_per_pixel
    )
    gt_rows, gt_boundary_index = summarize_instances(
        gt_mask, quantify_class_ids, class_names, sample_id, "gt", um_per_pixel
    )
    object_rows.extend(pred_rows)
    object_rows.extend(gt_rows)
    image_summary.update(_aggregate_numeric_rows(pred_rows, "pred"))
    image_summary.update(_aggregate_numeric_rows(gt_rows, "gt"))
    image_summary["pred_content_percent"] = (
        float((image_summary.get("pred_area_px_sum", 0.0) / image_area_px) * 100.0) if image_area_px else 0.0
    )
    image_summary["gt_content_percent"] = (
        float((image_summary.get("gt_area_px_sum", 0.0) / image_area_px) * 100.0) if image_area_px else 0.0
    )

    class_rows.extend(
        summarize_class_rows(pred_rows, quantify_class_ids, class_names, sample_id, "pred", image_area_px, um_per_pixel)
    )
    class_rows.extend(
        summarize_class_rows(gt_rows, quantify_class_ids, class_names, sample_id, "gt", image_area_px, um_per_pixel)
    )

    detection_rows = compute_detection_metrics(gt_mask, pred_mask, quantify_class_ids, class_names, sample_id)
    if detection_rows:
        precision_values = np.asarray([row["precision"] for row in detection_rows], dtype=np.float64)
        recall_values = np.asarray([row["recall"] for row in detection_rows], dtype=np.float64)
        f1_values = np.asarray([row["f1"] for row in detection_rows], dtype=np.float64)
        count_errors = np.asarray([row["count_abs_error"] for row in detection_rows], dtype=np.float64)
        image_summary["object_macro_precision"] = float(precision_values.mean())
        image_summary["object_macro_recall"] = float(recall_values.mean())
        image_summary["object_macro_f1"] = float(f1_values.mean())
        image_summary["object_count_mae"] = float(count_errors.mean())

    if task_name == "mito_sr":
        sr_class_id = max(quantify_class_ids)
        mito_class_ids = [class_id for class_id in quantify_class_ids if class_id != sr_class_id]
        pred_sr_rows = [row for row in pred_rows if row["class_id"] == sr_class_id]
        gt_sr_rows = [row for row in gt_rows if row["class_id"] == sr_class_id]
        pred_sr_distance = compute_sr_distance_metrics(
            pred_sr_rows,
            pred_boundary_index,
            pred_boundary_index,
            mito_class_ids,
            sr_class_id,
            um_per_pixel=um_per_pixel,
        )
        gt_sr_distance = compute_sr_distance_metrics(
            gt_sr_rows,
            gt_boundary_index,
            gt_boundary_index,
            mito_class_ids,
            sr_class_id,
            um_per_pixel=um_per_pixel,
        )
        image_summary.update({"pred_" + key: value for key, value in pred_sr_distance.items()})
        image_summary.update({"gt_" + key: value for key, value in gt_sr_distance.items()})
        if pred_sr_distance["sr_distance_px_mean"] is not None and gt_sr_distance["sr_distance_px_mean"] is not None:
            image_summary["sr_distance_px_mean_abs_error"] = abs(
                pred_sr_distance["sr_distance_px_mean"] - gt_sr_distance["sr_distance_px_mean"]
            )
        if pred_sr_distance["sr_distance_um_mean"] is not None and gt_sr_distance["sr_distance_um_mean"] is not None:
            image_summary["sr_distance_um_mean_abs_error"] = abs(
                pred_sr_distance["sr_distance_um_mean"] - gt_sr_distance["sr_distance_um_mean"]
            )

    if task_name == "sarcomere":
        image_summary["note"] = (
            "Current sarcomere quantification covers component-level Z/M/T/side-tubule "
            "length and morphology proxies. Sarcomere length, AsR, Cur, alpha, and H "
            "require validated pairing rules between adjacent Z/M lines."
        )

    if return_class_rows:
        return image_summary, object_rows, detection_rows, class_rows
    return image_summary, object_rows, detection_rows
