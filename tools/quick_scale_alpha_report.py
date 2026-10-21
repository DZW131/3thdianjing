import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = ROOT / "doctor_prediction_review_full_baseline" / "03_sarcomere_structure"
OUT_DIR = ROOT / "scale_alpha_results_20260702"

CLASS_COLORS = {
    1: (0, 109, 119),      # Z-line
    3: (58, 134, 255),    # M-line
    4: (107, 76, 154),    # side tubule
}
CLASS_NAMES = {
    1: "Z-line",
    3: "M-line",
    4: "Side tubule",
}


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_index():
    rows = []
    with open(REVIEW_ROOT / "index.csv", "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def label_components(binary_mask):
    structure = ndimage.generate_binary_structure(2, 2)
    return ndimage.label(binary_mask.astype(bool), structure=structure)


def component_stats(binary_mask, min_area=20):
    labeled, count = label_components(binary_mask)
    stats = []
    for component_id in range(1, int(count) + 1):
        coords = np.column_stack(np.nonzero(labeled == component_id))
        area = int(coords.shape[0])
        if area < min_area:
            continue
        y = coords[:, 0].astype(np.float64)
        x = coords[:, 1].astype(np.float64)
        centroid = np.array([float(x.mean()), float(y.mean())], dtype=np.float64)
        points = np.column_stack([x, y])
        centered = points - centroid[None, :]
        if points.shape[0] >= 2:
            covariance = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            direction = eigenvectors[:, order[0]]
            projection = centered @ direction
            major_length = float(projection.max() - projection.min() + 1.0)
        else:
            direction = np.array([1.0, 0.0], dtype=np.float64)
            major_length = 1.0
        if direction[0] < 0:
            direction = -direction
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)
        stats.append(
            {
                "component_id": component_id,
                "area_px": area,
                "centroid": centroid,
                "direction": direction,
                "major_length_px": major_length,
                "bbox": (int(min_col), int(min_row), int(max_col), int(max_row)),
            }
        )
    return stats


def mask_for_color(mask_rgb, color, tolerance=8):
    arr = mask_rgb.astype(np.int16)
    target = np.array(color, dtype=np.int16)
    return np.all(np.abs(arr - target[None, None, :]) <= tolerance, axis=2)


def detect_scale_bar_px(overlay_rgb):
    """Detect the black horizontal 1 um scale bar in the lower-right corner."""
    image = np.asarray(overlay_rgb.convert("RGB"))
    h, w = image.shape[:2]
    gray = image.mean(axis=2)
    x0 = int(w * 0.58)
    y0 = int(h * 0.62)
    x1 = int(w * 0.98)
    y1 = int(h * 0.98)
    roi = gray[y0:y1, x0:x1]
    dark = roi < 55
    labeled, count = label_components(dark)

    candidates = []
    for component_id in range(1, int(count) + 1):
        yy, xx = np.nonzero(labeled == component_id)
        if xx.size == 0:
            continue
        width = int(xx.max() - xx.min() + 1)
        height = int(yy.max() - yy.min() + 1)
        area = int(xx.size)
        if width < 60 or height > 35:
            continue
        if width / max(height, 1) < 6.0:
            continue
        density = area / float(width * height)
        if density < 0.25:
            continue
        candidates.append(
            {
                "width": width,
                "height": height,
                "area": area,
                "bbox": (int(xx.min() + x0), int(yy.min() + y0), int(xx.max() + x0), int(yy.max() + y0)),
            }
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["width"], item["area"]), reverse=True)
    return candidates[0]


def angle_between(v1, v2):
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def choose_alpha_triplet(z_components):
    components = [item for item in z_components if item["area_px"] >= 80 and item["major_length_px"] >= 20]
    if len(components) < 3:
        return None

    best = None
    for middle in components:
        others = [item for item in components if item is not middle]
        for i, left in enumerate(others):
            for right in others[i + 1 :]:
                va = left["centroid"] - middle["centroid"]
                vb = right["centroid"] - middle["centroid"]
                da = float(np.linalg.norm(va))
                db = float(np.linalg.norm(vb))
                if not (90 <= da <= 900 and 90 <= db <= 900):
                    continue
                opposite_angle = angle_between(va, vb)
                if opposite_angle is None or opposite_angle < 120:
                    continue
                alpha = 180.0 - opposite_angle
                if alpha > 70:
                    continue
                length_balance = min(da, db) / max(da, db)
                area_score = math.log1p(left["area_px"] + middle["area_px"] + right["area_px"])
                score = area_score * length_balance + alpha * 0.05
                candidate = {
                    "left": left,
                    "middle": middle,
                    "right": right,
                    "alpha_deg": float(alpha),
                    "distance_left_px": da,
                    "distance_right_px": db,
                    "score": score,
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate
    return best


def draw_line_with_points(draw, p1, p2, fill, width=7):
    draw.line([tuple(p1), tuple(p2)], fill=fill, width=width)
    radius = 10
    for point in (p1, p2):
        x, y = float(point[0]), float(point[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(255, 255, 255), width=2)


def make_visualization(row, triplet, scale_candidate, out_path):
    overlay_path = ROOT / "doctor_prediction_review_full_baseline" / row["prediction_overlay"]
    image = Image.open(overlay_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    left = triplet["left"]["centroid"]
    middle = triplet["middle"]["centroid"]
    right = triplet["right"]["centroid"]
    draw_line_with_points(draw, left, middle, fill=(255, 220, 0), width=8)
    draw_line_with_points(draw, middle, right, fill=(255, 70, 70), width=8)

    label = "alpha = {:.1f} deg".format(triplet["alpha_deg"])
    text_xy = (int(middle[0]) + 20, int(middle[1]) - 45)
    draw.rectangle((text_xy[0] - 8, text_xy[1] - 8, text_xy[0] + 150, text_xy[1] + 22), fill=(0, 0, 0))
    draw.text(text_xy, label, fill=(255, 255, 255), font=font)

    for name, point, color in [
        ("Z1", left, (255, 220, 0)),
        ("Z2", middle, (255, 255, 255)),
        ("Z3", right, (255, 70, 70)),
    ]:
        xy = (int(point[0]) + 12, int(point[1]) + 12)
        draw.rectangle((xy[0] - 4, xy[1] - 4, xy[0] + 42, xy[1] + 16), fill=(0, 0, 0))
        draw.text(xy, name, fill=color, font=font)

    if scale_candidate:
        x0, y0, x1, y1 = scale_candidate["bbox"]
        draw.rectangle((x0 - 4, y0 - 4, x1 + 4, y1 + 4), outline=(255, 0, 0), width=4)
        scale_text = "detected scale: 1 um = {} px, {:.5f} um/px".format(
            scale_candidate["width"], 1.0 / scale_candidate["width"]
        )
        draw.rectangle((x0 - 8, y0 - 40, x0 + 370, y0 - 14), fill=(0, 0, 0))
        draw.text((x0, y0 - 36), scale_text, fill=(255, 255, 255), font=font)

    image.save(out_path, quality=95)


def summarize_sample(row):
    overlay_path = ROOT / "doctor_prediction_review_full_baseline" / row["prediction_overlay"]
    mask_path = ROOT / "doctor_prediction_review_full_baseline" / row["prediction_mask"]
    overlay = Image.open(overlay_path).convert("RGB")
    mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    scale = detect_scale_bar_px(overlay)
    um_per_pixel = 1.0 / scale["width"] if scale else None

    class_stats = {}
    for class_id, color in CLASS_COLORS.items():
        binary = mask_for_color(mask_rgb, color)
        min_area = 30 if class_id != 1 else 50
        stats = component_stats(binary, min_area=min_area)
        class_stats[class_id] = stats

    row_summary = {
        "sample_no": row["sample_no"],
        "sample_id": row["sample_id"],
        "scale_bar_um": 1.0 if scale else None,
        "scale_bar_px": scale["width"] if scale else None,
        "um_per_pixel": um_per_pixel,
    }

    for class_id, stats in class_stats.items():
        prefix = CLASS_NAMES[class_id].lower().replace("-", "_").replace(" ", "_")
        areas_px = np.asarray([item["area_px"] for item in stats], dtype=np.float64)
        lengths_px = np.asarray([item["major_length_px"] for item in stats], dtype=np.float64)
        row_summary[prefix + "_count"] = int(len(stats))
        row_summary[prefix + "_area_px_sum"] = float(areas_px.sum()) if areas_px.size else 0.0
        row_summary[prefix + "_length_px_mean"] = float(lengths_px.mean()) if lengths_px.size else None
        if um_per_pixel:
            row_summary[prefix + "_area_um2_sum"] = float(areas_px.sum() * um_per_pixel * um_per_pixel)
            row_summary[prefix + "_length_um_mean"] = float(lengths_px.mean() * um_per_pixel) if lengths_px.size else None
        else:
            row_summary[prefix + "_area_um2_sum"] = None
            row_summary[prefix + "_length_um_mean"] = None

    triplet = choose_alpha_triplet(class_stats[1])
    if triplet:
        row_summary["alpha_deg"] = triplet["alpha_deg"]
        row_summary["alpha_left_distance_um"] = (
            triplet["distance_left_px"] * um_per_pixel if um_per_pixel else None
        )
        row_summary["alpha_right_distance_um"] = (
            triplet["distance_right_px"] * um_per_pixel if um_per_pixel else None
        )
    else:
        row_summary["alpha_deg"] = None
        row_summary["alpha_left_distance_um"] = None
        row_summary["alpha_right_distance_um"] = None

    return row_summary, class_stats, triplet, scale


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main():
    ensure_dir(OUT_DIR)
    rows = load_index()
    sample_rows = []
    visual_candidates = []

    for row in rows:
        summary, class_stats, triplet, scale = summarize_sample(row)
        sample_rows.append(summary)
        if triplet and scale:
            visual_candidates.append((summary, row, triplet, scale))

    write_csv(OUT_DIR / "scale_alpha_sample_summary.csv", sample_rows)

    class_summary_rows = []
    for class_id in CLASS_COLORS:
        prefix = CLASS_NAMES[class_id].lower().replace("-", "_").replace(" ", "_")
        class_summary_rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "sample_count": len(sample_rows),
                "component_count_mean": mean([row.get(prefix + "_count") for row in sample_rows]),
                "area_um2_sum_mean": mean([row.get(prefix + "_area_um2_sum") for row in sample_rows]),
                "length_um_mean": mean([row.get(prefix + "_length_um_mean") for row in sample_rows]),
            }
        )
    write_csv(OUT_DIR / "scale_alpha_class_summary.csv", class_summary_rows)

    alpha_values = [row["alpha_deg"] for row in sample_rows if row.get("alpha_deg") is not None]
    scale_values = [row["um_per_pixel"] for row in sample_rows if row.get("um_per_pixel") is not None]

    if visual_candidates:
        visual_candidates.sort(key=lambda item: item[0]["alpha_deg"], reverse=True)
        summary, row, triplet, scale = visual_candidates[0]
        make_visualization(row, triplet, scale, OUT_DIR / "alpha_angle_visualization.jpg")

    report = [
        "# Scale and alpha-angle report",
        "",
        "Scale detection assumes the visible scale bar label is 1 um.",
        "",
        "## Summary",
        "",
        "- samples: {}".format(len(sample_rows)),
        "- scale detected: {}/{}".format(len(scale_values), len(sample_rows)),
        "- mean um_per_pixel: {:.6f}".format(mean(scale_values) or 0.0),
        "- alpha computed: {}/{}".format(len(alpha_values), len(sample_rows)),
        "- mean alpha_deg: {:.2f}".format(mean(alpha_values) or 0.0),
        "",
        "## Output files",
        "",
        "- scale_alpha_sample_summary.csv",
        "- scale_alpha_class_summary.csv",
        "- alpha_angle_visualization.jpg",
    ]
    (OUT_DIR / "README_quick_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("Wrote:", OUT_DIR)
    print("sample table:", OUT_DIR / "scale_alpha_sample_summary.csv")
    print("class table:", OUT_DIR / "scale_alpha_class_summary.csv")
    print("visualization:", OUT_DIR / "alpha_angle_visualization.jpg")


if __name__ == "__main__":
    main()
