"""Scale bar detection and um/pixel calibration for TEM images.

The detector finds the dark horizontal scale bar in the lower-right corner
and combines pixel measurements with a physical scale. Resolution priority:

1. use an explicit ``um_per_pixel`` if provided;
2. use a per-sample/per-group calibration CSV if available;
3. optional RapidOCR label reading plus bar width;
4. legacy bar-only detection with a caller-provided/legacy default length.

The legacy 1 um default is NOT valid for mixed labels. OCR failure returns no
physical calibration, never a fallback 1 um. Review results before clinical use.
"""

import csv
from pathlib import Path

import numpy as np
from scipy import ndimage


DEFAULT_ROI_X = (0.50, 0.99)
DEFAULT_ROI_Y = (0.55, 0.99)
DEFAULT_DARK_THRESH = 55
DEFAULT_MIN_BAR_WIDTH = 60
DEFAULT_MAX_BAR_HEIGHT = 35
DEFAULT_MIN_ASPECT = 6.0
DEFAULT_MIN_DENSITY = 0.25
DEFAULT_SCALE_BAR_UM = 1.0


def sample_group_id(sample_id):
    """Return the case/group prefix used by jijie sample ids."""
    if sample_id is None:
        return None
    return str(sample_id).split("__", 1)[0]


def _parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if np.isfinite(number) and number > 0 else None


def load_scale_calibration(path):
    """Load optional sample/group scale calibration CSV.

    Accepted columns include:
      - sample_id / image_id / id
      - group_id / case_id
      - um_per_pixel / um_per_px / pixel_size_um
      - scale_bar_um / bar_um
      - scale_bar_px / bar_px / width

    A row may provide either ``um_per_pixel`` directly, or
    ``scale_bar_um`` with ``scale_bar_px``. A row with only
    ``scale_bar_um`` can still guide automatic width detection.
    """
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Scale calibration CSV not found: {}".format(path))

    calibration = {"sample": {}, "group": {}, "rows": []}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {str(key).strip().lower(): value for key, value in raw_row.items() if key is not None}
            if row.get('status') and row['status'] not in ('auto_consensus','auto_recovered','manual_verified','verified'):
                continue
            sample_id = row.get("sample_id") or row.get("image_id") or row.get("id")
            group_id = row.get("group_id") or row.get("case_id")
            record = {
                "sample_id": str(sample_id).strip() if sample_id else None,
                "group_id": str(group_id).strip() if group_id else None,
                "um_per_pixel": _parse_float(
                    row.get("um_per_pixel") or row.get("um_per_px") or row.get("pixel_size_um")
                ),
                "scale_bar_um": _parse_float(row.get("scale_bar_um") or row.get("bar_um")),
                "scale_bar_px": _parse_float(row.get("scale_bar_px") or row.get("bar_px") or row.get("width")),
                "source": str(path),
            }
            if not any(record.get(key) is not None for key in ("um_per_pixel", "scale_bar_um", "scale_bar_px")):
                continue
            calibration["rows"].append(record)
            if record["sample_id"]:
                calibration["sample"][record["sample_id"]] = record
            if record["group_id"]:
                calibration["group"][record["group_id"]] = record
    return calibration


def lookup_scale_calibration(calibration, sample_id):
    if not calibration or not sample_id:
        return None, None
    sample_id = str(sample_id)
    if sample_id in calibration.get("sample", {}):
        return calibration["sample"][sample_id], "sample"
    group_id = sample_group_id(sample_id)
    if group_id in calibration.get("group", {}):
        return calibration["group"][group_id], "group"
    return None, None


def detect_scale_bar_candidates(
    gray_image,
    dark_thresh=DEFAULT_DARK_THRESH,
    min_bar_width=DEFAULT_MIN_BAR_WIDTH,
    max_bar_height=DEFAULT_MAX_BAR_HEIGHT,
    min_aspect=DEFAULT_MIN_ASPECT,
    min_density=DEFAULT_MIN_DENSITY,
    roi_x=DEFAULT_ROI_X,
    roi_y=DEFAULT_ROI_Y,
):
    """Return candidate dark horizontal scale bars in the lower-right corner.

    The return list is sorted by confidence descending. ``bbox`` uses
    inclusive image coordinates: (x0, y0, x1, y1).
    """
    gray = np.asarray(gray_image)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    gray = gray.astype(np.uint8)
    h, w = gray.shape
    x0, x1 = int(w * roi_x[0]), int(w * roi_x[1])
    y0, y1 = int(h * roi_y[0]), int(h * roi_y[1])
    roi = gray[y0:y1, x0:x1]
    dark = roi < dark_thresh
    labeled, n = ndimage.label(dark, structure=ndimage.generate_binary_structure(2, 2))
    candidates = []
    for cid, component_slice in enumerate(ndimage.find_objects(labeled),start=1):
        if component_slice is None:
            continue
        yy, xx = np.nonzero(labeled[component_slice] == cid)
        yy=yy+component_slice[0].start
        xx=xx+component_slice[1].start
        if xx.size == 0:
            continue
        bw = int(xx.max() - xx.min() + 1)
        bh = int(yy.max() - yy.min() + 1)
        area = int(xx.size)
        if bw < min_bar_width or bh > max_bar_height:
            continue
        if bw / max(bh, 1) < min_aspect:
            continue
        if area / float(bw * bh) < min_density:
            continue
        abs_bbox = (
            int(xx.min() + x0),
            int(yy.min() + y0),
            int(xx.max() + x0),
            int(yy.max() + y0),
        )
        aspect = float(bw) / float(max(bh, 1))
        density = float(area) / float(max(bw * bh, 1))
        center_y = (abs_bbox[1] + abs_bbox[3]) / 2.0 / float(max(h, 1))
        # Prefer long, dense, low-positioned bars. This helps avoid choosing
        # dark label text fragments when widths are similar.
        score = bw * density * min(aspect / 10.0, 2.0) * (0.5 + center_y)
        candidates.append(
            {
                "width": bw,
                "height": bh,
                "area": area,
                "bbox": abs_bbox,
                "aspect": aspect,
                "density": density,
                "score": float(score),
            }
        )
    candidates.sort(key=lambda c: (c["score"], c["width"], c["area"]), reverse=True)
    return candidates


def detect_scale_bar(gray_image, **kwargs):
    """Detect the best scale bar candidate, or return None."""
    candidates = detect_scale_bar_candidates(gray_image, **kwargs)
    return candidates[0] if candidates else None


def compute_um_per_pixel(gray_image, scale_bar_um=DEFAULT_SCALE_BAR_UM, **detect_kwargs):
    """Detect the scale bar and return (um_per_pixel, bar_info)."""
    bar = detect_scale_bar(gray_image, **detect_kwargs)
    if bar is None:
        return None, None
    um_per_pixel = float(scale_bar_um) / float(bar["width"])
    bar["scale_bar_um"] = float(scale_bar_um)
    bar["um_per_pixel"] = um_per_pixel
    bar["method"] = "auto_scale_bar"
    return um_per_pixel, bar


def resolve_um_per_pixel(args, gray_image=None, sample_id=None, calibration=None, ocr=None):
    """Resolve the effective um_per_pixel for an evaluation sample.

    Priority:
      1. Explicit ``--um-per-pixel`` CLI argument (applies to all images).
      2. Per-sample/per-group calibration CSV, if provided.
      3. Auto-detect from the image's scale bar (if gray_image provided and
         ``--auto-scale-bar`` is enabled).

    Returns
    -------
    (um_per_pixel, bar_info)
        bar_info records how the scale was resolved.
    """
    explicit = getattr(args, "um_per_pixel", None)
    if explicit is not None and np.isfinite(explicit) and explicit > 0:
        return float(explicit), {"method": "explicit_um_per_pixel", "um_per_pixel": float(explicit)}

    record, level = lookup_scale_calibration(calibration, sample_id)
    if record is not None:
        method = "calibration_csv_{}".format(level)
        um_per_pixel = record.get("um_per_pixel")
        if um_per_pixel:
            info = dict(record)
            info["method"] = method
            return float(um_per_pixel), info
        if record.get("scale_bar_um") and record.get("scale_bar_px"):
            um_per_pixel = float(record["scale_bar_um"]) / float(record["scale_bar_px"])
            info = dict(record)
            info["width"] = float(record["scale_bar_px"])
            info["scale_bar_um"] = float(record["scale_bar_um"])
            info["um_per_pixel"] = um_per_pixel
            info["method"] = method
            return um_per_pixel, info
        if gray_image is not None and getattr(args, "auto_scale_bar", False) and record.get("scale_bar_um"):
            um_per_pixel, bar = compute_um_per_pixel(gray_image, scale_bar_um=record["scale_bar_um"])
            if bar is not None:
                bar["method"] = method + "_plus_auto_width"
            return um_per_pixel, bar

    if getattr(args,'auto_scale_ocr',False):
        if gray_image is None:
            return None,dict(status='needs_review',method='scale_ocr',reason='original_image_missing')
        if ocr is None:
            from .scale_ocr import ScaleOCR
            ocr=ScaleOCR(numeric_only=getattr(args,'scale_ocr_numeric_only',False))
        return ocr.resolve(gray_image)

    if gray_image is not None and getattr(args, "auto_scale_bar", False):
        scale_bar_um = getattr(args, "scale_bar_um", DEFAULT_SCALE_BAR_UM)
        return compute_um_per_pixel(gray_image, scale_bar_um=scale_bar_um)
    return None, None
