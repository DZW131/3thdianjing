"""Scale bar detection and um/pixel calibration for TEM images.

Detects the dark horizontal scale bar in the lower-right corner of TEM
images and converts pixel measurements to micrometers.

The default assumption is that the visible scale bar label is 1 um, which
matches the bulk of the jijie dataset. If a different scale is known, it
can be passed via ``scale_bar_um`` or ``um_per_pixel`` to override.
"""


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


def detect_scale_bar(
    gray_image,
    dark_thresh=DEFAULT_DARK_THRESH,
    min_bar_width=DEFAULT_MIN_BAR_WIDTH,
    max_bar_height=DEFAULT_MAX_BAR_HEIGHT,
    min_aspect=DEFAULT_MIN_ASPECT,
    min_density=DEFAULT_MIN_DENSITY,
    roi_x=DEFAULT_ROI_X,
    roi_y=DEFAULT_ROI_Y,
):
    """Detect the dark horizontal scale bar in the lower-right corner.

    Parameters
    ----------
    gray_image : np.ndarray (H, W)
        Grayscale image (uint8).
    dark_thresh : int
        Pixel value below which a pixel is considered "dark" (part of bar).
    min_bar_width, max_bar_height : int
        Size filters: bar must be wider than ``min_bar_width`` and shorter
        than ``max_bar_height`` (bars are thin horizontal strips).
    min_aspect : float
        width / height ratio lower bound.
    min_density : float
        area / (width * height) lower bound (filters out sparse noise).
    roi_x, roi_y : (float, float)
        Fractional ROI bounds where the bar is expected (lower-right corner).

    Returns
    -------
    dict or None
        {"width": int, "height": int, "area": int, "bbox": (x0,y0,x1,y1)}
        or None if no bar found.
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
    for cid in range(1, int(n) + 1):
        yy, xx = np.nonzero(labeled == cid)
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
        candidates.append({
            "width": bw,
            "height": bh,
            "area": area,
            "bbox": (int(xx.min() + x0), int(yy.min() + y0),
                     int(xx.max() + x0), int(yy.max() + y0)),
        })
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c["width"], c["area"]), reverse=True)
    return candidates[0]


def compute_um_per_pixel(gray_image, scale_bar_um=DEFAULT_SCALE_BAR_UM, **detect_kwargs):
    """Detect the scale bar and return (um_per_pixel, bar_info).

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image.
    scale_bar_um : float
        Physical length the bar represents in micrometers (default 1.0).

    Returns
    -------
    (float, dict) or (None, None)
        um_per_pixel and bar info dict, or (None, None) if bar not found.
    """
    bar = detect_scale_bar(gray_image, **detect_kwargs)
    if bar is None:
        return None, None
    um_per_pixel = float(scale_bar_um) / float(bar["width"])
    bar["scale_bar_um"] = float(scale_bar_um)
    bar["um_per_pixel"] = um_per_pixel
    return um_per_pixel, bar


def resolve_um_per_pixel(args, gray_image=None):
    """Resolve the effective um_per_pixel for an evaluation sample.

    Priority:
      1. Explicit ``--um-per-pixel`` CLI argument (applies to all images).
      2. Auto-detect from the image's scale bar (if gray_image provided and
         ``--auto-scale-bar`` is enabled).

    Returns
    -------
    (um_per_pixel, bar_info)
        bar_info is None when not auto-detected.
    """
    explicit = getattr(args, "um_per_pixel", None)
    if explicit is not None and explicit > 0:
        return float(explicit), None
    if gray_image is not None and getattr(args, "auto_scale_bar", False):
        scale_bar_um = getattr(args, "scale_bar_um", DEFAULT_SCALE_BAR_UM)
        return compute_um_per_pixel(gray_image, scale_bar_um=scale_bar_um)
    return None, None