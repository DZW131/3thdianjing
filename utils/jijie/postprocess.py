"""Post-processing for mito prediction masks.

Two-stage pipeline to improve 4-class mito damage classification:
  1) Instance unification: merge split mitochondria (one physical mito assigned
     multiple classes) into a single class via majority-core voting.
  2) Texture-classifier refinement: for instances where the model is uncertain
     (low margin between top-2 class probabilities), a texture+gray+morph
     classifier re-assigns the class.

The texture classifier is a RandomForest trained on per-component features
(GLCM / LBP / Gabor / wavelet / FFT / intra-mito gray / morphology).
"""
import math
import os
import numpy as np
from scipy import ndimage

try:
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    from skimage.filters import gabor
    from skimage.measure import regionprops
except ImportError:
    local_binary_pattern = graycomatrix = graycoprops = gabor = regionprops = None

try:
    import pywt
except ImportError:
    pywt = None

try:
    import joblib
except ImportError:
    joblib = None


DEFAULT_CLASSIFIER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "run", "jijie", "mito_texture_classifier.pkl",
)

# Candidate features used by the classifier (must match training).
CANDIDATE_FEATURES = [
    "intramito_gray_p75", "intramito_gray_median", "intramito_gray_mean",
    "intramito_gray_iqr", "intramito_gray_std", "intramito_gray_p25",
    "gray_ratio_to_healthy", "gray_ratio_to_healthy_log",
    "fft_low_ratio", "fft_mid_ratio", "fft_high_ratio",
    "wavelet_approx_energy", "wavelet_l2_H_energy", "wavelet_l2_D_energy",
    "glcm_correlation_d1_0", "glcm_correlation_d1_45", "glcm_correlation_d1_90",
    "glcm_contrast_d1_0", "glcm_homogeneity_d1_0", "glcm_energy_d1_0",
    "lbp_u8_r3_bin0", "lbp_u8_r2_bin0", "lbp_u8_r3_bin1", "lbp_u8_r3_bin4",
    "morph_solidity", "morph_extent", "morph_eccentricity", "morph_circularity",
    "morph_form_factor", "morph_swelling_proxy", "morph_area",
    "morph_aspect_major_minor", "morph_equivalent_diameter",
]

MITO_CLASSES = [1, 2, 3, 4]
CLASS_NAMES = {1: "severe", 2: "moderate", 3: "healthy", 4: "autophagic"}
MIN_INSTANCE_AREA = 50


# ---------------------------------------------------------------------------
# Stage 1: instance unification (majority-core voting)
# ---------------------------------------------------------------------------

def unify_instance_classes(pred_mask, mito_classes=None, min_area=MIN_INSTANCE_AREA,
                           erode_radius=3, gray_image=None, use_gray_tiebreak=False):
    """Merge split mitochondria: each physical instance gets ONE class.

    Parameters
    ----------
    pred_mask : np.ndarray (H, W) int
        Prediction mask with remapped class ids (1=severe, 2=moderate,
        3=healthy, 4=autophagic, 0=background).
    mito_classes : list[int]
        Class ids that belong to mitochondria.
    min_area : int
        Ignore instances smaller than this.
    erode_radius : int
        Erosion radius for core extraction. Boundary pixels are noisy.
    gray_image : np.ndarray (H, W) or None
        Original grayscale image, needed for gray tie-break.
    use_gray_tiebreak : bool
        If True, use intra-instance gray to break close races between classes.

    Returns
    -------
    unified_mask : np.ndarray same shape as pred_mask
        Post-processed mask where each physical instance has a single class.
    changed_instance_count : int
        How many instances had their class changed.
    """
    if mito_classes is None:
        mito_classes = MITO_CLASSES
    out = np.zeros_like(pred_mask)
    union = np.isin(pred_mask, mito_classes)
    labeled, n = ndimage.label(union, structure=np.ones((3, 3)))
    struct = np.ones((3, 3))
    changed = 0

    for cid in range(1, int(n) + 1):
        comp = labeled == cid
        if int(comp.sum()) < min_area:
            continue
        classes_in = [c for c in np.unique(pred_mask[comp]) if c in mito_classes]
        if not classes_in:
            continue
        counts = {c: int((pred_mask[comp] == c).sum()) for c in classes_in}
        sorted_classes = sorted(counts, key=counts.get, reverse=True)
        top = sorted_classes[0]
        second = sorted_classes[1] if len(sorted_classes) > 1 else None
        total_mito = sum(counts.values())

        # use eroded core for more stable vote
        if erode_radius > 0:
            eroded = ndimage.binary_erosion(comp, structure=struct, iterations=int(erode_radius), border_value=0)
            if not eroded.any():
                eroded = comp
            core_classes = [c for c in np.unique(pred_mask[eroded]) if c in mito_classes]
            if core_classes:
                counts = {c: int((pred_mask[eroded] == c).sum()) for c in core_classes}
                sorted_classes = sorted(counts, key=counts.get, reverse=True)
                top = sorted_classes[0]
                second = sorted_classes[1] if len(sorted_classes) > 1 else None
                total_mito = sum(counts.values())

        winner = top
        # gray tie-break for close races
        if use_gray_tiebreak and gray_image is not None and second is not None and counts[top] < 0.65 * total_mito:
            gray_means = {}
            for c in classes_in:
                cp = comp & (pred_mask == c)
                if cp.any():
                    gray_means[c] = float(gray_image[cp].mean())
            if top in gray_means and second in gray_means:
                if gray_means[second] > gray_means[top] * 1.05:
                    winner = second

        out[comp] = winner
        if winner != pred_mask[comp].max() or len(classes_in) > 1:
            # instance had multiple classes or winner differs from some pixels
            if len(classes_in) > 1:
                changed += 1

    return out, changed


# ---------------------------------------------------------------------------
# Stage 2: texture features + classifier refinement
# ---------------------------------------------------------------------------

def _glcm_features(patch_gray, levels=16):
    g = (patch_gray.astype(np.float32) / 255.0 * (levels - 1)).astype(np.int32)
    g = np.clip(g, 0, levels - 1)
    if g.shape[0] < 5 or g.shape[1] < 5:
        return None
    feats = {}
    for dist in [1, 2]:
        for ang, ang_name in [(0, "0"), (np.pi / 4, "45"), (np.pi / 2, "90"), (3 * np.pi / 4, "135")]:
            glcm = graycomatrix(g, [dist], [ang], levels=levels, symmetric=True, normed=True)
            for prop in ["contrast", "homogeneity", "energy", "correlation"]:
                feats["glcm_{}_d{}_{}".format(prop, dist, ang_name)] = float(graycoprops(glcm, prop)[0, 0])
    return feats


def _lbp_features(patch_gray, P=8):
    if patch_gray.shape[0] < 4 or patch_gray.shape[1] < 4:
        return None
    feats = {}
    for R in [1, 2, 3]:
        lbp = local_binary_pattern(patch_gray, P=P, R=R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        for i, v in enumerate(hist):
            feats["lbp_u{}_r{}_bin{}".format(P, R, i)] = float(v)
    return feats


def _gabor_features(patch_gray):
    if patch_gray.shape[0] < 8 or patch_gray.shape[1] < 8:
        return None
    feats = {}
    for freq in [0.1, 0.25, 0.4]:
        for ang in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            fr, fi = gabor(patch_gray.astype(np.float32), frequency=freq, theta=ang)
            mag = np.sqrt(fr ** 2 + fi ** 2)
            feats["gabor_mean_f{}_a{}".format(freq, int(np.degrees(ang)))] = float(mag.mean())
            feats["gabor_std_f{}_a{}".format(freq, int(np.degrees(ang)))] = float(mag.std())
    return feats


def _wavelet_features(patch_gray):
    if pywt is None or patch_gray.shape[0] < 8 or patch_gray.shape[1] < 8:
        return None
    arr = patch_gray.astype(np.float32) / 255.0
    feats = {}
    try:
        coeffs = pywt.wavedec2(arr, "db2", level=2)
        feats["wavelet_approx_energy"] = float(np.mean(coeffs[0] ** 2))
        for lvl, detail in enumerate(coeffs[1:], 1):
            for orient, name in zip(detail, ["H", "V", "D"]):
                feats["wavelet_l{}_{}_energy".format(lvl, name)] = float(np.mean(orient ** 2))
    except Exception:
        return None
    return feats


def _fft_features(patch_gray):
    p = patch_gray.astype(np.float32)
    if p.shape[0] < 8 or p.shape[1] < 8:
        return None
    F = np.fft.fft2(p)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    cy, cx = np.array(p.shape) // 2
    yy, xx = np.mgrid[0:p.shape[0], 0:p.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(cy, cx)
    if rmax < 2:
        return None
    total = mag.sum() + 1e-9
    return {
        "fft_low_ratio": float(mag[r <= rmax * 0.15].sum() / total),
        "fft_mid_ratio": float(mag[(r > rmax * 0.15) & (r <= rmax * 0.5)].sum() / total),
        "fft_high_ratio": float(mag[(r > rmax * 0.5) & (r <= rmax)].sum() / total),
    }


def _intramito_gray_features(patch_gray, comp_patch, healthy_mean_gray):
    vals = patch_gray[comp_patch].astype(np.float32)
    if vals.size == 0:
        return None
    mean = float(vals.mean())
    feats = {
        "intramito_gray_mean": mean,
        "intramito_gray_median": float(np.median(vals)),
        "intramito_gray_std": float(vals.std()),
        "intramito_gray_min": float(vals.min()),
        "intramito_gray_max": float(vals.max()),
        "intramito_gray_p25": float(np.percentile(vals, 25)),
        "intramito_gray_p75": float(np.percentile(vals, 75)),
        "intramito_gray_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
    }
    if healthy_mean_gray and healthy_mean_gray > 0:
        feats["gray_ratio_to_healthy"] = mean / healthy_mean_gray
        feats["gray_ratio_to_healthy_log"] = float(np.log(mean / healthy_mean_gray + 1e-9))
    else:
        feats["gray_ratio_to_healthy"] = None
        feats["gray_ratio_to_healthy_log"] = None
    return feats


def _morphology_features(comp_patch):
    if regionprops is None:
        return None
    labeled = comp_patch.astype(np.int32)
    props = regionprops(labeled)
    if not props:
        return None
    p = props[0]
    area = float(p.area)
    peri = float(p.perimeter)
    feats = {
        "morph_area": area,
        "morph_perimeter": peri,
        "morph_equivalent_diameter": float(p.equivalent_diameter),
        "morph_major_axis": float(p.major_axis_length),
        "morph_minor_axis": float(p.minor_axis_length),
        "morph_eccentricity": float(p.eccentricity),
        "morph_extent": float(p.extent),
        "morph_solidity": float(p.solidity),
        "morph_aspect_major_minor": float(p.major_axis_length / (p.minor_axis_length + 1e-9)),
    }
    if peri > 0 and area > 0:
        feats["morph_circularity"] = float((4 * math.pi * area) / (peri ** 2))
        feats["morph_form_factor"] = float((peri ** 2) / (4 * math.pi * area))
    feats["morph_swelling_proxy"] = float(area / (peri ** 2 + 1e-9))
    return feats


def extract_component_features(patch_gray, comp_patch, healthy_mean_gray=None):
    """Extract the full feature vector for one mitochondrion instance.

    Returns a dict with all CANDIDATE_FEATURES keys (or None if extraction fails).
    """
    feats = {}
    if not comp_patch.any() or patch_gray.shape[0] < 8 or patch_gray.shape[1] < 8:
        return None
    fill = float(patch_gray[comp_patch].mean()) if comp_patch.any() else float(patch_gray.mean())
    patch_masked = patch_gray.copy()
    patch_masked[~comp_patch] = fill

    for fn in [_glcm_features, _lbp_features, _gabor_features, _wavelet_features, _fft_features]:
        r = fn(patch_masked)
        if r is None:
            return None
        feats.update(r)

    r = _intramito_gray_features(patch_gray, comp_patch, healthy_mean_gray)
    if r is None:
        return None
    feats.update(r)

    r = _morphology_features(comp_patch)
    if r is None:
        return None
    feats.update(r)

    # keep only candidate features
    return {k: feats.get(k) for k in CANDIDATE_FEATURES}


def _feature_dict_to_vector(feats):
    vec = []
    for k in CANDIDATE_FEATURES:
        v = feats.get(k)
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return None
        vec.append(float(v))
    return np.array(vec, dtype=np.float64)


def compute_healthy_baseline_gray(gray_image, pred_mask, healthy_class_id=3):
    """Mean gray of pixels predicted as healthy — used as per-image baseline."""
    hp = gray_image[pred_mask == healthy_class_id]
    if hp.size == 0:
        # fall back to all mito pixels
        hp = gray_image[np.isin(pred_mask, MITO_CLASSES)]
    return float(hp.mean()) if hp.size > 0 else None


def refine_with_texture_classifier(pred_mask, gray_image, classifier_path=None,
                                   uncertainty_threshold=0.4, min_area=MIN_INSTANCE_AREA):
    """Second-stage refinement: re-classify uncertain instances using the
    texture+gray+morph classifier.

    An instance is "uncertain" if the model's top-2 class probability margin is
    small. Since we only have the argmax mask (not logits here), we approximate
    uncertainty by checking if the instance was a close race in stage-1 voting
    (top class < 65% of instance pixels). For those, the classifier predicts
    a class from texture features and overrides if it disagrees.

    Parameters
    ----------
    pred_mask : np.ndarray
        Already stage-1 unified mask.
    gray_image : np.ndarray
        Original grayscale image.
    classifier_path : str
        Path to the trained classifier .pkl (joblib).
    uncertainty_threshold : float
        Fraction threshold below which an instance is considered uncertain.
        (top class pixel fraction < threshold => uncertain)

    Returns
    -------
    refined_mask : np.ndarray
    refined_count : int
    """
    if joblib is None or classifier_path is None or not os.path.isfile(classifier_path):
        return pred_mask, 0

    bundle = joblib.load(classifier_path)
    clf = bundle["classifier"]
    scaler = bundle.get("scaler")
    # classifier predicts original class id in {1,2,3,4}
    healthy_mean = compute_healthy_baseline_gray(gray_image, pred_mask)

    out = pred_mask.copy()
    union = np.isin(pred_mask, MITO_CLASSES)
    labeled, n = ndimage.label(union, structure=np.ones((3, 3)))
    refined = 0

    for cid in range(1, int(n) + 1):
        comp = labeled == cid
        if int(comp.sum()) < min_area:
            continue
        current_class = int(pred_mask[comp][0])
        # uncertainty: fraction of current class pixels in instance
        frac = float((pred_mask[comp] == current_class).sum()) / float(comp.sum())
        if frac >= 1.0 - 1e-6:
            # pure instance, was never split — still allow classifier if it's a
            # "boundary" class (severe/moderate are the hard ones)
            if current_class not in (1, 2):
                continue
        # extract features
        ys, xs = np.nonzero(comp)
        y0, y1 = max(0, ys.min() - 2), min(gray_image.shape[0], ys.max() + 3)
        x0, x1 = max(0, xs.min() - 2), min(gray_image.shape[1], xs.max() + 3)
        patch = gray_image[y0:y1, x0:x1]
        comp_patch = comp[y0:y1, x0:x1]
        feats = extract_component_features(patch, comp_patch, healthy_mean)
        if feats is None:
            continue
        vec = _feature_dict_to_vector(feats)
        if vec is None:
            continue
        if scaler is not None:
            vec = scaler.transform(vec.reshape(1, -1))[0]
        predicted = int(clf.predict(vec.reshape(1, -1))[0])
        if predicted != current_class:
            out[comp] = predicted
            refined += 1

    return out, refined


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def postprocess_mito_mask(pred_mask, gray_image=None, classifier_path=None,
                          use_gray_tiebreak=True, use_texture_refine=True,
                          erode_radius=3):
    """Full two-stage post-processing for mito 4-class masks.

    Returns
    -------
    post_mask : np.ndarray
        Final post-processed mask.
    stats : dict
        {"unified_count": int, "refined_count": int}
    """
    # Stage 1: unify instance classes
    post_mask, unified = unify_instance_classes(
        pred_mask, gray_image=gray_image, use_gray_tiebreak=use_gray_tiebreak,
        erode_radius=erode_radius,
    )
    stats = {"unified_count": int(unified)}

    # Stage 2: texture classifier refinement
    if use_texture_refine and gray_image is not None and classifier_path:
        post_mask, refined = refine_with_texture_classifier(
            post_mask, gray_image, classifier_path=classifier_path,
        )
        stats["refined_count"] = int(refined)
    else:
        stats["refined_count"] = 0

    return post_mask, stats