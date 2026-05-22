import numpy as np
import torch
from scipy import ndimage


def _normalize_class_ids(class_ids):
    if class_ids is None:
        return []
    if isinstance(class_ids, int):
        return [class_ids]
    return [int(class_id) for class_id in class_ids]


def _component_endpoints(mask, class_ids, min_component_area=4):
    class_ids = _normalize_class_ids(class_ids)
    if not class_ids:
        return []

    binary = np.isin(mask, class_ids)
    labeled, component_count = ndimage.label(binary, structure=ndimage.generate_binary_structure(2, 2))
    endpoints = []

    for component_id in range(1, component_count + 1):
        coords = np.column_stack(np.nonzero(labeled == component_id))
        if coords.shape[0] < int(min_component_area):
            continue
        if coords.shape[0] == 1:
            endpoints.append(tuple(coords[0]))
            continue

        coords_float = coords.astype(np.float64)
        centered = coords_float - coords_float.mean(axis=0, keepdims=True)
        if not np.any(centered):
            endpoints.extend([tuple(coords[0]), tuple(coords[-1])])
            continue

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        projection = centered @ direction
        endpoints.append(tuple(coords[int(np.argmin(projection))]))
        endpoints.append(tuple(coords[int(np.argmax(projection))]))

    return endpoints


def _line_pixels(start, end):
    start_row, start_col = [int(value) for value in start]
    end_row, end_col = [int(value) for value in end]
    steps = int(max(abs(end_row - start_row), abs(end_col - start_col))) + 1
    if steps <= 1:
        return np.asarray([start_row], dtype=np.int64), np.asarray([start_col], dtype=np.int64)
    rows = np.rint(np.linspace(start_row, end_row, steps)).astype(np.int64)
    cols = np.rint(np.linspace(start_col, end_col, steps)).astype(np.int64)
    return rows, cols


def _draw_confidence_line(weight_map, start, end, confidence, radius=1):
    height, width = weight_map.shape
    rows, cols = _line_pixels(start, end)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if not np.any(valid):
        return

    line_mask = np.zeros_like(weight_map, dtype=bool)
    line_mask[rows[valid], cols[valid]] = True
    if radius and radius > 0:
        structure = ndimage.generate_binary_structure(2, 1)
        line_mask = ndimage.binary_dilation(line_mask, structure=structure, iterations=int(radius))

    weight_map[line_mask] = np.maximum(weight_map[line_mask], float(confidence))


def generate_endpoint_bridge_prior(
    mask,
    first_anchor_class_ids,
    second_anchor_class_ids,
    target_class_id,
    line_radius=1,
    min_distance=8,
    max_distance=384,
    min_component_area=4,
    min_confidence=0.25,
    exclude_non_background=True,
):
    """Build a soft candidate band between two anchor structures."""
    mask = np.asarray(mask).astype(np.int64)
    weight_map = np.zeros(mask.shape, dtype=np.float32)

    if target_class_id is None:
        return np.zeros(mask.shape, dtype=np.float32), weight_map

    first_endpoints = _component_endpoints(mask, first_anchor_class_ids, min_component_area=min_component_area)
    second_endpoints = _component_endpoints(mask, second_anchor_class_ids, min_component_area=min_component_area)
    if not first_endpoints or not second_endpoints:
        return np.zeros(mask.shape, dtype=np.float32), weight_map

    second_array = np.asarray(second_endpoints, dtype=np.float64)
    used_pairs = set()
    min_distance = float(min_distance)
    max_distance = float(max_distance)
    confidence_span = max(1.0, max_distance - min_distance)

    for first_point in first_endpoints:
        first_array = np.asarray(first_point, dtype=np.float64)
        distances = np.sqrt(np.sum((second_array - first_array) ** 2, axis=1))
        if distances.size == 0:
            continue

        for second_index in np.argsort(distances):
            distance = float(distances[second_index])
            if distance < min_distance or distance > max_distance:
                continue
            pair_key = (
                tuple(int(v) for v in first_point),
                tuple(int(v) for v in second_endpoints[int(second_index)]),
            )
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)
            confidence = min_confidence + (1.0 - min_confidence) * ((max_distance - distance) / confidence_span)
            _draw_confidence_line(
                weight_map,
                first_point,
                second_endpoints[int(second_index)],
                confidence=max(min_confidence, min(1.0, confidence)),
                radius=line_radius,
            )
            break

    if exclude_non_background:
        allowed = (mask == 0) | (mask == int(target_class_id))
        weight_map = weight_map * allowed.astype(np.float32)

    prior_mask = (weight_map > 0).astype(np.float32)
    return prior_mask, weight_map.astype(np.float32)


def generate_side_tubule_prior(
    mask,
    z_class_ids,
    m_class_ids,
    target_class_id,
    line_radius=1,
    min_distance=8,
    max_distance=384,
    min_component_area=4,
    min_confidence=0.25,
    exclude_non_background=True,
):
    """Build a soft side-tubule candidate band from Z/M component endpoints."""
    return generate_endpoint_bridge_prior(
        mask,
        first_anchor_class_ids=z_class_ids,
        second_anchor_class_ids=m_class_ids,
        target_class_id=target_class_id,
        line_radius=line_radius,
        min_distance=min_distance,
        max_distance=max_distance,
        min_component_area=min_component_area,
        min_confidence=min_confidence,
        exclude_non_background=exclude_non_background,
    )


def generate_t_tubule_prior(
    mask,
    z_class_ids,
    m_class_ids,
    target_class_id,
    line_radius=1,
    min_distance=8,
    max_distance=384,
    min_component_area=4,
    min_confidence=0.25,
    exclude_non_background=True,
):
    """Backward-compatible wrapper for older experimental configs."""
    return generate_endpoint_bridge_prior(
        mask,
        first_anchor_class_ids=z_class_ids,
        second_anchor_class_ids=m_class_ids,
        target_class_id=target_class_id,
        line_radius=line_radius,
        min_distance=min_distance,
        max_distance=max_distance,
        min_component_area=min_component_area,
        min_confidence=min_confidence,
        exclude_non_background=exclude_non_background,
    )


class AddEndpointBridgePrior(object):
    def __init__(
        self,
        first_anchor_class_ids,
        second_anchor_class_ids,
        target_class_id,
        sample_key_prefix,
        line_radius=1,
        min_distance=8,
        max_distance=384,
        min_component_area=4,
        min_confidence=0.25,
        exclude_non_background=True,
    ):
        self.first_anchor_class_ids = _normalize_class_ids(first_anchor_class_ids)
        self.second_anchor_class_ids = _normalize_class_ids(second_anchor_class_ids)
        self.target_class_id = None if target_class_id is None else int(target_class_id)
        self.sample_key_prefix = str(sample_key_prefix)
        self.line_radius = int(line_radius)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.min_component_area = int(min_component_area)
        self.min_confidence = float(min_confidence)
        self.exclude_non_background = bool(exclude_non_background)

    def __call__(self, sample):
        label = sample["label"]
        if torch.is_tensor(label):
            label_array = label.detach().cpu().numpy()
        else:
            label_array = np.asarray(label)

        prior_mask, weight_map = generate_endpoint_bridge_prior(
            label_array,
            first_anchor_class_ids=self.first_anchor_class_ids,
            second_anchor_class_ids=self.second_anchor_class_ids,
            target_class_id=self.target_class_id,
            line_radius=self.line_radius,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            min_component_area=self.min_component_area,
            min_confidence=self.min_confidence,
            exclude_non_background=self.exclude_non_background,
        )

        updated = dict(sample)
        updated[self.sample_key_prefix + "_label"] = torch.from_numpy(prior_mask).float()
        updated[self.sample_key_prefix + "_weight"] = torch.from_numpy(weight_map).float()
        return updated


class AddSideTubulePrior(AddEndpointBridgePrior):
    def __init__(
        self,
        z_class_ids,
        m_class_ids,
        target_class_id,
        line_radius=1,
        min_distance=8,
        max_distance=384,
        min_component_area=4,
        min_confidence=0.25,
        exclude_non_background=True,
    ):
        super().__init__(
            first_anchor_class_ids=z_class_ids,
            second_anchor_class_ids=m_class_ids,
            target_class_id=target_class_id,
            sample_key_prefix="side_tubule_prior",
            line_radius=line_radius,
            min_distance=min_distance,
            max_distance=max_distance,
            min_component_area=min_component_area,
            min_confidence=min_confidence,
            exclude_non_background=exclude_non_background,
        )


class AddTTubulePrior(AddEndpointBridgePrior):
    def __init__(
        self,
        z_class_ids,
        m_class_ids,
        target_class_id,
        line_radius=1,
        min_distance=8,
        max_distance=384,
        min_component_area=4,
        min_confidence=0.25,
        exclude_non_background=True,
    ):
        super().__init__(
            first_anchor_class_ids=z_class_ids,
            second_anchor_class_ids=m_class_ids,
            target_class_id=target_class_id,
            sample_key_prefix="t_tubule_prior",
            line_radius=line_radius,
            min_distance=min_distance,
            max_distance=max_distance,
            min_component_area=min_component_area,
            min_confidence=min_confidence,
            exclude_non_background=exclude_non_background,
        )
