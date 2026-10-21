"""Visual QC for editable mitochondrial instance export.

The left panel reproduces legacy per-class 8-connected contour export.  The
right panel uses the same post-processed mask but separates touching
mitochondria with watershed before contours are written to ``slice.db``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from export_prediction_masks_to_db import (
    component_to_contour,
    iter_components,
    split_mito_instances,
)


COLORS = {1: "#ff2020", 2: "#39d353", 3: "#00e5e5", 4: "#00ffb7"}
NAMES = {1: "严重损伤", 2: "中度损伤", 3: "健康", 4: "自噬"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare legacy and watershed mitochondrial DB contours.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mito-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--min-distance", type=int, default=12)
    parser.add_argument("--simplify-epsilon", type=float, default=1.5)
    parser.add_argument("--max-points", type=int, default=256)
    return parser.parse_args()


def plot_contour(axis, points, color, linewidth=1.4):
    points = np.asarray(points)
    axis.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth)


def main() -> None:
    args = parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    mask = np.asarray(Image.open(args.mito_mask), dtype=np.uint8)
    if image.shape[:2] != mask.shape:
        raise ValueError(f"image/mask shape mismatch: {image.shape[:2]} vs {mask.shape}")

    class_mask, instance_labels, stats = split_mito_instances(
        mask, min_area=args.min_area, min_distance=args.min_distance
    )
    legacy_count = 0
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Original EM image")
    axes[1].imshow(image)
    for class_id in (1, 2, 3, 4):
        for _, points in iter_components(
            mask,
            class_id,
            min_area=args.min_area,
            simplify_epsilon=args.simplify_epsilon,
            max_points=args.max_points,
        ):
            plot_contour(axes[1], points, COLORS[class_id])
            legacy_count += 1
    axes[1].set_title(f"Legacy DB contours: {legacy_count} objects\n(per-class 8-connected components)")

    axes[2].imshow(image)
    for instance_id in range(1, int(instance_labels.max()) + 1):
        component = instance_labels == instance_id
        class_id = int(class_mask[component][0])
        points = component_to_contour(component, args.simplify_epsilon, args.max_points)
        if len(points) >= 3:
            plot_contour(axes[2], points, COLORS[class_id])
    axes[2].set_title(
        "Post-processing retained + watershed instances: "
        f"{stats['exported_instances']} objects\n"
        f"{stats['watershed_split_source_components']} connected regions split"
    )

    for axis in axes:
        axis.set_axis_off()
    legend_handles = [plt.Line2D([0], [0], color=COLORS[class_id], lw=3, label=NAMES[class_id]) for class_id in COLORS]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print({"legacy_objects": legacy_count, **stats, "output": str(args.output)})


if __name__ == "__main__":
    main()
