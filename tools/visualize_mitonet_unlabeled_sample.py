"""Render a compact side-by-side preview for one unlabeled TEM prediction."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = np.array([[0, 0, 0], [255, 0, 0], [61, 192, 5],
                         [0, 162, 232], [163, 73, 164]], dtype=np.uint8)
CLASS_NAMES = {1: "Severe", 2: "Moderate", 3: "Healthy", 4: "Autophagic"}


def boundaries(labels):
    edge = np.zeros(labels.shape, dtype=bool)
    diff = (labels[:, :-1] != labels[:, 1:]) & ((labels[:, :-1] > 0) | (labels[:, 1:] > 0))
    edge[:, :-1] |= diff
    edge[:, 1:] |= diff
    diff = (labels[:-1, :] != labels[1:, :]) & ((labels[:-1, :] > 0) | (labels[1:, :] > 0))
    edge[:-1, :] |= diff
    edge[1:, :] |= diff
    return edge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--classification-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = Image.open(args.images / f"{args.sample_id}.jpg").convert("RGB")
    labels = np.load(args.instances / f"{args.sample_id}_instances.npz")["labels"]
    if image.size != (labels.shape[1], labels.shape[0]):
        raise ValueError("Input image and instance prediction sizes differ")

    class_by_id, uncertain_by_id = {}, {}
    with args.classification_csv.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["sample_id"] == args.sample_id:
                identifier = int(row["instance_id"])
                class_by_id[identifier] = int(row["mito_conditioned"])
                uncertain_by_id[identifier] = row["needs_review"].lower() == "true"

    h, w = labels.shape
    scale = min(1.0, 1100 / w, 900 / h)
    size = (round(w * scale), round(h * scale))
    original = image.resize(size, Image.Resampling.LANCZOS)
    source = np.asarray(original).astype(np.float32)
    ids = np.unique(labels)
    random_colors = np.random.default_rng(2026).integers(50, 256, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    if len(random_colors):
        random_colors[0] = 0
    instance_rgb = np.repeat(source.copy(), 1, axis=0)
    label_small = np.asarray(Image.fromarray(labels.astype(np.int32), mode="I").resize(size, Image.Resampling.NEAREST))
    instance_colors = random_colors[label_small]
    foreground = label_small > 0
    instance_rgb[foreground] = 0.55 * instance_rgb[foreground] + 0.45 * instance_colors[foreground]
    edge = boundaries(label_small) & foreground
    instance_rgb[edge] = 255

    class_map = np.zeros(labels.shape, dtype=np.uint8)
    uncertain_map = np.zeros(labels.shape, dtype=bool)
    for identifier, class_id in class_by_id.items():
        selected = labels == identifier
        class_map[selected] = class_id
        uncertain_map[selected] = uncertain_by_id[identifier]
    class_small = np.asarray(Image.fromarray(class_map, mode="L").resize(size, Image.Resampling.NEAREST))
    uncertain_small = np.asarray(Image.fromarray(uncertain_map.astype(np.uint8), mode="L").resize(size, Image.Resampling.NEAREST)) > 0
    class_rgb = source.copy()
    class_fg = (label_small > 0) & (class_small > 0)
    class_rgb[class_fg] = 0.62 * class_rgb[class_fg] + 0.38 * CLASS_COLORS[class_small[class_fg]]
    class_edge = boundaries(label_small) & foreground
    class_rgb[class_edge] = 255
    uncertain_edge = boundaries(label_small) & uncertain_small & foreground
    class_rgb[uncertain_edge] = [255, 215, 0]

    panels = [original, Image.fromarray(np.clip(instance_rgb, 0, 255).astype(np.uint8)),
              Image.fromarray(np.clip(class_rgb, 0, 255).astype(np.uint8))]
    header, footer, gap = 50, 60, 16
    width, height = size
    canvas = Image.new("RGB", (width * 3 + gap * 2, height + header + footer), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ["Original", f"MitoNet instances ({int(labels.max())})",
              "DeepLab class candidates"]
    for index, (title, panel) in enumerate(zip(titles, panels)):
        x = index * (width + gap)
        draw.text((x + 6, 8), title, fill="black")
        canvas.paste(panel, (x, header))
    legend = "Class colors: red severe | green moderate | blue healthy | purple autophagic | yellow outline: uncertain, review required"
    draw.text((6, header + height + 10), legend, fill="black")
    draw.text((6, header + height + 31),
              "Instance boundaries are white. Predictions are AI pre-annotations; no GT is available for this batch.", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
    print(args.output)


if __name__ == "__main__":
    main()
