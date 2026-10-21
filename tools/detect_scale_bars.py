#!/usr/bin/env python3
"""Batch-detect TEM scale bars and write a calibration CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.jijie.scale_bar import compute_um_per_pixel, sample_group_id


def read_split(path: Path) -> List[str]:
    ids: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            ids.append(item)
    return ids


def iter_image_ids(image_dir: Path, split_file: Path | None) -> Iterable[str]:
    if split_file is not None:
        yield from read_split(split_file)
        return
    for path in sorted(image_dir.glob("*.jpg")):
        yield path.stem
    for path in sorted(image_dir.glob("*.png")):
        yield path.stem


def resolve_image(image_dir: Path, sample_id: str) -> Path | None:
    for suffix in (".jpg", ".png", ".jpeg", ".tif", ".tiff"):
        path = image_dir / f"{sample_id}{suffix}"
        if path.exists():
            return path
    return None


def save_review_image(image_path: Path, output_path: Path, bbox) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    crop_box = (int(width * 0.45), int(height * 0.50), width, height)
    crop = image.crop(crop_box)
    draw = ImageDraw.Draw(crop)
    if bbox:
        x0, y0, x1, y1 = bbox
        rx0, ry0 = x0 - crop_box[0], y0 - crop_box[1]
        rx1, ry1 = x1 - crop_box[0], y1 - crop_box[1]
        draw.rectangle((rx0, ry0, rx1, ry1), outline=(255, 0, 0), width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--scale-bar-um", type=float, default=1.0)
    parser.add_argument("--review-dir", type=Path, default=None, help="optional directory for lower-right review crops")
    parser.add_argument("--max-review", type=int, default=200)
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    review_count = 0
    for sample_id in iter_image_ids(args.image_dir, args.split_file):
        image_path = resolve_image(args.image_dir, sample_id)
        row = {
            "sample_id": sample_id,
            "group_id": sample_group_id(sample_id),
            "image_path": str(image_path) if image_path else "",
            "detected": 0,
            "scale_bar_px": "",
            "scale_bar_um": args.scale_bar_um,
            "um_per_pixel": "",
            "bbox_x0": "",
            "bbox_y0": "",
            "bbox_x1": "",
            "bbox_y1": "",
            "method": "auto_scale_bar",
        }
        if image_path is not None:
            gray = Image.open(image_path).convert("L")
            um_per_pixel, bar = compute_um_per_pixel(gray, scale_bar_um=args.scale_bar_um)
            if bar is not None:
                row.update(
                    {
                        "detected": 1,
                        "scale_bar_px": bar["width"],
                        "um_per_pixel": um_per_pixel,
                        "bbox_x0": bar["bbox"][0],
                        "bbox_y0": bar["bbox"][1],
                        "bbox_x1": bar["bbox"][2],
                        "bbox_y1": bar["bbox"][3],
                        "method": bar.get("method", "auto_scale_bar"),
                    }
                )
                if args.review_dir is not None and review_count < args.max_review:
                    save_review_image(image_path, args.review_dir / f"{sample_id}.jpg", bar["bbox"])
                    review_count += 1
        rows.append(row)

    fieldnames = [
        "sample_id",
        "group_id",
        "image_path",
        "detected",
        "scale_bar_px",
        "scale_bar_um",
        "um_per_pixel",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "method",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    detected = sum(int(row["detected"]) for row in rows)
    print("Detected scale bars: {}/{}".format(detected, len(rows)))
    print("CSV saved to: {}".format(args.output_csv))
    if args.review_dir is not None:
        print("Review crops saved to: {}".format(args.review_dir))


if __name__ == "__main__":
    main()
