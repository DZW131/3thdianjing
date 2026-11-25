"""Create semantic class maps from saved instance IDs and object-level decisions."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.table.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    by_image = {}
    for row in rows:
        by_image.setdefault(row["sample_id"], []).append(row)
    with args.samples.open(newline="", encoding="utf-8-sig") as stream:
        sample_ids = [row["sample_id"] for row in csv.DictReader(stream)]
    for sample_id in sorted(sample_ids):
        objects = by_image.get(sample_id, [])
        labels = np.load(args.instances / f"{sample_id}_instances.npz")["labels"]
        for rule in ("mean_prob_all", "mito_conditioned", "review_gate"):
            class_map = np.zeros(labels.shape, dtype=np.uint8)
            for row in objects:
                class_map[labels == int(row["instance_id"])] = int(row[rule])
            Image.fromarray(class_map).save(args.output / f"{sample_id}_{rule}.png")
        print(sample_id, len(objects), flush=True)


if __name__ == "__main__":
    main()
