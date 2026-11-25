"""Stage unannotated TEM slice TIFF display pages for MitoNet inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be a fresh directory")
    jpeg_dir = args.output / "JPEGImages"
    jpeg_dir.mkdir(parents=True)
    records, excluded = [], []
    for tif in sorted(args.input.rglob("*.tif")):
        rel = tif.relative_to(args.input)
        parts = rel.parts
        if "slices" not in parts:
            excluded.append(dict(path=str(rel), reason="not_under_slices"))
            continue
        slice_index = parts.index("slices")
        if slice_index < 1 or len(parts) != slice_index + 3:
            excluded.append(dict(path=str(rel), reason="unexpected_slice_layout"))
            continue
        batch, slice_id = parts[slice_index - 1], parts[slice_index + 1]
        db = tif.parent / "slice.db"
        if not db.is_file():
            excluded.append(dict(path=str(rel), reason="missing_slice_db"))
            continue
        sample_id = f"{batch}__{slice_id}"
        target = jpeg_dir / f"{sample_id}.jpg"
        with Image.open(tif) as image:
            frames = getattr(image, "n_frames", 1)
            image.seek(0)
            rgb = image.convert("RGB")
            width, height = rgb.size
            rgb.save(target, format="JPEG", quality=95, subsampling=0, optimize=True)
        records.append(dict(sample_id=sample_id, case_id=batch, slice_id=slice_id,
                            source_tif=str(rel), source_db=str(db.relative_to(args.input)),
                            width=width, height=height, tif_frames=frames,
                            tif_sha256=sha256(tif), db_sha256=sha256(db),
                            inference_jpeg=f"JPEGImages/{sample_id}.jpg",
                            jpeg_sha256=sha256(target)))
        print(len(records), sample_id, f"{width}x{height}", flush=True)
    if not records:
        raise ValueError("No paired TIFF/DB slices found")
    with (args.output / "samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "inventory.json").write_text(json.dumps(dict(
        input_root=str(args.input), selected_slice_count=len(records), excluded=excluded,
        source_frame="TIFF page 0 converted to RGB; raw TIFF and source DB are not modified",
        records=records), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(selected=len(records), excluded=excluded), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
