"""Batch OCR of scale numbers and horizontal bars; never assume a 1 um bar."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.jijie.scale_ocr import ScaleOCR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--numeric-only', action='store_true', help='explicitly assume ALL labels are micrometers; read numbers only')
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args()
    if not a.image_dir.is_dir():
        p.error('image-dir must exist')
    if a.output_dir.exists():
        p.error('Use a new output directory')
    paths = sorted(x for x in a.image_dir.iterdir() if x.suffix.lower() in
                   {'.jpg', '.jpeg', '.png', '.tif', '.tiff'} and x.is_file())
    if a.limit is not None:
        if a.limit < 1:
            p.error('limit must be positive')
        paths = paths[:a.limit]
    if not paths:
        p.error('No images found (nonrecursive directory scan)')
    if len({x.stem for x in paths}) != len(paths):
        p.error('Duplicate sample stems; separate input directories first')
    engine = ScaleOCR(numeric_only=a.numeric_only)
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'review').mkdir()
    rows = []
    for i, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert('RGB')
        factor, info = engine.resolve(image)
        row = dict(sample_id=path.stem, status=info['status'], um_per_pixel=factor,
                   scale_bar_um=info.get('scale_bar_um'), scale_bar_px=info.get('scale_bar_px'),
                   ocr_text=info.get('ocr_text', ''), ocr_confidence=info.get('ocr_confidence'),
                   unit_source=info.get('unit_source'), reason=info.get('reason', ''),
                   image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   review_file=f'review/{i:05d}.jpg')
        w, h = image.size
        box = (int(w*.45), int(h*.5), w, h)
        crop = image.crop(box)
        if info.get('bbox'):
            x0, y0, x1, y1 = info['bbox']
            ImageDraw.Draw(crop).rectangle((x0-box[0], y0-box[1], x1-box[0], y1-box[1]), outline='red', width=2)
        crop.save(a.output_dir/row['review_file'])
        rows.append(row)
        print(f'{i+1}/{len(paths)} {path.name}: {info["status"]}', flush=True)
    with (a.output_dir/'calibration.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(total=len(rows), auto_consensus=sum(r['status']=='auto_consensus' for r in rows),
                   needs_review=sum(r['status']=='needs_review' for r in rows),
                   numeric_only=a.numeric_only, warning='Consensus is not human-verified accuracy; review crops before clinical use')
    (a.output_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
