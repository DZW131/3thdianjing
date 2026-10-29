"""Create a small case-diverse diagnostic archive from existing server outputs."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, default=8)
    a = p.parse_args()
    if a.output.exists():
        p.error('Archive exists; choose a new destination.')
    pred_root = a.project/'outputs/doctor_review_predictions_sarcomere_baseline_full_20260903/sarcomere/review_dataset'
    seen, selected = set(), []
    for pred in sorted((pred_root/'SegmentationClass').glob('*.png')):
        case = pred.stem.split('__')[0]
        if case in seen:
            continue
        gt = a.data_root/'SegmentationClass'/pred.name
        image = pred_root/'JPEGImages'/f'{pred.stem}.jpg'
        if not gt.exists() or not image.exists():
            continue
        annotation = np.asarray(Image.open(gt))
        prediction = np.asarray(Image.open(pred))
        if annotation.shape != prediction.shape or int((annotation==10).sum())<100:
            continue
        if 1 not in np.unique(prediction) and 10 not in np.unique(prediction):
            continue
        # Exporter stores original IDs; reject ambiguous/remapped masks explicitly.
        if 10 not in np.unique(prediction):
            continue
        seen.add(case)
        selected.append(dict(case=case,sample_id=pred.stem,annotation_z_pixels=int((annotation==10).sum()),
                             prediction_z_pixels=int((prediction==10).sum()),
                             paths=dict(annotation=gt,prediction=pred,image=image)))
        if len(selected)>=a.cases:
            break
    if not selected:
        raise RuntimeError('No paired original-ID masks found.')
    manifest=[]
    with tarfile.open(a.output,'w:gz') as archive:
        for i,s in enumerate(selected):
            entry={k:v for k,v in s.items() if k!='paths'}
            entry['development_id']=f'dev{i:02d}'
            entry['sha256']={}
            for kind,path in s['paths'].items():
                archive.add(path,arcname=f'dev{i:02d}/{kind}{path.suffix}')
                entry['sha256'][kind]=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.append(entry)
        import io
        payload=json.dumps(dict(selection='First eligible image per sorted case; development only, not random test',
                                source_project=str(a.project),samples=manifest),indent=2).encode()
        info=tarfile.TarInfo('manifest.json')
        info.size=len(payload)
        archive.addfile(info,io.BytesIO(payload))
    print(json.dumps(dict(cases=len(selected),archive=str(a.output),bytes=a.output.stat().st_size)))


if __name__=='__main__':
    main()
