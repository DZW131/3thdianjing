"""Prepare a deterministic prediction-only domain-transfer pilot, not a test set."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import regionprops


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--prediction-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-class',type=int,default=4)
    a=p.parse_args()
    if a.output.exists(): p.error('Choose a fresh output directory')
    sys.path.insert(0,str(a.project/'tools'))
    from export_prediction_masks_to_db import split_mito_instances
    summary=json.loads((a.prediction_root/'summary.json').read_text())
    if summary.get('postprocess_mito') is not True: raise ValueError('Postprocessing provenance required')
    a.output.mkdir(parents=True)
    rows=[]; counts={1:0,2:0,3:0}; used={k:set() for k in counts}
    for path in sorted((a.prediction_root/'review_dataset/SegmentationClass').glob('*.png')):
        m=np.asarray(Image.open(path)); sid=path.stem
        case=sid.split('__')[0]
        imagepath=a.prediction_root/'review_dataset/JPEGImages'/(sid+'.jpg')
        image=Image.open(imagepath).convert('RGB')
        if image.size!=(m.shape[1],m.shape[0]): raise ValueError('Image/mask mismatch')
        cm,inst,_=split_mito_instances(m,min_area=30,min_distance=20)
        for r in sorted(regionprops(inst),key=lambda r:(-r.area,r.label)):
            y0,x0,y1,x1=r.bbox
            cl=int(np.bincount(cm[r.slice][r.image]).argmax())
            if cl not in counts or counts[cl]>=a.per_class or case in used[cl]:continue
            if min(y0,x0)==0 or y1==m.shape[0] or x1==m.shape[1] or r.area<400:continue
            box=(max(0,x0-24),max(0,y0-24),min(m.shape[1],x1+24),min(m.shape[0],y1+24))
            stem=f'pilot_{len(rows):02d}'
            crop=image.crop(box); crop.save(a.output/(stem+'_image.png'))
            binary=np.asarray(inst==r.label,dtype=np.uint8)*255
            mask=Image.fromarray(binary).crop(box);mask.save(a.output/(stem+'_mask.png'))
            arr=np.asarray(crop).copy();arr[np.asarray(mask)==0]=0
            Image.fromarray(arr).save(a.output/(stem+'_masked.png'))
            rows.append(dict(pilot_id=stem,sample_id=sid,case_prefix=case,instance_id=r.label,predicted_class=cl,label_source='AI_prediction_NOT_GT',area_px=int(r.area),bbox=json.dumps(box),image_sha256=hashlib.sha256(imagepath.read_bytes()).hexdigest()))
            counts[cl]+=1;used[cl].add(case)
        if all(n>=a.per_class for n in counts.values()):break
    if rows:
        with (a.output/'manifest.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    result=dict(counts=counts,total=len(rows),selection='largest nonborder instance >=400px, at most one per class/case prefix, sorted inputs',purpose='qualitative transfer check only; predictions are not clinical labels',source=summary)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
