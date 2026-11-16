"""Replay class unification only on saved masks; never changes production output."""
import argparse,csv,sys
from pathlib import Path
import json
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser()
    for k in ['project','bundle','predictions','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.project))
    if a.output.exists():p.error('Use new file')
    from utils.jijie.postprocess import postprocess_mito_mask
    rows=list(csv.DictReader((a.bundle/'objects.csv').open()));out=[]
    for sid in sorted(set(r['sample_id'] for r in rows)):
        raw=np.array(Image.open(a.predictions/(sid+'_raw.png')))
        image=Image.open(a.bundle/(sid+'.jpg')).convert('L')
        post,_=postprocess_mito_mask(raw,gray_image=np.array(image),classifier_path=None,use_texture_refine=False)
        for r in [r for r in rows if r['sample_id']==sid]:
            x0,y0,x1,y1=json.loads(r['bbox']);mask=np.array(Image.open(a.bundle/r['mask_file']))>0
            out.append(dict(sample_id=sid,component_id=r['component_id'],class_id=r['class_id'],unification_class=int(np.bincount(post[y0:y1,x0:x1][mask],minlength=5).argmax())))
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
    print('audited',len(out),'objects',flush=True)

if __name__=='__main__':main()
