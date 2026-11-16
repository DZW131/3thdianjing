"""Package only selected private images/regions for baseline inference."""
import argparse,csv,json,tarfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import regionprops

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['states','cardiac','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use a fresh output directory')
    a.output.mkdir(parents=True)
    rows=[r for r in csv.DictReader((a.states/'oof_predictions.csv').open()) if r['experiment']=='appearance_control']
    if not rows:raise ValueError('No reference regions')
    for i,r in enumerate(rows):
        arr=np.array(Image.open(a.cardiac/'SegmentationClass'/(r['sample_id']+'.png')))
        cc,_=ndimage.label(arr==int(r['class_id']))
        reg=next(x for x in regionprops(cc) if x.label==int(r['component_id']))
        if int(reg.area)!=int(r['area']):raise ValueError('Region changed since feature extraction')
        y0,x0,y1,x1=reg.bbox;r['bbox']=json.dumps([x0,y0,x1,y1]);r['mask_file']=f'object_{i:03d}.png'
        Image.fromarray(reg.image.astype('uint8')*255).save(a.output/r['mask_file'])
    with (a.output/'objects.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with tarfile.open(a.output/'private_bundle.tar.gz','w:gz') as tar:
        for path in [a.output/'objects.csv',*(a.output/r['mask_file'] for r in rows)]:tar.add(path,arcname='baseline_bundle/'+path.name)
        for sid in sorted(set(r['sample_id'] for r in rows)):tar.add(a.cardiac/'JPEGImages'/(sid+'.jpg'),arcname='baseline_bundle/'+sid+'.jpg')
    print('Private bundle contains',len(rows),'regions. Do not publish it.')

if __name__=='__main__':main()
