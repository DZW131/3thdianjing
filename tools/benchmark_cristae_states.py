"""Exploratory public paired evaluation; no cardiac damage labels are changed."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage import exposure

def metrics(cm):
    rows=[]
    for k in range(14):
        tp=cm[k,k];gt=cm[k].sum();pr=cm[:,k].sum()
        rows.append(dict(class_id=k,gt_pixels=int(gt),pred_pixels=int(pr),iou=float(tp/(gt+pr-tp)) if gt+pr-tp else None,dice=float(2*tp/(gt+pr)) if gt+pr else None,recall=float(tp/gt) if gt else None))
    present=[r for r in rows[2:] if r['gt_pixels']]
    return dict(per_class=rows,macro_present_crista_iou=float(np.mean([r['iou'] for r in present])),macro_present_crista_dice=float(np.mean([r['dice'] for r in present])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--tile',type=int,choices=[512,1024],default=512);p.add_argument('--modes',nargs='+',choices=['raw','flip_tta','clahe'],default=['raw','flip_tta','clahe']);a=p.parse_args()
    if a.output.exists():p.error('Use fresh output')
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    model=torch.jit.load(str(a.model),map_location=a.device).eval();results=[]
    with torch.inference_mode():
        for mode in a.modes:
            total=np.zeros((14,14),dtype=np.int64)
            for gtpath in sorted((a.data/'CRISTAE_manual_annotation').glob('*.png')):
                original=np.array(Image.open(a.data/'CRISTAE_orig_images'/(gtpath.stem+'.tif')).convert('L'))
                gt=np.array(Image.open(gtpath));gt=gt.astype('int64')//1000
                if original.shape!=gt.shape:raise ValueError(f'Shape mismatch {original.shape} vs {gt.shape}')
                if gt.max()>13:raise ValueError('Unexpected panoptic encoding')
                image=original.astype('float32')/255
                if mode=='clahe':image=exposure.equalize_adapthist(image,clip_limit=.01).astype('float32')
                h,w=image.shape;scores=np.zeros((14,h,w),dtype='float32');counts=np.zeros((h,w),dtype='float32')
                tile=a.tile;stride=tile*3//4
                ys=sorted(set(list(range(0,max(1,h-tile+1),stride))+[max(0,h-tile)]));xs=sorted(set(list(range(0,max(1,w-tile+1),stride))+[max(0,w-tile)]))
                for y in ys:
                    for x in xs:
                        crop=image[y:y+tile,x:x+tile];hh,ww=crop.shape
                        t=torch.from_numpy(crop.copy())[None,None].to(a.device);t=(t-.57571)/.12765;t=F.pad(t,(0,(-ww)%128,0,(-hh)%128))
                        logits=model(t)['sem_logits'];prob=logits.softmax(1)
                        if mode=='flip_tta':prob=(prob+model(t.flip(-1))['sem_logits'].softmax(1).flip(-1))/2
                        if prob.shape[-2:]!=t.shape[-2:]:prob=F.interpolate(prob,size=t.shape[-2:],mode='bilinear',align_corners=False)
                        scores[:,y:y+hh,x:x+ww]+=prob[0,:,:hh,:ww].cpu().numpy();counts[y:y+hh,x:x+ww]+=1
                assert counts.min()>0
                pred=(scores/counts[None]).argmax(0).astype('uint8')
                cm=np.bincount((gt*14+pred).ravel(),minlength=196).reshape(14,14);total+=cm
                Image.fromarray(pred).save(a.output/f'{mode}_{gtpath.stem}.png')
                print(mode,gtpath.stem,metrics(cm)['macro_present_crista_iou'],flush=True)
            m=metrics(total);m['condition']=mode;results.append(m)
            np.savetxt(a.output/f'{mode}_confusion.csv',total,delimiter=',',fmt='%d')
        (a.output/'results.json').write_text(json.dumps(results,indent=2))
        with (a.output/'comparison.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=['condition','macro_present_crista_iou','macro_present_crista_dice']);wr.writeheader();wr.writerows({k:r[k] for k in wr.fieldnames} for r in results)
    (a.output/'provenance.json').write_text(json.dumps(dict(model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),n=len(list((a.data/'CRISTAE_manual_annotation').glob('*.png'))),torch=torch.__version__,gt_encoding='class=floor(panoptic_id/1000), zero remains background',tile=a.tile,stride=a.tile*3//4,scope='same public exploratory subset for all conditions; not independent model selection test; no cardiac GT metrics',input_sha256={str(p.relative_to(a.data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.data.rglob('*') if p.suffix in ['.png','.tif']}),indent=2))

if __name__=='__main__':main()
