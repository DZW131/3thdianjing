"""Qualitative LMCF TorchScript semantic transfer check; no damage grading."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--crops',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--scale',type=int,choices=[1,2,4],default=1)
    p.add_argument('--background',type=int,choices=[0,255],default=0)
    a=p.parse_args()
    if a.output.exists():p.error('Use a fresh directory')
    torch.set_num_threads(4)
    model=torch.jit.load(str(a.model),map_location=a.device).eval()
    manifest=list(csv.DictReader((a.crops/'manifest.csv').open()))
    a.output.mkdir(parents=True)
    palette=np.array([[0,0,0],[255,255,255],[31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189],[140,86,75],[227,119,194],[127,127,127],[188,189,34],[23,190,207],[255,210,60],[255,80,130]],dtype=np.uint8)
    rows=[];panels=[]
    with torch.inference_mode():
        for row in manifest:
            stem=row['pilot_id'];img=Image.open(a.crops/(stem+'_masked.png')).convert('L')
            mask=np.array(Image.open(a.crops/(stem+'_mask.png')))>0
            if a.scale!=1:
                img=img.resize((img.width*a.scale,img.height*a.scale),Image.Resampling.BILINEAR)
                mask=np.array(Image.fromarray(mask).resize(img.size,Image.Resampling.NEAREST))
            pixels=np.array(img).copy();pixels[~mask]=a.background
            x=torch.from_numpy(pixels.astype('float32')/255)[None,None].to(a.device)
            x=(x-.57571)/.12765
            h,w=x.shape[-2:];x=F.pad(x,(0,(-w)%128,0,(-h)%128))
            raw=model(x)
            logits=raw['sem_logits']
            if logits.shape[-2:]!=x.shape[-2:]:logits=F.interpolate(logits,size=x.shape[-2:],mode='bilinear',align_corners=False)
            prob=logits.softmax(1)[0,:,:h,:w].cpu().numpy()
            if prob.shape[0]!=14:raise ValueError(f'Unexpected semantic channels: {prob.shape}')
            sem=prob.argmax(0).astype('uint8');confidence=prob.max(0)
            Image.fromarray(sem).save(a.output/(stem+'_semantic.png'))
            np.savez_compressed(a.output/(stem+'_heads.npz'),**{k:v.cpu().numpy() for k,v in raw.items()})
            features=model.encoder(x)
            np.savez_compressed(a.output/(stem+'_encoder.npz'),**{f'level_{i}':v.mean((-2,-1)).cpu().numpy() for i,v in enumerate(features)})
            r=dict(row,mean_confidence=float(confidence[mask].mean()),low_confidence_fraction=float((confidence[mask]<.5).mean()))
            for cl in range(14):r[f'class_{cl}_area_fraction']=float((sem[mask]==cl).mean())
            rows.append(r)
            rgb=np.array(Image.open(a.crops/(stem+'_image.png')).convert('RGB'))
            if a.scale!=1:rgb=np.array(Image.fromarray(rgb).resize(img.size,Image.Resampling.BILINEAR))
            overlay=rgb.copy();active=mask&(sem>=2)
            overlay[active]=(rgb[active]*.45+palette[sem[active]]*.55).astype('uint8')
            Image.fromarray(overlay).save(a.output/(stem+'_overlay.png'))
            panel=Image.new('RGB',(640,352),'white');d=ImageDraw.Draw(panel)
            d.text((8,4),f'{stem}: AI class={row["predicted_class"]}; conf={r["mean_confidence"]:.3f}; NOT GT',fill='black')
            for j,im in enumerate([Image.fromarray(rgb),Image.fromarray(overlay)]):
                im.thumbnail((312,310));panel.paste(im,(j*320+4,32))
            panel.save(a.output/(stem+'_comparison.png'));panels.append(panel)
            print(stem,r['mean_confidence'],flush=True)
    with (a.output/'crista_features.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
    gallery=Image.new('RGB',(1280,352*((len(panels)+1)//2)),'white')
    for i,panel in enumerate(panels):gallery.paste(panel,((i%2)*640,(i//2)*352))
    gallery.save(a.output/'gallery.png')
    (a.output/'provenance.json').write_text(json.dumps(dict(model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),torch=torch.__version__,device=a.device,normalization=dict(mean=.57571,std=.12765),padding_factor=128,count=len(rows),scale=a.scale,background=a.background,mode='masked grayscale; semantic-only forward, not official full panoptic pipeline',features='global average pooling per encoder level including padded area; exploratory only',metrics='no manual cristae GT; no classification accuracy measured'),indent=2))

if __name__=='__main__':main()
