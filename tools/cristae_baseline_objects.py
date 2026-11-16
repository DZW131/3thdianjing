"""Full-image existing DeepLab inference pooled on fixed physician regions."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

def main():
    p=argparse.ArgumentParser()
    for k in ['project','bundle','checkpoint','classifier','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use new output')
    if not a.classifier.is_file():raise FileNotFoundError(a.classifier)
    a.output.mkdir(parents=True);sys.path.insert(0,str(a.project));torch.set_num_threads(4)
    from modeling.backbone import resnet
    # All parameters are loaded strictly below; do not fetch unrelated ImageNet weights.
    resnet.ResNet._load_pretrained_model=lambda self:None
    from modeling.deeplab import DeepLab
    from utils.jijie.inference import predict_logits
    from utils.jijie.postprocess import postprocess_mito_mask
    model=DeepLab(backbone='resnet',output_stride=16,num_classes=5,sync_bn=False).cuda().eval()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    sd=checkpoint.get('state_dict',checkpoint);model.load_state_dict({k.removeprefix('module.'):v for k,v in sd.items()},strict=True)
    rows=list(csv.DictReader((a.bundle/'objects.csv').open()));result=[]
    with torch.inference_mode():
        for sid in sorted(set(r['sample_id'] for r in rows)):
            image=Image.open(a.bundle/(sid+'.jpg')).convert('RGB');arr=np.array(image)
            x=(arr.astype('float32')/255-np.array([.485,.456,.406],dtype='float32'))/np.array([.229,.224,.225],dtype='float32')
            x=torch.from_numpy(x.transpose(2,0,1).copy())[None].cuda()
            logits=predict_logits(model,x,5,inference_mode='sliding',window_size=640,stride=320)
            probs=logits.softmax(1)[0].cpu().numpy();raw=probs.argmax(0).astype('uint8')
            post,stats=postprocess_mito_mask(raw.astype('int32'),gray_image=np.array(image.convert('L')),classifier_path=str(a.classifier),use_texture_refine=True)
            Image.fromarray(raw).save(a.output/(sid+'_raw.png'));Image.fromarray(post.astype('uint8')).save(a.output/(sid+'_post.png'))
            for row in [r for r in rows if r['sample_id']==sid]:
                x0,y0,x1,y1=json.loads(row['bbox']);mask=np.array(Image.open(a.bundle/row['mask_file']))>0
                pr=probs[:,y0:y1,x0:x1][:,mask];rp=raw[y0:y1,x0:x1][mask];pp=post[y0:y1,x0:x1][mask]
                r=dict(row,raw_class=int(np.bincount(rp,minlength=5).argmax()),post_class=int(np.bincount(pp,minlength=5).argmax()))
                for cl in range(5):r[f'prob_{cl}']=float(pr[cl].mean());r[f'raw_fraction_{cl}']=float((rp==cl).mean());r[f'post_fraction_{cl}']=float((pp==cl).mean())
                result.append(r)
            print('finished',sid,stats,flush=True)
            del x,logits,probs
    with (a.output/'baseline_objects.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=result[0]);w.writeheader();w.writerows(result)
    (a.output/'provenance.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),texture_classifier=str(a.classifier),texture_sha256=hashlib.sha256(a.classifier.read_bytes()).hexdigest(),n=len(result),images=len(set(r['sample_id'] for r in result)),window=640,stride=320,GT_regions_used_for_pooling=True,independent_training_history_unverified=True),indent=2))

if __name__=='__main__':main()
