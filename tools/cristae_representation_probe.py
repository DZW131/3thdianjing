"""Frozen public encoder, supervised morphology probe and grouped cardiac pilot."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import regionprops
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','public','cardiac','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--local-states',action='store_true',help='train square-context public probe; aggregate local cardiac windows')
    a=p.parse_args()
    if a.output.exists():p.error('Use fresh output')
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    model=torch.jit.load(str(a.model),map_location='cpu').eval()
    def encode(image,mask):
        # Fixed-size pilot, not physical-scale harmonization.
        arr=np.array(image.convert('L').resize((128,128),Image.Resampling.BILINEAR)).astype('float32')/255
        mm=np.array(Image.fromarray(mask).resize((128,128),Image.Resampling.NEAREST))>0
        arr[~mm]=1
        x=torch.from_numpy((arr-.57571)/.12765)[None,None]
        with torch.inference_mode():
            levels=model.encoder(x);v=levels[-1]
            m=torch.nn.functional.interpolate(torch.from_numpy(mm.astype('float32'))[None,None],size=v.shape[-2:],mode='nearest')
            emb=((v*m).sum((-2,-1))/m.sum().clamp_min(1)).numpy()[0]
        pix=arr[mm]
        hist=np.histogram(pix,bins=16,range=(0,1),density=False)[0].astype(float);hist/=max(1,hist.sum())
        base=np.r_[hist,pix.mean(),pix.std(),mask.mean(),mask.shape[0]/mask.shape[1]]
        return emb,base
    px=[];py=[];pg=[]
    for path in sorted((a.public/'CRISTAE_manual_annotation').glob('*.png')):
        labels=np.array(Image.open(path));image=Image.open(a.public/'CRISTAE_orig_images'/(path.stem+'.tif')).convert('L')
        for r in regionprops(labels.astype('int32')):
            cl=r.label//1000
            if cl<2 or r.area<30:continue
            y0,x0,y1,x1=r.bbox
            box=(max(0,x0-12),max(0,y0-12),min(image.width,x1+12),min(image.height,y1+12))
            if a.local_states:
                size=max(64,x1-x0+24,y1-y0+24);cx=(x0+x1)//2;cy=(y0+y1)//2
                box=(max(0,cx-size//2),max(0,cy-size//2),min(image.width,cx+(size+1)//2),min(image.height,cy+(size+1)//2))
            # Include adjacent texture in morphology probe rather than isolated label pixels.
            crop=image.crop(box);mask=np.ones((crop.height,crop.width),dtype=bool)
            emb,_=encode(crop,mask);px.append(emb);py.append(cl);pg.append(path.stem)
    px=np.array(px);py=np.array(py);pg=np.array(pg)
    def classifier():return make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=42))
    public_oof=[]
    for g in np.unique(pg):
        train=pg!=g;test=~train;clf=classifier().fit(px[train],py[train]);pred=clf.predict(px[test])
        public_oof.append(dict(image=str(g),n=int(test.sum()),macro_f1=float(f1_score(py[test],pred,labels=np.unique(py),average='macro',zero_division=0))))
    probe=classifier().fit(px,py)
    import joblib
    joblib.dump(probe,a.output/'public_morphology_probe.joblib')
    meta=list(csv.DictReader((a.cardiac/'metadata/samples.csv').open(encoding='utf-8-sig')))
    folds={r['case_id']:int(r['fold']) for r in csv.DictReader((a.cardiac/'metadata/cv_fold_assignment.csv').open(encoding='utf-8-sig'))}
    rows=[];xx=[];base=[];state=[];local=[];local_std=[];yy=[];ff=[];counts={}
    for record in meta:
        sid=record['sample_id'];case=record['case_id']
        labels=np.array(Image.open(a.cardiac/'SegmentationClass'/(sid+'.png')))
        weight=np.array(Image.open(a.cardiac/'SupervisionWeight'/(sid+'.png')))
        image=Image.open(a.cardiac/'JPEGImages'/(sid+'.jpg')).convert('L')
        if labels.shape!=(image.height,image.width):raise ValueError('Size mismatch')
        for cl in [1,2,3]:
            cc,_=ndimage.label(labels==cl)
            for r in sorted(regionprops(cc),key=lambda x:-x.area):
                key=(case,cl)
                if counts.get(key,0)>=8:break
                if r.area<100 or (weight[r.slice][r.image]>=250).mean()<.95:continue
                y0,x0,y1,x1=r.bbox
                if y0==0 or x0==0 or y1==image.height or x1==image.width:continue
                crop=image.crop((x0,y0,x1,y1));emb,b=encode(crop,r.image)
                if a.local_states:
                    patches=[]
                    hh,ww=r.image.shape;tile=min(64,hh,ww)
                    ys=np.unique(np.linspace(0,hh-tile,min(4,max(1,hh//32)),dtype=int))
                    xs=np.unique(np.linspace(0,ww-tile,min(4,max(1,ww//32)),dtype=int))
                    for y in ys:
                        for x in xs:
                            mm=r.image[y:y+tile,x:x+tile]
                            if mm.mean()<.6:continue
                            v,_=encode(crop.crop((int(x),int(y),int(x+tile),int(y+tile))),mm)
                            patches.append(v)
                    if not patches:patches=[emb]
                    probabilities=probe.predict_proba(np.array(patches));local.append(probabilities.mean(0));local_std.append(probabilities.std(0))
                xx.append(emb);base.append(b);state.append(probe.predict_proba(emb[None])[0]);yy.append(cl);ff.append(folds[case]);counts[key]=counts.get(key,0)+1
                rows.append(dict(sample_id=sid,case_id=case,class_id=cl,component_id=r.label,area=int(r.area),doctor_fraction=float((weight[r.slice][r.image]>=250).mean()),fold=folds[case]))
                print('cardiac',len(rows),flush=True)
    xx=np.array(xx);base=np.array(base);state=np.array(state);yy=np.array(yy);ff=np.array(ff)
    np.savez_compressed(a.output/'features.npz',appearance=base,embedding=xx,candidate_state_probabilities=state,labels=yy,folds=ff,public_features=px,public_labels=py,public_image_groups=pg,public_state_classes=probe.classes_)
    experiments={'appearance_control':base,'appearance_plus_candidate_states':np.c_[base,state],'appearance_plus_crista_embedding':np.c_[base,xx]}
    if a.local_states:
        local=np.array(local);local_std=np.array(local_std)
        np.savez_compressed(a.output/'local_states.npz',mean=local,std=local_std,classes=probe.classes_)
        with (a.output/'local_candidate_states.csv').open('w',newline='') as handle:
            records=[dict(row,**{f'local_class_{cl}_mean':float(mu) for cl,mu in zip(probe.classes_,means)},**{f'local_class_{cl}_std':float(sd) for cl,sd in zip(probe.classes_,stds)}) for row,means,stds in zip(rows,local,local_std)]
            writer=csv.DictWriter(handle,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
        experiments['appearance_plus_local_state_mean']=np.c_[base,local]
        experiments['appearance_plus_local_state_mean_std']=np.c_[base,local,local_std]
    results=[];predrows=[]
    for name,x in experiments.items():
        pred=np.zeros(len(yy),dtype=int)
        for fold in np.unique(ff):
            tr=ff!=fold;te=~tr
            if len(np.unique(yy[tr]))<2:raise ValueError('Insufficient train classes')
            fitted=classifier().fit(x[tr],yy[tr]);pred[te]=fitted.predict(x[te])
            joblib.dump(fitted,a.output/f'{name}_fold{fold}.joblib')
        r=dict(experiment=name,n=len(yy),macro_f1=float(f1_score(yy,pred,labels=[1,2,3],average='macro',zero_division=0)),moderate_f1=float(f1_score(yy==2,pred==2,zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(yy,pred)),per_class_f1=f1_score(yy,pred,labels=[1,2,3],average=None,zero_division=0).tolist(),confusion=confusion_matrix(yy,pred,labels=[1,2,3]).tolist());results.append(r)
        for row,prediction in zip(rows,pred):predrows.append(dict(row,experiment=name,prediction=int(prediction)))
    with (a.output/'oof_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=predrows[0]);w.writeheader();w.writerows(predrows)
    with (a.output/'candidate_states.csv').open('w',newline='') as f:
        records=[dict(row,**{f'public_morphology_{cl}_prob':float(pr) for cl,pr in zip(probe.classes_,probs)}) for row,probs in zip(rows,state)]
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    (a.output/'results.json').write_text(json.dumps(dict(public_probe_cv=public_oof,public_n=len(py),public_images=len(np.unique(pg)),local_states=a.local_states,public_classes=probe.classes_.tolist(),cardiac_counts={str(k):v for k,v in counts.items()},experiments=results,limitations=['bounded public pilot only; same source as encoder training, original-image train/test history not independently audited','candidate morphology is not validated cardiac crista state','grouped OOF; not independent external test; repeated exploratory comparisons on same cardiac folds','control is handcrafted appearance, NOT original DeepLab model','components may merge adjacent same-class doctor marks','original baseline may have seen these patients; no original-model generalization claim','encoder frozen; supervised public linear representation probe only','fixed128px inputs not physically calibrated'],model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest()),indent=2))
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
