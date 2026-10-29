"""Multiscale image evidence for candidate connections in row/column coordinates."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.feature import structure_tensor


def texture_fields(image, scales=(6.,12.,24.)):
    gray=np.asarray(image,dtype=float)
    if gray.ndim==3:
        gray=gray[...,:3] @ np.array([.2126,.7152,.0722])
    lo,hi=np.percentile(gray,[1,99])
    gray=np.clip((gray-lo)/max(hi-lo,1e-12),0,1)
    rr,rc,cc=structure_tensor(gray,sigma=1,order='rc')
    fields=[]
    for scale in scales:
        a,b,c=(gaussian_filter(v,scale) for v in (rr,rc,cc))
        delta=np.sqrt((a-c)**2+4*b*b)
        coherence=delta/(a+c+1e-12)
        theta=.5*np.arctan2(2*b,a-c)
        # Direction of minimum gradient variation (texture tangent), row/column.
        fields.append(tuple(v.astype(np.float32) for v in (coherence,-np.sin(theta),np.cos(theta))))
    return fields


def texture_audit(objects,pairs,fields,coherence_threshold=.2):
    records=[]
    moments=[(coherence,vr*vr,vr*vc,vc*vc) for coherence,vr,vc in fields]
    for e in pairs.to_dict('records'):
        a,b=objects.iloc[int(e['z1'])],objects.iloc[int(e['z2'])]
        start=np.array([a.x,a.y])
        delta=np.array([b.x-a.x,b.y-a.y])
        distance=np.linalg.norm(delta)
        if distance<1e-9:
            raise ValueError('Coincident centers are not measurable links.')
        along=delta/distance
        normal=np.array([-along[1],along[0]])
        t=np.linspace(.15,.85,max(8,min(512,int(distance))))
        offsets=np.linspace(-3,3,5)
        points=start[:,None,None]+delta[:,None,None]*t[None,:,None]+normal[:,None,None]*offsets[None,None,:]
        coords=points.reshape(2,-1)
        scores,coverages=[],[]
        for coherence,rr,rc,cc in moments:
            weight=map_coordinates(coherence,coords,order=1,mode='nearest')
            # Interpolate sign-invariant alignment, not signed eigenvectors.
            alignment=(map_coordinates(rr,coords,order=1,mode='nearest')*along[0]**2+
                       2*map_coordinates(rc,coords,order=1,mode='nearest')*along[0]*along[1]+
                       map_coordinates(cc,coords,order=1,mode='nearest')*along[1]**2)
            valid=weight>=coherence_threshold
            coverages.append(float(valid.mean()))
            scores.append(float(np.average(alignment[valid],weights=weight[valid])) if valid.any() else float('nan'))
        valid_scores=[s for s,v in zip(scores,coverages) if v>=.4 and np.isfinite(s)]
        if len(valid_scores)<2:
            status='uncertain_texture'
        elif min(valid_scores)>=.65:
            status='texture_supported'
        elif max(valid_scores)<=.35:
            status='texture_conflict'
        else:
            status='uncertain_texture'
        e.update(texture_status=status,texture_alignment=float(np.mean(valid_scores)) if valid_scores else float('nan'),
                 texture_coverage=float(np.mean(coverages)))
        for i,(s,c) in enumerate(zip(scores,coverages)):
            e[f'texture_scale{i}_alignment']=s
            e[f'texture_scale{i}_coverage']=c
        records.append(e)
    extra=['texture_status','texture_alignment','texture_coverage']
    extra += [f'texture_scale{i}_{name}' for i in range(len(fields)) for name in ('alignment','coverage')]
    return pd.DataFrame(records,columns=list(pairs.columns)+extra)
