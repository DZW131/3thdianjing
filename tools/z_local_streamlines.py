"""Local texture streamlines from existing Z segments; all hits are hypotheses."""
import numpy as np

def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]

def sample(field,p):
    r,c=p
    if r<0 or c<0 or r>=field.shape[0]-1 or c>=field.shape[1]-1:return None
    i,j=int(r),int(c);u,v=r-i,c-j
    return float((1-u)*(1-v)*field[i,j]+u*(1-v)*field[i+1,j]+(1-u)*v*field[i,j+1]+u*v*field[i+1,j+1])

def first_hit(a,b,segments,source):
    d=b-a;q=segments[:,0];v=segments[:,1]-q;den=cross(d,v)
    valid=np.abs(den)>1e-9;safe=np.where(valid,den,1)
    t=cross(q-a,v)/safe;s=cross(q-a,d)/safe
    valid&=(t>1e-6)&(t<=1)&(s>=0)&(s<=1)
    valid[source]=False
    ids=np.flatnonzero(valid)
    if not len(ids):return None
    ids=ids[np.argsort(t[ids])];idx=int(ids[0])
    if len(ids)>1 and abs(t[ids[1]]-t[idx])*np.linalg.norm(d)<2:return ('ambiguous_hit',None,None)
    return ('hit',idx,a+t[idx]*d)

def trace(seed,direction,fields,segments,source,max_length=400,step=3,departure_px=0):
    p=np.asarray(seed,float);prev=np.asarray(direction,float);prev/=max(np.linalg.norm(prev),1e-9)
    points=[p.copy()];travel=0.;gap=0.;supported=0.;status='max_length';target=None
    while travel+step<=max_length:
        vectors=[]
        for co,vr,vc in fields:
            confidence=sample(co,p)
            if confidence is None:status='image_boundary';break
            v=np.array([sample(vr,p),sample(vc,p)])
            if confidence>=.15 and np.linalg.norm(v)>1e-8:
                v/=np.linalg.norm(v)
                if v@prev<0:v=-v
                vectors.append(v)
        if status=='image_boundary':break
        if travel<departure_px:
            v=prev
        elif len(vectors)==len(fields) and abs(vectors[0]@vectors[-1])>=.85:
            v=np.mean(vectors,axis=0);v/=np.linalg.norm(v)
            if v@prev<np.cos(np.radians(30)):status='sharp_turn';break
            gap=0.;supported+=step
        else:
            v=prev;gap+=step
            if gap>18:status='unsupported_gap';break
        new=p+step*v
        hit=first_hit(p,new,segments,source)
        if hit:
            status,target,point=hit
            if status=='hit':
                travel+=float(np.linalg.norm(point-p));points.append(point)
                if travel<20:status='too_close_hit'
            break
        points.append(new.copy());travel+=step;p=new;prev=v
    return dict(status=status,target=target,length_px=travel,supported_fraction=min(1.,supported/max(travel,1)),points=np.asarray(points).tolist())
