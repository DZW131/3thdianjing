"""Separate instance identity, damage classification and shape measurements."""
import numpy as np
from skimage.measure import regionprops


def shape_features(region, image_shape):
    area=float(region.area)
    perimeter=float(region.perimeter_crofton)
    y0,x0,y1,x1=region.bbox
    major,minor=float(region.axis_major_length),float(region.axis_minor_length)
    return dict(area_px2=area,perimeter_px=perimeter,feret_px=float(region.feret_diameter_max),
        major_axis_px=major,minor_axis_px=minor,aspect_ratio=major/minor if minor>0 else None,
        circularity=4*np.pi*area/perimeter**2 if perimeter>0 else None,
        form_factor=perimeter**2/(4*np.pi*area),bbox_width_px=x1-x0,bbox_height_px=y1-y0,
        border_truncated=bool(y0==0 or x0==0 or y1==image_shape[0] or x1==image_shape[1]))


def pool_and_measure(probs, labels):
    if probs.shape!=(5,*labels.shape) or labels.ndim!=2 or np.any(labels<0):
        raise ValueError("Expected five probabilities and nonnegative instance IDs")
    if not np.issubdtype(labels.dtype,np.integer):
        raise ValueError("Instance IDs must be integers")
    if not np.isfinite(probs).all() or (probs<0).any() or not np.allclose(probs.sum(0),1,atol=1e-4):
        raise ValueError("Invalid class probabilities")
    raw=probs.argmax(0)
    counts=np.bincount(labels.ravel())
    means=np.zeros((5,len(counts)),dtype=float)
    votes=np.zeros_like(means)
    for cl in range(5):
        means[cl]=np.bincount(labels.ravel(),weights=probs[cl].ravel(),minlength=len(counts))/np.maximum(1,counts)
        votes[cl]=np.bincount(labels[raw==cl],minlength=len(counts))/np.maximum(1,counts)
    rows=[]
    for region in regionprops(labels):
        identifier=region.label
        p=means[:,identifier]
        fg=float(p[1:].sum())
        conditional=p[1:]/max(fg,1e-12)
        order=np.sort(conditional)
        margin=float(order[-1]-order[-2])
        conditioned_class=int(p[1:].argmax()+1)
        needs_review=bool(fg<.5 or margin<.1)
        row=dict(instance_id=int(identifier),hard_vote=int(votes[:,identifier].argmax()),
                 mean_prob_all=int(p.argmax()),mito_conditioned=conditioned_class,
                 review_gate=0 if needs_review else conditioned_class,
                 needs_review=needs_review,foreground_support=fg,conditional_margin=margin)
        row.update({f"prob_{cl}":float(p[cl]) for cl in range(5)})
        row.update(shape_features(region,labels.shape))
        rows.append(row)
    return rows
