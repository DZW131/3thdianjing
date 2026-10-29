"""Recompute morphology from postprocessed full-image masks for doctor review."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import regionprops


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--prediction-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error('Output exists; choose a new directory.')
    sys.path.insert(0,str(a.project/'tools'))
    from export_prediction_masks_to_db import split_mito_instances
    source=json.loads((a.prediction_root/'summary.json').read_text())
    if source.get('postprocess_mito') is not True:
        raise ValueError('Explicit postprocess_mito=true provenance is required.')
    masks=sorted((a.prediction_root/'review_dataset/SegmentationClass').glob('*.png'))
    images=a.prediction_root/'review_dataset/JPEGImages'
    a.output.mkdir(parents=True)
    classes={1:'严重损伤',2:'中度损伤',3:'健康',4:'自噬'}
    rows=[]
    coverage=[]
    for index,path in enumerate(masks):
        m=np.asarray(Image.open(path))
        if m.ndim!=2 or not set(np.unique(m)).issubset({0,1,2,3,4}):
            raise ValueError(f'Invalid original-ID mask {path.name}')
        image=images/(path.stem+'.jpg')
        if not image.is_file() or Image.open(image).size != (m.shape[1],m.shape[0]):
            raise ValueError(f'Image missing or mismatched: {path.stem}')
        class_mask,labels,stats=split_mito_instances(m,min_area=30,min_distance=20)
        before=len(rows)
        for r in regionprops(labels):
            sl=r.slice
            binary=r.image
            values=class_mask[sl][binary]
            cid=int(np.bincount(values,minlength=5)[1:].argmax()+1)
            perimeter=float(r.perimeter_crofton)
            area=int(r.area)
            minr,minc,maxr,maxc=r.bbox
            border=minr==0 or minc==0 or maxr==m.shape[0] or maxc==m.shape[1]
            major,minor=float(r.axis_major_length),float(r.axis_minor_length)
            rows.append(dict(sample_id=path.stem,case_id=path.stem.split('__')[0],class_id=cid,
                class_name=classes[cid],object_id=int(r.label),area_px2=area,
                perimeter_px=perimeter,feret_px=float(r.feret_diameter_max),
                major_axis_px=major,minor_axis_px=minor,
                aspect_ratio=major/minor if minor>0 else None,
                circularity=4*np.pi*area/perimeter**2 if perimeter>0 else None,
                form_factor=perimeter**2/(4*np.pi*area),bbox_width_px=maxc-minc,
                bbox_height_px=maxr-minr,border_truncated=bool(border)))
        retained=sum(x['area_px2'] for x in rows[before:])
        coverage.append(dict(sample_id=path.stem,case_id=path.stem.split('__')[0],
            image_area_px2=int(m.size),foreground_before_px2=int((m>0).sum()),
            retained_area_px2=retained,object_count=len(rows)-before,
            discarded_foreground_px2=int((m>0).sum())-retained,
            mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if (index+1)%50==0:
            print(f'{index+1}/{len(masks)} images; {len(rows)} objects',flush=True)
    objects=pd.DataFrame(rows)
    image_df=pd.DataFrame(coverage)
    objects.to_csv(a.output/'objects.csv.gz',index=False,compression='gzip')
    image_df.to_csv(a.output/'image_coverage.csv',index=False)
    aggregates=[]
    for level,key in [('全量',None),('病例','case_id'),('图像','sample_id')]:
        groups=[('全部',objects,image_df)] if key is None else [
            (str(v),objects[objects[key]==v],image_df[image_df[key]==v]) for v in image_df[key].unique()]
        for group,part,imgs in groups:
            for cid,name in classes.items():
                selected=part[part.class_id==cid]
                count=len(selected)
                area=int(selected.area_px2.sum())
                aggregates.append(dict(统计层级=level,分组=group,类别=name,图像数=len(imgs),对象数=count,
                    数量比例=count/len(part) if len(part) else None,面积合计_px2=area,
                    占线粒体面积比例=area/part.area_px2.sum() if len(part) else None,
                    占全图面积比例=area/imgs.image_area_px2.sum(),
                    面积均值_px2=selected.area_px2.mean(),面积中位数_px2=selected.area_px2.median(),
                    周长均值_px=selected.perimeter_px.mean(),Feret均值_px=selected.feret_px.mean(),
                    主轴均值_px=selected.major_axis_px.mean(),次轴均值_px=selected.minor_axis_px.mean(),
                    长宽比均值=selected.aspect_ratio.mean(),圆度均值=selected.circularity.mean(),
                    形态因子均值=selected.form_factor.mean(),边缘截断对象数=int(selected.border_truncated.sum())))
    summary=pd.DataFrame(aggregates)
    for level,name in [('全量','overall'),('病例','cases'),('图像','images')]:
        summary[summary['统计层级']==level].to_csv(a.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    assert int(objects.area_px2.sum())==int(image_df.retained_area_px2.sum())
    assert len(objects)==int(image_df.object_count.sum())
    provenance=dict(prediction_summary=source,image_count=len(masks),case_count=int(image_df.case_id.nunique()),
        objects=len(objects),min_area_px=30,watershed_min_distance_px=20,
        retained_area_px2=int(objects.area_px2.sum()),
        border_objects=int(objects.border_truncated.sum()),
        perimeter_definition='skimage Crofton perimeter, four directions',
        axis_definition='equivalent ellipse axes from second central moments',
        feret_definition='skimage maximum convex hull Feret diameter',
        physical_scale='not applied; no assumed default micrometer calibration',
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        instance_script_sha256=hashlib.sha256((a.project/'tools/export_prediction_masks_to_db.py').read_bytes()).hexdigest())
    (a.output/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    import tarfile
    with tarfile.open(str(a.output)+'.tar.gz','w:gz') as tar:
        for path in a.output.iterdir():
            tar.add(path,arcname=path.name)
    print(json.dumps(provenance,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
