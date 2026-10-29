"""Compare graph hypotheses and export image crops for reviewing real anatomy."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sarcgraph_normal_neighbors import normal_neighbors
from sarcgraph_texture import texture_fields, texture_audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error('Output exists; select a fresh directory.')
    manifest=json.loads((a.inputs/'manifest.json').read_text())
    a.output.mkdir(parents=True)
    (a.output/'review_crops').mkdir()
    summaries=[]
    review=[]
    for sample in manifest['samples']:
        dev=sample['development_id']
        image=np.asarray(Image.open(a.inputs/dev/'image.jpg').convert('RGB'))
        objects=pd.read_csv(a.results/dev/'prediction/z_objects.csv')
        original=pd.read_csv(a.results/dev/'prediction/texture_audit.csv')
        proposals=normal_neighbors(objects)
        alternate=texture_audit(objects,proposals,texture_fields(image))
        alternate.to_csv(a.output/f'{dev}_alternative_candidates.csv',index=False)
        selected=alternate[alternate.normal_neighbor_selected==True]
        original_keys={tuple(sorted((int(e.z1),int(e.z2)))) for e in original.itertuples()}
        selected_keys={tuple(sorted((int(e.z1),int(e.z2)))) for e in selected.itertuples()}
        summaries.append(dict(development_id=dev,original_pairs=len(original),
            original_geometry_pass=int(original.bundle_geometry_pass.sum()),
            normal_proposals=len(alternate),normal_selected=len(selected),
            new_selected=len(selected_keys-original_keys),
            selected_texture_supported=int((selected.texture_status=='texture_supported').sum()),
            selected_texture_conflict=int((selected.texture_status=='texture_conflict').sum()),
            selected_texture_uncertain=int((selected.texture_status=='uncertain_texture').sum())))
        fig,axes=plt.subplots(1,2,figsize=(14,7))
        for ax,table,title in zip(axes,[original,selected],['SarcGraph candidates','Mutual normal neighbors']):
            ax.imshow(image)
            for e in table.itertuples():
                points=objects.iloc[[int(e.z1),int(e.z2)]]
                color={'texture_supported':'lime','texture_conflict':'red','uncertain_texture':'gold'}[e.texture_status]
                if title.startswith('SarcGraph') and not e.bundle_geometry_pass:
                    color='silver'
                ax.plot(points.y,points.x,color=color,linewidth=1)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'{dev}: image evidence (green=support, red=conflict, yellow=uncertain); unverified',fontsize=10,y=.99)
        fig.tight_layout(rect=(0,0,1,.94))
        fig.savefig(a.output/f'{dev}_comparison.png',dpi=150)
        plt.close(fig)
        # Deterministic stratified review: one proposed link per texture status per case.
        for status in ('texture_supported','texture_conflict','uncertain_texture'):
            subset=selected[selected.texture_status==status].sort_values(['length_px','z1','z2'])
            if subset.empty:
                continue
            e=subset.iloc[len(subset)//2]
            ids=[int(e.z1),int(e.z2)]
            points=objects.iloc[ids]
            all_points=points[['x','y']].to_numpy()
            x0,y0=np.maximum(0,np.floor(all_points.min(axis=0)-80)).astype(int)
            x1,y1=np.minimum(image.shape[:2],np.ceil(all_points.max(axis=0)+80)).astype(int)
            name=f'{dev}_{status}_{ids[0]}_{ids[1]}.png'
            fig,ax=plt.subplots(figsize=(6,6))
            ax.imshow(image[x0:x1,y0:y1],extent=(y0,y1,x1,x0))
            ax.plot(points.y,points.x,color='yellow',linewidth=1.5)
            for r in points.itertuples():
                ax.plot([r.p1_y,r.p2_y],[r.p1_x,r.p2_x],color='cyan')
                ax.text(r.y,r.x,str(r.particle),color='white',fontsize=9)
            ax.set_title(f'{dev} Z{ids[0]} - Z{ids[1]} | {status}',fontsize=10)
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(a.output/'review_crops'/name,dpi=140)
            plt.close(fig)
            review.append(dict(development_id=dev,z1=ids[0],z2=ids[1],
                texture_status=status,length_px=float(e.length_px),crop=f'review_crops/{name}',
                endpoints_are_real_z='',same_myofibril='',adjacent_z='',reviewer_comment=''))
        print(json.dumps(summaries[-1]),flush=True)
    pd.DataFrame(summaries).to_csv(a.output/'alternative_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(review).to_csv(a.output/'review_form.csv',index=False,encoding='utf-8-sig')
    overview=pd.read_csv(a.results/'development_summary.csv')
    totals=overview.groupby('source').sum(numeric_only=True)
    totals.to_csv(a.output/'aggregate_diagnostics.csv',encoding='utf-8-sig')
    text='# Eight-case development diagnostic\n\n'
    text+='Selection: first eligible image per sorted case. No independent pair reference labels.\n'
    text+='Pixel ranges and texture scales are fixed development heuristics; physical scale has not been calibrated.\n\n'
    text+='Original paired-mask diagnostic totals:\n\n'+totals.to_string()+'\n\n'
    text+='Alternative normal-neighbor graph totals:\n\n'+pd.DataFrame(summaries).sum(numeric_only=True).to_string()+'\n\n'
    text+='Green image evidence does not establish anatomy. Silver links fail geometry. '
    text+='Alternative proposals search all nodes within 20..400px, apply geometry checks, '
    text+='and select mutual nearest candidates on each normal side. A second candidate '
    text+='within a 1.15 distance ratio flags ambiguity. This can reject genuine branches.\n\n'
    text+='review_form.csv contains deterministic examples from each available texture status per case. '
    text+='Review fields are intentionally empty pending anatomy review. No accuracy, true sarcomere length, '
    text+='or alpha-angle claim can be made from these diagnostics.\n'
    (a.output/'README.md').write_text(text,encoding='utf-8')
    import hashlib
    import importlib.metadata
    import platform
    provenance=dict(python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in (
        'numpy','pandas','scipy','scikit-image','matplotlib','networkx','trackpy','sk-video')},
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
        'report_sarcgraph_development.py','sarcgraph_normal_neighbors.py','sarcgraph_texture.py','sarcgraph_geometry.py')})
    (a.output/'environment_and_sources.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
