"""Compare center pairs to independent local streamlines, without modifying Z masks."""
import argparse,json,csv,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from sarcgraph_texture import texture_fields
from z_local_streamlines import trace

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--nodes',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['dev00','dev05']);p.add_argument('--departure-px',type=float,default=0);a=p.parse_args()
    if a.output.exists():p.error('Use fresh output')
    a.output.mkdir(parents=True);summaries=[]
    for dev in a.cases:
        im=Image.open(a.inputs/dev/'image.jpg').convert('RGB')
        objects=pd.read_csv(a.nodes/dev/'prediction/z_objects.csv')
        segments=objects[['p1_x','p1_y','p2_x','p2_y']].to_numpy().reshape(-1,2,2)
        fields=texture_fields(np.array(im),scales=(6.,12.));results=[];edge_seeds=defaultdict(list)
        for source,seg in enumerate(segments):
            tangent=seg[1]-seg[0]
            if np.linalg.norm(tangent)<5:continue
            normal=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
            for fraction in [.2,.5,.8]:
                for side in [-1,1]:
                    r=trace(seg[0]+fraction*tangent,normal*side,fields,segments,source,departure_px=a.departure_px)
                    r.update(source=source,fraction=fraction,side=side);results.append(r)
                    if r['status']=='hit':edge_seeds[tuple(sorted([source,r['target']]))].append(r)
        baseline=pd.read_csv(a.pairs/f'{dev}_alternative_candidates.csv');baseline=baseline[baseline.normal_neighbor_selected==True]
        oldkeys={tuple(sorted([int(e.z1),int(e.z2)])) for e in baseline.itertuples()}
        edges=[]
        for (i,j),rs in edge_seeds.items():
            edges.append(dict(z1=i,z2=j,trace_count=len(rs),reciprocal=len({r['source'] for r in rs})==2,already_in_center_baseline=(i,j) in oldkeys,mean_path_length_px=float(np.mean([r['length_px'] for r in rs])),review_same_bundle='',review_adjacent=''))
        with (a.output/f'{dev}_candidate_pairs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(edges[0]) if edges else ['z1','z2']);w.writeheader();w.writerows(edges)
        (a.output/f'{dev}_traces.json').write_text(json.dumps(results))
        panels=[]
        for local in [False,True]:
            canvas=im.copy();d=ImageDraw.Draw(canvas)
            for seg in segments:d.line([(p[1],p[0]) for p in seg],fill='cyan',width=2)
            if local:
                for r in results:
                    if r['status']=='hit':d.line([(p[1],p[0]) for p in r['points']],fill='#35DC70',width=2)
            else:
                for e in baseline.itertuples():
                    nodes=objects.iloc[[int(e.z1),int(e.z2)]];d.line(list(zip(nodes.y,nodes.x)),fill='white',width=2)
            canvas.thumbnail((760,760));panel=Image.new('RGB',(780,810),'white');panel.paste(canvas,(10,40));ImageDraw.Draw(panel).text((10,10),dev+(': local texture hits, NOT validated bundles' if local else ': center-based baseline pairs'),fill='black');panels.append(panel)
        joined=Image.new('RGB',(1560,810),'white');joined.paste(panels[0],(0,0));joined.paste(panels[1],(780,0));joined.save(a.output/f'{dev}_comparison.png')
        statuses=Counter(r['status'] for r in results)
        summary=dict(case=dev,nodes=len(objects),center_baseline_pairs=len(baseline),traces=len(results),hit_traces=statuses.get('hit',0),local_candidate_pairs=len(edges),reciprocal_pairs=sum(e['reciprocal'] for e in edges),new_pairs=sum(not e['already_in_center_baseline'] for e in edges),statuses=dict(statuses))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    (a.output/'summary.json').write_text(json.dumps(summaries,indent=2))
    (a.output/'provenance.json').write_text(json.dumps(dict(status='development hypotheses only; unchanged source nodes/masks; no actual muscle-region segmentation',parameters=dict(scales=[6,12],coherence=.15,max_scale_angle_cos=.85,max_turn_deg=30,max_gap_px=18,min_hit_length_px=20,max_length_px=400,departure_px=a.departure_px,seeds=[.2,.5,.8],sides=[-1,1]),source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['pilot_z_local_paths.py','z_local_streamlines.py','sarcgraph_texture.py']}),indent=2))

if __name__=='__main__':main()
