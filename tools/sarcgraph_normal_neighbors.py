"""Alternative spatial hypotheses to diagnose neighbors lost by graph pruning."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sarcgraph_geometry import axis


def normal_neighbors(objects,min_length=20.,max_length=400.,ambiguity_ratio=1.15):
    from analyze_sarcomere_sarcgraph import direction_audit
    from sarcgraph_geometry import bundle_audit
    columns=['z1','z2','length_px','length_um','status']
    if len(objects)<2:
        return pd.DataFrame(columns=columns+['normal_neighbor_selected','ambiguous_endpoint'])
    centers=objects[['x','y']].to_numpy(float)
    records=[]
    for i,j in sorted(cKDTree(centers).query_pairs(max_length)):
        length=float(np.linalg.norm(centers[i]-centers[j]))
        if length>=min_length:
            records.append(dict(z1=i,z2=j,length_px=length,length_um=None,status='alternative_requires_review'))
    audit=bundle_audit(objects,direction_audit(objects,pd.DataFrame(records,columns=columns)))
    eligible=audit[audit.bundle_geometry_pass].copy()
    choices={}
    ambiguous=set()
    for i in range(len(objects)):
        v=axis(objects.iloc[i])
        normal=np.array([-v[1],v[0]])
        for side in (-1,1):
            entries=[]
            for index,e in eligible.iterrows():
                if i not in (int(e.z1),int(e.z2)):
                    continue
                other=int(e.z2) if int(e.z1)==i else int(e.z1)
                direction=float((centers[other]-centers[i])@normal)
                if direction*side>0:
                    entries.append((float(e.length_px),index))
            entries.sort()
            if entries:
                choices[(i,side)]=entries[0][1]
                if len(entries)>1 and entries[1][0]/max(entries[0][0],1e-12)<ambiguity_ratio:
                    ambiguous.add((i,side))
    counts={}
    uncertain_edges=set()
    for key,index in choices.items():
        counts[index]=counts.get(index,0)+1
        if key in ambiguous:
            uncertain_edges.add(index)
    eligible['normal_neighbor_selected']=[counts.get(i,0)==2 and i not in uncertain_edges for i in eligible.index]
    eligible['ambiguous_endpoint']=[i in uncertain_edges for i in eligible.index]
    return eligible.reset_index(drop=True)
