"""Conservative geometry hypotheses; never modifies the source mask."""
import numpy as np
import pandas as pd


def endpoints(row):
    return np.array([[row.p1_x, row.p1_y], [row.p2_x, row.p2_y]], dtype=float)


def axis(row):
    v = endpoints(row)[1] - endpoints(row)[0]
    return v / max(np.linalg.norm(v), 1e-12)


def merge_fragments(objects, gap=8., lateral=2., angle=15.):
    """Merge only disjoint pairs with unique compatible partners (no chains)."""
    proposals = []
    for i in range(len(objects)):
        a = objects.iloc[i]
        va = axis(a)
        if np.linalg.norm(endpoints(a)[1]-endpoints(a)[0]) < 5:
            continue
        for j in range(i+1, len(objects)):
            b = objects.iloc[j]
            vb = axis(b)
            if np.linalg.norm(endpoints(b)[1]-endpoints(b)[0]) < 5:
                continue
            parallel = np.degrees(np.arccos(np.clip(abs(va@vb),0,1)))
            if parallel > angle:
                continue
            ea, eb = endpoints(a), endpoints(b)
            endpoint_gap = float(np.linalg.norm(ea[:,None]-eb[None,:],axis=2).min())
            center_delta = np.array([b.x-a.x,b.y-a.y])
            offset = max(abs(v[0]*center_delta[1]-v[1]*center_delta[0]) for v in (va,vb))
            # Projected intervals must not overlap: avoid merging parallel adjacent lines.
            pa, pb = sorted(ea@va), sorted(eb@va)
            overlap = min(pa[1],pb[1])-max(pa[0],pb[0])
            if endpoint_gap <= gap and offset <= lateral and overlap <= 0:
                proposals.append(dict(source1=i,source2=j,gap_px=endpoint_gap,
                                      lateral_px=float(offset),angle_deg=float(parallel)))
    counts = {}
    for e in proposals:
        for k in ('source1','source2'):
            counts[e[k]] = counts.get(e[k],0)+1
    partners = {}
    for e in proposals:
        e['accepted'] = counts[e['source1']]==counts[e['source2']]==1
        if e['accepted']:
            partners[e['source1']] = e['source2']
            partners[e['source2']] = e['source1']
    rows, mapping, seen = [], [], set()
    for i in range(len(objects)):
        if i in seen:
            continue
        ids = [i,partners[i]] if i in partners else [i]
        seen.update(ids)
        parts = objects.iloc[ids]
        row = parts.iloc[0].to_dict()
        if len(ids)>1:
            points = np.concatenate([endpoints(r) for r in parts.itertuples()])
            center = np.average(parts[['x','y']],axis=0,weights=parts.area_px)
            _,_,v = np.linalg.svd(points-points.mean(axis=0),full_matrices=False)
            projections = (points-center)@v[0]
            e1,e2 = center+projections.min()*v[0],center+projections.max()*v[0]
            row.update(x=center[0],y=center[1],p1_x=e1[0],p1_y=e1[1],
                       p2_x=e2[0],p2_y=e2[1],area_px=int(parts.area_px.sum()),
                       touches_border=bool(parts.touches_border.any()))
        row['particle'] = len(rows)
        mapping.extend(dict(source_id=s,merged_id=len(rows)) for s in ids)
        rows.append(row)
    return (pd.DataFrame(rows,columns=objects.columns),
            pd.DataFrame(proposals,columns=['source1','source2','gap_px','lateral_px','angle_deg','accepted']),
            pd.DataFrame(mapping,columns=['source_id','merged_id']))


def bundle_audit(objects, audit, min_overlap=.25, max_parallel_angle=30.):
    records = []
    for e in audit.to_dict('records'):
        a,b = objects.iloc[int(e['z1'])],objects.iloc[int(e['z2'])]
        va,vb = axis(a),axis(b)
        angle = float(np.degrees(np.arccos(np.clip(abs(va@vb),0,1))))
        if va@vb < 0:
            vb = -vb
        v = va+vb
        v /= max(np.linalg.norm(v),1e-12)
        pa,pb = sorted(endpoints(a)@v), sorted(endpoints(b)@v)
        overlap = max(0.,min(pa[1],pb[1])-max(pa[0],pb[0]))
        ratio = overlap/max(min(pa[1]-pa[0],pb[1]-pb[0]),1e-12)
        passed = bool(e['direction_pass'] and angle<=max_parallel_angle and ratio>=min_overlap)
        e.update(parallel_angle_deg=angle,lateral_overlap_fraction=float(ratio),
                 bundle_geometry_pass=passed,
                 bundle_reason='geometry_only_not_verified' if passed else 'geometry_conflict')
        records.append(e)
    return pd.DataFrame(records,columns=list(audit.columns)+[
        'parallel_angle_deg','lateral_overlap_fraction','bundle_geometry_pass','bundle_reason'])
