"""Analyze existing class-ID masks without changing segmentation predictions."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops
from sarcgraph_geometry import merge_fragments, bundle_audit


def direction_audit(objects, pairs, tolerance=30.0):
    """Flag geometry only; passing does not establish anatomical membership."""
    records = []
    for e in pairs.to_dict('records'):
        a, b = objects.iloc[int(e['z1'])], objects.iloc[int(e['z2'])]
        link = np.array([b.x-a.x, b.y-a.y], dtype=float)
        axes = [np.array([r.p2_x-r.p1_x, r.p2_y-r.p1_y]) for r in (a,b)]
        if np.linalg.norm(link) == 0 or any(np.linalg.norm(v) == 0 for v in axes):
            deviations = [float('nan'), float('nan')]
            passed = False
        else:
            deviations = [float(np.degrees(np.arcsin(np.clip(abs(np.dot(v,link)) /
                (np.linalg.norm(v)*np.linalg.norm(link)),0,1)))) for v in axes]
            passed = max(deviations) <= tolerance
        e.update(normal_deviation_z1_deg=deviations[0],
                 normal_deviation_z2_deg=deviations[1], direction_pass=passed,
                 review_reason='geometry_only_not_verified' if passed else 'direction_conflict')
        records.append(e)
    return pd.DataFrame(records, columns=list(pairs.columns)+[
        'normal_deviation_z1_deg','normal_deviation_z2_deg','direction_pass','review_reason'])


def objects_from_mask(mask, class_id, min_area):
    if mask.ndim != 2:
        raise ValueError('Expected a 2D class-ID mask, not an RGB visualization.')
    rows = []
    for r in regionprops(label(mask == class_id, connectivity=2)):
        if r.area < min_area or len(r.coords) < 2:
            continue
        center = r.coords.mean(axis=0)
        _, _, vectors = np.linalg.svd(r.coords - center, full_matrices=False)
        axis = vectors[0]
        proj = (r.coords - center) @ axis
        p1, p2 = center + proj.min() * axis, center + proj.max() * axis
        rows.append(dict(frame=0, x=center[0], y=center[1],
                         p1_x=p1[0], p1_y=p1[1], p2_x=p2[0], p2_y=p2[1],
                         particle=len(rows), area_px=int(r.area),
                         touches_border=bool(np.any(r.coords == 0) or
                             np.any(r.coords[:, 0] == mask.shape[0]-1) or
                             np.any(r.coords[:, 1] == mask.shape[1]-1))))
    return pd.DataFrame(rows, columns=['frame','x','y','p1_x','p1_y',
        'p2_x','p2_y','particle','area_px','touches_border'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mask', type=Path, required=True)
    p.add_argument('--image', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sarcgraph-source', type=Path, required=True)
    p.add_argument('--z-class', type=int, required=True,
                   help='10 for original labels; 1 for remapped sarcomere labels')
    p.add_argument('--min-area', type=int, default=4)
    p.add_argument('--length-range-px', nargs=2, type=float, required=True)
    p.add_argument('--expected-length-px', type=float, required=True)
    p.add_argument('--um-per-pixel', type=float)
    p.add_argument('--direction-tolerance-deg', type=float, default=30)
    p.add_argument('--merge-gap-px', type=float, default=0, help='0 disables virtual fragment merging')
    p.add_argument('--texture-audit', action='store_true', help='Measure local image evidence at 6/12/24 pixel scales')
    p.add_argument('--compact-output', action='store_true', help='Skip the three diagnostic figures for batch runs')
    a = p.parse_args()
    lo, hi = a.length_range_px
    if not 0 <= lo < a.expected_length_px < hi or a.min_area < 2:
        p.error('Require 0 <= min < expected < max and min-area >= 2.')
    if a.um_per_pixel is not None and (not np.isfinite(a.um_per_pixel) or a.um_per_pixel <= 0):
        p.error('um-per-pixel must be positive.')
    if not 0 <= a.direction_tolerance_deg <= 90:
        p.error('direction-tolerance-deg must be between 0 and 90.')
    if not np.isfinite(a.merge_gap_px) or a.merge_gap_px < 0:
        p.error('merge-gap-px must be finite and nonnegative.')
    if a.texture_audit and a.image is None:
        p.error('--texture-audit requires --image.')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output must be new or empty; existing results are not overwritten.')
    sys.path.insert(0, str(a.sarcgraph_source.resolve()))
    from sarcgraph import SarcGraph

    class EmptySafeSarcGraph(SarcGraph):
        def _process_sarcomeres(self, graph, tracked_zdiscs):
            if graph.number_of_edges()==0:
                return pd.DataFrame(columns=['frame','sarc_id','zdiscs','x','y','length','width','angle'])
            return super()._process_sarcomeres(graph,tracked_zdiscs)
    mask = np.asarray(Image.open(a.mask))
    objects = objects_from_mask(mask, a.z_class, a.min_area)
    bg = np.asarray(Image.open(a.image).convert('RGB')) if a.image else mask
    if bg.shape[:2] != mask.shape:
        raise ValueError('Image/mask shape mismatch; resizing is not implicit.')
    a.output.mkdir(parents=True, exist_ok=True)
    original_count = len(objects)
    objects.to_csv(a.output / 'source_objects.csv', index=False)
    if a.merge_gap_px > 0:
        objects, proposals, mapping = merge_fragments(objects, gap=a.merge_gap_px)
        proposals.to_csv(a.output/'merge_proposals.csv',index=False)
        mapping.to_csv(a.output/'object_mapping.csv',index=False)
    edges = []
    if len(objects) >= 2:
        sg = EmptySafeSarcGraph(input_type='image', output_dir=str(a.output),
                       save_output=False, num_neighbors=min(3, len(objects)-1))
        # Upstream setters validate against other fields, so expand bounds first.
        sg.config.max_sarc_length = max(hi, 30.0)
        sg.config.avg_sarc_length = a.expected_length_px
        sg.config.max_sarc_length = hi
        sg.config.min_sarc_length = lo
        sarcs, graphs = sg.sarcomere_detection(tracked_zdiscs=objects.copy())
        sarcs.to_csv(a.output / 'sarcgraph_raw.csv', index=False)
        for graph in graphs:
            for u, v in graph.edges:
                id1 = int(graph.nodes[u]['particle_id'])
                id2 = int(graph.nodes[v]['particle_id'])
                length = float(np.linalg.norm(graph.nodes[u]['pos'] - graph.nodes[v]['pos']))
                edges.append(dict(z1=id1, z2=id2, length_px=length,
                    length_um=length*a.um_per_pixel if a.um_per_pixel else None,
                    status='candidate_requires_review'))
    pairs = pd.DataFrame(edges, columns=['z1','z2','length_px','length_um','status'])
    paired = {e[k] for e in edges for k in ('z1','z2')}
    objects['has_candidate_pair'] = objects.particle.isin(paired)
    objects.to_csv(a.output / 'z_objects.csv', index=False)
    pairs.to_csv(a.output / 'candidate_pairs.csv', index=False)
    audit = direction_audit(objects, pairs, a.direction_tolerance_deg)
    audit.to_csv(a.output / 'direction_audit.csv', index=False)
    bundle = bundle_audit(objects,audit)
    bundle.to_csv(a.output/'bundle_audit.csv',index=False)
    texture_counts={}
    if a.texture_audit:
        from sarcgraph_texture import texture_fields, texture_audit
        texture=texture_audit(objects,bundle,texture_fields(bg))
        texture.to_csv(a.output/'texture_audit.csv',index=False)
        texture_counts={str(k):int(v) for k,v in texture.texture_status.value_counts().items()}
    if not a.compact_output:
        save_figures(bg,mask,objects,edges,audit,bundle,a)
    result = dict(source_objects=original_count, virtual_merge_count=original_count-len(objects),
        bundle_geometry_pass_count=int(bundle.bundle_geometry_pass.sum()) if len(bundle) else 0,
        texture_status_counts=texture_counts,
        objects=len(objects), candidate_pairs=len(edges),
        unpaired_objects=len(objects)-len(paired),
        direction_pass_count=int(audit.direction_pass.sum()) if len(audit) else 0,
        direction_conflict_count=int((~audit.direction_pass).sum()) if len(audit) else 0,
        coordinate_convention='x=row,y=column',
        input_sha256=hashlib.sha256(a.mask.read_bytes()).hexdigest(),
        upstream_sg_sha256=hashlib.sha256((a.sarcgraph_source/'sarcgraph/sg.py').read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        texture_sha256=hashlib.sha256(Path(__file__).with_name('sarcgraph_texture.py').read_bytes()).hexdigest() if a.texture_audit else None,
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        warning='No clinical validation. Components may be fragments or merged Z-lines. Length is 2D center distance.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


def save_figures(bg,mask,objects,edges,audit,bundle,a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(bg, cmap='gray')
    if np.any(mask == a.z_class) and np.any(mask != a.z_class):
        ax.contour(mask == a.z_class, levels=[0.5], colors=['cyan'], linewidths=.6)
    for e in edges:
        r = objects.iloc[[e['z1'], e['z2']]]
        ax.plot(r.y, r.x, color='yellow', linewidth=1)
    for r in objects.itertuples():
        ax.text(r.y, r.x, str(r.particle), fontsize=6, color='red')
    ax.set_title('Z contours (cyan); candidate links (yellow), NOT verified sarcomeres')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(a.output / 'candidate_overlay.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(bg, cmap='gray')
    if np.any(mask == a.z_class) and np.any(mask != a.z_class):
        ax.contour(mask == a.z_class, levels=[0.5], colors=['cyan'], linewidths=.6)
    for e in audit.to_dict('records'):
        r = objects.iloc[[e['z1'], e['z2']]]
        ax.plot(r.y, r.x, color='lime' if e['direction_pass'] else 'red',
                linestyle='-' if e['direction_pass'] else '--', linewidth=1)
    ax.set_title('Direction audit: green=passes geometry; red=conflict; neither is GT')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(a.output / 'direction_overlay.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(bg,cmap='gray')
    for e in bundle.to_dict('records'):
        r = objects.iloc[[e['z1'],e['z2']]]
        ax.plot(r.y,r.x,color='lime' if e['bundle_geometry_pass'] else 'red',
                linestyle='-' if e['bundle_geometry_pass'] else '--',linewidth=1)
    for r in objects.itertuples():
        ax.plot([r.p1_y,r.p2_y],[r.p1_x,r.p2_x],color='cyan',linewidth=1)
    ax.set_title('Virtual Z objects; green=geometry pass, red=review; NOT confirmed bundles')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(a.output/'bundle_overlay.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
