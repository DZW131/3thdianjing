"""Fuse unchanged Z hypotheses with learned semantic regions; no validated instances."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.draw import polygon
from skimage.graph import route_through_array
from skimage.morphology import closing, disk, skeletonize


def common_strip(a, b):
    va, vb = a[1]-a[0], b[1]-b[0]
    va = va/max(np.linalg.norm(va), 1e-9)
    vb = vb/max(np.linalg.norm(vb), 1e-9)
    angle = np.degrees(np.arccos(np.clip(abs(va@vb), 0, 1)))
    if va@vb < 0:
        vb = -vb
    tangent = va+vb
    tangent /= max(np.linalg.norm(tangent), 1e-9)
    pa, pb = a@tangent, b@tangent
    lo, hi = max(pa.min(), pb.min()), min(pa.max(), pb.max())
    if hi-lo < 5 or angle > 30:
        return None
    def point(seg, projections, value):
        return seg[0]+(value-projections[0])/(projections[1]-projections[0])*(seg[1]-seg[0])
    aa = np.array([point(a, pa, v) for v in [lo, hi]])
    bb = np.array([point(b, pb, v) for v in [lo, hi]])
    delta = bb.mean(0)-aa.mean(0)
    distance = np.linalg.norm(delta)
    if distance < 20 or distance > 400:
        return None
    deviation = np.degrees(np.arcsin(np.clip(abs(delta@tangent)/distance, 0, 1)))
    if deviation > 35:
        return None
    return aa, bb, float(angle), float(distance)


def raster(vertices, shape):
    rr, cc = polygon(vertices[:, 0], vertices[:, 1], shape)
    mask = np.zeros(shape, bool)
    mask[rr, cc] = True
    return mask


def prepare_regions(labels, closing_radius=2):
    raw = labels == 1
    # Limited repair of region gaps, not blanket union with long Z lines.
    region = closing(raw, disk(closing_radius), mode='ignore') if closing_radius else raw.copy()
    components, _ = ndi.label(region, structure=np.ones((3, 3)))
    sizes = np.bincount(components.ravel())
    region &= sizes[components] >= 16
    components, _ = ndi.label(region, structure=np.ones((3, 3)))
    return raw, region, components, skeletonize(region)


def audit_pair(a, b, segments, exclude_ids, raw, components, skeleton, scale):
    result = dict(status='uncertain', reason='geometry_conflict', region_component=0,
                  region_coverage=0., dominant_component_fraction=0., intervening_z_count=0,
                  skeleton_interior_length_px=None, boundary_midpoint_distance_px=None)
    geometry = common_strip(a, b)
    if geometry is None:
        return result, None, None
    aa, bb, angle, distance = geometry
    vertices = np.array([aa[0], aa[1], bb[1], bb[0]])
    result.update(boundary_midpoint_distance_px=distance, parallel_angle_deg=angle)
    inner = raster(np.array([(.85*aa+.15*bb)[0], (.85*aa+.15*bb)[1],
                             (.15*aa+.85*bb)[1], (.15*aa+.85*bb)[0]])*scale, raw.shape)
    if inner.sum() < 4:
        result['reason'] = 'tiny_interior'
        return result, vertices, None
    result['region_coverage'] = float(raw[inner].mean())
    other_hits = 0
    for k, segment in enumerate(segments):
        if k in exclude_ids:
            continue
        points = np.linspace(segment[0], segment[1], max(3, int(np.linalg.norm(segment[1]-segment[0])/4)+1))*scale
        indices = np.rint(points).astype(int)
        valid = ((indices >= 0) & (indices < np.array(raw.shape))).all(1)
        if np.count_nonzero(inner[tuple(indices[valid].T)]) >= 2:
            other_hits += 1
    result['intervening_z_count'] = other_hits
    ids, counts = np.unique(components[inner], return_counts=True)
    valid = ids != 0
    if not valid.any():
        result['reason'] = 'no_region'
        return result, vertices, None
    component = int(ids[valid][np.argmax(counts[valid])])
    result['region_component'] = component
    result['dominant_component_fraction'] = float(counts[ids == component][0]/max(counts[valid].sum(), 1))
    if other_hits:
        result['reason'] = 'intervening_z'
        return result, vertices, None
    if result['region_coverage'] < .6 or result['dominant_component_fraction'] < .7:
        result['reason'] = 'weak_or_fragmented_region'
        return result, vertices, None
    corridor = raster(vertices*scale, raw.shape)
    allowed = skeleton & (components == component) & corridor
    coordinates = np.argwhere(allowed)
    if len(coordinates) < 2:
        result['reason'] = 'no_medial_skeleton'
        return result, vertices, None
    ends = np.array([.85*aa.mean(0)+.15*bb.mean(0), .15*aa.mean(0)+.85*bb.mean(0)])*scale
    anchors = []
    for end in ends:
        distances = np.linalg.norm(coordinates-end, axis=1)
        if distances.min()/scale > 32:
            result['reason'] = 'skeleton_endpoint_too_far'
            return result, vertices, None
        anchors.append(tuple(coordinates[np.argmin(distances)]))
    if anchors[0] == anchors[1]:
        result['reason'] = 'collapsed_skeleton_path'
        return result, vertices, None
    costs = np.where(allowed, 1., np.inf)
    try:
        path, cost = route_through_array(costs, anchors[0], anchors[1], fully_connected=True)
    except ValueError:
        path, cost = [], np.inf
    if not np.isfinite(cost):
        result['reason'] = 'disconnected_medial_skeleton'
        return result, vertices, None
    path = np.asarray(path, dtype=float)/scale
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    result['skeleton_interior_length_px'] = length
    if length > 1.6*distance:
        result['reason'] = 'excessive_skeleton_detour'
        return result, vertices, path
    result.update(status='region_supported_hypothesis', reason='requires_anatomical_review')
    return result, vertices, path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--nodes', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--regions', type=Path, required=True)
    p.add_argument('--local-pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=['dev00', 'dev05'])
    p.add_argument('--scale', type=float, default=.25)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    if args.scale <= 0:
        p.error('scale must be positive')
    args.output.mkdir(parents=True)
    summaries = []
    for case in args.cases:
        im = Image.open(args.inputs/case/'image.jpg').convert('RGB')
        label_path = args.regions/f'{case}_scale{args.scale}_labels.png'
        labels = np.array(Image.open(label_path))
        if labels.shape != (int(im.height*args.scale), int(im.width*args.scale)):
            raise ValueError('Coordinate scale does not match learned region raster')
        raw, region, components, skeleton = prepare_regions(labels)
        objects = pd.read_csv(args.nodes/case/'prediction/z_objects.csv')
        segments = objects[['p1_x', 'p1_y', 'p2_x', 'p2_y']].to_numpy().reshape(-1, 2, 2)
        original = pd.read_csv(args.pairs/f'{case}_alternative_candidates.csv')
        original = original[original.normal_neighbor_selected == True]
        local_path = args.local_pairs/f'{case}_candidate_pairs.csv'
        local = pd.read_csv(local_path)
        candidates = {}
        for source, frame in [('normal_neighbors', original), ('local_paths', local)]:
            for row in frame.itertuples():
                key = tuple(sorted([int(row.z1), int(row.z2)]))
                candidates.setdefault(key, []).append(source)
        records, shapes, paths = [], [], []
        for (i, j), sources in sorted(candidates.items()):
            record, vertices, path = audit_pair(segments[i], segments[j], segments, {i, j},
                                               raw, components, skeleton, args.scale)
            record.update(z1=i, z2=j, sources=';'.join(sources))
            records.append(record); shapes.append(vertices); paths.append(path)
        # Distinct cell hypotheses may share a Z boundary but not substantial interior area.
        good = [i for i, r in enumerate(records) if r['status'] == 'region_supported_hypothesis']
        masks = {i: raster(shapes[i]*args.scale, raw.shape) for i in good}
        conflicts = set()
        for index, i in enumerate(good):
            for j in good[index+1:]:
                overlap = (masks[i] & masks[j]).sum()/max(min(masks[i].sum(), masks[j].sum()), 1)
                if overlap > .2:
                    conflicts.update([i, j])
        for i in conflicts:
            records[i].update(status='uncertain', reason='overlapping_cell_hypotheses')
        final = [i for i, r in enumerate(records) if r['status'] == 'region_supported_hypothesis']
        pd.DataFrame(records).to_csv(args.output/f'{case}_candidate_audit.csv', index=False)
        component_image = Image.fromarray(components.astype(np.uint16))
        component_image.save(args.output/f'{case}_region_components.tif')
        Image.fromarray(skeleton.astype(np.uint8)*255).save(args.output/f'{case}_region_skeleton.png')
        exported = []
        instance_ids = np.zeros(raw.shape, dtype=np.uint16)
        overlap_pixels = np.zeros(raw.shape, bool)
        occupied = np.zeros(raw.shape, bool)
        for instance, i in enumerate(final, 1):
            mask = raster(shapes[i]*args.scale, raw.shape)
            overlap_pixels |= occupied & mask
            instance_ids[mask] = instance; occupied |= mask
            exported.append(dict(candidate_id=i, **records[i], vertices_rc_native=shapes[i].tolist(),
                                 medial_path_rc_native=paths[i].tolist(), side_boundaries='inferred_not_observed'))
        instance_ids[overlap_pixels] = 0
        Image.fromarray(instance_ids).save(args.output/f'{case}_candidate_cells.tif')
        (args.output/f'{case}_candidate_cells.json').write_text(json.dumps(exported, indent=2))
        panels = []
        for mode in ['regions', 'skeleton', 'cells']:
            canvas = im.copy()
            if mode == 'regions':
                mask = np.array(Image.fromarray(raw).resize(im.size, Image.Resampling.NEAREST))
                arr = np.array(canvas); arr[mask] = (.6*arr[mask]+.4*np.array([30, 210, 110])).astype(np.uint8)
                canvas = Image.fromarray(arr)
            draw = ImageDraw.Draw(canvas)
            for seg in segments:
                draw.line([(float(v[1]), float(v[0])) for v in seg], fill='cyan', width=3)
            if mode == 'skeleton':
                for y, x in np.argwhere(skeleton):
                    cx, cy = int(x/args.scale), int(y/args.scale)
                    draw.ellipse((cx-1, cy-1, cx+1, cy+1), fill='yellow')
            if mode == 'cells':
                for i in final:
                    xy = [(float(v[1]), float(v[0])) for v in shapes[i]]
                    draw.line(xy+[xy[0]], fill='lime', width=5)
                    draw.line([(float(v[1]), float(v[0])) for v in paths[i]], fill='yellow', width=4)
            canvas.thumbnail((700, 700))
            panel = Image.new('RGB', (720, 750), 'white'); panel.paste(canvas, (10, 40))
            titles = dict(regions='Z cyan + learned region green', skeleton='Z cyan + region medial skeleton yellow',
                          cells=f'{len(final)} supported hypotheses / inferred sides')
            ImageDraw.Draw(panel).text((10, 10), titles[mode], fill='black')
            panels.append(panel)
        sheet = Image.new('RGB', (2160, 750), 'white')
        for k, panel in enumerate(panels):
            sheet.paste(panel, (k*720, 0))
        sheet.save(args.output/f'{case}_fusion_comparison.png')
        summary = dict(case=case, z_nodes=len(objects), region_components=int(components.max()),
                       input_pair_candidates=len(records), supported_cell_hypotheses=len(final),
                       overlap_ambiguous_pixels=int(overlap_pixels.sum()),
                       reasons=pd.Series([r['reason'] for r in records]).value_counts().to_dict(),
                       region_input_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),
                       z_nodes_sha256=hashlib.sha256((args.nodes/case/'prediction/z_objects.csv').read_bytes()).hexdigest(),
                       pair_input_sha256=hashlib.sha256((args.pairs/f'{case}_alternative_candidates.csv').read_bytes()).hexdigest(),
                       local_pair_input_sha256=hashlib.sha256(local_path.read_bytes()).hexdigest())
        summaries.append(summary); print(json.dumps(summary), flush=True)
    (args.output/'summary.json').write_text(json.dumps(dict(cases=summaries, scale=args.scale,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='development hypotheses, NOT validated complete sarcomeres',
        coordinates='native row,column; TIFF candidate masks on scaled grid; IDs 1-based; shared polygon pixels 0',
        parameters=dict(closing_radius_grid_px=2, min_region_area_grid_px=16, region_coverage=.6,
                        dominant_component_fraction=.7, max_endpoint_snap_native_px=32,
                        max_skeleton_detour_ratio=1.6, max_candidate_interior_overlap=.2),
        limitations=['Uncertain learned myofibril masks', 'No instance/boundary GT',
                     'Closing can merge neighboring bundles', 'Skeleton follows segmentation errors',
                     'Side boundaries inferred from common Z extent, not observed membranes',
                     'Shared Z lines only analyzed on common overlap; no exhaustive local assignment',
                     'Candidates limited to existing neighbor/local-path hypotheses',
                     'No physical calibration or clinical length/alpha output']), indent=2))


if __name__ == '__main__':
    main()
