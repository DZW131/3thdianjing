"""Add semantic region evidence to the ORIGINAL SarcGraph edges, never box cells."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

from pilot_fused_sarcomere_regions import common_strip


def region_evidence(start, end, probabilities, scale):
    start, end = np.asarray(start, float), np.asarray(end, float)
    delta = end-start
    distance = np.linalg.norm(delta)
    if distance < 1e-6:
        raise ValueError('Coincident Z centers')
    across = np.array([-delta[1], delta[0]])/distance
    t = np.linspace(.15, .85, max(8, int(distance)))
    points = start[:, None, None]+delta[:, None, None]*t[None, :, None]
    points = points+across[:, None, None]*np.array([-8, -4, 0, 4, 8])[None, None, :]
    coords = (points*scale).reshape(2, -1)
    sampled = np.stack([map_coordinates(channel, coords, order=1, mode='nearest')
                        for channel in probabilities]).reshape(3, len(t), 5)
    support = (sampled.argmax(0) == 1).sum(1) >= 3
    longest = current = 0
    for value in support:
        current = 0 if value else current+1
        longest = max(longest, current)
    coverage = float(support.mean())
    gap = longest/len(support)
    status = ('region_supported' if coverage >= .6 and gap <= .25 else
              'weak_region_evidence' if coverage < .2 else 'uncertain_region_evidence')
    return dict(region_status=status, corridor_coverage=coverage, max_gap_fraction=gap,
                mean_myofibril_probability=float(sampled[1].mean()),
                region_support_score=float(coverage*(1-gap)))


def edge_key(a, b):
    return tuple(sorted([int(a), int(b)]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--regions', type=Path, required=True)
    p.add_argument('--alternatives', type=Path, required=True)
    p.add_argument('--local-pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=['dev00', 'dev05'])
    p.add_argument('--scale', type=float, default=.25)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    summaries = []
    for case in args.cases:
        root = args.graphs/case/'prediction'
        nodes = pd.read_csv(root/'z_objects.csv')
        base = pd.read_csv(root/'bundle_audit.csv')
        base_keys = {edge_key(e.z1, e.z2) for e in base.itertuples()}
        centers = nodes[['x', 'y']].to_numpy(float)
        segments = nodes[['p1_x', 'p1_y', 'p2_x', 'p2_y']].to_numpy().reshape(-1, 2, 2)
        probability_path = args.regions/f'{case}_scale{args.scale}_probabilities.npy'
        probabilities = np.load(probability_path)
        image = Image.open(args.inputs/case/'image.jpg').convert('RGB')
        if probabilities.shape != (3, int(image.height*args.scale), int(image.width*args.scale)):
            raise ValueError('Region probability/coordinate shape mismatch')
        records = []
        for row in base.to_dict('records'):
            i, j = int(row['z1']), int(row['z2'])
            row.update(region_evidence(centers[i], centers[j], probabilities, args.scale))
            row.update(edge_role='original_sarcgraph', retained=True)
            records.append(row)
        proposed = set()
        alternatives = pd.read_csv(args.alternatives/f'{case}_alternative_candidates.csv')
        for row in alternatives[alternatives.normal_neighbor_selected == True].itertuples():
            proposed.add(edge_key(row.z1, row.z2))
        local_path = args.local_pairs/f'{case}_candidate_pairs.csv'
        for row in pd.read_csv(local_path).itertuples():
            proposed.add(edge_key(row.z1, row.z2))
        additions = []
        for i, j in sorted(proposed-base_keys):
            evidence = region_evidence(centers[i], centers[j], probabilities, args.scale)
            geometry_pass = common_strip(segments[i], segments[j]) is not None
            additions.append(dict(z1=i, z2=j, **evidence, geometry_pass=geometry_pass,
                                  edge_role='proposal_only_not_added', retained=False,
                                  review_candidate=bool(geometry_pass and evidence['region_status'] == 'region_supported')))
        annotated = pd.DataFrame(records)
        assert {edge_key(r['z1'], r['z2']) for r in records} == base_keys
        assert len(records) == len(base)
        annotated.to_csv(args.output/f'{case}_original_edges_with_region_evidence.csv', index=False)
        pd.DataFrame(additions, columns=['z1', 'z2', 'region_status', 'corridor_coverage',
                     'max_gap_fraction', 'mean_myofibril_probability', 'region_support_score',
                     'geometry_pass', 'edge_role', 'retained', 'review_candidate']).to_csv(
                         args.output/f'{case}_addition_proposals.csv', index=False)
        graph = dict(nodes=nodes.to_dict('records'), edges=records,
                     proposed_edges_not_in_graph=additions, coordinates='native row,column; x=row,y=column',
                     unchanged_topology=True, region_score='heuristic evidence, NOT anatomical confidence')
        (args.output/f'{case}_region_annotated_graph.json').write_text(
            json.dumps(graph, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))
        panels = []
        for mode in ['original', 'region', 'annotated']:
            canvas = image.copy()
            if mode == 'region':
                mask = Image.fromarray((probabilities.argmax(0) == 1).astype(np.uint8)*255)
                mask = np.array(mask.resize(image.size, Image.Resampling.NEAREST)) > 0
                rgb = np.array(canvas)
                rgb[mask] = (.72*rgb[mask]+.28*np.array([45, 210, 110])).astype(np.uint8)
                canvas = Image.fromarray(rgb)
            draw = ImageDraw.Draw(canvas)
            for segment in segments:
                draw.line([(p[1], p[0]) for p in segment], fill='cyan', width=3)
            for row in records:
                color = 'yellow'
                if mode == 'annotated':
                    color = ('#F35E65' if not row['bundle_geometry_pass'] else
                             '#42E77F' if row['region_status'] == 'region_supported' else '#FFCE50')
                i, j = int(row['z1']), int(row['z2'])
                draw.line([(v[1], v[0]) for v in centers[[i, j]]], fill=color, width=5)
            if mode == 'annotated':
                for row in additions:
                    if not row['review_candidate']:
                        continue
                    a, b = centers[[row['z1'], row['z2']]]
                    for t in np.arange(0, 1, .12):
                        segment = np.array([a+(b-a)*t, a+(b-a)*min(t+.06, 1)])
                        draw.line([(v[1], v[0]) for v in segment], fill='#549CFF', width=5)
            canvas.thumbnail((760, 760))
            panel = Image.new('RGB', (780, 825), 'white'); panel.paste(canvas, (10, 55))
            titles = dict(original=f'Original SarcGraph / ALL {len(base)} edges',
                          region='SAME graph + learned myofibril region',
                          annotated='SAME graph + region evidence / blue=draft additions')
            draw = ImageDraw.Draw(panel); draw.text((10, 8), titles[mode], fill='black')
            if mode == 'annotated':
                draw.text((10, 30), 'green=supported  yellow=uncertain/weak  red=geometry review', fill='black')
            panels.append(panel)
        sheet = Image.new('RGB', (2340, 825), 'white')
        for k, panel in enumerate(panels):
            sheet.paste(panel, (k*780, 0))
        sheet.save(args.output/f'{case}_graph_region_comparison.png')
        summary = dict(case=case, nodes=len(nodes), original_edges=len(base), retained_edges=len(records),
                       region_status_counts=annotated.region_status.value_counts().to_dict(),
                       original_geometry_review=int((~annotated.bundle_geometry_pass).sum()),
                       extra_pair_proposals=len(additions),
                       region_supported_extra_proposals=sum(r['review_candidate'] for r in additions),
                       topology_unchanged=True,
                       input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                     [root/'z_objects.csv', root/'bundle_audit.csv', probability_path,
                                      args.alternatives/f'{case}_alternative_candidates.csv', local_path]})
        summaries.append(summary); print(json.dumps(summary), flush=True)
    (args.output/'summary.json').write_text(json.dumps(dict(cases=summaries,
        method='Region evidence layer on original SarcGraph topology; no cell rectangles; no edge deletion',
        scale=args.scale, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parameters=dict(corridor_half_width_native_px=8, ignore_endpoint_fraction=.15,
                        supported_coverage=.6, supported_max_gap=.25, weak_coverage=.2),
        limitations=['Heuristic edge scores, not calibrated confidence', 'Region network has domain errors',
                     'No new training, topology optimization or demonstrated accuracy gain',
                     'Proposed edges need intervening-Z/crossing/branch review before actual addition',
                     'Unchanged nodes may still be Z fragments or merged Z lines',
                     'Not verified complete sarcomeres; no clinical length/angle output']), indent=2))


if __name__ == '__main__':
    main()
