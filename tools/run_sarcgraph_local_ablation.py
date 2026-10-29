"""Run the four local diagnostic conditions without changing existing outputs."""
import csv
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
base=root/'outputs/research'
results=[]
for source in ('annotation','prediction'):
    for gap in (0,8):
        out=base/'sarcgraph_geometry_ablation'/f'{source}_gap{gap}'
        subprocess.run([sys.executable,str(root/'tools/analyze_sarcomere_sarcgraph.py'),
            '--mask',str(base/'sarcgraph_inputs'/f'{source}.png'),
            '--image',str(base/'sarcgraph_inputs/image.jpg'),'--output',str(out),
            '--sarcgraph-source',str(base/'sarcgraph_upstream'),'--z-class','10',
            '--length-range-px','20','400','--expected-length-px','150',
            '--merge-gap-px',str(gap)],check=True)
        data=json.loads((out/'summary.json').read_text())
        results.append(dict(source=source,gap=gap,**{k:data[k] for k in (
            'source_objects','objects','virtual_merge_count','candidate_pairs',
            'direction_pass_count','bundle_geometry_pass_count')}))
with (base/'sarcgraph_geometry_ablation/ablation.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]))
    w.writeheader()
    w.writerows(results)
print(json.dumps(results,indent=2))
