"""Run paired-mask development diagnostics; no training or reference pair labels."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sarcgraph-source',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error('Output exists; use a new directory.')
    manifest=json.loads((a.inputs/'manifest.json').read_text())
    a.output.mkdir(parents=True)
    results=[]
    for sample in manifest['samples']:
        for source in ('annotation','prediction'):
            dev=sample['development_id']
            folder=a.output/dev/source
            command=[sys.executable,str(Path(__file__).with_name('analyze_sarcomere_sarcgraph.py')),
                     '--mask',str(a.inputs/dev/f'{source}.png'),
                     '--image',str(a.inputs/dev/'image.jpg'),'--output',str(folder),
                     '--sarcgraph-source',str(a.sarcgraph_source),'--z-class','10',
                     '--length-range-px','20','400','--expected-length-px','150',
                     '--texture-audit','--compact-output']
            process=subprocess.run(command,capture_output=True,text=True)
            (folder.parent/f'{source}.log').write_text(process.stdout+process.stderr,encoding='utf-8')
            if process.returncode:
                raise RuntimeError(f'{dev}/{source} failed; see log')
            summary=json.loads((folder/'summary.json').read_text())
            record=dict(development_id=dev,source=source,**{k:summary[k] for k in (
                'objects','candidate_pairs','direction_pass_count','bundle_geometry_pass_count')})
            record.update({k:summary['texture_status_counts'].get(k,0) for k in (
                'texture_supported','texture_conflict','uncertain_texture')})
            results.append(record)
            print(json.dumps(record),flush=True)
    with (a.output/'development_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    (a.output/'protocol.json').write_text(json.dumps(dict(
        inputs=str(a.inputs),selection=manifest['selection'],cases=len(manifest['samples']),
        length_parameters_px=[20,150,400],physical_scale_verified=False,
        note='Development diagnostics; texture status is not anatomical truth. No pair accuracy available.'),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
