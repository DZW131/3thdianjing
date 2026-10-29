"""Exercise the optional SarcGraph pipeline using generated, non-patient data."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sarcgraph-source',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error('Output exists; choose a new directory.')
    a.output.mkdir(parents=True)
    rr,_=np.mgrid[:200,:200]
    image=(128+60*np.cos(rr/3)).astype(np.uint8)
    mask=np.zeros((200,200),dtype=np.uint8)
    for col in (30,90,150):
        mask[40:160,col:col+3]=10
        image[40:160,col:col+3]=20
    Image.fromarray(mask).save(a.output/'mask.png')
    Image.fromarray(image).save(a.output/'image.png')
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_sarcomere_sarcgraph.py')),
        '--mask',str(a.output/'mask.png'),'--image',str(a.output/'image.png'),
        '--output',str(a.output/'analysis'),'--sarcgraph-source',str(a.sarcgraph_source),
        '--z-class','10','--length-range-px','20','100','--expected-length-px','60',
        '--texture-audit'],check=True)


if __name__=='__main__':
    main()
