"""Isolated zero-shot MitoNet inference using the official empanada render engine."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch


def read_csv(path):
    import csv
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def panoptic_to_instances(panoptic):
    if any(v != 0 and v//1000 != 1 for v in np.unique(panoptic)):
        raise ValueError("Unexpected MitoNet panoptic labels")
    values,inverse=np.unique(panoptic,return_inverse=True)
    labels=inverse.reshape(panoptic.shape)
    if values[0]!=0:
        labels=labels+1
    return labels.astype(np.int32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "images", "references", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--downsample", type=int, choices=[1,2,4], default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    p.add_argument("--samples-csv",type=Path,help="Optional full image manifest, including images without eligible reference polygons")
    a = p.parse_args()
    if a.output.exists():
        p.error("Use fresh output")
    if hashlib.md5(a.model.read_bytes()).hexdigest() != "b56fce28d877a3c35b7ca970cec73bc9":
        raise ValueError("Official model checksum mismatch")
    sys.path.insert(0,str(a.source))
    from empanada.inference.engines import PanopticDeepLabRenderEngine
    torch.set_num_threads(4)
    model = torch.jit.load(str(a.model), map_location=a.device).eval()
    engine = PanopticDeepLabRenderEngine(model,thing_list=[1],label_divisor=1000,
        stuff_area=64,void_label=0,nms_threshold=.1,nms_kernel=7,confidence_thr=.5,
        padding_factor=16,coarse_boundaries=True)
    sids = sorted({r["sample_id"] for r in read_csv(a.samples_csv or a.references/"objects.csv")})
    if a.limit is not None:
        sids = sids[:a.limit]
    a.output.mkdir(parents=True)
    provenance = dict(model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
        model_md5="b56fce28d877a3c35b7ca970cec73bc9",
        source_commit=subprocess.check_output(["git","-C",str(a.source),"rev-parse","HEAD"],text=True).strip(),
        downsample=a.downsample,official_upsampling=a.downsample,engine="PanopticDeepLabRenderEngine",
        engine_parameters=dict(thing_list=[1],label_divisor=1000,stuff_area=64,void_label=0,
                               nms_threshold=.1,nms_kernel=7,confidence_thr=.5,padding_factor=16,coarse_boundaries=True),
        normalization=dict(mean=.57571,std=.12765),torch=torch.__version__,device=a.device,
        label_source="pretrained MitoNet_v1; no cardiac fine-tuning; no damage classification",
        images={},complete=False)
    with torch.inference_mode():
        for sid in sids:
            t0=time.monotonic()
            path=a.images/(sid+".jpg")
            image=Image.open(path).convert("L");w,h=image.size
            small=image.resize((math.ceil(w/a.downsample),math.ceil(h/a.downsample)),Image.Resampling.BILINEAR)
            tensor=torch.from_numpy((np.array(small).astype("float32")/255-.57571)/.12765)[None,None].to(a.device)
            if a.device=="cuda":torch.cuda.reset_peak_memory_stats()
            panoptic=engine(tensor,(h,w),upsampling=a.downsample).squeeze().cpu().numpy().astype(np.int32)
            if panoptic.shape!=(h,w):raise ValueError("Output size mismatch")
            # Relabel IDs only: never recompute connected components on the union.
            labels=panoptic_to_instances(panoptic)
            np.savez_compressed(a.output/(sid+"_instances.npz"),labels=labels)
            provenance["images"][sid]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size=[h,w],input_size=[small.height,small.width],instances=int(labels.max()),
                foreground_pixels=int((labels>0).sum()),seconds=time.monotonic()-t0,
                peak_gpu_bytes=int(torch.cuda.max_memory_allocated()) if a.device=="cuda" else None)
            print(sid,provenance["images"][sid],flush=True)
            (a.output/"provenance.json").write_text(json.dumps(provenance,indent=2))
            del tensor,panoptic,labels
    provenance["complete"]=True
    (a.output/"provenance.json").write_text(json.dumps(provenance,indent=2))


if __name__=="__main__":
    main()
