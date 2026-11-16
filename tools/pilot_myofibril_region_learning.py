"""Independent region-learning pilot; published CellSMB outputs are NOT manual GT."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageSequence
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1),
                         nn.GroupNorm(4, outgoing), nn.ReLU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1),
                         nn.GroupNorm(4, outgoing), nn.ReLU())


class RegionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = Block(1, 8)
        self.enc2 = Block(8, 16)
        self.enc3 = Block(16, 32)
        self.bottleneck = Block(32, 64)
        self.dec3 = Block(96, 32)
        self.dec2 = Block(48, 16)
        self.dec1 = Block(24, 8)
        self.head = nn.Conv2d(8, 3, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        d = self.bottleneck(F.max_pool2d(c, 2))
        d = self.dec3(torch.cat([F.interpolate(d, size=c.shape[-2:], mode='nearest'), c], 1))
        d = self.dec2(torch.cat([F.interpolate(d, size=b.shape[-2:], mode='nearest'), b], 1))
        d = self.dec1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='nearest'), a], 1))
        return self.head(d)


def read_stack(path):
    with Image.open(path) as image:
        return np.stack([np.array(frame.copy()) for frame in ImageSequence.Iterator(image)])


def load_pairs(root):
    raw = read_stack(root / 'raw.tif')
    masks = []
    for structure in ['myofibrils', 'mitochondria']:
        cells = []
        for cell in [1, 2]:
            original = read_stack(root / f'cell{cell}-{structure}-substack.tif')
            if original.shape != (raw.shape[0], raw.shape[1]*2, raw.shape[2]*2):
                raise ValueError('Unexpected public mask layout')
            cells.append(np.stack([np.array(Image.fromarray(m).resize(
                (raw.shape[2], raw.shape[1]), Image.Resampling.NEAREST)) > 0 for m in original]))
        masks.append(cells[0] | cells[1])
    labels = np.zeros(raw.shape, dtype=np.uint8)
    labels[masks[0]] = 1
    labels[masks[1]] = 2
    labels[masks[0] & masks[1]] = 255
    return raw, labels


def normalize(image, percentiles=None):
    lo, hi = np.percentile(image, [1, 99]) if percentiles is None else percentiles
    return np.clip((image.astype(np.float32) - lo) / max(float(hi-lo), 1), 0, 1).astype(np.float32)


def starts(length, tile, stride):
    return sorted(set(list(range(0, max(length-tile+1, 1), stride)) + [max(length-tile, 0)]))


@torch.no_grad()
def predict(model, image, tile=128, stride=96):
    model.eval()
    h, w = image.shape
    padded = np.pad(image, ((0, max(0, tile-h)), (0, max(0, tile-w))), mode='edge')
    total = np.zeros((3, *padded.shape), dtype=np.float32)
    count = np.zeros(padded.shape, dtype=np.float32)
    for y in starts(padded.shape[0], tile, stride):
        for x in starts(padded.shape[1], tile, stride):
            sample = torch.from_numpy(padded[y:y+tile, x:x+tile].copy())[None, None].float()
            prob = model(sample).softmax(1)[0].numpy()
            total[:, y:y+tile, x:x+tile] += prob
            count[y:y+tile, x:x+tile] += 1
    return (total / count[None])[:, :h, :w]


def confusion(target, predicted):
    valid = target != 255
    return np.bincount((target[valid].astype(int)*3 + predicted[valid]), minlength=9).reshape(3, 3)


def scores(matrix):
    output = {}
    for k, name in enumerate(['other', 'myofibril', 'mitochondria']):
        tp = int(matrix[k, k]); gt = int(matrix[k].sum()); pred = int(matrix[:, k].sum())
        output[name] = dict(iou=tp/max(gt+pred-tp, 1), dice=2*tp/max(gt+pred, 1),
                            recall=tp/max(gt, 1), precision=tp/max(pred, 1), gt_pixels=gt)
    return output


def overlay(image, labels):
    rgb = np.repeat((np.clip(image, 0, 1)*255).astype(np.uint8)[..., None], 3, -1)
    for k, color in [(1, [30, 210, 110]), (2, [235, 90, 170])]:
        mask = labels == k
        rgb[mask] = (0.55*rgb[mask] + 0.45*np.array(color)).astype(np.uint8)
    return Image.fromarray(rgb)


def panel(images, titles, path):
    width = 620
    height = max(round(im.height*width/im.width) for im in images)
    canvas = Image.new('RGB', (width*len(images), height+36), 'white')
    draw = ImageDraw.Draw(canvas)
    for k, (im, title) in enumerate(zip(images, titles)):
        draw.text((k*width+8, 10), title, fill='black')
        im = im.resize((width, round(im.height*width/im.width)))
        canvas.paste(im, (k*width, 36))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--private-inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    raw, labels = load_pairs(args.data)
    # Keep all adjacent planes in the same spatial partition; leave a 256px XY gap.
    train_end, validation_start = 1280, 1536
    norm_limits = np.percentile(raw[:, :, :train_end], [1, 99]).tolist()
    images = normalize(raw, norm_limits)
    model = RegionUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    log = []
    start_time = time.time()
    for step in range(1, args.steps+1):
        model.train()
        xx, yy = [], []
        for _ in range(4):
            z = rng.integers(len(raw)); y = rng.integers(raw.shape[1]-128+1)
            x = rng.integers(train_end-128+1)
            im = images[z, y:y+128, x:x+128]
            lab = labels[z, y:y+128, x:x+128]
            rotation = int(rng.integers(4))
            im, lab = np.rot90(im, rotation), np.rot90(lab, rotation)
            if rng.random() < .5:
                im, lab = im[:, ::-1], lab[:, ::-1]
            im = np.clip(im*rng.uniform(.8, 1.2)+rng.uniform(-.1, .1), 0, 1)
            xx.append(im.copy()); yy.append(lab.copy())
        x = torch.from_numpy(np.stack(xx))[:, None].float()
        y = torch.from_numpy(np.stack(yy)).long()
        logits = model(x)
        ce = F.cross_entropy(logits, y, ignore_index=255)
        valid = y != 255
        onehot = F.one_hot(y.clamp(max=2), 3).permute(0, 3, 1, 2)*valid[:, None]
        prob = logits.softmax(1)*valid[:, None]
        axes = (0, 2, 3)
        dice = (2*(prob*onehot).sum(axes)+1)/(prob.sum(axes)+onehot.sum(axes)+1)
        loss = ce + (1-dice[1:]).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        log.append(dict(step=step, loss=float(loss.detach())))
        if step % 50 == 0:
            print(f'step={step} loss={log[-1]["loss"]:.4f} seconds={time.time()-start_time:.1f}', flush=True)
    torch.save(dict(model=model.state_dict(), classes=['other', 'myofibril', 'mitochondria'],
                    normalization=norm_limits, seed=args.seed, steps=args.steps), args.output/'region_pilot.pt')
    matrix = np.zeros((3, 3), dtype=np.int64)
    rows = []
    for z in range(len(raw)):
        image = images[z, :, validation_start:]
        truth = labels[z, :, validation_start:]
        probability = predict(model, image)
        prediction = probability.argmax(0).astype(np.uint8)
        current = confusion(truth, prediction); matrix += current
        for name, metric in scores(current).items():
            rows.append(dict(slice=z, class_name=name, **metric))
        if z == 5:
            panel([overlay(image, np.zeros_like(truth)), overlay(image, truth), overlay(image, prediction)],
                  ['Public raw / spatial holdout', 'Published output (NOT manual GT)', 'Learned region prediction'],
                  args.output/'public_spatial_holdout.png')
    private = []
    for case in ['dev00', 'dev05']:
        original = Image.open(args.private_inputs/case/'image.jpg').convert('L')
        panels = [original.convert('RGB')]; titles = ['Private TEM / no myofibril GT']
        for scale in [.25, .125]:
            image = normalize(np.array(original.resize((int(original.width*scale), int(original.height*scale)), Image.Resampling.BILINEAR)))
            probability = predict(model, image)
            prediction = probability.argmax(0).astype(np.uint8)
            np.save(args.output/f'{case}_scale{scale}_probabilities.npy', probability)
            Image.fromarray(prediction).save(args.output/f'{case}_scale{scale}_labels.png')
            panels.append(overlay(image, prediction)); titles.append(f'Scale {scale}: green=myof / pink=mito')
            private.append(dict(case=case, scale=scale, myofibril_fraction=float((prediction == 1).mean()),
                                mitochondria_fraction=float((prediction == 2).mean()), evaluation='qualitative_only'))
        panel(panels, titles, args.output/f'{case}_region_comparison.png')
    checkpoint = args.output/'region_pilot.pt'
    manifest = dict(task='public-output distillation / feasibility pilot, NOT pretrained author model',
                    label_provenance='CellSMB published segmentation outputs; not confirmed manual GT',
                    source='https://github.com/CellSMB/sbfsem-cardiac-cell-segmenter-v1',
                    spatial_partition=dict(train_x=[0, train_end], unused_x=[train_end, validation_start],
                                           holdout_x=[validation_start, raw.shape[2]], all_z=list(range(len(raw)))),
                    limitations=['Single volume, correlated planes; not independent case-level validation',
                                 '2x nearest-neighbor mask downsample, registration requires anatomical review',
                                 'No private myofibril GT; scales exploratory, not physical calibration',
                                 'Semantic mask does not establish myofibril instance or Z-line membership'],
                    classes=['other', 'myofibril', 'mitochondria'], normalization=norm_limits,
                    ignored_overlap_pixels=int((labels == 255).sum()), confusion=matrix.tolist(),
                    spatial_holdout_agreement=scores(matrix), private=private, steps=args.steps,
                    seed=args.seed, torch_version=str(torch.__version__), seconds=time.time()-start_time,
                    checkpoint=dict(bytes=checkpoint.stat().st_size, sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
                    inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data.glob('*.tif')})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    aggregate = [dict(class_name=name, **metric) for name, metric in scores(matrix).items()]
    for filename, records in [('training_loss.csv', log), ('spatial_holdout_per_slice.csv', rows),
                              ('spatial_holdout_summary.csv', aggregate)]:
        with (args.output/filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    print(json.dumps(manifest['spatial_holdout_agreement']), flush=True)


if __name__ == '__main__':
    main()
