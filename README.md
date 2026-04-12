# Jijie DeepLabV3+ Segmentation

## What This Repository Is For

This repository trains and evaluates a DeepLabV3+ semantic segmentation model on the custom `jijie` dataset.

The current project standardizes four things that were previously scattered across ad hoc scripts:

- one training entrypoint: `train.py`
- one evaluation entrypoint: `evaluate.py`
- one standard `jijie` split policy
- one documented server workflow

## Recommended Environment

Recommended server environment:

- OS: Linux server with NVIDIA GPU
- Python: 3.10 or 3.11
- CUDA: install the PyTorch wheel that matches your CUDA version

Install PyTorch first from the official selector on [pytorch.org](https://pytorch.org/get-started/locally/), then install the rest:

```bash
pip install -r requirements.txt
```

This code uses `torchvision`, `Pillow`, `matplotlib`, `seaborn`, and TensorBoard logging.

## Repository Structure

```text
.
├── train.py                         # Standard training entrypoint
├── evaluate.py                      # Standard evaluation / visualization entrypoint
├── train_jijie.txt                  # Example jijie training command
├── mypath.py                        # Dataset root resolution
├── dataloaders/
│   ├── __init__.py                  # Standard dataloader package entry
│   ├── custom_transforms.py         # Resize / crop / normalization transforms
│   └── datasets/
│       └── jijie.py                 # jijie dataset definition
├── modeling/
│   └── ...                          # DeepLabV3+ backbone / ASPP / decoder
├── utils/
│   ├── checkpoint.py                # Robust checkpoint load/save helpers
│   ├── metrics.py                   # Pixel accuracy / IoU metrics
│   ├── saver.py                     # Run directory management
│   └── summaries.py                 # TensorBoard logging helpers
├── tools/
│   └── prepare_jijie_dataset.py     # Build data/jijie from raw tif + sqlite exports
└── data/
    └── jijie/
        ├── JPEGImages/
        ├── SegmentationClass/
        ├── ImageSets/Segmentation/
        └── metadata/
```

## Jijie Data Organization

Prepared `jijie` data is expected at:

```text
data/jijie/
├── JPEGImages/
├── SegmentationClass/
├── ImageSets/Segmentation/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── train_all_usable.txt
│   ├── val_all_usable.txt
│   └── test_all_usable.txt
└── metadata/
    ├── summary.json
    ├── class_mapping.json
    ├── samples.csv
    └── duplicates_removed.csv
```

### Standard Split Policy

The standard project default is:

- split profile: `annotated`
- source files: `train.txt`, `val.txt`, `test.txt`
- split rule: group-aware `70 / 15 / 15`
- training set only contains non-zero annotated masks by default

If you explicitly want to include zero-mask candidates, switch to:

```bash
--split-profile all_usable
```

### Standard Size Handling

The `jijie` dataset contains mixed resolutions. The standardized default is:

- `--resize-mode pad`
- `--crop-size 512`

That means every image is resized so its longest side becomes `512`, then padded to a `512 x 512` square. This keeps batching stable on the server without relying on every raw image having the same shape.

Other available modes:

- `pad`: resize longest side and pad to square
- `crop`: DeepLab-style random scale crop for training and center crop for evaluation
- `resize`: direct square resize
- `none`: no size normalization

## Preparing the Jijie Dataset

The raw exporter format is `tif + sqlite db`. To rebuild the prepared dataset:

```bash
python tools/prepare_jijie_dataset.py \
  --input-root /path/to/raw_exports \
  --output-root ./data/jijie \
  --seed 3407 \
  --force
```

The output metadata file `data/jijie/metadata/summary.json` records:

- scanned samples
- deduplicated samples
- usable samples
- annotated samples
- zero-mask candidates
- split counts

## Training

### Standard Full-Class Training

```bash
python train.py \
  --dataset jijie \
  --backbone resnet \
  --workers 4 \
  --epochs 150 \
  --batch-size 8 \
  --gpu-ids 0 \
  --checkname jijie_resnet_pad512 \
  --lr 0.007 \
  --crop-size 512 \
  --resize-mode pad \
  --split-profile annotated \
  --eval-interval 1
```

### Class-Filtered Training

`jijie.py` supports remapping a selected subset of original class ids into a compact label space.

Example:

```bash
python train.py \
  --dataset jijie \
  --backbone resnet \
  --workers 4 \
  --epochs 150 \
  --batch-size 8 \
  --gpu-ids 0 \
  --checkname jijie_subset_resnet \
  --lr 0.007 \
  --crop-size 512 \
  --resize-mode pad \
  --split-profile annotated \
  --selected-classes 1,2,3,4,5,7,8,10,11,12,13
```

### Training Artifacts

Training outputs are written under:

```text
run/<dataset>/<checkname>/
├── model_best.pth.tar
└── experiment_<id>/
    ├── checkpoint.pth.tar
    ├── best_pred.txt
    ├── parameters.txt
    └── events.out.tfevents...
```

## Evaluation

Use the standardized evaluation entrypoint:

```bash
python evaluate.py \
  --dataset jijie \
  --split test \
  --backbone resnet \
  --resume run/jijie/jijie_resnet_pad512/model_best.pth.tar \
  --test-batch-size 1 \
  --crop-size 512 \
  --resize-mode pad \
  --split-profile annotated \
  --save-dir outputs/eval/jijie_resnet_pad512
```

To also save visualization panels:

```bash
python evaluate.py \
  --dataset jijie \
  --split test \
  --backbone resnet \
  --resume run/jijie/jijie_resnet_pad512/model_best.pth.tar \
  --test-batch-size 1 \
  --crop-size 512 \
  --resize-mode pad \
  --split-profile annotated \
  --save-dir outputs/eval/jijie_resnet_pad512 \
  --visualize \
  --num-vis-samples 20
```

## Evaluation Outputs

`evaluate.py` saves:

- `metrics_summary.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `visualizations/sample_*.png` when `--visualize` is enabled

## Metrics

The standard reported metrics are:

- pixel accuracy
- mean class accuracy
- mIoU
- frequency-weighted IoU
- per-class IoU / precision / recall

## Visualization

Each saved visualization panel contains:

- original image
- ground-truth mask
- predicted mask
- original + ground truth overlay
- original + prediction overlay
- prediction error map

## Results Table Template

Use this table to track server runs:

| Run Name | Dataset / Split | Model | Key Changes | Dice | IoU | Precision | Recall | Specificity | Accuracy | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| jijie_resnet_pad512 | jijie / test | DeepLabV3+ ResNet | Standard pad-512 baseline | - | - | - | - | - | - | Fill after server evaluation |

## Suggested Server Workflow

1. Prepare or copy the dataset to the server.
2. Create the Python environment and install PyTorch plus `requirements.txt`.
3. Run `python train.py ...` with a named `--checkname`.
4. After training, evaluate `model_best.pth.tar` with `python evaluate.py ...`.
5. Archive the run folder and the evaluation output folder together.

## Local Validation That Was Added In This Refactor

- package import path standardized to `from dataloaders import make_data_loader`
- evaluation logic unified into `evaluate.py`
- checkpoint loading now accepts both wrapped and unwrapped state dicts
- `jijie` split selection is explicit through `--split-profile`
- `jijie` mixed image sizes are normalized with a documented policy

## Known Limits

- Full training still needs to be run on the server; this repository does not include a completed training result.
- The default model remains DeepLabV3+; this refactor does not change the core architecture.
- Non-`jijie` datasets were left close to their original structure unless needed for package cleanup.
