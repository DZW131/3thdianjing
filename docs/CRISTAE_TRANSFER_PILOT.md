# Cristae transfer pilot, 2026-11-03

## Scope and safety

The production DeepLab model, postprocessing and labels are unchanged. The
current damage label is an instance majority vote over the multiclass semantic
prediction, not a pre-existing independent crop classifier.

This experiment checks whether public cristae predictions are visually useful.
It does not measure damage classification improvement, and AI classes used for
selection are NOT human ground truth. No thresholds are fitted on this sample.
The sample deliberately favors large nonborder objects and is not representative
of all mitochondria, especially small or truncated ones.

## Public resource audit

- Repository: https://github.com/LMCF-IMG/Analysis_Mitochondrial_Ultrastructure_and_Morphology
- Existing local source commit: 5985ee4ceb12ae7d65eb4b3640a8fe9ddd90c5f2.
- STEP 2 share: https://owncloud.cesnet.cz/index.php/s/KLP6jyZzAU1OJdV
- Official TorchScript model: PanopticDeepLabPR_Panoptic200epochs.pth.
- Size: 220896471 bytes.
- Locally computed SHA-256: 43d53f014d6238a9b731e2bebbd0f973ff73ddce0344affc2c18a5416120ff5c.
  This is an integrity identifier, not a publisher-provided signature.
- Downloaded YAML: grayscale, mean .57571, std .12765, padding factor 128,
  label divisor 1000, semantic label 1 and instance labels 2 through 13.
- YAML contains the author's Windows model path; the pilot passes the downloaded
  model explicitly rather than editing the original configuration.
- README describes 271 original images, 3716 overlapping training patches and
  28 test images. These are author-reported, not re-counted here.
- Licensing for weights/data and hospital redistribution is not yet confirmed;
  public access is not a grant of unrestricted deployment rights.
- UCPH was not trained or incorporated in this first LMCF transfer pilot.

## Labels verified against the official poster table

The text label file only names lab01 through lab13. Descriptions below are
transcribed from Volfova_poster_IPL_2025_final.pdf, not invented from lab names.

| ID | Poster description |
|---|---|
| 0 | Model void/background index; not one of the 13 named labels |
| 1 | Outer membrane (semantic/stuff) |
| 2 | Lamellar, straight, one junction |
| 3 | Lamellar, straight, double junction |
| 4 | Lamellar, tilted, one junction |
| 5 | Lamellar, tilted, double junction |
| 6 | Lamellar, tilted, triple junction |
| 7 | Lamellar, disconnected, no junction |
| 8 | Globular, connected, one junction |
| 9 | Globular, connected, double junction |
| 10 | Globular, connected, multiple junctions |
| 11 | Globular, disconnected, no junction |
| 12 | Onion shape |
| 13 | Junction |

Thus twelve morphology/connection classes plus outer membrane produce thirteen
named labels. Class 13 must not be interpreted as a complete independent crista.
Disconnected/globular/onion categories are candidate structural descriptors,
not automatic Moderate or Severe labels. A missing prediction can also result
from poor contrast, section orientation or domain shift.

## Reproduction

Use the project's existing scientific Python environment with torch, Pillow,
numpy, scipy and scikit-image. Empanada is not needed for this limited semantic
TorchScript check. It is not a reproduction of the official panoptic pipeline.

```sh
python tools/prepare_cristae_transfer_pilot.py --project . --prediction-root outputs/doctor_review_predictions_mito_mito_sr/mito --output outputs/cristae_crops_new
python tools/run_cristae_transfer_pilot.py --model /path/to/downloaded_model.pth --crops outputs/cristae_crops_new --output outputs/cristae_semantic_new --device cpu
```

The preparation uses the existing watershed parameters (30px minimum area,
20px peak distance), selects four objects per AI class across distinct case
prefixes per class, and preserves native pixel scale. Each crop has a mask,
unmasked image and masked image. The mask is not optimized by this experiment.

The semantic runner saves raw heads, class-index semantic PNGs, overlays,
per-label area fractions, prediction confidence and pooled encoder features.
Raw heads are NOT a panoptic instance map. Encoder pooling currently includes
padding and background and must be revised before use in a trained classifier.

No intact-crista ratio, abnormal-crista percentage, damage thresholds or
classification accuracy are asserted. Pixel area density is not equivalent to
the area occupied by structurally intact cristae in published injury grading.

## Interpretation and next gate

First inspect whether predictions follow visible cristae rather than intensity
or border artifacts, compare against an official public test image, and test
physical-scale sensitivity. Confidence is NOT accuracy. Only after that gate
should physician-labeled grouped train/validation folds be used to compare
baseline instance predictions against a small auxiliary classifier. Preserve
autophagy separately; never silently map it to Severe.
