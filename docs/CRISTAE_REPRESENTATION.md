# Cristae representation and damage-classification pilot

Updated 2026-11-16; experiments executed 2026-11-12. These independent scripts
do not replace production inference, change masks, or edit physician labels.
The goal is instance-level structural information, not improving crista pixel IoU.

## Resources and dependencies

See [resource and label audit](CRISTAE_TRANSFER_PILOT.md). Public images are
downloaded from the author's LMCF STEP 2 share. Eleven paired images yielded
458 annotated crista instances in the expanded pilot. Labels represented were
2,3,4,5,6,7,8,10,11; absent categories must not be treated as validated.
The encoder is the author's frozen TorchScript model. Only a linear supervised
morphology probe is learned, not a new neural encoder or segmenter.

Use Python 3.10+ and install the appropriate torch build separately. The local
CPU probes ran with torch 2.12.1+cpu and sklearn 1.7.1; full-image baseline
inference ran in the existing server environment. A clean environment rebuild
has not been independently verified. Do not change a working GPU environment
just to match the CPU version.

```bash
python -m pip install -r requirements-cristae.txt
python -m unittest discover -s tests -v
```

Download model and YAML from the public share linked in the audit. Expected
weight file is `PanopticDeepLabPR_Panoptic200epochs.pth`, 220896471 bytes,
SHA-256 `43d53f014d6238a9b731e2bebbd0f973ff73ddce0344affc2c18a5416120ff5c`.
Weights are NOT uploaded here. Licensing for redistribution/hospital deployment
is not established; public access is not deployment authorization. No patient
image, mask, private manifest, per-object CSV or fitted probe is published.

## Data contract

Public data downloader creates `CRISTAE_orig_images/*.tif` and matching
`CRISTAE_manual_annotation/*.png`. Panoptic ID // 1000 is the semantic class;
0 remains background. Private cardiac data require `JPEGImages/*.jpg`,
`SegmentationClass/*.png`, `SupervisionWeight/*.png`, and
`metadata/{samples.csv,cv_fold_assignment.csv}`. These prepared private files
must be obtained from the project owner. Weight pixels >=250 indicate strong
doctor supervision in the current dataset; this encoding must be rechecked for
new datasets. Class 1/2/3 = severe/moderate/healthy; autophagy is excluded rather
than relabeled as severe. Public shape labels are not clinical damage labels.

## Reproduce the local-state experiment

Run from repository root; replace paths with authorized local files. Outputs
must be fresh directories. These commands were checked against argparse; the
underlying experiments ran with equivalent paths, not from a fresh clone.

```bash
python tools/fetch_cristae_public_pilot.py --output outputs/cristae_public --count 11
python tools/cristae_representation_probe.py --model /path/to/official_model.pth --public outputs/cristae_public --cardiac /path/to/prepared_revised52 --output outputs/cristae_states --local-states
```

Public probe uses square context around annotated crista instances. Cardiac
regions use at most 4x4 local windows (up to 64 native pixels, >=60% foreground).
Windows are encoded at 128px; morphology probabilities are aggregated as mean
and standard deviation. These are candidate public morphology descriptions,
NOT validated cardiac intactness/disorganization labels. Physical resolution
harmonization remains unfinished. `features.npz`, `local_states.npz`, readable
`local_candidate_states.csv`, OOF predictions, results and fitted fold models
are private outputs. Never unpickle untrusted joblib files.

## Reproduce baseline-information fusion

```bash
python tools/prepare_cristae_baseline_bundle.py --states outputs/cristae_states --cardiac /path/to/prepared_revised52 --output outputs/cristae_private_bundle
# Transfer private_bundle.tar.gz only through authorized channels, then extract:
tar -xzf outputs/cristae_private_bundle/private_bundle.tar.gz -C outputs/cristae_private_bundle
python tools/cristae_baseline_objects.py --project . --bundle outputs/cristae_private_bundle/baseline_bundle --checkpoint /path/to/original_baseline.pth.tar --classifier /path/to/mito_texture_classifier.pkl --output outputs/cristae_baseline_objects
python tools/audit_cristae_postprocess_stages.py --project . --bundle outputs/cristae_private_bundle/baseline_bundle --predictions outputs/cristae_baseline_objects --output outputs/cristae_unification.csv
python tools/evaluate_cristae_baseline_fusion.py --baseline outputs/cristae_baseline_objects --states outputs/cristae_states --output outputs/cristae_fusion
```

Baseline inference requires CUDA and the existing project's model dependencies;
640 window / 320 stride, original normalization and existing weighted-logit
sliding inference are used. The backbone's unnecessary ImageNet download is
disabled in-process; the actual checkpoint is loaded strictly. Model and texture
classifier hashes are recorded. Production files are untouched.
The bundle helper is newly packaged from the local preparation procedure;
syntax/help and synthetic packaging checks are used, not a repeat of GPU work.

Fusion pools original probabilities and raw/post label fractions on the same
101 doctor-dominant regions. Calibration and fusion heads use identical C=.1,
balanced logistic regression; scaling is fit only on training cases in each
fold. Exactly 101 regions are required by this fixed pilot evaluator; a new
cohort needs a separately audited protocol, not silent modification of the test.

## Verified aggregate results

Sources: private `cristae_local_representation_20261112/results.json` and
`cristae_baseline_fusion_20261112/{comparison.csv,raw_positive_subset.csv}`,
generated by the above scripts on 2026-11-12. Only aggregate values are released.

| Method | Macro-F1 | Moderate F1 |
|---|---:|---:|
| Handcrafted appearance control | .5762 | .4706 |
| Appearance + local state mean | .6588 | .5714 |
| Appearance + local mean/std | .6565 | .6111 |
| Original DeepLab raw, fixed regions | .5571 | .5079 |
| Existing postprocessing, fixed regions | .3401 | .1026 |
| Baseline-information calibrator | .5558 | .4384 |
| Baseline information + local mean | .6016 | .5263 |
| Baseline information + local mean/std | .6968 | .6087 |

On the fixed 64-region subset originally assigned H/M/S rather than background
or autophagy: Moderate F1 raw=.6531, fusion=.6957. The postprocessing reduction
is specific to this selected revised cohort, not a claim that historical
postprocessing never helps. No production setting was changed.

## Boundaries and next work

- GT regions define objects: this is conditional classification, NOT end-to-end
  detection/segmentation improvement. Missed objects cannot receive GT crops in
  deployment. Connected regions may merge adjacent same-class mitochondria.
- Six case prefixes / five folds, 101 selected regions (25 severe,35 moderate,
  41 healthy); large/complete/doctor-dominant selection introduces bias.
- Repeated exploratory comparisons reuse the same folds; independent new cases
  are needed before claiming generalization. Baseline/texture training overlap
  with these cases has not been excluded; raw predictions are not OOF training.
- Frozen public encoder may have seen its public source data. Public leave-image
  probe results do not establish independently pretrained encoder generalization.
- No cardiac crista-state GT, clinical cutoffs, or hospital deployment validation.
- Optional `run_cristae_transfer_pilot.py` and `benchmark_cristae_states.py`
  retain earlier semantic diagnostics, not the primary optimization target.
- Synthetic tests verify code contracts, not medical accuracy. Next: freeze the
  proposed method, validate new cases, then test predicted-instance deployment.
