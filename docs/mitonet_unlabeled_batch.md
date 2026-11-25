# MitoNet Unlabeled TEM Batch Inference

This workflow creates two review outputs for paired cardiac TEM slices:

1. MitoNet full-resolution instance segmentation without damage classification.
2. The same MitoNet instance IDs with per-instance candidates from the existing five-class DeepLabV3+ model.

The source TIFF and `slice.db` files are read-only. Input preparation uses TIFF page 0 converted to RGB JPEG for inference and writes a `samples.csv` crosswalk with source paths and SHA-256 hashes. Only paired slices under `slices/` are selected; attachments and unsupported slide formats are excluded.

## Commands

Prepare native-resolution inputs:

```bash
python tools/prepare_unlabeled_mito_batch.py --input /data/batch --output /work/batch/input
```

Run official MitoNet v1 panoptic inference:

```bash
python tools/run_mitonet_pilot.py \
  --source /models/empanada_source \
  --model /models/MitoNet_v1.pth \
  --images /work/batch/input/JPEGImages \
  --references /work/batch/input \
  --samples-csv /work/batch/input/samples.csv \
  --output /work/batch/mitonet_instances \
  --downsample 1 --device cuda
```

Classify the preserved instances with the existing DeepLabV3+ checkpoint:

```bash
python tools/run_mitonet_unlabeled_classification.py \
  --project /path/to/3thdianjing \
  --checkpoint /path/to/model_best.pth.tar \
  --images /work/batch/input/JPEGImages \
  --mitonet /work/batch/mitonet_instances \
  --samples-csv /work/batch/input/samples.csv \
  --output /work/batch/mitonet_with_class
```

Create class maps, a summary, or a representative preview with:

```bash
python tools/rebuild_mitonet_class_maps.py --instances /work/batch/mitonet_instances --table /work/batch/mitonet_with_class/instances_with_class.csv --samples /work/batch/input/samples.csv --output /work/batch/mitonet_with_class/class_maps
python tools/summarize_mitonet_unlabeled_batch.py --samples /work/batch/input/samples.csv --instances /work/batch/mitonet_instances --classification /work/batch/mitonet_with_class --output /work/batch/report
```

Editable `slice.db` copies can be generated in both variants (original TIFFs are not copied):

```bash
python tools/export_mitonet_unlabeled_to_db.py \
  --raw-root /data/batch \
  --samples /work/batch/input/samples.csv \
  --instances /work/batch/mitonet_instances \
  --classification-csv /work/batch/mitonet_with_class/instances_with_class.csv \
  --output /work/batch/editable_db --mode both
```

The DB exporter preserves source databases, adds standard editable AI marks to output copies, and emits a manifest for pairing each output DB with its source TIFF. Validate before packaging:

```bash
python tools/validate_mitonet_db_exports.py --exports /work/batch/editable_db --samples /work/batch/input/samples.csv --raw-root /data/batch --output /work/batch/editable_db/validation.json
```

## Class IDs And Interpretation

- `0`: background in the instance map; in the `review_gate` class map it also represents an uncertain object, so use the instance map and object CSV together.
- `1`: severe damage.
- `2`: moderate damage.
- `3`: healthy.
- `4`: autophagic.

The `mito_conditioned` rule assigns every predicted instance a foreground class candidate. `review_gate` is a fixed heuristic based on foreground support and top-two class margin, not calibrated confidence. These outputs are for physician review, not clinical use. Do not report segmentation or classification accuracy without ground-truth annotations.

The class maps are semantic visualizations; the NPZ instance maps and object CSV preserve instance identity. Do not recompute connected components from the semantic maps when object identity matters.
