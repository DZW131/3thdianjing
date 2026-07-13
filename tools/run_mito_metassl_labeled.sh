#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

TRAIN_LOG="logs/train_mito_v0_6_metassl_labeled.log"
EVAL_LOG="logs/eval_mito_v0_6_metassl_labeled.log"

echo "[MetaSSL] training mito labeled-only experiment..."
nohup python train.py \
  --config configs/mito/mito_v0_6_metassl_labeled.json \
  --no-progress \
  > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID=$!
echo "[MetaSSL] train pid=${TRAIN_PID}"
echo "[MetaSSL] log=${TRAIN_LOG}"
echo "[MetaSSL] after training finishes, run:"
cat <<'CMD'
python evaluate.py \
  --config configs/mito/mito_v0_6_metassl_labeled.json \
  --split test \
  --resume run/jijie/jijie_mito_v0_6_metassl_labeled/model_best.pth.tar \
  --save-dir outputs/eval/jijie_mito_v0_6_metassl_labeled_postprocess_um_scale \
  --postprocess-mito \
  --postprocess-classifier run/jijie/mito_texture_classifier.pkl \
  --auto-scale-bar \
  --scale-calibration-csv outputs/scale_bar/mito_test_scale_bar_measurements.csv \
  --scale-bar-um 1.0 \
  --no-progress
CMD
