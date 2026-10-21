#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dianjing5090

mkdir -p logs

TRAIN_LOG="logs/train_mito_v0_7_metassl_rdrop_semi.log"

echo "[MetaSSL] training mito R-Drop semi-supervised experiment..."
nohup python train.py \
  --config configs/mito/mito_v0_7_metassl_rdrop_semi.json \
  > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID=$!
echo "[MetaSSL] train pid=${TRAIN_PID}"
echo "[MetaSSL] log=${TRAIN_LOG}"
echo "[MetaSSL] after training finishes, evaluate with:"
cat <<'CMD'
python evaluate.py \
  --config configs/mito/mito_v0_7_metassl_rdrop_semi.json \
  --split test \
  --resume run/jijie/jijie_mito_v0_7_metassl_rdrop_semi/model_best.pth.tar \
  --save-dir outputs/eval/jijie_mito_v0_7_metassl_rdrop_semi_postprocess_um_scale \
  --postprocess-mito \
  --postprocess-classifier run/jijie/mito_texture_classifier.pkl \
  --auto-scale-bar \
  --scale-calibration-csv outputs/scale_bar/mito_test_scale_bar_measurements.csv \
  --scale-bar-um 1.0 \
  --no-progress
CMD
