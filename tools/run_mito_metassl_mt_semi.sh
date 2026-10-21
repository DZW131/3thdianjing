#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-dianjing5090}"

cd "$PROJECT_DIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

mkdir -p logs
TRAIN_LOG="logs/train_mito_v0_8_metassl_mt_semi.log"

python train.py \
  --config configs/mito/mito_v0_8_metassl_mt_semi.json \
  2>&1 | tee "$TRAIN_LOG"

echo
echo "Training log: $TRAIN_LOG"
echo "Evaluate with:"
echo "python evaluate.py \\"
echo "  --config configs/mito/mito_v0_8_metassl_mt_semi.json \\"
echo "  --split test \\"
echo "  --resume run/jijie/jijie_mito_v0_8_metassl_mt_semi/model_best.pth.tar \\"
echo "  --save-dir outputs/eval/jijie_mito_v0_8_metassl_mt_semi_postprocess_um_scale \\"
echo "  --postprocess-mito \\"
echo "  --scale-bar-mode auto"
