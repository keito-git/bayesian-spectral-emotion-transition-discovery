#!/usr/bin/env bash
# Reproduce the main BSETD results on EmotionLines.
# Pre-bundled outputs in experiments/stage{1,2}_emotionlines/ should match the
# re-generated outputs to within floating-point tolerance.

set -euo pipefail

cd "$(dirname "$0")"

echo "[1/3] Stage 1: Hierarchical DM posterior on EmotionLines ..."
python -m bsetd.stage1_dirichlet \
    --input data_processed/emotionlines_softlabels_v2_bsetd.parquet \
    --out  experiments/stage1_emotionlines/

echo "[2/3] Stage 2: Symmetrized spectral decomposition ..."
python -m bsetd.stage2_spectral \
    --stage1-npz experiments/stage1_emotionlines/stage1_total.npz \
    --out        experiments/stage2_emotionlines/

echo "[3/3] Synthetic ground-truth recovery (144 configurations) ..."
python -m bsetd.synthetic_ablation

echo "Done. Compare experiments/stage{1,2}_emotionlines/*.json with the bundled outputs."
