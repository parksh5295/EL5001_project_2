#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EVTX_ROOT="${1:-evtx_samples}"
DATASET_OUT="${2:-results/threat_agent_data.json}"
SEED="${3:-42}"
TABULAR_EPISODES="${4:-3000}"
DEEP_EPISODES="${5:-2000}"
EVAL_EPISODES="${6:-100}"

mkdir -p results checkpoints

echo "[1/2] Build dataset -> ${DATASET_OUT}"
pipenv run python threat_agent/build_dataset.py \
  --evtx-root "$EVTX_ROOT" \
  --evtx-lib-dir "$EVTX_ROOT/EVTX_ATT&CK_Metadata" \
  -o "$DATASET_OUT"

echo "[2/2] Run all comparison experiments"
pipenv run python -m threat_agent.experiment_compare \
  --dataset "$DATASET_OUT" \
  --seed "$SEED" \
  --tabular-episodes "$TABULAR_EPISODES" \
  --deep-episodes "$DEEP_EPISODES" \
  --eval-episodes "$EVAL_EPISODES"

echo "Done."
echo "Summary JSON: results/compare_summary.json"
echo "Summary CSV : results/compare_summary.csv"
