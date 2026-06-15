#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INPUT_EVENTS="${1:-events.ndjson}"
WEAK_LABELED_OUT="${2:-results/events_weak_labeled.ndjson}"
STREAM_OUT="${3:-results/stream_events.ndjson}"
NUM_STREAMS="${4:-100000}"
EVENTS_PER_STREAM="${5:-100}"
SEED="${6:-42}"
TABULAR_EPISODES="${7:-1000}"
DEEP_EPISODES="${8:-1000}"
EVAL_EPISODES="${9:-100}"

mkdir -p results checkpoints

echo "[1/3] Weak label events -> ${WEAK_LABELED_OUT}"
pipenv run python -m threat_agent.stream_labeler \
  --input "$INPUT_EVENTS" \
  --output "$WEAK_LABELED_OUT" \
  --summary-json results/events_weak_label_summary.json

echo "[2/3] Build stream episodes -> ${STREAM_OUT}"
pipenv run python -m threat_agent.stream_builder \
  --input "$WEAK_LABELED_OUT" \
  --output "$STREAM_OUT" \
  --summary-json results/stream_summary.json \
  --num-streams "$NUM_STREAMS" \
  --events-per-stream "$EVENTS_PER_STREAM" \
  --seed "$SEED"

echo "[3/3] Run stream comparison experiments"
pipenv run python -m threat_agent.stream_experiment_compare \
  --stream-data "$STREAM_OUT" \
  --seed "$SEED" \
  --tabular-episodes "$TABULAR_EPISODES" \
  --deep-episodes "$DEEP_EPISODES" \
  --eval-episodes "$EVAL_EPISODES"

echo "Done."
echo "Summary JSON: results/stream_compare_summary.json"
echo "Summary CSV : results/stream_compare_summary.csv"
