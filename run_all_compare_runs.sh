#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PIPENV_VENV_IN_PROJECT=1

if python3 -m pipenv --version >/dev/null 2>&1; then
  PIPENV=(python3 -m pipenv)
elif command -v pipenv >/dev/null 2>&1; then
  PIPENV=(pipenv)
else
  echo "pipenv를 찾을 수 없습니다." >&2
  echo "먼저 실행하세요: python3 -m pip install --user pipenv" >&2
  exit 1
fi

INPUT_EVENTS="events.ndjson"
WEAK_LABELED_OUT="results/events_weak_labeled.ndjson"
RUNS_SUMMARY_OUT="results/confident_runs_summary.json"
RUNS_OUT="results/confident_runs.ndjson"
STREAM_OUT="results/stream_events_runs.ndjson"
NUM_STREAMS="3000"
EVENTS_PER_STREAM="120"
SPLIT_MODE="source"
SPLIT_RATIO="0.7,0.15,0.15"
MIN_CONFIDENCE="medium"
USE_TACTICS=""
ATTACK_RUNS_MAX="3"
INTRA_RUN_BENIGN_PROB="0.15"
SEED="42"
TABULAR_EPISODES="1000"
DEEP_EPISODES="1000"
EVAL_EPISODES="30"
DECISION_STRIDE="1"
SKIP_BUILD="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-events) INPUT_EVENTS="$2"; shift 2 ;;
    --weak-labeled-out) WEAK_LABELED_OUT="$2"; shift 2 ;;
    --runs-out) RUNS_OUT="$2"; shift 2 ;;
    --stream-out) STREAM_OUT="$2"; shift 2 ;;
    --num-streams) NUM_STREAMS="$2"; shift 2 ;;
    --events-per-stream) EVENTS_PER_STREAM="$2"; shift 2 ;;
    --split-mode) SPLIT_MODE="$2"; shift 2 ;;
    --split-ratio) SPLIT_RATIO="$2"; shift 2 ;;
    --min-confidence) MIN_CONFIDENCE="$2"; shift 2 ;;
    --use-tactics) USE_TACTICS="$2"; shift 2 ;;
    --attack-runs-max) ATTACK_RUNS_MAX="$2"; shift 2 ;;
    --intra-run-benign-prob) INTRA_RUN_BENIGN_PROB="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --tabular-episodes) TABULAR_EPISODES="$2"; shift 2 ;;
    --deep-episodes) DEEP_EPISODES="$2"; shift 2 ;;
    --eval-episodes) EVAL_EPISODES="$2"; shift 2 ;;
    --decision-stride) DECISION_STRIDE="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD="true"; shift 1 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p results checkpoints

stream_dir="$(dirname "$STREAM_OUT")"
stream_base="$(basename "$STREAM_OUT")"
if [[ "$stream_base" == *.* ]]; then
  stream_stem="${stream_base%.*}"
  stream_ext=".${stream_base##*.}"
else
  stream_stem="$stream_base"
  stream_ext=""
fi
TRAIN_STREAM_OUT="${stream_dir}/${stream_stem}_train${stream_ext}"
VAL_STREAM_OUT="${stream_dir}/${stream_stem}_val${stream_ext}"
TEST_STREAM_OUT="${stream_dir}/${stream_stem}_test${stream_ext}"

if [[ ! -d ".venv" ]]; then
  echo "[prep] .venv가 없어 setup_pipenv.sh 실행"
  bash "$ROOT_DIR/setup_pipenv.sh"
fi

if ! "${PIPENV[@]}" run python -c "import numpy, torch, six, hexdump" >/dev/null 2>&1; then
  echo "[prep] .venv 의 핵심 패키지가 없어 setup_pipenv.sh 재실행"
  bash "$ROOT_DIR/setup_pipenv.sh"
fi

if [[ "$SKIP_BUILD" != "true" ]]; then
  echo "[1/4] Weak label events -> ${WEAK_LABELED_OUT}"
  "${PIPENV[@]}" run python -m threat_agent.stream_labeler \
    --input "$INPUT_EVENTS" \
    --output "$WEAK_LABELED_OUT" \
    --summary-json results/events_weak_label_summary.json

  echo "[2/4] Extract confident runs -> ${RUNS_OUT}"
  "${PIPENV[@]}" run python -m threat_agent.extract_confident_runs \
    --input "$WEAK_LABELED_OUT" \
    --output-json "$RUNS_SUMMARY_OUT" \
    --output-ndjson "$RUNS_OUT"

  echo "[3/4] Build run-based stream episodes -> ${STREAM_OUT}"
  "${PIPENV[@]}" run python -m threat_agent.stream_builder_from_runs \
    --events-input "$WEAK_LABELED_OUT" \
    --runs-input "$RUNS_OUT" \
    --output "$STREAM_OUT" \
    --summary-json results/stream_runs_summary.json \
    --num-streams "$NUM_STREAMS" \
    --events-per-stream "$EVENTS_PER_STREAM" \
    --split-mode "$SPLIT_MODE" \
    --split-ratio "$SPLIT_RATIO" \
    --min-confidence "$MIN_CONFIDENCE" \
    --use-tactics "$USE_TACTICS" \
    --attack-runs-max "$ATTACK_RUNS_MAX" \
    --intra-run-benign-prob "$INTRA_RUN_BENIGN_PROB" \
    --seed "$SEED"
else
  if [[ ! -f "$STREAM_OUT" ]]; then
    echo "--skip-build was specified, but stream data file not found: $STREAM_OUT" >&2
    exit 1
  fi
  if [[ ! -f "$TRAIN_STREAM_OUT" || ! -f "$VAL_STREAM_OUT" || ! -f "$TEST_STREAM_OUT" ]]; then
    echo "--skip-build was specified, but split stream files are missing." >&2
    echo "expected: $TRAIN_STREAM_OUT, $VAL_STREAM_OUT, $TEST_STREAM_OUT" >&2
    exit 1
  fi
  echo "[skip] Reusing existing run-based stream data -> ${STREAM_OUT}"
fi

echo "[4/4] Run stream comparison experiments"
"${PIPENV[@]}" run python -m threat_agent.stream_experiment_compare \
  --stream-data "$STREAM_OUT" \
  --train-stream-data "$TRAIN_STREAM_OUT" \
  --val-stream-data "$VAL_STREAM_OUT" \
  --test-stream-data "$TEST_STREAM_OUT" \
  --seed "$SEED" \
  --decision-stride "$DECISION_STRIDE" \
  --tabular-episodes "$TABULAR_EPISODES" \
  --deep-episodes "$DEEP_EPISODES" \
  --eval-episodes "$EVAL_EPISODES" \
  --output-json "results/stream_compare_summary_runs.json" \
  --output-csv "results/stream_compare_summary_runs.csv"

echo "Done."
echo "Summary JSON: results/stream_compare_summary_runs.json"
echo "Summary CSV : results/stream_compare_summary_runs.csv"

