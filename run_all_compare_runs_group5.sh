#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Base configuration (requested)
SPLIT_MODE="source"
SPLIT_RATIO="0.7,0.15,0.15"
NUM_STREAMS="100000"
EVENTS_PER_STREAM="120"
MIN_CONFIDENCE="medium"
DECISION_STRIDE="8"
TABULAR_EPISODES="10000"
DEEP_EPISODES="10000"
EVAL_EPISODES="100"
INTRA_RUN_BENIGN_PROB="0.05"
ATTACK_RUNS_MAX="3"

# Group tactic sets
GROUP1_TACTICS="Execution,Discovery,Defense Evasion,Command and Control,AutomatedTestingTools"
GROUP2_TACTICS="Credential Access,Privilege Escalation"
GROUP3_TACTICS="Persistence,Lateral Movement"
GROUP4_TACTICS="Other"

# Representative one-per-group (Other excluded)
# Execution / Credential Access / Persistence
REP_TACTICS="Execution,Credential Access,Persistence"

run_one() {
  local run_name="$1"
  local tactics="$2"
  echo ""
  echo "============================================================"
  echo "[RUN] ${run_name}"
  echo "[TACTICS] ${tactics}"
  echo "============================================================"

  ./run_all_compare_runs.sh \
    --split-mode "$SPLIT_MODE" \
    --split-ratio "$SPLIT_RATIO" \
    --num-streams "$NUM_STREAMS" \
    --events-per-stream "$EVENTS_PER_STREAM" \
    --min-confidence "$MIN_CONFIDENCE" \
    --decision-stride "$DECISION_STRIDE" \
    --tabular-episodes "$TABULAR_EPISODES" \
    --deep-episodes "$DEEP_EPISODES" \
    --eval-episodes "$EVAL_EPISODES" \
    --intra-run-benign-prob "$INTRA_RUN_BENIGN_PROB" \
    --attack-runs-max "$ATTACK_RUNS_MAX" \
    --use-tactics "$tactics" \
    --output-dir "results/group5_runs/${run_name}"
}

mkdir -p "results/group5_runs"

# 1) Four group runs
run_one "group1_execution_recon_evasion" "$GROUP1_TACTICS"
run_one "group2_credential_privilege" "$GROUP2_TACTICS"
run_one "group3_persistence_lateral" "$GROUP3_TACTICS"
run_one "group4_residual_other" "$GROUP4_TACTICS"

# 2) Representative mixed run (excluding Other)
run_one "group5_representative_mix" "$REP_TACTICS"

echo ""
echo "All 5 grouped runs completed."
echo "Output root: results/group5_runs"

