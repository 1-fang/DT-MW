#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HUB_OFFLINE=1
export DTMW_SYNTHETIC=1
export DTMW_USE_REAL_ORDERS=0
export DTMW_USE_LLM=0
export DTMW_CPU=1
export DTMW_GRID_NUM="${DTMW_GRID_NUM:-9}"
export DTMW_DRIVER_NUM="${DTMW_DRIVER_NUM:-36}"
export DTMW_EPISODE_STEPS="${DTMW_EPISODE_STEPS:-24}"
export DTMW_MAX_ITER="${DTMW_MAX_ITER:-2}"
export DTMW_WARMUP_EPISODES="${DTMW_WARMUP_EPISODES:-1}"
export DTMW_MAX_ORDER_NUM="${DTMW_MAX_ORDER_NUM:-12}"
export DTMW_FEATURE_NORMAL=1
unset DTMW_DATA_PATH DTMW_EVAL_DATA_PATH COOPRIDE_DATA_PATH DTMW_RESUME_PATH COOPRIDE_RESUME_PATH

if [[ -z "${DTMW_LOG_DIR:-}" ]]; then
    mkdir -p logs
    DTMW_LOG_DIR="$(mktemp -d "$REPO_ROOT/logs/smoke.XXXXXX")"
fi
export DTMW_LOG_DIR

python -u -m run.train
DTMW_RESUME_PATH="$DTMW_LOG_DIR/param.pkl" \
    DTMW_EVAL_OUTPUT="$DTMW_LOG_DIR/eval.csv" \
    python -u -m run.eval
