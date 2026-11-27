#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DTMW_SIMULATOR_ROOT:?Set DTMW_SIMULATOR_ROOT to your unpacked experiment simulator. See data/README.md.}"
export DTMW_DATA_DIR="${DTMW_DATA_DIR:-$DTMW_SIMULATOR_ROOT/data}"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export DTMW_USE_REAL_ORDERS="${DTMW_USE_REAL_ORDERS:-1}"
export DTMW_REAL_ORDER_SAMPLE_RATE="${DTMW_REAL_ORDER_SAMPLE_RATE:-0.35}"
export DTMW_GRID_NUM="${DTMW_GRID_NUM:-289}"
export DTMW_DRIVER_NUM="${DTMW_DRIVER_NUM:-2000}"
export DTMW_MAX_ITER="${DTMW_MAX_ITER:-400}"
export DTMW_SAVE_INTERVAL="${DTMW_SAVE_INTERVAL:-20}"
export DTMW_DATA_PATH="${DTMW_DATA_PATH:-$DTMW_DATA_DIR/NYC2015Jan05_h3_289_real_orders.pkl}"
export DTMW_WEATHER_CONFIG="${DTMW_WEATHER_CONFIG:-$REPO_ROOT/config/weather_days.json}"

cd "$REPO_ROOT"

run_variant() {
  local name="$1"
  shift
  echo "=== Running $name ==="
  local log_dir=""
  if [[ -n "${DTMW_LOG_DIR:-}" ]]; then
    log_dir="$DTMW_LOG_DIR/$name"
  fi
  DTMW_RUN_TAG="$name" DTMW_LOG_DIR="$log_dir" "$@" python -u -m run.train
}

run_variant full env
run_variant noDual env DTMW_NO_DUAL=1
run_variant noLLMprior env DTMW_USE_LLM_PRIOR=0
run_variant noSSA env DTMW_NO_SSA=1
run_variant noDRS env DTMW_DRS_WEIGHT=0
run_variant noISA env DTMW_INTRINSIC_WEIGHT=0 DTMW_CONTRASTIVE_WEIGHT=0
