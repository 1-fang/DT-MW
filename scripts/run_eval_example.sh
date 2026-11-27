#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DTMW_SIMULATOR_ROOT:?Set DTMW_SIMULATOR_ROOT to your unpacked experiment simulator. See data/README.md.}"
export DTMW_DATA_DIR="${DTMW_DATA_DIR:-$DTMW_SIMULATOR_ROOT/data}"

: "${DTMW_RESUME_PATH:?Set DTMW_RESUME_PATH to a trained checkpoint/log directory.}"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export DTMW_USE_REAL_ORDERS="${DTMW_USE_REAL_ORDERS:-1}"
export DTMW_REAL_ORDER_SAMPLE_RATE="${DTMW_REAL_ORDER_SAMPLE_RATE:-0.35}"
export DTMW_GRID_NUM="${DTMW_GRID_NUM:-289}"
export DTMW_DRIVER_NUM="${DTMW_DRIVER_NUM:-2000}"
export DTMW_EVAL_DATA_PATH="${DTMW_EVAL_DATA_PATH:-$DTMW_DATA_DIR/NYC2015Jan06_h3_289_real_orders.pkl}"
export DTMW_EVAL_TAG="${DTMW_EVAL_TAG:-nyc_normal}"

cd "$REPO_ROOT"
python -u -m run.eval
