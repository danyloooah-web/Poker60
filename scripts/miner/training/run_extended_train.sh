#!/usr/bin/env bash
# Long CPU training run for miner chunk_model.joblib (hours on large --samples).
#
# Usage (foreground):
#   ./scripts/miner/training/run_extended_train.sh
#
# Leave running many hours in background on VPS:
#   cd /path/to/Poker44-subnet && nohup ./scripts/miner/training/run_extended_train.sh >>scripts/miner/training/artifacts/nohup_train.log 2>&1 &
#   tail -f scripts/miner/training/artifacts/nohup_train.log
#
# Override scale via env:
#   EXTENDED_TRAIN_SAMPLES=400000 EXTENDED_TRAIN_SEED=99 ./scripts/miner/training/run_extended_train.sh
# Extra train_model.py flags:
#   ./scripts/miner/training/run_extended_train.sh --disk-weight 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "Need python3 or .venv/bin/python" >&2
  exit 1
fi

SAMPLES="${EXTENDED_TRAIN_SAMPLES:-250000}"
SEED="${EXTENDED_TRAIN_SEED:-42}"

ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
mkdir -p "$ARTIFACTS_DIR"
LOG="$ARTIFACTS_DIR/train_extended_${SAMPLES}_${SEED}_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== Extended train start $(date -uIs) ==="
  echo "REPO_ROOT=$REPO_ROOT"
  echo "PYTHON=$PYTHON"
  echo "samples=$SAMPLES seed=$SEED"
  echo "log=$LOG"
  echo "========================================"
} | tee -a "$LOG"

set +e
"$PYTHON" "$SCRIPT_DIR/train_model.py" \
  --samples "$SAMPLES" \
  --seed "$SEED" \
  --real-weight 6 \
  --human-sample-boost 1.15 \
  --calibrate \
  "$@" 2>&1 | tee -a "$LOG"
EC=${PIPESTATUS[0]}
set -e

echo "=== Exit code $EC at $(date -uIs) | full log: $LOG ===" | tee -a "$LOG"
exit "$EC"
