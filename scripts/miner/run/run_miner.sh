#!/bin/bash

# Poker44 Miner Startup Script (PM2 + venv Python)

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-poker44-miner-ck}"
HOTKEY="${HOTKEY:-poker44-miner-hk}"
NETWORK="${NETWORK:-finney}"
MINER_SCRIPT="${MINER_SCRIPT:-./neurons/miner.py}"
PM2_NAME="${PM2_NAME:-poker44_miner11}"
AXON_PORT="${AXON_PORT:-8091}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [ ! -f "$MINER_SCRIPT" ]; then
    echo "Error: Miner script not found at $MINER_SCRIPT"
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed (npm install -g pm2)"
    exit 1
fi

pm2 delete "$PM2_NAME" 2>/dev/null || true

export PYTHONPATH="$(pwd)"

# Manifest traceability: publish repo URL + commit that match THIS checkout (competition verification).
# Override with POKER44_MODEL_REPO_URL / POKER44_MODEL_REPO_COMMIT if needed.
# Custom miners: point `origin` at your public fork (or set POKER44_MODEL_REPO_URL). If `origin`
# is unset, same normalization is applied to the `danyloooah` remote when present.
export POKER44_MODEL_REPO_URL="${POKER44_MODEL_REPO_URL:-https://github.com/danyloooah-web/PokerNew.git}"
_poker44_export_repo_url_from_remote() {
  local raw="$1"
  [[ -z "$raw" ]] && return 1
  case "$raw" in
    git@github.com:*)
      suf="${raw#git@github.com:}"
      export POKER44_MODEL_REPO_URL="https://github.com/${suf%.git}"
      ;;
    git@gitlab.com:*)
      suf="${raw#git@gitlab.com:}"
      export POKER44_MODEL_REPO_URL="https://gitlab.com/${suf%.git}"
      ;;
    *)
      export POKER44_MODEL_REPO_URL="${raw%.git}"
      ;;
  esac
  # Trim trailing slash for stable manifest / policy checks
  POKER44_MODEL_REPO_URL="${POKER44_MODEL_REPO_URL%/}"
  export POKER44_MODEL_REPO_URL
  return 0
}

if [[ -z "${POKER44_MODEL_REPO_COMMIT:-}" ]] && [[ -d "$REPO_ROOT/.git" ]]; then
  if FULL_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)"; then
    export POKER44_MODEL_REPO_COMMIT="$FULL_SHA"
  fi
fi
if [[ -z "${POKER44_MODEL_REPO_URL:-}" ]] && [[ -d "$REPO_ROOT/.git" ]]; then
  ORIGIN="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
  if [[ -n "$ORIGIN" ]]; then
    _poker44_export_repo_url_from_remote "$ORIGIN" || true
  fi
  if [[ -z "${POKER44_MODEL_REPO_URL:-}" ]]; then
    DANY="$(git -C "$REPO_ROOT" remote get-url danyloooah 2>/dev/null || true)"
    [[ -n "$DANY" ]] && _poker44_export_repo_url_from_remote "$DANY" || true
  fi
fi

# Default trained chunk model (override if needed)
export POKER44_CHUNK_MODEL_PATH="${POKER44_CHUNK_MODEL_PATH:-$REPO_ROOT/scripts/miner/training/artifacts/chunk_model.joblib}"

# Public miner identity in the model_manifest. Override via POKER44_MODEL_NAME if you fork.
export POKER44_MODEL_NAME="${POKER44_MODEL_NAME:-poker44-baseline-v2}"

# Custom ML + upstream repo_url breaks transparent policy (repo_url_must_point_to_model_repo).
if [[ -n "${POKER44_MODEL_REPO_URL:-}" ]] && [[ -f "$POKER44_CHUNK_MODEL_PATH" ]]; then
  if ! head -1 "$POKER44_CHUNK_MODEL_PATH" | grep -q "git-lfs.github.com/spec" 2>/dev/null; then
    case "$POKER44_MODEL_REPO_URL" in
      *github.com/Poker44/Poker44-subnet*)
        echo "Warning: Manifest repo_url points at upstream Poker44 while a real (non-LFS-pointer) chunk model is present." >&2
        echo "  Custom ML miners need POKER44_MODEL_REPO_URL / origin (or danyloooah) = your public fork, e.g. https://github.com/<you>/Poker" >&2
        ;;
    esac
  fi
fi

# Optional inference tuning (see neurons/miner.py docstring):
# export POKER44_RISK_TEMPERATURE=1.15   # >1 softens risk_scores toward 0.5 (logit temperature)
# export POKER44_BOT_THRESHOLD=0.55      # threshold for synapse.predictions only (validators score risk_scores)

MINER_ARGS=(
  "$MINER_SCRIPT"
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --axon.port "$AXON_PORT"
  --logging.debug
)

if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VALIDATOR_HOTKEY_ARRAY <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  MINER_ARGS+=(--blacklist.allowed_validator_hotkeys "${VALIDATOR_HOTKEY_ARRAY[@]}")
else
  MINER_ARGS+=(--blacklist.force_validator_permit)
fi

pm2 start "$PYTHON_BIN" \
  --name "$PM2_NAME" \
  --cwd "$REPO_ROOT" \
  --interpreter none \
  -- "${MINER_ARGS[@]}"

pm2 save

echo "Miner started: $PM2_NAME"
echo "Python: $PYTHON_BIN"
echo "Model: $POKER44_CHUNK_MODEL_PATH"
echo "Manifest repo_url (if set): ${POKER44_MODEL_REPO_URL:-}"
echo "Manifest repo_commit (if set): ${POKER44_MODEL_REPO_COMMIT:-}"
echo "View logs: pm2 logs $PM2_NAME"
echo "Config: netuid=$NETUID network=$NETWORK wallet=$WALLET_NAME hotkey=$HOTKEY axon_port=$AXON_PORT"
if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
    echo "Access mode: validator allowlist"
else
    echo "Access mode: validator_permit fallback"
fi
