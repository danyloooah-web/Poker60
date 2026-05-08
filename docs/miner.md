# Poker44 Miner Guide

Production-facing miner guide for Poker44 subnet `126`.

## What Miners Are Solving Today

Poker44 validators currently evaluate miners with sanitized behavioral payloads derived from
live Poker44 benchmark tables.

Current production path:

1. live benchmark tables run on Poker44 platform infrastructure;
2. those hands are persisted in platform SQL;
3. `poker44-platform-backend` builds sanitized labeled evaluation batches from those hands;
4. the validator fetches the active batch set through `/internal/eval/current`;
5. the validator sends those batches to miners through `DetectionSynapse`;
6. miners return one risk score per received chunk;
7. the validator scores the miner and sets weights on-chain.

Important: the miner does **not** receive labels.

## Current Miner Contract

Miners receive `DetectionSynapse(chunks=...)`.

Current semantics:

- `chunks` is a list of chunks;
- each chunk is a list of sanitized hand payloads;
- each chunk may contain one or many sanitized hands;
- the validator expects exactly one `risk_score` per chunk.

So today the practical task is:

- receive many chunks per request;
- score each chunk independently;
- return one probability-like bot score per chunk.

Relevant code:

- [DetectionSynapse](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/synapse.py)
- [reference miner](/Users/mac/poker44-launch/poker44-subnet/neurons/miner.py)
- [validator forward cycle](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/forward.py)

## Important Precision About Chunk Structure

There are two different layers:

1. source hands on benchmark tables
2. chunks delivered to miners

Today, platform source hands are collected from live benchmark tables where humans and bots sit
together.

But the chunk format delivered to miners is still aligned with the current scoring path:

- the backend builds labeled batches from sanitized benchmark-table hands;
- the validator groups those batches into `DetectionSynapse(chunks=...)`;
- miners return one score per batch/chunk.

So:

- the overall validator request can contain both human-labeled and bot-labeled chunks;
- each individual chunk is homogeneous, so the hands inside a chunk are all human or all bot;
- miners should not assume a fixed number of hands per chunk.

Do not build your miner assuming this exact granularity will never evolve, but document and
optimize against the contract that is live today: one score per received chunk.

## Sanitized Payload Boundary

The payload sent to miners is sanitized before inference.

Current provider-runtime sanitization includes:

- `metadata`
- `players`
- `streets`
- `actions`
- `outcome`
- no direct identity fields
- no explicit ground-truth label

Recent hardening removed the most obvious timing leakage from the miner-visible payload.

See:

- [Anti-Leakage Policy](./anti-leakage.md)

## Expected Miner Output

Return fields:

- `risk_scores: List[float]`
- `predictions: List[bool]` optional but recommended
- `model_manifest: Dict[str, Any]` optional but recommended

Rules:

- length of `risk_scores` must equal number of received chunks;
- each score should be in `[0, 1]`;
- `predictions` should align one-to-one with `risk_scores` when provided.

Optional environment variables (reference miner):

- `POKER44_RISK_TEMPERATURE` — default `1.0`. Applies logit temperature to raw ensemble probabilities before returning `risk_scores`. Values greater than `1.0` pull scores toward `0.5`, which can reduce false positives on human-labeled chunks if your model is over-confident (validators grade `risk_scores`, not `predictions`).
- `POKER44_BOT_THRESHOLD` — default `0.5`. Threshold on each returned `risk_score` when building `predictions` (`risk_score >= threshold` ⇒ bot). Validators still consume continuous `risk_scores` for scoring.

The reference miner treats each chunk as one scoring unit and returns:

- low score for human-like behavior
- high score for bot-like behavior

## Install

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install bittensor-cli
```

Or use:

```bash
./scripts/miner/setup.sh
```

## Wallet and Registration

`btcli` is provided by the separate `bittensor-cli` package.

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey --wallet.name my_cold --wallet.hotkey my_poker44_hotkey

btcli subnet register \
  --wallet.name my_cold \
  --wallet.hotkey my_poker44_hotkey \
  --netuid 126 \
  --subtensor.network finney

btcli wallet overview --wallet.name my_cold --subtensor.network finney
```

## Run Miner

Script path:

- `scripts/miner/run/run_miner.sh`

Example:

```bash
WALLET_NAME=my_cold \
HOTKEY=my_poker44_hotkey \
AXON_PORT=8091 \
ALLOWED_VALIDATOR_HOTKEYS="validator_hotkey_1 validator_hotkey_2" \
./scripts/miner/run/run_miner.sh
```

Before using the script, set at least:

- `WALLET_NAME`
- `HOTKEY`
- `AXON_PORT`
- `ALLOWED_VALIDATOR_HOTKEYS` for the recommended allowlist mode

If `ALLOWED_VALIDATOR_HOTKEYS` is empty, the script falls back to
`--blacklist.force_validator_permit`.

Direct CLI example:

```bash
python neurons/miner.py \
  --netuid 126 \
  --wallet.name my_cold \
  --wallet.hotkey my_poker44_hotkey \
  --subtensor.network finney \
  --axon.port 8091 \
  --blacklist.allowed_validator_hotkeys <validator_hotkey_1> <validator_hotkey_2>
```

## Production Access Policy

Recommended mode:

- `--blacklist.allowed_validator_hotkeys <validator_hotkey...>`

Fallback mode:

- `--blacklist.force_validator_permit`

Operationally:

- if an allowlist is set, only those validators may query your miner;
- otherwise the miner falls back to the metagraph `validator_permit` rule.

## Model Manifest

Poker44 miners can publish a lightweight `model_manifest` without changing the remote-inference
scoring path.

**Competition traceability:** for top daily miners, declared **public repository** and **commit**
must match the code actually serving the axon (see subnet operator communications). Set
`POKER44_MODEL_REPO_URL` and **`POKER44_MODEL_REPO_COMMIT`** (full SHA) to that public tree.
`scripts/miner/run/run_miner.sh` exports both from `git remote get-url origin` and
`git rev-parse HEAD` when unset (and a `.git` directory exists); if **`origin`** has no URL, it tries
the **`danyloooah`** remote next. Point `origin` at **your public fork** for a predictable setup; declaring the upstream Poker44 URL together with a non-reference
`model_name` fails the transparent policy check in `evaluate_manifest_compliance`.

The reference miner’s manifest includes **`implementation_sha256`** over `neurons/miner.py` plus
`poker44/training/features.py`, `calibration.py`, and `risk_postprocess.py` when present, so
auditors can verify the inference stack you ship.

Recommended fields:

- `open_source`
- `repo_url`
- `repo_commit`
- `model_name`
- `model_version`
- `framework`
- `license`
- `training_data_statement`
- `training_data_sources`
- `private_data_attestation`
- `artifact_url`
- `artifact_sha256`
- `implementation_sha256`

Minimum fields for `transparent` compliance (see `evaluate_manifest_compliance` in
`poker44/utils/model_manifest.py`):

- `open_source=true`
- non-empty `repo_url` consistent with `model_name` (custom models cannot declare only the upstream Poker44 subtree URL unless `model_name` is the reference heuristic)
- valid git `repo_commit`
- `model_name`, `model_version`
- `training_data_statement`, `private_data_attestation`
- `implementation_files`, `implementation_sha256`

The validator still scores your `risk_scores`; the manifest is for transparency and
anti-leakage tracking.

## Production Evaluation Boundary

Production evaluation is not derived from local helper artifacts.

Production validators now target:

- live hands generated on Poker44 platform tables
- centralized SQL persistence
- sanitizer-built batches served by the eval API

Miners should optimize against the live contract and the current chunk-level scoring path, not
against assumptions about any local reference corpus.

## PM2

```bash
pm2 logs poker44_miner
pm2 restart poker44_miner
pm2 stop poker44_miner
pm2 delete poker44_miner
```

## Health Checklist

- Miner hotkey registered on netuid `126`
- Axon served and reachable
- Validator queries accepted
- Returned `risk_scores` length matches chunk count
- Miner remains stable under repeated validator polling
