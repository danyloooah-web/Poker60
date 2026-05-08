# Training data: step by step

This folder helps you turn **local PokerStars-style hand histories** into **miner-aligned** JSONL (same sanitization as production: `prepare_hand_for_miner`).

Production validators use **live** batches from the platform API. Treat this as **development data** only, and describe real training sources in your miner `model_manifest`.

---

## Step 0 — Environment

From the repo root:

```bash
cd /path/to/Poker44-subnet
source .venv/bin/activate   # or miner_env
pip install -r requirements.txt
pip install -e .
```

---

## Step 1 — Add raw human hands

Put PokerStars text exports (`.txt`) anywhere under:

`hands_generator/human_hands/poker_hands/`

(You can use year subfolders; the merge script walks the tree recursively.)

---

## Step 2 — Merge text files into one corpus

```bash
python scripts/miner/training/prepare_training_data.py merge
```

This runs `hands_generator/human_hands/data_parser.py` and writes:

`hands_generator/human_hands/massive_data.txt`

If there are no files under `poker_hands/`, put a single file at `hands_generator/human_hands/data.txt` instead and **skip Step 2**.

---

## Step 3 — Parse text → canonical JSON

```bash
python scripts/miner/training/prepare_training_data.py parse
```

- Reads `massive_data.txt` if it exists, otherwise `data.txt`.
- Writes `hands_generator/human_hands/human_hands.json` (canonical Poker44 hands, `label: human`).

Optional:

```bash
python scripts/miner/training/prepare_training_data.py parse --input path/to/hands.txt --output path/out.json
```

---

## Step 4 — Export miner-visible training JSONL

```bash
python scripts/miner/training/prepare_training_data.py export
```

Writes `hands_generator/human_hands/training_prepared.jsonl`.

Each line is one JSON object:

- `chunk_label`: `0` = human, `1` = bot (aligned with validator-side labels).
- `chunk_size`: number of hands in the row.
- `hands`: list of dicts after `prepare_hand_for_miner` (what your miner should see at runtime).

Optional chunking (to mimic “one score per chunk” with multiple hands):

```bash
python scripts/miner/training/prepare_training_data.py export --chunk-size 4
```

Only consecutive hands with the **same** `label` are grouped; mixed chunks are skipped with a warning.

---

## Step 5 — One-shot pipeline

```bash
python scripts/miner/training/prepare_training_data.py all
```

Same as merge → parse → export with defaults. Add `--chunk-size N` if needed.

---

## Step 6 — Bot class and modeling

- This repo’s human parser only emits **`label: human`**. For supervised **bot** class (`chunk_label: 1`), you need **separate** canonical hand JSON (same top-level schema as `human_hands.json`) with `"label": "bot"` on each hand, then either:
  - append/merge into one JSON list before `export`, or
  - run `export` on a combined file.
- `hands_generator/bot_hands/hole_strengths.csv` is **not** full hand histories; it cannot replace bot JSON.

Train your model on `training_prepared.jsonl`, then in `neurons/miner.py` replace the heuristic block with **inference** that:

1. Calls `prepare_hand_for_miner` only if you still have canonical payloads (JSONL rows are already prepared).
2. Returns **one float in `[0, 1]` per chunk** in `synapse.risk_scores`.

---

## Outputs (default paths)

| Step   | Output file |
|--------|-------------|
| merge  | `hands_generator/human_hands/massive_data.txt` |
| parse  | `hands_generator/human_hands/human_hands.json` |
| export | `hands_generator/human_hands/training_prepared.jsonl` |

Add large generated files to `.gitignore` locally if you do not want them committed.

---

## Train the chunk classifier (ensemble)

Generate optional **bulk synthetic** JSONL (large file; gitignored):

```bash
python scripts/miner/training/generate_synthetic_jsonl.py --chunks 32000 --seed 99
```

Train — auto-loads **`training_prepared.jsonl`** (real) and **`synthetic_prepared.jsonl`** (disk) when present:

```bash
python scripts/miner/training/train_model.py --samples 75000 --seed 99 --real-weight 6
```

- **Features:** v3 (`FEATURE_VERSION=3`, **34-D**): entropy / aggression / preflop-fold share / pot variance (see `poker44/training/features.py`).
- **In-memory synthetic:** `--samples` balanced chunks from an upgraded simulator (**regimes**, wider stakes).
- **Disk synthetic:** rows from `generate_synthetic_jsonl.py` (weight `--disk-weight`, default 1).
- **Real parser export:** `training_prepared.jsonl` (weight `--real-weight`, default 6).
- **Model:** soft **VotingClassifier** — RF (800 trees) + **HistGradientBoosting**; artifact saved with **zlib** compression.
- **Subnet-aligned metrics:** holdout reports `subnet_reward` using the same formula as `poker44/score/scoring.py` (AP + bot recall, gated by human FPR ≤ 10%).
- **`--human-sample-boost N`** (default `1.0`): multiply sample weights on **human** chunks during training to pressure the model away from false positives on humans (`N > 1`).
- **`--calibrate`**: optional sigmoid (Platt) calibration on a held-out slice of the training split; bundle stores `"calibrated": true`.

Example tuning run:

```bash
python scripts/miner/training/train_model.py \
  --samples 75000 --seed 99 --real-weight 6 \
  --human-sample-boost 1.25 --calibrate
```

Flags to isolate sources:

```bash
python scripts/miner/training/train_model.py --no-real-jsonl --no-disk-synthetic --samples 80000
```

Writes `scripts/miner/training/artifacts/chunk_model.joblib` (gitignored). Override path: `POKER44_CHUNK_MODEL_PATH`.

---

## Long CPU run (VPS / overnight)

Use **`run_extended_train.sh`** when you want a heavier fit (large `--samples`, calibration, human boost). It logs under `scripts/miner/training/artifacts/train_extended_*.log`.

Default scale is **`EXTENDED_TRAIN_SAMPLES=250000`** (override via env). Wall-clock is CPU-dependent (often **hours**).

Foreground:

```bash
./scripts/miner/training/run_extended_train.sh
```

Background with **`nohup`**:

```bash
cd /path/to/Poker44-subnet
nohup ./scripts/miner/training/run_extended_train.sh >>scripts/miner/training/artifacts/nohup_extended_train.log 2>&1 &
tail -f scripts/miner/training/artifacts/nohup_extended_train.log
```

Larger sweep:

```bash
EXTENDED_TRAIN_SAMPLES=400000 EXTENDED_TRAIN_SEED=99 ./scripts/miner/training/run_extended_train.sh
```

Extra flags are forwarded to `train_model.py` (e.g. `--disk-weight 2`).
