#!/usr/bin/env python3
"""Evaluate a trained chunk_model.joblib on the full prepared benchmark JSONL.

Reports overall and per-`source_date` metrics:
    accuracy, roc_auc, average_precision, confusion_matrix,
    bot_recall, fpr, subnet_reward (poker44.score.scoring.reward).

Writes a JSON summary alongside the model and prints the headline numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from poker44.score.scoring import reward as subnet_reward  # noqa: E402
from poker44.training.features import N_FEATURES, featurize_chunk  # noqa: E402

DEFAULT_BENCHMARK = REPO / "scripts" / "miner" / "training" / "benchmark" / "benchmark_prepared.jsonl"
DEFAULT_MODEL = REPO / "scripts" / "miner" / "training" / "artifacts" / "chunk_model.joblib"


def _load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _block_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    neg = max(tn + fp, 1)
    pos = max(tp + fn, 1)
    fpr = fp / neg
    bot_recall = tp / pos
    rew, detail = subnet_reward(proba.astype(np.float64), y_true.astype(np.int64))
    return {
        "rows": int(y_true.size),
        "label_counts": {"human": int(neg), "bot": int(pos)},
        "accuracy": float(accuracy_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": float(fpr),
        "bot_recall": float(bot_recall),
        "subnet_reward": float(rew),
        "reward_detail": detail,
        "score_summary": {
            "min": float(proba.min()),
            "p25": float(np.percentile(proba, 25)),
            "median": float(np.median(proba)),
            "p75": float(np.percentile(proba, 75)),
            "max": float(proba.max()),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Trained chunk_model.joblib bundle")
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK, help="Prepared benchmark JSONL")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Decision threshold for predicted-bot. Lowering increases bot_recall at the cost of FPR; "
            "the subnet reward function rounds at 0.5 internally so this only affects the printed CM."
        ),
    )
    p.add_argument("--out", type=Path, default=None, help="JSON report path (default: <model>.benchmark_eval.json)")
    args = p.parse_args()

    if not args.model.is_file():
        print(f"Model bundle not found: {args.model}", file=sys.stderr)
        return 2
    if not args.benchmark.is_file():
        print(f"Benchmark JSONL not found: {args.benchmark}", file=sys.stderr)
        return 2

    t0 = time.time()
    bundle = joblib.load(args.model)
    clf = bundle["classifier"]
    feat_v = bundle.get("feature_version")
    print(f"Loaded model {args.model.name} (feature_version={feat_v}, n_features={bundle.get('n_features')})")

    rows = _load_jsonl(args.benchmark)
    print(f"Loaded {len(rows)} benchmark rows from {args.benchmark.name} in {time.time() - t0:.1f}s")

    X = np.zeros((len(rows), N_FEATURES), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int64)
    by_date = defaultdict(list)
    for i, row in enumerate(rows):
        X[i] = featurize_chunk(row.get("hands") or [])
        y[i] = int(row.get("chunk_label", 0))
        by_date[str(row.get("source_date", "unknown"))].append(i)

    proba = clf.predict_proba(X)[:, 1]

    overall = _block_metrics(y, proba, args.threshold)
    by_date_metrics = {}
    for date in sorted(by_date):
        idx = np.array(by_date[date], dtype=np.int64)
        by_date_metrics[date] = _block_metrics(y[idx], proba[idx], args.threshold)

    report = {
        "model_path": str(args.model),
        "benchmark_path": str(args.benchmark),
        "threshold": float(args.threshold),
        "model_bundle_summary": {
            k: bundle.get(k)
            for k in (
                "feature_version",
                "n_features",
                "train_samples",
                "calibrated",
                "holdout_roc_auc",
                "holdout_subnet_reward",
                "benchmark_jsonl_chunks",
                "synth_like_chunks",
                "bot_source_chunks",
                "pokerstars_jsonl_chunks",
                "human_sample_boost",
            )
            if k in bundle
        },
        "overall": overall,
        "by_source_date": by_date_metrics,
    }

    out_path = args.out or args.model.with_suffix(".benchmark_eval.json")
    out_path.write_text(json.dumps(report, indent=2))

    o = overall
    print()
    print(f"=== Overall on {len(rows)} benchmark rows ===")
    print(
        f"  acc={o['accuracy']:.4f}  AUC={o['roc_auc']:.4f}  AP={o['average_precision']:.4f}  "
        f"bot_recall={o['bot_recall']:.4f}  fpr={o['fpr']:.4f}  "
        f"subnet_reward={o['subnet_reward']:.4f}"
    )
    cm = o["confusion_matrix"]
    print(f"  CM: tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}")
    print()
    print("=== Per source_date ===")
    for date, m in by_date_metrics.items():
        cm = m["confusion_matrix"]
        print(
            f"  {date}: rows={m['rows']:>4}  "
            f"acc={m['accuracy']:.4f}  AP={m['average_precision']:.4f}  "
            f"bot_recall={m['bot_recall']:.4f}  fpr={m['fpr']:.4f}  "
            f"reward={m['subnet_reward']:.4f}  CM=({cm['tn']},{cm['fp']},{cm['fn']},{cm['tp']})"
        )
    print()
    print(f"Wrote report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
