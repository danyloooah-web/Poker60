#!/usr/bin/env python3
"""
Train a chunk-level bot-risk model: balanced synthetic chunks + optional JSONL files.

Auto-loads when present:
  - hands_generator/human_hands/training_prepared.jsonl   (real parser export)
  - hands_generator/human_hands/synthetic_prepared.jsonl    (bulk synthetic disk)

  python scripts/miner/training/train_model.py
  python scripts/miner/training/train_model.py --samples 80000 --real-weight 6

Writes ``scripts/miner/training/artifacts/chunk_model.joblib`` (zlib-compressed, gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DEFAULT_OUT = ARTIFACTS / "chunk_model.joblib"
REAL_JSONL = REPO / "hands_generator" / "human_hands" / "training_prepared.jsonl"
DISK_SYNTHETIC_JSONL = REPO / "hands_generator" / "human_hands" / "synthetic_prepared.jsonl"


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    sys.path.insert(0, str(REPO))
    from poker44.training.calibration import PlattCalibratedClassifier
    from poker44.training.features import FEATURE_VERSION, N_FEATURES, featurize_chunk
    from poker44.training.synthetic import generate_chunk
    from poker44.validator.sanitization import prepare_hand_for_miner
    from poker44.score.scoring import reward as subnet_reward

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--samples",
        type=int,
        default=60000,
        help="In-memory synthetic chunks total (balanced human/bot). Default 60000.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help=f"Extra prepared JSONL (optional). Default real export: {REAL_JSONL.name} if present.",
    )
    p.add_argument(
        "--no-real-jsonl",
        action="store_true",
        help="Skip training_prepared.jsonl (real parser export).",
    )
    p.add_argument(
        "--no-disk-synthetic",
        action="store_true",
        help="Skip synthetic_prepared.jsonl bulk file if present.",
    )
    p.add_argument(
        "--real-weight",
        type=float,
        default=6.0,
        help="sample_weight for rows from training_prepared.jsonl.",
    )
    p.add_argument(
        "--disk-weight",
        type=float,
        default=1.0,
        help="sample_weight for rows from synthetic_prepared.jsonl.",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--human-sample-boost",
        type=float,
        default=1.0,
        help=(
            "Multiply training sample_weight on human chunks (label 0). Values > 1.0 penalize "
            "human false positives more heavily (aligns with subnet FPR gate). Default 1.0."
        ),
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Fit sigmoid Platt calibration on a held-out slice of the training split "
            "(LogisticRegression on base ``decision_function``, compatible with current sklearn)."
        ),
    )
    p.add_argument(
        "--calibration-holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of training rows held out for Platt calibration when --calibrate.",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    w_list: list[float] = []

    n = max(100, args.samples // 2)
    print(f"Building {n * 2} in-memory synthetic chunks (balanced human/bot)...", flush=True)
    for label_name, y in (("human", 0), ("bot", 1)):
        for _ in range(n):
            sz = int(rng.integers(1, 6))
            hands, _ = generate_chunk(rng, label_name, sz)  # type: ignore[arg-type]
            prepared = [prepare_hand_for_miner(dict(h)) for h in hands]
            X_list.append(featurize_chunk(prepared))
            y_list.append(y)
            w_list.append(1.0)

    real_rows = 0
    if not args.no_real_jsonl:
        paths = []
        if args.jsonl is not None:
            paths.append(args.jsonl)
        elif REAL_JSONL.is_file():
            paths.append(REAL_JSONL)
        for jp in paths:
            if not jp.is_file():
                continue
            rows = _load_jsonl_rows(jp)
            real_rows += len(rows)
            print(f"Adding {len(rows)} rows from {jp.relative_to(REPO)} (weight={args.real_weight})", flush=True)
            for row in rows:
                hands = row.get("hands") or []
                y = int(row.get("chunk_label", 0))
                if not hands:
                    continue
                X_list.append(featurize_chunk(hands))
                y_list.append(y)
                w_list.append(float(args.real_weight))

    disk_syn = 0
    if not args.no_disk_synthetic and DISK_SYNTHETIC_JSONL.is_file():
        rows = _load_jsonl_rows(DISK_SYNTHETIC_JSONL)
        disk_syn = len(rows)
        print(
            f"Adding {disk_syn} disk-synthetic rows from {DISK_SYNTHETIC_JSONL.relative_to(REPO)} "
            f"(weight={args.disk_weight})",
            flush=True,
        )
        dw = float(args.disk_weight)
        for row in rows:
            hands = row.get("hands") or []
            y = int(row.get("chunk_label", 0))
            if not hands:
                continue
            X_list.append(featurize_chunk(hands))
            y_list.append(y)
            w_list.append(dw)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    sw = np.array(w_list, dtype=np.float64)

    print(
        f"Matrix shape {X.shape} | mem_synth={n * 2} real_jsonl={real_rows} "
        f"disk_synth={disk_syn} | fitting ensemble...",
        flush=True,
    )
    if X.shape[1] != N_FEATURES:
        raise SystemExit(f"Feature dim mismatch: got {X.shape[1]} expected {N_FEATURES}")

    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.15, random_state=args.seed, stratify=y
    )
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    sw_tr = sw[idx_tr].copy()
    hb = float(args.human_sample_boost)
    if hb != 1.0:
        human_mask = y_tr == 0
        sw_tr[human_mask] *= hb
        print(f"Applied human-sample-boost={hb} to {int(human_mask.sum())} human rows.", flush=True)

    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=28,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=args.seed,
        class_weight="balanced_subsample",
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=550,
        max_depth=16,
        learning_rate=0.045,
        l2_regularization=0.28,
        min_samples_leaf=12,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=35,
        random_state=args.seed,
        class_weight="balanced",
    )
    clf = VotingClassifier(
        estimators=[("rf", rf), ("hgb", hgb)],
        voting="soft",
        weights=[1.0, 1.15],
        n_jobs=-1,
    )

    cal_frac = float(args.calibration_holdout_fraction)
    cal_frac = max(0.05, min(0.45, cal_frac))

    if args.calibrate and len(X_tr) >= 500:
        idx_fit, idx_cal = train_test_split(
            np.arange(len(X_tr)),
            test_size=cal_frac,
            random_state=args.seed,
            stratify=y_tr,
        )
        X_fit, X_cal = X_tr[idx_fit], X_tr[idx_cal]
        y_fit, y_cal = y_tr[idx_fit], y_tr[idx_cal]
        sw_fit = sw_tr[idx_fit]
        clf.fit(X_fit, y_fit, sample_weight=sw_fit)
        try:
            scores_cal = clf.decision_function(X_cal)
            if scores_cal.ndim > 1:
                scores_cal = scores_cal[:, 1]
            lr = LogisticRegression(
                C=1e12,
                solver="lbfgs",
                max_iter=2000,
                random_state=args.seed,
            )
            lr.fit(scores_cal.reshape(-1, 1), y_cal)
            clf_out = PlattCalibratedClassifier(clf, lr)
            calibrated_flag = True
        except Exception as exc:
            print(f"Calibration failed ({exc}); saving uncalibrated ensemble.", flush=True)
            clf.fit(X_tr, y_tr, sample_weight=sw_tr)
            clf_out = clf
            calibrated_flag = False
    else:
        if args.calibrate and len(X_tr) < 500:
            print(
                "Skipping --calibrate (training set too small for stable calibration).",
                flush=True,
            )
        clf.fit(X_tr, y_tr, sample_weight=sw_tr)
        clf_out = clf
        calibrated_flag = False

    proba = clf_out.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    acc = float((clf_out.predict(X_te) == y_te).mean())
    rew, rew_detail = subnet_reward(proba.astype(np.float64), y_te.astype(np.int64))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    bundle = {
        "classifier": clf_out,
        "feature_version": FEATURE_VERSION,
        "n_features": N_FEATURES,
        "train_samples": int(X.shape[0]),
        "mem_synthetic_chunks": int(n * 2),
        "real_jsonl_chunks": int(real_rows),
        "disk_synthetic_chunks": int(disk_syn),
        "real_weight": float(args.real_weight),
        "disk_weight": float(args.disk_weight),
        "human_sample_boost": hb,
        "calibrated": calibrated_flag,
        "holdout_roc_auc": float(auc),
        "holdout_accuracy": acc,
        "holdout_subnet_reward": float(rew),
        "holdout_reward_detail": rew_detail,
    }
    joblib.dump(bundle, args.out, compress=("zlib", 3))
    print(
        f"Saved {args.out} | samples={X.shape[0]} features={N_FEATURES} "
        f"holdout_acc={acc:.4f} roc_auc={auc:.4f} subnet_reward={rew:.4f} "
        f"fpr={rew_detail['fpr']:.4f} human_penalty={rew_detail['human_safety_penalty']:.4f}"
    )


if __name__ == "__main__":
    main()
