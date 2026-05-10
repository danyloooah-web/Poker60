#!/usr/bin/env python3
"""
Train a chunk-level bot-risk model: balanced synthetic chunks + optional JSONL files.

Auto-loads when present:
  - scripts/miner/training/benchmark/benchmark_prepared.jsonl (public benchmark)
  - scripts/miner/training/real_pokerstars/pokerstars_prepared.jsonl (capped real hands)
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
BENCHMARK_JSONL = Path(__file__).resolve().parent / "benchmark" / "benchmark_prepared.jsonl"
POKERSTARS_JSONL = Path(__file__).resolve().parent / "real_pokerstars" / "pokerstars_prepared.jsonl"
SYNTH_LIKE_JSONL = (
    Path(__file__).resolve().parent
    / "synthetic_benchmark_like"
    / "synthetic_benchmark_like_train.jsonl"
)
BOT_SOURCE_JSONLS = [
    Path(__file__).resolve().parent / "bot_sources" / "bot_train_chunks_benchmark_like.jsonl",
    Path(__file__).resolve().parent / "bot_sources" / "hf_bot_sim_train_chunks_60_prepared.jsonl",
    Path(__file__).resolve().parent / "bot_sources" / "pluribus_train_chunks_60_prepared.jsonl",
]


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
    from poker44.training.calibration import PlattCalibratedClassifier, calibration_scores
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
        "--no-benchmark-jsonl",
        action="store_true",
        help="Skip benchmark/benchmark_prepared.jsonl.",
    )
    p.add_argument(
        "--benchmark-weight",
        type=float,
        default=12.0,
        help=(
            "sample_weight for rows from benchmark/benchmark_prepared.jsonl. "
            "Default 12 keeps the small benchmark set influential."
        ),
    )
    p.add_argument(
        "--pokerstars-jsonl",
        type=Path,
        default=POKERSTARS_JSONL,
        help="Prepared PokerStars/RealStars JSONL to load when present.",
    )
    p.add_argument(
        "--no-pokerstars-jsonl",
        action="store_true",
        help="Skip the prepared PokerStars/RealStars JSONL.",
    )
    p.add_argument(
        "--max-pokerstars-rows",
        type=int,
        default=10000,
        help=(
            "Random cap for PokerStars/RealStars rows before training. "
            "Use 0 to load all rows. Default 10000 avoids drowning benchmark data."
        ),
    )
    p.add_argument(
        "--pokerstars-weight",
        type=float,
        default=1.0,
        help="sample_weight for rows from --pokerstars-jsonl.",
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
    p.add_argument(
        "--no-synth-like-jsonl",
        action="store_true",
        help="Skip synthetic_benchmark_like/synthetic_benchmark_like_train.jsonl.",
    )
    p.add_argument(
        "--synth-like-weight",
        type=float,
        default=4.0,
        help=(
            "sample_weight for rows from synthetic_benchmark_like_train.jsonl. "
            "These are large (40-80 hand) balanced chunks shaped like the benchmark."
        ),
    )
    p.add_argument(
        "--max-synth-like-rows",
        type=int,
        default=0,
        help="Random cap on synth-like rows (0 = use all).",
    )
    p.add_argument(
        "--no-bot-sources",
        action="store_true",
        help="Skip the prepared bot_sources/*.jsonl files.",
    )
    p.add_argument(
        "--bot-sources-weight",
        type=float,
        default=3.0,
        help="sample_weight for rows from bot_sources/*.jsonl (all bot-only).",
    )
    p.add_argument(
        "--max-bot-source-rows",
        type=int,
        default=0,
        help="Random cap per bot-source file (0 = use all).",
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

    def _display_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO))
        except ValueError:
            return str(path)

    def _append_prepared_rows(
        *,
        path: Path,
        rows: list[dict],
        weight: float,
        source_name: str,
        max_rows: int | None = None,
    ) -> int:
        original_rows = len(rows)
        if max_rows is not None and max_rows > 0 and original_rows > max_rows:
            selected = rng.choice(original_rows, size=max_rows, replace=False)
            rows = [rows[int(i)] for i in selected]

        used_rows = 0
        for row in rows:
            hands = row.get("hands") or []
            y = int(row.get("chunk_label", 0))
            if not hands:
                continue
            X_list.append(featurize_chunk(hands))
            y_list.append(y)
            w_list.append(float(weight))
            used_rows += 1

        cap_msg = ""
        if max_rows is not None and max_rows > 0 and original_rows > max_rows:
            cap_msg = f" (sampled {used_rows}/{original_rows})"
        print(
            f"Adding {used_rows} rows from {_display_path(path)} as {source_name} "
            f"(weight={weight}){cap_msg}",
            flush=True,
        )
        return used_rows

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

    benchmark_rows = 0
    if not args.no_benchmark_jsonl and BENCHMARK_JSONL.is_file():
        benchmark_rows = _append_prepared_rows(
            path=BENCHMARK_JSONL,
            rows=_load_jsonl_rows(BENCHMARK_JSONL),
            weight=float(args.benchmark_weight),
            source_name="benchmark",
        )

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
            real_rows += _append_prepared_rows(
                path=jp,
                rows=rows,
                weight=float(args.real_weight),
                source_name="real-jsonl",
            )

    pokerstars_rows = 0
    if not args.no_pokerstars_jsonl and args.pokerstars_jsonl is not None and args.pokerstars_jsonl.is_file():
        pokerstars_cap = None if int(args.max_pokerstars_rows) <= 0 else int(args.max_pokerstars_rows)
        pokerstars_rows = _append_prepared_rows(
            path=args.pokerstars_jsonl,
            rows=_load_jsonl_rows(args.pokerstars_jsonl),
            weight=float(args.pokerstars_weight),
            source_name="pokerstars",
            max_rows=pokerstars_cap,
        )

    synth_like_rows = 0
    if not args.no_synth_like_jsonl and SYNTH_LIKE_JSONL.is_file():
        synth_like_cap = None if int(args.max_synth_like_rows) <= 0 else int(args.max_synth_like_rows)
        synth_like_rows = _append_prepared_rows(
            path=SYNTH_LIKE_JSONL,
            rows=_load_jsonl_rows(SYNTH_LIKE_JSONL),
            weight=float(args.synth_like_weight),
            source_name="synth-like",
            max_rows=synth_like_cap,
        )

    bot_source_rows = 0
    if not args.no_bot_sources:
        bot_cap = None if int(args.max_bot_source_rows) <= 0 else int(args.max_bot_source_rows)
        for bot_path in BOT_SOURCE_JSONLS:
            if not bot_path.is_file():
                continue
            bot_source_rows += _append_prepared_rows(
                path=bot_path,
                rows=_load_jsonl_rows(bot_path),
                weight=float(args.bot_sources_weight),
                source_name=f"bot-source[{bot_path.stem}]",
                max_rows=bot_cap,
            )

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
        f"Matrix shape {X.shape} | mem_synth={n * 2} benchmark={benchmark_rows} "
        f"real_jsonl={real_rows} pokerstars={pokerstars_rows} synth_like={synth_like_rows} "
        f"bot_sources={bot_source_rows} disk_synth={disk_syn} "
        "| fitting ensemble...",
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
            scores_cal = calibration_scores(clf, X_cal)
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
        "benchmark_jsonl_chunks": int(benchmark_rows),
        "real_jsonl_chunks": int(real_rows),
        "pokerstars_jsonl_chunks": int(pokerstars_rows),
        "synth_like_chunks": int(synth_like_rows),
        "bot_source_chunks": int(bot_source_rows),
        "disk_synthetic_chunks": int(disk_syn),
        "benchmark_weight": float(args.benchmark_weight),
        "real_weight": float(args.real_weight),
        "pokerstars_weight": float(args.pokerstars_weight),
        "synth_like_weight": float(args.synth_like_weight),
        "bot_sources_weight": float(args.bot_sources_weight),
        "max_pokerstars_rows": int(args.max_pokerstars_rows),
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
