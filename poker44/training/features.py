"""
Fixed-size features from **sanitized** hand dicts (post `prepare_hand_for_miner`).

Keep `FEATURE_VERSION` in sync when changing layout; training bundles store it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np

FEATURE_VERSION = 3
N_FEATURES = 34

_MEANINGFUL = frozenset({"call", "check", "bet", "raise", "fold"})
_ENT_KEYS = ("call", "check", "bet", "raise", "fold")


def _entropy_norm(counts: Dict[str, int]) -> float:
    total = sum(counts.get(k, 0) for k in _ENT_KEYS)
    if total <= 0:
        return 0.0
    ent = 0.0
    for k in _ENT_KEYS:
        p = counts.get(k, 0) / total
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return float(ent / math.log(len(_ENT_KEYS)))

_order = (
    "r_call",
    "r_check",
    "r_bet",
    "r_raise",
    "r_fold",
    "n_actions",
    "n_players",
    "n_streets",
    "mean_norm_bb",
    "max_norm_bb",
    "mean_pot_bb",
    "street_depth",
    "entropy_actions",
    "aggression",
    "preflop_fold_share",
    "pot_bb_std",
)


def _hand_vector(hand: Dict[str, Any]) -> np.ndarray:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    meta = hand.get("metadata") or {}
    bb = float(meta.get("bb") or 0.02)
    if bb < 1e-9:
        bb = 0.02

    counts: Dict[str, int] = {}
    norms: List[float] = []
    pots: List[float] = []
    street_set: set[str] = set()
    fold_preflop = 0
    fold_total = 0

    for a in actions:
        t = str(a.get("action_type") or "")
        counts[t] = counts.get(t, 0) + 1
        st = str(a.get("street") or "")
        street_set.add(st)
        try:
            norms.append(float(a.get("normalized_amount_bb") or 0.0))
        except (TypeError, ValueError):
            norms.append(0.0)
        try:
            pots.append(float(a.get("pot_after") or 0.0))
        except (TypeError, ValueError):
            pots.append(0.0)
        if t == "fold":
            fold_total += 1
            if st == "preflop":
                fold_preflop += 1

    meaningful = max(1, sum(counts.get(k, 0) for k in _MEANINGFUL))

    def ratio(k: str) -> float:
        return counts.get(k, 0) / meaningful

    flop = 1.0 if "flop" in street_set else 0.0
    turn = 1.0 if "turn" in street_set else 0.0
    river = 1.0 if "river" in street_set else 0.0
    street_depth = (flop + turn + river) / 3.0

    agg = (counts.get("raise", 0) + counts.get("bet", 0)) / meaningful
    pfs = fold_preflop / max(1, fold_total) if fold_total else 0.0
    pot_bb_std = float(np.std(pots)) / bb if pots else 0.0

    v = np.array(
        [
            ratio("call"),
            ratio("check"),
            ratio("bet"),
            ratio("raise"),
            ratio("fold"),
            float(len(actions)),
            float(len(players)),
            float(len(streets)),
            float(np.mean(norms)) if norms else 0.0,
            float(np.max(norms)) if norms else 0.0,
            (float(np.mean(pots)) / bb) if pots else 0.0,
            street_depth,
            _entropy_norm(counts),
            agg,
            pfs,
            pot_bb_std,
        ],
        dtype=np.float64,
    )
    return v


def featurize_chunk(prepared_hands: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Return shape (N_FEATURES,) float32 vector for one chunk."""
    out = np.zeros(N_FEATURES, dtype=np.float32)
    if not prepared_hands:
        return out

    rows = np.stack([_hand_vector(h) for h in prepared_hands], axis=0)
    mean_v = rows.mean(axis=0)
    std_v = rows.std(axis=0) if rows.shape[0] > 1 else np.zeros_like(mean_v)
    out[:16] = mean_v.astype(np.float32)
    out[16:32] = std_v.astype(np.float32)
    out[32] = min(1.0, math.log1p(len(prepared_hands)) / math.log1p(32))
    out[33] = float(FEATURE_VERSION) / 10.0
    return out


def feature_names() -> List[str]:
    names = [f"mean_{k}" for k in _order]
    names += [f"std_{k}" for k in _order]
    names += ["log_chunk_size_norm", "feature_version_tag"]
    return names
