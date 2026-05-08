#!/usr/bin/env python3
"""
Materialize a large prepared JSONL dataset from the synthetic simulator.

Each line: {\"chunk_label\": 0|1, \"chunk_size\": k, \"hands\": [...sanitized...]}

Example:
  python scripts/miner/training/generate_synthetic_jsonl.py --chunks 40000 --seed 123

Output default: hands_generator/human_hands/synthetic_prepared.jsonl (large; gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "hands_generator" / "human_hands" / "synthetic_prepared.jsonl"


def main() -> None:
    sys.path.insert(0, str(REPO))
    from poker44.training.synthetic import generate_chunk
    from poker44.validator.sanitization import prepare_hand_for_miner

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=int, default=30000, help="Total chunk rows (balanced human/bot).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size-lo", type=int, default=1)
    p.add_argument("--chunk-size-hi", type=int, default=6)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n_half = max(1, args.chunks // 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out.open("w", encoding="utf-8") as out_f:
        for label_name, y in (("human", 0), ("bot", 1)):
            for _ in range(n_half):
                sz = int(rng.integers(args.chunk_size_lo, args.chunk_size_hi + 1))
                hands, _ = generate_chunk(rng, label_name, sz)  # type: ignore[arg-type]
                prepared = [prepare_hand_for_miner(dict(h)) for h in hands]
                rec = {"chunk_label": y, "chunk_size": len(prepared), "hands": prepared}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if written % 5000 == 0:
                    print(f"... {written} chunks", flush=True)

    print(f"Wrote {written} rows to {args.out.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
