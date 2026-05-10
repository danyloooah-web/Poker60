#!/usr/bin/env python3
"""Convert downloaded benchmark daily JSON -> miner-ready training JSONL.

Reads every `data/benchmark/<sourceDate>.json` produced by
`scripts/download_benchmark.py` and emits one training row per *inner* chunk
(i.e. one ground-truth label per row) into the JSONL file consumed by
`scripts/miner/training/train_model.py`.

Each output row is shape:

    {
        "source_date":      "2026-05-08",
        "release_version":  "v1.1",
        "source_chunk_id":  "<window chunkId>::<inner_index>",
        "chunk_hash":       "<window chunkHash>",
        "chunk_label":      0 | 1,
        "chunk_label_name": "human" | "bot",
        "hands":            [<sanitized hand dict>, ...],
    }

Hands are passed through `prepare_hand_for_miner` to match what miners actually
see at inference time (12 evenly-spaced actions per hand, BB-normalized
amounts, leakage fields stripped).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from poker44.validator.sanitization import prepare_hand_for_miner  # noqa: E402


def emit_day(day_path: Path, out_fp) -> dict:
    payload = json.loads(day_path.read_text())
    source_date = payload.get("sourceDate") or day_path.stem
    release_version = payload.get("releaseVersion", "")
    rows = 0
    label_counts: Counter[int] = Counter()
    hands_total = 0

    for window in payload.get("chunks", []):
        chunk_id = window.get("chunkId", "")
        chunk_hash = window.get("chunkHash", "")
        inner_chunks = window.get("chunks") or []
        gt = window.get("groundTruth") or []
        gt_names = window.get("groundTruthLabels") or []

        if len(inner_chunks) != len(gt):
            print(
                f"  WARNING {source_date} {chunk_id[:8]}: "
                f"{len(inner_chunks)} inner chunks vs {len(gt)} labels — skipping window",
                file=sys.stderr,
            )
            continue

        for idx, (inner, label) in enumerate(zip(inner_chunks, gt)):
            try:
                label_int = int(label)
            except (TypeError, ValueError):
                continue
            if label_int not in (0, 1):
                continue

            hands = [prepare_hand_for_miner(h) for h in inner if isinstance(h, dict)]
            if not hands:
                continue

            label_name = (
                str(gt_names[idx]).lower()
                if idx < len(gt_names)
                else ("bot" if label_int == 1 else "human")
            )
            row = {
                "source_date": source_date,
                "release_version": release_version,
                "source_chunk_id": f"{chunk_id}::{idx}",
                "chunk_hash": chunk_hash,
                "chunk_label": label_int,
                "chunk_label_name": label_name,
                "hands": hands,
            }
            out_fp.write(json.dumps(row, separators=(",", ":")))
            out_fp.write("\n")
            rows += 1
            label_counts[label_int] += 1
            hands_total += len(hands)

    return {
        "sourceDate": source_date,
        "rows": rows,
        "labelCounts": {int(k): int(v) for k, v in label_counts.items()},
        "hands": hands_total,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", default="data/benchmark", help="Directory of downloaded daily JSON")
    p.add_argument(
        "--out",
        default="scripts/miner/training/benchmark/benchmark_prepared.jsonl",
        help="Destination JSONL (overwrites)",
    )
    p.add_argument(
        "--summary",
        default=None,
        help="Optional summary JSON path (default: <out>.summary.json)",
    )
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    day_files = sorted(p for p in in_dir.glob("*.json") if p.name not in {"manifest.json"})
    if not day_files:
        print(f"No daily JSONs found under {in_dir}", file=sys.stderr)
        return 1
    print(f"Preparing {len(day_files)} daily files -> {out_path}")

    tmp = out_path.with_suffix(".jsonl.tmp")
    summary = {"days": [], "totals": {"rows": 0, "hands": 0, "labelCounts": {0: 0, 1: 0}}}
    with tmp.open("w", encoding="utf-8") as fp:
        for day_path in day_files:
            stats = emit_day(day_path, fp)
            summary["days"].append(stats)
            summary["totals"]["rows"] += stats["rows"]
            summary["totals"]["hands"] += stats["hands"]
            for k, v in stats["labelCounts"].items():
                summary["totals"]["labelCounts"][k] = (
                    summary["totals"]["labelCounts"].get(k, 0) + v
                )
            print(
                f"  {stats['sourceDate']}: rows={stats['rows']:>4}  "
                f"hands={stats['hands']:>6}  "
                f"labels={dict(stats['labelCounts'])}"
            )
    tmp.replace(out_path)

    summary_path = Path(args.summary) if args.summary else out_path.with_suffix(".jsonl.summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    totals = summary["totals"]
    print()
    print(
        f"Wrote {totals['rows']} rows / {totals['hands']} hands "
        f"(humans={totals['labelCounts'].get(0, 0)} bots={totals['labelCounts'].get(1, 0)}) "
        f"-> {out_path}"
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
