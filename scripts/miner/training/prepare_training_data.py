#!/usr/bin/env python3
"""
Build miner-aligned training artifacts from local PokerStars text hands.

Run from repository root (Poker44-subnet):

  python scripts/miner/training/prepare_training_data.py <command>

Requires: pip install -e . (so poker44.validator.sanitization is importable).
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def human_hands_dir() -> Path:
    return repo_root() / "hands_generator" / "human_hands"


def load_human_hands_parser():
    import importlib.util

    path = human_hands_dir() / "human_hands_parser.py"
    spec = importlib.util.spec_from_file_location("human_hands_parser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load parser from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_merge(_: argparse.Namespace) -> None:
    """Step 1: Concatenate poker_hands/**/*.txt -> massive_data.txt"""
    data_parser = human_hands_dir() / "data_parser.py"
    if not data_parser.is_file():
        print(f"Missing {data_parser}", file=sys.stderr)
        sys.exit(1)
    runpy.run_path(str(data_parser), run_name="__main__")


def cmd_parse(ns: argparse.Namespace) -> None:
    """Step 2: PokerStars text -> canonical JSON (human_hands.json)."""
    root = repo_root()
    hdir = human_hands_dir()
    input_path = Path(ns.input) if ns.input else None
    if input_path is None:
        massive = hdir / "massive_data.txt"
        fallback = hdir / "data.txt"
        input_path = massive if massive.is_file() else fallback
    output_path = Path(ns.output) if ns.output else hdir / "human_hands.json"

    if not input_path.is_file():
        print(
            f"No input file at {input_path}. Place .txt under poker_hands/ and run: merge",
            file=sys.stderr,
        )
        sys.exit(1)

    mod = load_human_hands_parser()
    hands = mod.parse_file(input_path)
    if not ns.no_anonymize:
        hands = mod.anonymize_all_hands(hands)
    output_path.write_text(json.dumps(hands, indent=2), encoding="utf-8")
    print(f"Wrote {len(hands)} hands to {output_path.relative_to(root)}")


def cmd_export(ns: argparse.Namespace) -> None:
    """Step 3: Canonical JSON -> JSONL with miner-visible sanitized hands."""
    sys.path.insert(0, str(repo_root()))
    from poker44.validator.sanitization import prepare_hand_for_miner

    root = repo_root()
    hdir = human_hands_dir()
    input_path = Path(ns.input) if ns.input else hdir / "human_hands.json"
    output_path = Path(ns.output) if ns.output else hdir / "training_prepared.jsonl"

    if not input_path.is_file():
        print(f"Missing {input_path}; run: parse", file=sys.stderr)
        sys.exit(1)

    hands: list[dict] = json.loads(input_path.read_text(encoding="utf-8"))
    chunk_size = max(1, int(ns.chunk_size))
    # 0 = human, 1 = bot (matches validator batch_label convention in forward.py)
    label_map = {"human": 0, "bot": 1}

    lines_out = 0
    with output_path.open("w", encoding="utf-8") as out:
        for i in range(0, len(hands), chunk_size):
            group = hands[i : i + chunk_size]
            raw_labels = [str(h.get("label", "human")).lower() for h in group]
            if len(set(raw_labels)) != 1:
                print(
                    f"Warning: chunk at index {i} mixes labels {raw_labels!r}; skipping.",
                    file=sys.stderr,
                )
                continue
            chunk_label = label_map.get(raw_labels[0], 0)
            prepared = [prepare_hand_for_miner(dict(h)) for h in group]
            rec = {
                "chunk_label": chunk_label,
                "chunk_size": len(prepared),
                "hands": prepared,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lines_out += 1

    print(
        f"Wrote {lines_out} chunk records ({len(hands)} hands, chunk_size={chunk_size}) "
        f"to {output_path.relative_to(root)}"
    )


def cmd_all(ns: argparse.Namespace) -> None:
    cmd_merge(ns)
    cmd_parse(argparse.Namespace(input=None, output=None, no_anonymize=ns.no_anonymize))
    export_ns = argparse.Namespace(
        input=None,
        output=None,
        chunk_size=ns.chunk_size,
    )
    cmd_export(export_ns)


def main() -> None:
    root = repo_root()
    if not (root / "poker44").is_dir():
        print("Run this script from the Poker44-subnet repository root.", file=sys.stderr)
        sys.exit(1)

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("merge", help="Merge poker_hands/**/*.txt into massive_data.txt")

    pp = sub.add_parser("parse", help="Parse text file to human_hands.json")
    pp.add_argument(
        "--input",
        help=f"Default: massive_data.txt or data.txt under {human_hands_dir()}",
    )
    pp.add_argument("--output", help="Default: human_hands.json")
    pp.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Skip player anonymization (not recommended).",
    )

    pe = sub.add_parser("export", help="Sanitize to training_prepared.jsonl")
    pe.add_argument("--input", help="Default: human_hands.json")
    pe.add_argument("--output", help="Default: training_prepared.jsonl")
    pe.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Hands per training row (homogeneous chunk; default 1).",
    )

    pa = sub.add_parser("all", help="Run merge, then parse, then export")
    pa.add_argument("--no-anonymize", action="store_true")
    pa.add_argument("--chunk-size", type=int, default=1)

    ns = p.parse_args()
    if ns.command == "merge":
        cmd_merge(ns)
    elif ns.command == "parse":
        cmd_parse(ns)
    elif ns.command == "export":
        cmd_export(ns)
    elif ns.command == "all":
        cmd_all(ns)


if __name__ == "__main__":
    main()
