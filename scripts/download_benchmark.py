#!/usr/bin/env python3
"""Download the public Poker44 benchmark dataset.

Usage:
    python scripts/download_benchmark.py [--out data/benchmark] [--page-size 4]

Downloads every released `sourceDate` returned by
`https://api.poker44.net/api/v1/benchmark/releases` and saves one file per day:

    <out>/<sourceDate>.json   -> full per-day API response with all chunks merged
    <out>/manifest.json       -> top-level summary (totals + per-day stats)

The fetch is paginated (cursor = chunkId of the last item) so we stream large
days in pieces rather than asking the server for one massive blob. Re-running
the script skips days whose chunk count and hand count already match the API.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API_BASE = "https://api.poker44.net/api/v1/benchmark"
USER_AGENT = "poker44-benchmark-downloader/1.0"


def _get_json(url: str, *, retries: int = 5, backoff: float = 2.0) -> dict[str, Any]:
    """GET `url` and return parsed JSON, retrying on transient errors."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if not payload.get("success", False):
                raise RuntimeError(f"API error for {url}: {payload}")
            return payload["data"]
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            wait = backoff ** attempt
            print(f"  ! attempt {attempt}/{retries} failed ({exc}); retrying in {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def list_releases() -> list[dict[str, Any]]:
    data = _get_json(f"{API_BASE}/releases")
    releases = data.get("releases", [])
    # Sort oldest -> newest so progress feels chronological
    releases.sort(key=lambda r: r["sourceDate"])
    return releases


def download_day(source_date: str, page_size: int) -> dict[str, Any]:
    """Page through every chunk for `source_date` and return the merged response."""
    cursor: str | None = None
    merged: dict[str, Any] | None = None
    all_chunks: list[Any] = []
    page = 0

    while True:
        params: dict[str, Any] = {"sourceDate": source_date, "limit": page_size}
        if cursor is not None:
            params["cursor"] = cursor
        url = f"{API_BASE}/chunks?{urlencode(params)}"
        page += 1
        t0 = time.time()
        data = _get_json(url)
        page_chunks = data.get("chunks", [])
        all_chunks.extend(page_chunks)
        elapsed = time.time() - t0
        print(
            f"  page {page:>3}: +{len(page_chunks):>3} chunks "
            f"(running total {len(all_chunks):>4})  in {elapsed:5.1f}s"
        )

        if merged is None:
            merged = {k: v for k, v in data.items() if k != "chunks"}

        cursor = data.get("nextCursor")
        if not cursor:
            break

    assert merged is not None
    merged["chunks"] = all_chunks
    return merged


def count_hands(chunks: list[dict[str, Any]]) -> int:
    return sum(int(c.get("handCount", 0)) for c in chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/benchmark", help="Output directory (default: data/benchmark)")
    parser.add_argument(
        "--page-size",
        type=int,
        default=4,
        help="Chunks per API page; smaller pages reduce per-request memory (default: 4)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if existing file matches the API counts")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching release index from {API_BASE} ...")
    top = _get_json(API_BASE)
    print(
        f"  releaseVersion={top['releaseVersion']}  "
        f"totalChunks={top['totalChunks']}  totalHands={top['totalHands']}  "
        f"latestSourceDate={top['latestSourceDate']}"
    )

    releases = list_releases()
    print(f"Found {len(releases)} released day(s); downloading into {out_dir.resolve()}")

    manifest: dict[str, Any] = {
        "api_base": API_BASE,
        "releaseVersion": top["releaseVersion"],
        "cutoffWindowStart": top["cutoffWindowStart"],
        "downloadedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "expectedTotals": {"chunks": top["totalChunks"], "hands": top["totalHands"]},
        "days": [],
    }

    grand_chunks = 0
    grand_hands = 0

    for release in releases:
        date = release["sourceDate"]
        expected_chunks = int(release["chunkCount"])
        expected_hands = int(release["handCount"])
        target = out_dir / f"{date}.json"

        if target.exists() and not args.force:
            try:
                existing = json.loads(target.read_text())
                got_chunks = len(existing.get("chunks", []))
                got_hands = count_hands(existing.get("chunks", []))
            except (json.JSONDecodeError, OSError):
                got_chunks = got_hands = -1
            if got_chunks == expected_chunks and got_hands == expected_hands:
                print(f"[skip] {date}: already complete ({got_chunks} chunks / {got_hands} hands)")
                manifest["days"].append({
                    "sourceDate": date,
                    "chunks": got_chunks,
                    "hands": got_hands,
                    "file": target.name,
                    "skipped": True,
                })
                grand_chunks += got_chunks
                grand_hands += got_hands
                continue

        print(f"[get ] {date}: expecting {expected_chunks} chunks / {expected_hands} hands")
        day = download_day(date, page_size=args.page_size)
        chunks = day.get("chunks", [])
        got_chunks = len(chunks)
        got_hands = count_hands(chunks)
        if got_chunks != expected_chunks or got_hands != expected_hands:
            print(
                f"  WARNING: count mismatch for {date} "
                f"(got {got_chunks} chunks/{got_hands} hands, "
                f"expected {expected_chunks}/{expected_hands})",
                file=sys.stderr,
            )

        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(day, separators=(",", ":")))
        tmp.replace(target)
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"  saved -> {target}  ({size_mb:.1f} MB)")

        manifest["days"].append({
            "sourceDate": date,
            "chunks": got_chunks,
            "hands": got_hands,
            "file": target.name,
            "skipped": False,
        })
        grand_chunks += got_chunks
        grand_hands += got_hands

    manifest["actualTotals"] = {"chunks": grand_chunks, "hands": grand_hands}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"Downloaded {grand_chunks} chunks / {grand_hands} hands across {len(releases)} day(s).")
    expected = manifest["expectedTotals"]
    if grand_chunks == expected["chunks"] and grand_hands == expected["hands"]:
        print("Totals match the API metadata exactly.")
        return 0
    print(
        f"Totals DIFFER from API metadata "
        f"(expected {expected['chunks']} chunks / {expected['hands']} hands).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
