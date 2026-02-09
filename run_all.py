#!/usr/bin/env python3
"""Execute all options-screener notebooks in place.

Usage
-----
    python run_all.py                 # run the full pipeline
    python run_all.py --only bto      # run only bto_call_put_screener
    python run_all.py --only leaps    # run leaps_discovery → leaps_trade_readiness
    python run_all.py --timeout 1800  # custom per-cell timeout (seconds)
    python run_all.py --dry-run       # list notebooks without executing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# -------------------------------------------------------------------
# Notebook execution order.
# leaps_discovery feeds into leaps_trade_readiness, so ordering matters.
# -------------------------------------------------------------------
NOTEBOOKS: list[dict[str, str]] = [
    {"key": "bto", "file": "bto_call_put_screener.ipynb"},
    {"key": "leaps", "file": "leaps_discovery_screener.ipynb"},
    {"key": "trade", "file": "leaps_trade_readiness.ipynb"},
    {"key": "iv", "file": "iv_strategy_analysis.ipynb"},
    {"key": "fan", "file": "call_fan_discovery.ipynb"},
]

ROOT = Path(__file__).resolve().parent


def run_notebook(path: Path, timeout: int = 600) -> float:
    """Execute a notebook in place and return elapsed seconds."""
    nb = nbformat.read(path, as_version=4)
    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name="python3",
    )
    start = time.perf_counter()
    ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
    elapsed = time.perf_counter() - start
    nbformat.write(nb, path)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run options-screener notebooks.")
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="KEY",
        help="Run only notebooks matching these keys: "
        + ", ".join(nb["key"] for nb in NOTEBOOKS),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell execution timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List notebooks that would be executed, then exit.",
    )
    args = parser.parse_args()

    # Resolve which notebooks to run.
    if args.only:
        selected = [nb for nb in NOTEBOOKS if nb["key"] in args.only]
        unknown = set(args.only) - {nb["key"] for nb in selected}
        if unknown:
            print(f"Unknown keys: {', '.join(sorted(unknown))}")
            print(f"Valid keys: {', '.join(nb['key'] for nb in NOTEBOOKS)}")
            sys.exit(1)
    else:
        selected = NOTEBOOKS

    if args.dry_run:
        print("Notebooks that would be executed:")
        for nb in selected:
            print(f"  [{nb['key']}] {nb['file']}")
        return

    total_start = time.perf_counter()
    results: list[tuple[str, str, float | None]] = []

    for nb in selected:
        path = ROOT / nb["file"]
        if not path.exists():
            print(f"SKIP  {nb['file']} (not found)")
            results.append((nb["key"], nb["file"], None))
            continue

        print(f"RUN   {nb['file']} ...", flush=True)
        try:
            elapsed = run_notebook(path, timeout=args.timeout)
            print(f"  OK  {elapsed:.1f}s")
            results.append((nb["key"], nb["file"], elapsed))
        except Exception as exc:
            print(f"  FAIL  {exc}")
            results.append((nb["key"], nb["file"], None))

    total = time.perf_counter() - total_start

    # Summary
    print("\n" + "=" * 52)
    print(f"{'Key':<8} {'Notebook':<38} {'Time':>6}")
    print("-" * 52)
    for key, name, t in results:
        status = f"{t:.1f}s" if t is not None else "FAIL"
        print(f"{key:<8} {name:<38} {status:>6}")
    print("=" * 52)
    print(f"Total wall time: {total:.1f}s")


if __name__ == "__main__":
    main()
