"""
CLI entry point.  Usage:

    python -m cli scan
    python -m cli scan -t AAPL,MSFT,GOOG
    python -m cli scan --excel
    python -m cli scan --top 10 --no-trend-filter
    python -m cli cache
    python -m cli cache --clear
"""

import argparse
import sys
from datetime import date


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alpha-scan",
        description="Equity + options alpha scanner. Screen, score, export.",
    )
    sub = p.add_subparsers(dest="command")

    # -- scan ---------------------------------------------------------------
    scan = sub.add_parser("scan", help="Run the full alpha scan.")
    scan.add_argument(
        "-t",
        "--tickers",
        help="Comma-separated ticker list (overrides screener).",
    )
    scan.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of results to show (default: 15).",
    )
    scan.add_argument(
        "--max",
        type=int,
        default=25,
        help="Max tickers to process from screener (default: 25).",
    )
    scan.add_argument(
        "--excel",
        nargs="?",
        const="auto",
        default=None,
        help="Export to Excel. Optionally specify filename.",
    )
    scan.add_argument(
        "--no-trend-filter",
        action="store_true",
        help="Disable the trend filter.",
    )
    scan.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )

    # -- cache --------------------------------------------------------------
    cache = sub.add_parser("cache", help="Manage the IV history cache.")
    cache.add_argument("--clear", action="store_true", help="Wipe all cached data.")

    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "cache":
        from . import data

        if args.clear:
            data.clear_cache()
            print("Cache cleared.")
        else:
            stats = data.cache_stats()
            if not stats:
                print("Cache is empty.")
            else:
                print(f"{'TICKER':<10} {'READINGS':>8}  {'FIRST':>12}  {'LAST':>12}")
                print("-" * 48)
                for s in stats:
                    print(
                        f"{s['ticker']:<10} {s['readings']:>8}  {s['first']:>12}  {s['last']:>12}"
                    )
        return

    if args.command == "scan":
        from . import engine, report

        tickers = None
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

        df = engine.run_scan(
            tickers=tickers,
            top_n=args.top,
            max_tickers=args.max,
            trend_filter=not args.no_trend_filter,
            quiet=args.quiet,
        )

        if df.empty:
            print("No results. Adjust filters or check data availability.")
            sys.exit(1)

        # Terminal output
        engine.print_report(df)

        # Excel export
        if args.excel is not None:
            path = args.excel
            if path == "auto":
                path = f"alpha_scan_{date.today().isoformat()}.xlsx"
            out = report.generate_excel(df, output_path=path)
            print(f"Report saved: {out}")


if __name__ == "__main__":
    main()
