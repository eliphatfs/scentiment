"""Fetch intraday 5-min OHLCV from Twelve Data and plot with full pattern
detection, multi-scale trend, confirmation flow, and strength scores.

Usage
-----
    python plot_realtime.py                  # defaults to SPY, last 3 trading days
    python plot_realtime.py AAPL             # specify ticker
    python plot_realtime.py AAPL 5           # specify ticker and number of days

API key
-------
Set TWELVEDATA_API_KEY in a `.env` file at the project root (already gitignored):

    TWELVEDATA_API_KEY=your_key_here
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")

from plot_common import load_api_key, fetch_ohlcv, run_all_patterns, plot_chart


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot candlestick patterns on real-time 5-min data from Twelve Data"
    )
    parser.add_argument("symbol", nargs="?", default="SPY",
                        help="Ticker symbol (default: SPY)")
    parser.add_argument("days", nargs="?", type=int, default=3,
                        help="Number of trading days to fetch (default: 3)")
    args = parser.parse_args()

    api_key = load_api_key()
    # ~78 five-min bars per trading day (6.5 hours)
    outputsize = min(args.days * 78, 5000)
    df = fetch_ohlcv(args.symbol, "5min", outputsize, api_key)

    if len(df) < 10:
        print("Not enough data to analyze.", file=sys.stderr)
        sys.exit(1)

    # Tighter tolerance for intraday (0.1%)
    RAW, CONF, PATTERNS, STRENGTHS = run_all_patterns(df, match_tolerance=0.001)
    plot_chart(
        df, args.symbol, RAW, CONF, PATTERNS, STRENGTHS,
        interval_label="5-min bars",
        date_fmt="%Y-%m-%d %H:%M",
        tick_fmt="%m/%d\n%H:%M",
        out_prefix="realtime",
    )
