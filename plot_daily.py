"""Fetch daily OHLCV from Twelve Data and plot with full pattern detection,
multi-scale trend, confirmation flow, and strength scores.

Usage
-----
    python plot_daily.py                     # defaults to SPY, 6 months
    python plot_daily.py AAPL                # specify ticker
    python plot_daily.py AAPL 12             # specify ticker and months

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


def _weekly_tick(i, dt, prev_state):
    """Place a tick at the first bar of each week."""
    week = dt.isocalendar()[1]
    place = (week != prev_state)
    return place, week


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot candlestick patterns on daily data from Twelve Data"
    )
    parser.add_argument("symbol", nargs="?", default="SPY",
                        help="Ticker symbol (default: SPY)")
    parser.add_argument("months", nargs="?", type=int, default=6,
                        help="Number of months to fetch (default: 6)")
    args = parser.parse_args()

    api_key = load_api_key()
    # ~21 trading days per month
    outputsize = min(args.months * 21, 5000)
    df = fetch_ohlcv(args.symbol, "1day", outputsize, api_key)

    if len(df) < 10:
        print("Not enough data to analyze.", file=sys.stderr)
        sys.exit(1)

    # Daily uses default tolerances (0.3% for three_mountains/rivers via
    # their function defaults; match_tolerance controls the rest at 0.1%)
    RAW, CONF, PATTERNS, STRENGTHS = run_all_patterns(df)
    plot_chart(
        df, args.symbol, RAW, CONF, PATTERNS, STRENGTHS,
        interval_label="daily bars",
        date_fmt="%Y-%m-%d",
        tick_fmt="%b %d",
        tick_test=_weekly_tick,
        out_prefix="daily",
    )
