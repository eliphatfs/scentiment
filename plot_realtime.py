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
import os
import sys
import json
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from patterns.reversal import (
    hammer, hanging_man, engulfing, dark_cloud_cover, piercing_pattern,
)
from patterns.stars import (
    inverted_hammer, shooting_star, morning_star, evening_star,
    morning_doji_star, evening_doji_star,
)
from patterns.doji import (
    doji_at_top, doji_at_bottom, long_legged_doji, rickshaw_man,
    gravestone_doji, tri_star,
)
from patterns.more_reversals import (
    harami, harami_cross, tweezers_top, tweezers_bottom, belt_hold,
    upside_gap_two_crows, three_black_crows, counterattack_lines,
    three_mountains, three_rivers, dumpling_top, fry_pan_bottom,
    tower_top, tower_bottom,
)
from patterns.continuation import (
    window_up, window_down, rising_three_methods, falling_three_methods,
    three_white_soldiers, separating_lines,
)
from patterns.confirmation import (
    confirmed_hanging_man, confirmed_shooting_star, confirmed_inverted_hammer,
    confirmed_doji_at_top, confirmed_doji_at_bottom, confirmed_gravestone_doji,
)
from patterns.scoring import pattern_strength
from trend import multi_scale_trend, trend_terminations


# ── Load API key ─────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    """Read TWELVEDATA_API_KEY from environment or .env file."""
    key = os.environ.get("TWELVEDATA_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(__file__) or ".", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "TWELVEDATA_API_KEY":
                    return v.strip().strip("\"'")
    print("Error: TWELVEDATA_API_KEY not found.", file=sys.stderr)
    print("Set it in .env or as an environment variable.", file=sys.stderr)
    sys.exit(1)


# ── Fetch data ───────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, outputsize: int, api_key: str) -> pd.DataFrame:
    """Fetch 5-min OHLCV bars from Twelve Data."""
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval=5min&outputsize={outputsize}"
        f"&apikey={api_key}&format=JSON"
    )
    req = Request(url, headers={"User-Agent": "candlestick-plotter/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except HTTPError as e:
        print(f"API error: {e.code} {e.reason}", file=sys.stderr)
        sys.exit(1)

    if "values" not in data:
        msg = data.get("message", data.get("status", "unknown error"))
        print(f"API error: {msg}", file=sys.stderr)
        sys.exit(1)

    rows = data["values"]
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "volume" not in df.columns:
        df["volume"] = 0.0

    print(f"Fetched {len(df)} bars for {symbol} "
          f"({df.index[0]} → {df.index[-1]})")
    return df


# ── Pattern detection ────────────────────────────────────────────────────────

def run_all_patterns(df):
    RAW = {
        "hammer":               hammer(df),
        "hanging_man":          hanging_man(df),
        "engulfing":            engulfing(df),
        "dark_cloud_cover":     dark_cloud_cover(df),
        "piercing_pattern":     piercing_pattern(df),
        "inverted_hammer":      inverted_hammer(df),
        "shooting_star":        shooting_star(df),
        "morning_star":         morning_star(df),
        "evening_star":         evening_star(df),
        "morning_doji_star":    morning_doji_star(df),
        "evening_doji_star":    evening_doji_star(df),
        "doji_at_top":          doji_at_top(df),
        "doji_at_bottom":       doji_at_bottom(df),
        "long_legged_doji":     long_legged_doji(df),
        "rickshaw_man":         rickshaw_man(df),
        "gravestone_doji":      gravestone_doji(df),
        "tri_star":             tri_star(df),
        "harami":               harami(df),
        "harami_cross":         harami_cross(df),
        "tweezers_top":         tweezers_top(df),
        "tweezers_bottom":      tweezers_bottom(df),
        "belt_hold":            belt_hold(df),
        "upside_gap_two_crows": upside_gap_two_crows(df),
        "three_black_crows":    three_black_crows(df),
        "counterattack_lines":  counterattack_lines(df),
        "three_mountains":      three_mountains(df),
        "three_rivers":         three_rivers(df),
        "dumpling_top":         dumpling_top(df),
        "fry_pan_bottom":       fry_pan_bottom(df),
        "tower_top":            tower_top(df),
        "tower_bottom":         tower_bottom(df),
        "window_up":            window_up(df),
        "window_down":          window_down(df),
        "rising_three_methods": rising_three_methods(df),
        "falling_three_methods":falling_three_methods(df),
        "three_white_soldiers": three_white_soldiers(df),
        "separating_lines":     separating_lines(df),
    }
    CONF = {
        "hanging_man":      confirmed_hanging_man(df),
        "shooting_star":    confirmed_shooting_star(df),
        "inverted_hammer":  confirmed_inverted_hammer(df),
        "doji_at_top":      confirmed_doji_at_top(df),
        "doji_at_bottom":   confirmed_doji_at_bottom(df),
        "gravestone_doji":  confirmed_gravestone_doji(df),
    }
    PATTERNS = {name: CONF.get(name, sig) for name, sig in RAW.items()}
    STRENGTHS = {name: pattern_strength(df, name, sig) for name, sig in RAW.items()}
    return RAW, CONF, PATTERNS, STRENGTHS


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_chart(df, symbol, RAW, CONF, PATTERNS, STRENGTHS):
    scales = multi_scale_trend(df)
    terminations = trend_terminations(df, PATTERNS)
    dates = df.index
    price_range = df["high"].max() - df["low"].min()
    # Adaptive offsets scaled to price range
    marker_offset = price_range * 0.02
    text_offset = price_range * 0.04
    text_spacing = price_range * 0.015
    term_offset = price_range * 0.05

    # Print detected signals
    n_signals = 0
    for name, sig in RAW.items():
        hits = sig.dropna()
        if not len(hits):
            continue
        confirmed_sig = CONF.get(name)
        for dt, val in hits.items():
            n_signals += 1
            score = STRENGTHS[name].get(dt, float("nan"))
            score_str = f"  strength={score:.2f}" if not np.isnan(score) else ""
            if confirmed_sig is not None:
                conf_dates = set(confirmed_sig.dropna().index)
                status = "  (needs confirmation)"
                print(f"  {dt}  {name:25s}  {val}{score_str}{status}")
                idx = df.index.get_loc(dt)
                for k in range(1, 3):
                    if idx + k < len(df):
                        cdt = df.index[idx + k]
                        if cdt in conf_dates:
                            print(f"  {cdt}  {'':25s}  ↳ CONFIRMED")
                            break
            else:
                print(f"  {dt}  {name:25s}  {val}{score_str}")
    print(f"\n{n_signals} signals detected")

    if len(terminations):
        print(f"\nTrend termination signals ({len(terminations)}):")
        for _, row in terminations.iterrows():
            print(f"  {row['date']}  {row['pattern']:25s}  "
                  f"{row['signal']} vs {row['scale']} {row['trend_direction']}")

    # Build figure
    fig, axes = plt.subplots(
        2, 1, figsize=(20, 12), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.05},
    )
    ax = axes[0]
    ax_trend = axes[1]

    # Trend background
    TREND_COLORS = {"up": "#d4efdf", "down": "#fadbd8"}
    for i, dt in enumerate(dates):
        med = scales.loc[dt, "medium"]
        if med in TREND_COLORS:
            ax.axvspan(i - 0.5, i + 0.5, color=TREND_COLORS[med], alpha=0.4, zorder=0)

    # Candlesticks
    body_width = 0.7
    for i, (dt, row) in enumerate(df.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#2ecc71" if c >= o else "#e74c3c"
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)
        body_lo, body_hi = min(o, c), max(o, c)
        rect = mpatches.FancyBboxPatch(
            (i - body_width / 2, body_lo), body_width,
            max(body_hi - body_lo, price_range * 0.001),
            boxstyle="square,pad=0",
            linewidth=0.5, edgecolor=color, facecolor=color, zorder=3,
        )
        ax.add_patch(rect)

    # Pattern annotations
    BULLISH_COLOR = "#3498db"
    BEARISH_COLOR = "#e67e22"
    TERM_COLOR = "#9b59b6"
    PENDING_COLOR = "#95a5a6"
    CONFIRMED_COLOR = "#27ae60"

    annotation_lines: dict[int, dict[str, list[tuple[str, str]]]] = {}
    hit_strengths: dict[tuple[int, str], float] = {}

    for name, sig in RAW.items():
        needs_conf = name in CONF
        confirmed_sig = CONF.get(name)
        confirmed_dates = set(confirmed_sig.dropna().index) if confirmed_sig is not None else set()

        for dt, val in sig.dropna().items():
            i = dates.get_loc(dt)
            score = STRENGTHS[name].get(dt, float("nan"))
            label = name.replace("_", " ")
            if val not in ("bullish", "bearish"):
                continue
            key = (i, val)
            if not np.isnan(score):
                hit_strengths[key] = max(hit_strengths.get(key, 0), score)

            bar_entry = annotation_lines.setdefault(i, {})
            if needs_conf:
                bar_entry.setdefault(val, []).append((f"{label} (pending)", PENDING_COLOR))
                for k in range(1, 3):
                    if i + k < len(df) and df.index[i + k] in confirmed_dates:
                        conf_entry = annotation_lines.setdefault(i + k, {})
                        conf_entry.setdefault(val, []).append(
                            (f"\u2713 confirmed: {label}", CONFIRMED_COLOR)
                        )
                        break
            else:
                color = BULLISH_COLOR if val == "bullish" else BEARISH_COLOR
                bar_entry.setdefault(val, []).append((label, color))

    for bar_idx, directions in sorted(annotation_lines.items()):
        for direction, lines in directions.items():
            if not lines:
                continue

            strength = hit_strengths.get((bar_idx, direction), 0.5)
            msize = 4 + 6 * strength

            text_lines = [text for text, _ in lines]
            text_lines.append(f"[{strength:.0%}]")
            label_text = "\n".join(text_lines)

            colors_present = [c for _, c in lines]
            if CONFIRMED_COLOR in colors_present:
                ann_color = marker_color = marker_face = CONFIRMED_COLOR
            elif all(c == PENDING_COLOR for c in colors_present):
                ann_color = marker_color = PENDING_COLOR
                marker_face = "none"
            else:
                ann_color = BULLISH_COLOR if direction == "bullish" else BEARISH_COLOR
                marker_color = marker_face = ann_color

            n_lines = len(text_lines)

            if direction == "bullish":
                y = df.iloc[bar_idx]["low"] - text_offset
                ax.annotate(
                    label_text,
                    xy=(bar_idx, df.iloc[bar_idx]["low"]),
                    xytext=(bar_idx, y - text_spacing * n_lines),
                    fontsize=5.5, color=ann_color,
                    ha="center", va="top",
                    arrowprops=dict(arrowstyle="-", color=ann_color, lw=0.5),
                )
                ax.plot(bar_idx, df.iloc[bar_idx]["low"] - marker_offset, "^",
                        color=marker_color, markerfacecolor=marker_face,
                        markeredgecolor=marker_color, markeredgewidth=1.2,
                        markersize=msize, zorder=5)
            else:
                y = df.iloc[bar_idx]["high"] + text_offset
                ax.annotate(
                    label_text,
                    xy=(bar_idx, df.iloc[bar_idx]["high"]),
                    xytext=(bar_idx, y + text_spacing * n_lines),
                    fontsize=5.5, color=ann_color,
                    ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=ann_color, lw=0.5),
                )
                ax.plot(bar_idx, df.iloc[bar_idx]["high"] + marker_offset, "v",
                        color=marker_color, markerfacecolor=marker_face,
                        markeredgecolor=marker_color, markeredgewidth=1.2,
                        markersize=msize, zorder=5)

    # Trend termination diamonds
    if len(terminations):
        term_set = set()
        for _, row in terminations.iterrows():
            i = dates.get_loc(row["date"])
            if i not in term_set:
                term_set.add(i)
                ax.plot(i, df.iloc[i]["high"] + term_offset, "D",
                        color=TERM_COLOR, markersize=7, zorder=6)

    ax.set_ylabel("Price ($)", fontsize=10)
    date_range = f"{dates[0].strftime('%Y-%m-%d %H:%M')} → {dates[-1].strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(
        f"{symbol} — 5-min bars — {date_range}\n"
        f"Patterns with confirmation flow and strength scores",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=BULLISH_COLOR,
               markersize=8, label="Bullish signal"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=BEARISH_COLOR,
               markersize=8, label="Bearish signal"),
        Line2D([0], [0], marker="^", color=PENDING_COLOR, markerfacecolor="none",
               markeredgewidth=1.2, markersize=7, label="Pending confirmation"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=CONFIRMED_COLOR,
               markersize=8, label="Confirmed"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=TERM_COLOR,
               markersize=7, label="Trend termination"),
        mpatches.Patch(color=TREND_COLORS["up"], alpha=0.4, label="Pivot uptrend"),
        mpatches.Patch(color=TREND_COLORS["down"], alpha=0.4, label="Pivot downtrend"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    # Bottom panel: multi-scale trend
    SCALE_LABELS = {"micro": 0, "short": 1, "medium": 2, "long": 3}
    SCALE_NAMES = {
        "micro": "Micro (body runs)",
        "short": "Short (reg. slope)",
        "medium": "Medium (order 1)",
        "long": "Long (order 2)",
    }
    for scale_name, y_pos in SCALE_LABELS.items():
        for i, dt in enumerate(dates):
            val = scales.loc[dt, scale_name]
            if val == "up":
                color = "#27ae60"
            elif val == "down":
                color = "#c0392b"
            else:
                color = "#bdc3c7"
            alpha = 0.7 if val in ("up", "down") else 0.3
            ax_trend.barh(y_pos, 1, left=i - 0.5, height=0.7,
                          color=color, alpha=alpha, linewidth=0)

    ax_trend.set_yticks(list(SCALE_LABELS.values()))
    ax_trend.set_yticklabels([SCALE_NAMES[k] for k in SCALE_LABELS], fontsize=8)
    ax_trend.set_ylabel("Trend scale", fontsize=10)
    ax_trend.set_ylim(-0.5, 3.5)

    # X-axis: show hour boundaries
    tick_positions, tick_labels = [], []
    prev_label = None
    for i, dt in enumerate(dates):
        lbl = dt.strftime("%m/%d %H:%M")
        hour_lbl = dt.strftime("%m/%d %H:00")
        if hour_lbl != prev_label and dt.minute < 5:
            tick_positions.append(i)
            tick_labels.append(dt.strftime("%m/%d\n%H:%M"))
            prev_label = hour_lbl
    ax_trend.set_xticks(tick_positions)
    ax_trend.set_xticklabels(tick_labels, fontsize=7)
    ax_trend.set_xlim(-1, len(df))
    ax_trend.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = f"realtime_{symbol.lower()}.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved → {out}")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot candlestick patterns on real-time 5-min data from Twelve Data"
    )
    parser.add_argument("symbol", nargs="?", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("days", nargs="?", type=int, default=3,
                        help="Number of trading days to fetch (default: 3)")
    args = parser.parse_args()

    api_key = _load_api_key()
    # ~78 five-min bars per trading day (6.5 hours)
    outputsize = min(args.days * 78, 5000)
    df = fetch_ohlcv(args.symbol, outputsize, api_key)

    if len(df) < 10:
        print("Not enough data to analyze.", file=sys.stderr)
        sys.exit(1)

    RAW, CONF, PATTERNS, STRENGTHS = run_all_patterns(df)
    plot_chart(df, args.symbol, RAW, CONF, PATTERNS, STRENGTHS)
