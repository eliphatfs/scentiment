"""Load exhibit 9.2 crude oil data, run all pattern detections, and plot
with multi-scale trend background, trend-termination signals, confirmation
markers, and pattern strength scores."""

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
    confirmed_harami, confirmed_harami_cross,
)
from patterns.scoring import pattern_strength
from trend import multi_scale_trend, trend_terminations


# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/exhibit_9_2.csv", parse_dates=["Date"])
df = df.set_index("Date").sort_index()
df.index.name = "datetime"
if "volume" not in df.columns:
    df["volume"] = 0.0
df.columns = [c.lower() for c in df.columns]

# ── Run all patterns ─────────────────────────────────────────────────────────
# Raw (unconfirmed) signals — every pattern fires on its own bar.
RAW_PATTERNS = {
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

# Patterns that require next-bar confirmation per Nison.
# Confirmed signals fire on the confirmation bar, not the pattern bar.
NEEDS_CONFIRMATION = {
    "hanging_man":      confirmed_hanging_man(df),
    "shooting_star":    confirmed_shooting_star(df),
    "inverted_hammer":  confirmed_inverted_hammer(df),
    "doji_at_top":      confirmed_doji_at_top(df),
    "doji_at_bottom":   confirmed_doji_at_bottom(df),
    "gravestone_doji":  confirmed_gravestone_doji(df),
    "harami":           confirmed_harami(df),
    "harami_cross":     confirmed_harami_cross(df),
}

# PATTERNS = confirmed where available, raw otherwise (used for trend terminations)
PATTERNS = {}
for name, sig in RAW_PATTERNS.items():
    PATTERNS[name] = NEEDS_CONFIRMATION.get(name, sig)

# ── Compute strength scores ─────────────────────────────────────────────────
STRENGTHS = {}
for name, sig in RAW_PATTERNS.items():
    STRENGTHS[name] = pattern_strength(df, name, sig)

# ── Multi-scale trend ────────────────────────────────────────────────────────
scales = multi_scale_trend(df)
terminations = trend_terminations(df, PATTERNS)

print("Detected signals:")
for name, sig in RAW_PATTERNS.items():
    hits = sig.dropna()
    if not len(hits):
        continue
    confirmed_sig = NEEDS_CONFIRMATION.get(name)
    for dt, val in hits.items():
        score = STRENGTHS[name].get(dt, float("nan"))
        score_str = f"  strength={score:.2f}" if not np.isnan(score) else ""
        if confirmed_sig is not None:
            # Check if this raw signal was confirmed
            conf_dates = set(confirmed_sig.dropna().index)
            status = "  (needs confirmation)"
            print(f"  {dt.date()}  {name:25s}  {val}{score_str}{status}")
            # Find the confirmation bar (confirmed signal within max_wait bars)
            idx = df.index.get_loc(dt)
            for k in range(1, 3):
                if idx + k < len(df):
                    cdt = df.index[idx + k]
                    if cdt in conf_dates:
                        print(f"  {cdt.date()}  {'':25s}  ↳ CONFIRMED")
                        break
        else:
            print(f"  {dt.date()}  {name:25s}  {val}{score_str}")

print(f"\nTrend termination signals ({len(terminations)}):")
if len(terminations):
    for _, row in terminations.iterrows():
        print(f"  {row['date'].date()}  {row['pattern']:25s}  "
              f"{row['signal']} vs {row['scale']} {row['trend_direction']}")

# ── Plot ─────────────────────────────────────────────────────────────────────
dates = df.index

fig, axes = plt.subplots(
    2, 1, figsize=(18, 12), height_ratios=[3, 1],
    sharex=True, gridspec_kw={"hspace": 0.05},
)
ax = axes[0]
ax_trend = axes[1]

# ── Top panel: candlesticks + patterns ────────────────────────────────────────

# Trend background shading on the candlestick chart (using medium/pivot trend)
TREND_COLORS = {"up": "#d4efdf", "down": "#fadbd8"}  # light green / light red
for i, dt in enumerate(dates):
    med = scales.loc[dt, "medium"]
    if med in TREND_COLORS:
        ax.axvspan(i - 0.5, i + 0.5, color=TREND_COLORS[med], alpha=0.4, zorder=0)

# Candlesticks
for i, (dt, row) in enumerate(df.iterrows()):
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    color = "#2ecc71" if c >= o else "#e74c3c"
    ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)
    body_lo, body_hi = min(o, c), max(o, c)
    rect = mpatches.FancyBboxPatch(
        (i - 0.35, body_lo), 0.7, max(body_hi - body_lo, 0.5),
        boxstyle="square,pad=0",
        linewidth=0.5, edgecolor=color, facecolor=color, zorder=3,
    )
    ax.add_patch(rect)

# Pattern annotations
BULLISH_COLOR = "#3498db"
BEARISH_COLOR = "#e67e22"
TERM_COLOR = "#9b59b6"  # purple for termination signals

# Collect raw (pending) hits and confirmed hits separately
bullish_hits = {}       # bar_idx → [pattern names] for patterns NOT needing confirmation
bearish_hits = {}
pending_bullish = {}    # bar_idx → [pattern names] for unconfirmed signals (needs confirmation)
pending_bearish = {}
confirmed_bullish = {}  # bar_idx → [pattern names] for confirmation bar annotations
confirmed_bearish = {}
hit_strengths = {}      # (bar_index, direction) → max strength score

PENDING_COLOR_BULL = "#85c1e9"   # lighter blue for pending
PENDING_COLOR_BEAR = "#f0b27a"   # lighter orange for pending
CONFIRMED_COLOR = "#27ae60"      # green for "confirmed" label

for name, sig in RAW_PATTERNS.items():
    needs_conf = name in NEEDS_CONFIRMATION
    confirmed_sig = NEEDS_CONFIRMATION.get(name)
    confirmed_dates = set(confirmed_sig.dropna().index) if confirmed_sig is not None else set()

    for dt, val in sig.dropna().items():
        i = dates.get_loc(dt)
        score = STRENGTHS[name].get(dt, float("nan"))
        label = name.replace("_", " ")

        if val == "bullish":
            key = (i, "bullish")
            if needs_conf:
                pending_bullish.setdefault(i, []).append(label)
                # Find confirmation bar
                for k in range(1, 3):
                    if i + k < len(df) and df.index[i + k] in confirmed_dates:
                        confirmed_bullish.setdefault(i + k, []).append(label)
                        break
            else:
                bullish_hits.setdefault(i, []).append(label)
        elif val == "bearish":
            key = (i, "bearish")
            if needs_conf:
                pending_bearish.setdefault(i, []).append(label)
                for k in range(1, 3):
                    if i + k < len(df) and df.index[i + k] in confirmed_dates:
                        confirmed_bearish.setdefault(i + k, []).append(label)
                        break
            else:
                bearish_hits.setdefault(i, []).append(label)
        else:
            continue
        if not np.isnan(score):
            hit_strengths[key] = max(hit_strengths.get(key, 0), score)

# --- Plot pending (unconfirmed) signals: lighter color, hollow marker, "(pending)" ---
for i, names in pending_bullish.items():
    strength = hit_strengths.get((i, "bullish"), 0.5)
    label_parts = [f"{n} (pending)" for n in names]
    label_parts.append(f"[{strength:.0%}]")
    y = df.iloc[i]["low"] - 15
    ax.annotate(
        "\n".join(label_parts),
        xy=(i, df.iloc[i]["low"]),
        xytext=(i, y - 5 * len(label_parts)),
        fontsize=5.0, color=PENDING_COLOR_BULL,
        ha="center", va="top",
        arrowprops=dict(arrowstyle="-", color=PENDING_COLOR_BULL, lw=0.5),
    )
    ax.plot(i, df.iloc[i]["low"] - 8, "^",
            color=PENDING_COLOR_BULL, markersize=5, zorder=5,
            markeredgecolor=PENDING_COLOR_BULL, markerfacecolor="none", markeredgewidth=1.2)

for i, names in pending_bearish.items():
    strength = hit_strengths.get((i, "bearish"), 0.5)
    label_parts = [f"{n} (pending)" for n in names]
    label_parts.append(f"[{strength:.0%}]")
    y = df.iloc[i]["high"] + 15
    ax.annotate(
        "\n".join(label_parts),
        xy=(i, df.iloc[i]["high"]),
        xytext=(i, y + 5 * len(label_parts)),
        fontsize=5.0, color=PENDING_COLOR_BEAR,
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=PENDING_COLOR_BEAR, lw=0.5),
    )
    ax.plot(i, df.iloc[i]["high"] + 8, "v",
            color=PENDING_COLOR_BEAR, markersize=5, zorder=5,
            markeredgecolor=PENDING_COLOR_BEAR, markerfacecolor="none", markeredgewidth=1.2)

# --- Plot confirmed signals: solid marker + "confirmed: <name>" ---
for i, names in confirmed_bullish.items():
    # Merge with any non-confirmation bullish hits at the same bar
    all_names = bullish_hits.pop(i, [])
    label_parts = [n for n in all_names]
    for n in names:
        label_parts.append(f"confirmed: {n}")
    strength = hit_strengths.get((i, "bullish"), 0.5)
    msize = 4 + 6 * strength
    label_parts.append(f"[{strength:.0%}]")
    y = df.iloc[i]["low"] - 15
    ax.annotate(
        "\n".join(label_parts),
        xy=(i, df.iloc[i]["low"]),
        xytext=(i, y - 5 * len(label_parts)),
        fontsize=5.5, color=CONFIRMED_COLOR,
        ha="center", va="top",
        arrowprops=dict(arrowstyle="-", color=CONFIRMED_COLOR, lw=0.5),
    )
    ax.plot(i, df.iloc[i]["low"] - 8, "^",
            color=CONFIRMED_COLOR, markersize=msize, zorder=5)

for i, names in confirmed_bearish.items():
    all_names = bearish_hits.pop(i, [])
    label_parts = [n for n in all_names]
    for n in names:
        label_parts.append(f"confirmed: {n}")
    strength = hit_strengths.get((i, "bearish"), 0.5)
    msize = 4 + 6 * strength
    label_parts.append(f"[{strength:.0%}]")
    y = df.iloc[i]["high"] + 15
    ax.annotate(
        "\n".join(label_parts),
        xy=(i, df.iloc[i]["high"]),
        xytext=(i, y + 5 * len(label_parts)),
        fontsize=5.5, color=CONFIRMED_COLOR,
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=CONFIRMED_COLOR, lw=0.5),
    )
    ax.plot(i, df.iloc[i]["high"] + 8, "v",
            color=CONFIRMED_COLOR, markersize=msize, zorder=5)

# --- Plot remaining non-confirmation patterns (no confirmation needed) ---
for i, names in bullish_hits.items():
    strength = hit_strengths.get((i, "bullish"), 0.5)
    msize = 4 + 6 * strength
    label_parts = list(names)
    label_parts.append(f"[{strength:.0%}]")
    y = df.iloc[i]["low"] - 15
    ax.annotate(
        "\n".join(label_parts),
        xy=(i, df.iloc[i]["low"]),
        xytext=(i, y - 5 * len(label_parts)),
        fontsize=5.5, color=BULLISH_COLOR,
        ha="center", va="top",
        arrowprops=dict(arrowstyle="-", color=BULLISH_COLOR, lw=0.5),
    )
    ax.plot(i, df.iloc[i]["low"] - 8, "^",
            color=BULLISH_COLOR, markersize=msize, zorder=5)

for i, names in bearish_hits.items():
    strength = hit_strengths.get((i, "bearish"), 0.5)
    msize = 4 + 6 * strength
    label_parts = list(names)
    label_parts.append(f"[{strength:.0%}]")
    y = df.iloc[i]["high"] + 15
    ax.annotate(
        "\n".join(label_parts),
        xy=(i, df.iloc[i]["high"]),
        xytext=(i, y + 5 * len(label_parts)),
        fontsize=5.5, color=BEARISH_COLOR,
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=BEARISH_COLOR, lw=0.5),
    )
    ax.plot(i, df.iloc[i]["high"] + 8, "v",
            color=BEARISH_COLOR, markersize=msize, zorder=5)

# Mark trend termination signals with purple diamonds
if len(terminations):
    term_set = set()
    for _, row in terminations.iterrows():
        i = dates.get_loc(row["date"])
        if i not in term_set:
            term_set.add(i)
            ax.plot(i, df.iloc[i]["high"] + 20, "D",
                    color=TERM_COLOR, markersize=7, zorder=6)

ax.set_ylabel("Price (JPY/kl)", fontsize=10)
ax.set_title(
    "Exhibit 9.2 — Crude Oil (TOCOM, JPY) Jan–Apr 1990\n"
    "Patterns with confirmation flow and strength scores",
    fontsize=12,
)
ax.grid(axis="y", alpha=0.3)

legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor=BULLISH_COLOR,
           markersize=8, label="Bullish (no confirm needed)"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor=BEARISH_COLOR,
           markersize=8, label="Bearish (no confirm needed)"),
    Line2D([0], [0], marker="^", color=PENDING_COLOR_BULL, markerfacecolor="none",
           markeredgewidth=1.2, markersize=7, label="Pending (awaiting confirm)"),
    Line2D([0], [0], marker="v", color=PENDING_COLOR_BEAR, markerfacecolor="none",
           markeredgewidth=1.2, markersize=7, label="Pending (awaiting confirm)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=CONFIRMED_COLOR,
           markersize=8, label="Confirmed"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=TERM_COLOR,
           markersize=7, label="Trend termination"),
    mpatches.Patch(color=TREND_COLORS["up"], alpha=0.4, label="Pivot uptrend"),
    mpatches.Patch(color=TREND_COLORS["down"], alpha=0.4, label="Pivot downtrend"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

# ── Bottom panel: multi-scale trend state ────────────────────────────────────
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

# X-axis labels
tick_positions, tick_labels = [], []
prev_month = None
for i, dt in enumerate(dates):
    if dt.month != prev_month:
        tick_positions.append(i)
        tick_labels.append(dt.strftime("%b %Y"))
        prev_month = dt.month
ax_trend.set_xticks(tick_positions)
ax_trend.set_xticklabels(tick_labels, fontsize=9)
ax_trend.set_xlim(-1, len(df))
ax_trend.grid(axis="x", alpha=0.3)

plt.tight_layout()
out = "exhibit_9_2_patterns.png"
plt.savefig(out, dpi=150)
print(f"\nSaved → {out}")
