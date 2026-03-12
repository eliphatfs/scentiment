"""Load exhibit 9.2 crude oil data, run all pattern detections, and plot."""

import pandas as pd
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


# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/exhibit_9_2.csv", parse_dates=["Date"])
df = df.set_index("Date").sort_index()
df.index.name = "datetime"
# No volume column — add zeros so pattern functions that might use it don't fail
if "volume" not in df.columns:
    df["volume"] = 0.0
# Lowercase columns to match project convention
df.columns = [c.lower() for c in df.columns]

# ── Run all patterns ─────────────────────────────────────────────────────────
PATTERNS = {
    # reversal
    "hammer":               hammer(df),
    "hanging_man":          hanging_man(df),
    "engulfing":            engulfing(df),
    "dark_cloud_cover":     dark_cloud_cover(df),
    "piercing_pattern":     piercing_pattern(df),
    # stars
    "inverted_hammer":      inverted_hammer(df),
    "shooting_star":        shooting_star(df),
    "morning_star":         morning_star(df),
    "evening_star":         evening_star(df),
    "morning_doji_star":    morning_doji_star(df),
    "evening_doji_star":    evening_doji_star(df),
    # doji
    "doji_at_top":          doji_at_top(df),
    "doji_at_bottom":       doji_at_bottom(df),
    "long_legged_doji":     long_legged_doji(df),
    "rickshaw_man":         rickshaw_man(df),
    "gravestone_doji":      gravestone_doji(df),
    "tri_star":             tri_star(df),
    # more reversals
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
    # continuation
    "window_up":            window_up(df),
    "window_down":          window_down(df),
    "rising_three_methods": rising_three_methods(df),
    "falling_three_methods":falling_three_methods(df),
    "three_white_soldiers": three_white_soldiers(df),
    "separating_lines":     separating_lines(df),
}

# Print detected signals
print("Detected signals:")
for name, sig in PATTERNS.items():
    hits = sig.dropna()
    if len(hits):
        for dt, val in hits.items():
            print(f"  {dt.date()}  {name:25s}  {val}")

# ── Plot ─────────────────────────────────────────────────────────────────────
xs = list(range(len(df)))
dates = df.index

fig, ax = plt.subplots(figsize=(18, 8))

# Draw candlesticks
for i, (dt, row) in enumerate(df.iterrows()):
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    color = "#2ecc71" if c >= o else "#e74c3c"
    # Wick
    ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)
    # Body
    body_lo = min(o, c)
    body_hi = max(o, c)
    rect = mpatches.FancyBboxPatch(
        (i - 0.35, body_lo), 0.7, max(body_hi - body_lo, 0.5),
        boxstyle="square,pad=0",
        linewidth=0.5, edgecolor=color, facecolor=color, zorder=3,
    )
    ax.add_patch(rect)

# Overlay pattern signals as markers above/below bars
BULLISH_COLOR = "#3498db"
BEARISH_COLOR = "#e67e22"

# Collect unique pattern labels per bar for annotation
bullish_hits = {}  # x_index -> list of pattern names
bearish_hits = {}

for name, sig in PATTERNS.items():
    for dt, val in sig.dropna().items():
        i = dates.get_loc(dt)
        if val == "bullish":
            bullish_hits.setdefault(i, []).append(name.replace("_", " "))
        elif val == "bearish":
            bearish_hits.setdefault(i, []).append(name.replace("_", " "))

for i, names in bullish_hits.items():
    y = df.iloc[i]["low"] - 15
    ax.annotate(
        "\n".join(names),
        xy=(i, df.iloc[i]["low"]),
        xytext=(i, y - 5 * len(names)),
        fontsize=5.5,
        color=BULLISH_COLOR,
        ha="center", va="top",
        arrowprops=dict(arrowstyle="-", color=BULLISH_COLOR, lw=0.5),
    )
    ax.plot(i, df.iloc[i]["low"] - 8, "^", color=BULLISH_COLOR, markersize=6, zorder=5)

for i, names in bearish_hits.items():
    y = df.iloc[i]["high"] + 15
    ax.annotate(
        "\n".join(names),
        xy=(i, df.iloc[i]["high"]),
        xytext=(i, y + 5 * len(names)),
        fontsize=5.5,
        color=BEARISH_COLOR,
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=BEARISH_COLOR, lw=0.5),
    )
    ax.plot(i, df.iloc[i]["high"] + 8, "v", color=BEARISH_COLOR, markersize=6, zorder=5)

# X-axis: monthly tick labels
tick_positions = []
tick_labels = []
prev_month = None
for i, dt in enumerate(dates):
    if dt.month != prev_month:
        tick_positions.append(i)
        tick_labels.append(dt.strftime("%b %Y"))
        prev_month = dt.month

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_xlim(-1, len(df))
ax.set_ylabel("Price (JPY/kl)", fontsize=10)
ax.set_title("Exhibit 9.2 — Crude Oil (TOCOM, JPY) Jan–Apr 1990\nAll pattern signals overlaid", fontsize=12)
ax.grid(axis="y", alpha=0.3)

# Legend
legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor=BULLISH_COLOR,
           markersize=8, label="Bullish signal"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor=BEARISH_COLOR,
           markersize=8, label="Bearish signal"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

plt.tight_layout()
out = "exhibit_9_2_patterns.png"
plt.savefig(out, dpi=150)
print(f"\nSaved → {out}")
