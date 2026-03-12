"""Multi-scale trend analysis combining pivot structure and candlestick signals.

Provides three trend time-scales (Grimes-inspired) plus a mechanism to detect
trend *termination* when a reversal candlestick pattern fires against an
active trend at any scale.

Time-scales
-----------
short   — close-vs-close lookback (sensitive, catches 3–5 bar moves)
medium  — 1st-order pivot structure (confirmed HH+HL / LH+LL)
long    — 2nd-order pivot structure (major swing points)

Trend termination
-----------------
When a reversal pattern fires *against* an active trend at a given scale,
that constitutes an early termination signal — the candle pattern detects
exhaustion before the pivot structure formally breaks.
"""

import pandas as pd

from patterns._candle import (
    is_downtrend_by_pivots,
    is_uptrend_by_pivots,
    signal_series,
)


# ---------------------------------------------------------------------------
# Per-scale trend detection
# ---------------------------------------------------------------------------

def short_trend(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Simple close-vs-close directional bias.

    Returns 'up', 'down', or None.  Sensitive to short moves but noisy.
    """
    up = df["close"] > df["close"].shift(lookback)
    down = df["close"] < df["close"].shift(lookback)
    result = signal_series(df.index)
    result[up] = "up"
    result[down] = "down"
    return result


def pivot_trend(df: pd.DataFrame, order: int = 1) -> pd.Series:
    """Pivot-structure trend (Grimes HH+HL / LH+LL).

    Returns 'up', 'down', or None.  Requires two confirmed pivot pairs so
    it lags — but once it fires, the trend is structurally confirmed.
    """
    up = is_uptrend_by_pivots(df, order)
    down = is_downtrend_by_pivots(df, order)
    result = signal_series(df.index)
    result[up] = "up"
    result[down] = "down"
    return result


def multi_scale_trend(df: pd.DataFrame, short_lookback: int = 5) -> pd.DataFrame:
    """Return a DataFrame with short / medium / long trend columns.

    Each column contains 'up', 'down', or None per bar.
    """
    return pd.DataFrame({
        "short": short_trend(df, short_lookback),
        "medium": pivot_trend(df, order=1),
        "long": pivot_trend(df, order=2),
    })


# ---------------------------------------------------------------------------
# Effective trend: pivot when available, short-term fallback
# ---------------------------------------------------------------------------

def effective_trend(df: pd.DataFrame, short_lookback: int = 5) -> pd.Series:
    """Best-available trend: use pivot structure when confirmed, otherwise
    fall back to short-term close comparison.

    This is intended as the primary trend input for pattern detection —
    it catches brief moves that haven't produced two confirmed pivot pairs
    yet, but prefers the more reliable pivot signal when it exists.
    """
    piv = pivot_trend(df, order=1)
    sht = short_trend(df, short_lookback)
    # Prefer pivot; fill gaps with short-term
    result = piv.copy()
    result[result.isna()] = sht[result.isna()]
    return result


# ---------------------------------------------------------------------------
# Trend termination signals
# ---------------------------------------------------------------------------

def trend_terminations(
    df: pd.DataFrame,
    patterns: dict[str, pd.Series],
    short_lookback: int = 5,
) -> pd.DataFrame:
    """Identify where reversal patterns fire against an active trend.

    For each trend scale (short, medium, long), checks whether a reversal
    pattern contradicts the current trend direction:
      - bearish pattern during 'up' trend  → termination warning
      - bullish pattern during 'down' trend → termination warning

    Returns a DataFrame with columns:
      date, pattern, signal, scale, trend_direction
    """
    scales = multi_scale_trend(df, short_lookback)
    records = []

    for name, sig in patterns.items():
        for dt, val in sig.dropna().items():
            for scale in ("short", "medium", "long"):
                trend_dir = scales.loc[dt, scale]
                if trend_dir is None:
                    continue
                # Bearish pattern in uptrend, or bullish pattern in downtrend
                if (trend_dir == "up" and val == "bearish") or \
                   (trend_dir == "down" and val == "bullish"):
                    records.append({
                        "date": dt,
                        "pattern": name,
                        "signal": val,
                        "scale": scale,
                        "trend_direction": trend_dir,
                    })

    return pd.DataFrame(records)
