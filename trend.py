"""Multi-scale trend analysis combining pivot structure and candlestick signals.

Provides five trend time-scales plus a mechanism to detect trend *termination*
when a reversal candlestick pattern fires against an active trend at any scale.

Time-scales (fastest → slowest)
-------------------------------
micro   — candle body runs (2–3 consecutive same-color bodies; Nison-style)
short   — linear regression slope over a rolling window (Grimes-style)
medium  — 1st-order pivot structure (confirmed HH+HL / LH+LL)
long    — 2nd-order pivot structure (major swing points)

Trend termination
-----------------
When a reversal pattern fires *against* an active trend at a given scale,
that constitutes an early termination signal — the candle pattern detects
exhaustion before the pivot structure formally breaks.
"""

import numpy as np
import pandas as pd

from patterns._candle import (
    body,
    is_downtrend_by_pivots,
    is_uptrend_by_pivots,
    signal_series,
)


# ---------------------------------------------------------------------------
# Per-scale trend detection
# ---------------------------------------------------------------------------

def body_run_trend(df: pd.DataFrame, min_run: int = 2) -> pd.Series:
    """Micro-trend from consecutive same-color candle bodies (Nison-style).

    A run of ``min_run`` or more consecutive bullish (white) bodies signals
    'up'; a run of bearish (black) bodies signals 'down'.  This is the most
    sensitive trend measure — Nison frequently uses "after several declining
    sessions" to mean just 2–3 black candles in a row.

    The signal persists for every bar within the run (not just the last).
    A doji (body == 0) breaks the run.
    """
    b = body(df)
    bullish = b > 0
    bearish = b < 0

    result = signal_series(df.index)

    # Count consecutive same-direction bodies using groupby on direction changes
    direction = pd.Series(0, index=df.index, dtype=int)
    direction[bullish] = 1
    direction[bearish] = -1

    # Group consecutive same-direction bars
    group = (direction != direction.shift(1)).cumsum()

    for _, grp in direction.groupby(group):
        if len(grp) >= min_run:
            val = grp.iloc[0]
            # Only label from the min_run-th bar onward — at bar k within the
            # run, we only know k consecutive same-color bodies have occurred.
            causal_idx = grp.index[min_run - 1:]
            if val == 1:
                result[causal_idx] = "up"
            elif val == -1:
                result[causal_idx] = "down"

    return result


def regression_slope_trend(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Trend direction from rolling linear regression slope of closes.

    Fits a least-squares line to the last ``window`` closes at each bar.
    Positive slope → 'up', negative → 'down'.  Uses all bars in the window
    rather than just endpoints, making it more robust to single-bar noise
    than a simple close-vs-close comparison.

    Reference: Grimes, *The Art and Science of Technical Analysis*, discusses
    linear regression as a quantitative trend measure.
    """
    closes = df["close"]

    # Vectorised rolling OLS slope: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
    # With x = 0..n-1, x̄ = (n-1)/2, Σ(x - x̄)² = n(n²-1)/12
    n = window
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()  # = n(n²-1)/12

    # Rolling dot product of (x - x_mean) with (y - y_mean)
    # = Σ x_i * y_i - n * x_mean * y_mean
    # = Σ (i * close[t-n+1+i]) - x_mean * Σ close[window]
    # We compute via two rolling sums:
    #   sum_xy = Σ i * y_i  (weighted rolling sum)
    #   sum_y  = Σ y_i      (simple rolling sum)
    # slope = (sum_xy - x_mean * sum_y) / x_var

    # Build weight series for the weighted rolling sum
    # For each window ending at position t: weights are [0, 1, 2, ..., n-1]
    # applied to [close[t-n+1], close[t-n+2], ..., close[t]]
    sum_y = closes.rolling(n, min_periods=n).sum()
    # Weighted sum: reverse-convolve with [0, 1, ..., n-1]
    sum_xy = pd.Series(0.0, index=closes.index)
    for i in range(n):
        sum_xy += i * closes.shift(n - 1 - i)

    slope = (sum_xy - x_mean * sum_y) / x_var

    result = signal_series(df.index)
    result[slope > 0] = "up"
    result[slope < 0] = "down"
    return result, slope


def short_trend(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Regression-slope trend over ``lookback`` bars.

    Replaces the old close-vs-close comparison with a more robust linear
    regression slope.  Returns 'up', 'down', or None.
    """
    result, _slope = regression_slope_trend(df, window=lookback)
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


def multi_scale_trend(
    df: pd.DataFrame,
    body_run: int = 2,
    reg_window: int = 5,
) -> pd.DataFrame:
    """Return a DataFrame with micro / short / medium / long trend columns.

    Each column contains 'up', 'down', or None per bar.

    Scales (fastest → slowest):
      micro  — candle body runs (``body_run`` consecutive same-color bodies)
      short  — linear regression slope over ``reg_window`` bars
      medium — 1st-order pivot structure
      long   — 2nd-order pivot structure
    """
    return pd.DataFrame({
        "micro": body_run_trend(df, body_run),
        "short": short_trend(df, reg_window),
        "medium": pivot_trend(df, order=1),
        "long": pivot_trend(df, order=2),
    })


# ---------------------------------------------------------------------------
# Effective trend: pivot when available, short-term fallback
# ---------------------------------------------------------------------------

def effective_trend(df: pd.DataFrame, reg_window: int = 5) -> pd.Series:
    """Best-available trend: pivot → regression slope → body run fallback.

    This is intended as the primary trend input for pattern detection —
    it catches brief moves that haven't produced two confirmed pivot pairs
    yet, but prefers the more reliable pivot signal when it exists.
    """
    piv = pivot_trend(df, order=1)
    sht = short_trend(df, reg_window)
    mic = body_run_trend(df, min_run=2)
    # Prefer pivot; fill gaps with regression slope; then body run
    result = piv.copy()
    mask = result.isna()
    result[mask] = sht[mask]
    mask = result.isna()
    result[mask] = mic[mask]
    return result


# ---------------------------------------------------------------------------
# Trend termination signals
# ---------------------------------------------------------------------------

def trend_terminations(
    df: pd.DataFrame,
    patterns: dict[str, pd.Series],
    body_run: int = 2,
    reg_window: int = 5,
) -> pd.DataFrame:
    """Identify where reversal patterns fire against an active trend.

    For each trend scale (micro, short, medium, long), checks whether a
    reversal pattern contradicts the current trend direction:
      - bearish pattern during 'up' trend  → termination warning
      - bullish pattern during 'down' trend → termination warning

    Returns a DataFrame with columns:
      date, pattern, signal, scale, trend_direction
    """
    scales = multi_scale_trend(df, body_run=body_run, reg_window=reg_window)
    records = []

    for name, sig in patterns.items():
        for dt, val in sig.dropna().items():
            for scale in ("micro", "short", "medium", "long"):
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
