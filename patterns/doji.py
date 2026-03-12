"""Doji candlestick patterns — Nison chapters 7–8.

Implements the doji patterns covered in *Japanese Candlestick Charting
Techniques* by Steve Nison.

Chapter 7 — Doji:
  doji_at_top      : bearish warning at an uptrend high
  doji_at_bottom   : bullish warning at a downtrend low
  long_legged_doji : doji with long upper and lower shadows
  rickshaw_man     : long-legged doji where the body is near the midpoint
  gravestone_doji  : doji with long upper shadow and no lower shadow

Chapter 8 — Tri-Star:
  tri_star         : three consecutive doji candles (very rare reversal)

All functions accept a DataFrame with columns (open, high, low, close) and
return a pd.Series of signal strings: 'bullish', 'bearish', or None.

Doji definition
---------------
A doji occurs when the open and close are equal (or nearly so).  The
``doji_threshold`` parameter controls how small the body must be relative
to the total range: ``body_size ≤ doji_threshold × range``.

Trend semantics
---------------
For single-bar doji patterns the trend is evaluated at the current bar
(the doji itself).  For the tri-star the trend is evaluated at the first
of the three doji candles (``shift(2)``).
"""

import pandas as pd

from patterns._candle import (
    body_bottom,
    body_size,
    body_top,
    candle_range,
    is_doji,
    is_downtrend,
    is_downtrend_by_pivots,
    is_uptrend,
    is_uptrend_by_pivots,
    lower_shadow,
    signal_series,
    upper_shadow,
)


# ---------------------------------------------------------------------------
# Chapter 7 — Doji at tops and bottoms
# ---------------------------------------------------------------------------

def doji_at_top(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    doji_threshold: float = 0.1,
) -> pd.Series:
    """Bearish warning: doji appearing in an uptrend.

    A doji after a sustained rally warns of exhaustion.  It is not a
    reversal confirmation by itself — Nison stresses that a weak candle
    on the next session or a gap down is needed for confirmation.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 7.
    """
    signal = (
        is_doji(df, threshold=doji_threshold)
        & is_uptrend(df, trend_lookback)
    ).fillna(False)
    result = signal_series(df.index)
    result[signal] = "bearish"
    return result


def doji_at_bottom(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    doji_threshold: float = 0.1,
) -> pd.Series:
    """Bullish warning: doji appearing in a downtrend.

    A doji after a prolonged decline suggests selling pressure may be
    waning.  Confirmation from the following session is advised.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 7.
    """
    signal = (
        is_doji(df, threshold=doji_threshold)
        & is_downtrend(df, trend_lookback)
    ).fillna(False)
    result = signal_series(df.index)
    result[signal] = "bullish"
    return result


# ---------------------------------------------------------------------------
# Chapter 7 — Long-legged doji and rickshaw man
# ---------------------------------------------------------------------------

def long_legged_doji(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    doji_threshold: float = 0.1,
    shadow_ratio: float = 0.3,
) -> pd.Series:
    """Reversal warning: long-legged doji (long shadows on both sides).

    In addition to the standard doji body criterion, both the upper and
    lower shadows must each be at least ``shadow_ratio`` of the total
    range.  The pattern signals potential reversal in whichever trend
    is present.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 7.
    """
    rng = candle_range(df)
    long_upper = upper_shadow(df) >= shadow_ratio * rng
    long_lower = lower_shadow(df) >= shadow_ratio * rng

    is_uptrend_now  = is_uptrend(df, trend_lookback)
    is_downtrend_now = is_downtrend(df, trend_lookback)

    base = is_doji(df, doji_threshold) & long_upper & long_lower & (rng > 0)

    result = signal_series(df.index)
    result[(base & is_uptrend_now).fillna(False)]  = "bearish"
    result[(base & is_downtrend_now).fillna(False)] = "bullish"
    return result


def rickshaw_man(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    doji_threshold: float = 0.1,
    shadow_ratio: float = 0.3,
    center_tolerance: float = 0.25,
) -> pd.Series:
    """Reversal warning: rickshaw man (long-legged doji, body near midpoint).

    A special case of the long-legged doji where the body is near the
    midpoint of the range (within ``center_tolerance`` of the exact
    midpoint, expressed as a fraction of the range).

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 7.
    """
    rng = candle_range(df)
    midpoint = df["low"] + 0.5 * rng
    body_mid = 0.5 * (body_top(df) + body_bottom(df))
    near_center = (body_mid - midpoint).abs() <= center_tolerance * rng

    long_upper = upper_shadow(df) >= shadow_ratio * rng
    long_lower = lower_shadow(df) >= shadow_ratio * rng

    is_uptrend_now   = is_uptrend(df, trend_lookback)
    is_downtrend_now = is_downtrend(df, trend_lookback)

    base = (
        is_doji(df, doji_threshold)
        & long_upper & long_lower
        & near_center
        & (rng > 0)
    )

    result = signal_series(df.index)
    result[(base & is_uptrend_now).fillna(False)]   = "bearish"
    result[(base & is_downtrend_now).fillna(False)]  = "bullish"
    return result


# ---------------------------------------------------------------------------
# Chapter 7 — Gravestone doji
# ---------------------------------------------------------------------------

def gravestone_doji(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    doji_threshold: float = 0.1,
    lower_shadow_ratio: float = 0.05,
    upper_shadow_ratio: float = 0.5,
) -> pd.Series:
    """Bearish reversal: gravestone doji (long upper shadow, no lower shadow).

    The open and close are at or near the session low; the price rallied
    but gave back all gains.  Most significant as a bearish warning at
    market tops in an uptrend.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 7.
    """
    rng = candle_range(df)
    no_lower = lower_shadow(df) <= lower_shadow_ratio * rng
    long_upper = upper_shadow(df) >= upper_shadow_ratio * rng

    signal = (
        is_doji(df, doji_threshold)
        & no_lower
        & long_upper
        & (rng > 0)
        & is_uptrend(df, trend_lookback)
    ).fillna(False)

    result = signal_series(df.index)
    result[signal] = "bearish"
    return result


# ---------------------------------------------------------------------------
# Chapter 8 — Tri-Star
# ---------------------------------------------------------------------------

def tri_star(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    doji_threshold: float = 0.1,
) -> pd.Series:
    """Very rare and strong reversal: tri-star (three consecutive doji).

    Three consecutive doji candles form the tri-star.  In a downtrend it
    is bullish; in an uptrend it is bearish.  The trend is evaluated at
    the first doji (``t-2``) to avoid contamination from the pattern's
    own bars.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 8.
    """
    all_doji = (
        is_doji(df.shift(2), doji_threshold)
        & is_doji(df.shift(1), doji_threshold)
        & is_doji(df, doji_threshold)
    )

    trend_at_first_up   = is_uptrend_by_pivots(df).shift(2).fillna(False)
    trend_at_first_down = is_downtrend_by_pivots(df).shift(2).fillna(False)

    bullish = (all_doji & trend_at_first_down).fillna(False)
    bearish = (all_doji & trend_at_first_up).fillna(False)

    result = signal_series(df.index)
    result[bullish] = "bullish"
    result[bearish] = "bearish"
    return result
