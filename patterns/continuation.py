"""Continuation candlestick patterns — Nison chapters 13–14.

Implements the patterns covered in the continuation-pattern chapters of
*Japanese Candlestick Charting Techniques* by Steve Nison.

Chapter 13 — Windows (gaps):
  window_up, window_down

Chapter 14 — Continuation patterns:
  rising_three_methods, falling_three_methods,
  three_white_soldiers,
  separating_lines

All functions accept a DataFrame with columns (open, high, low, close) and
return a pd.Series of signal strings: 'bullish', 'bearish', or None.

Trend semantics
---------------
Continuation patterns are checked in the context of the prevailing trend at
the *first* candle of the pattern (shift by the pattern length minus one) so
that the pattern's own candles do not corrupt the trend reading.  Windows have
no trend-lookback requirement by default because the gap itself is the
directional signal.
"""

import pandas as pd

from patterns._candle import (
    body,
    body_bottom,
    body_size,
    body_top,
    candle_range,
    is_downtrend,
    is_downtrend_by_pivots,
    is_uptrend,
    is_uptrend_by_pivots,
    lower_shadow,
    signal_series,
    upper_shadow,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _long_body(df: pd.DataFrame, ratio: float) -> pd.Series:
    rng = candle_range(df)
    return (body_size(df) >= ratio * rng) & (rng > 0)


def _small_body(df: pd.DataFrame, ratio: float) -> pd.Series:
    rng = candle_range(df)
    return (body_size(df) <= ratio * rng) & (rng > 0)


# ---------------------------------------------------------------------------
# Chapter 13 — Windows (gaps)
# ---------------------------------------------------------------------------

def window_up(
    df: pd.DataFrame,
    trend_lookback: int = 0,
) -> pd.Series:
    """Bullish continuation: rising window (upside gap).

    Current bar's low is strictly above the prior bar's high.  The gap is
    itself the signal; no trend check is applied by default
    (``trend_lookback=0``).  Set a positive value to require a prior uptrend.

    Reference: Nison, ch. 13.
    """
    gap = df["low"] > df["high"].shift(1)
    if trend_lookback > 0:
        # Check trend at the bar before the gap so the gap itself doesn't
        # corrupt the trend reading (a large gap can flip close vs. lookback).
        prior_uptrend = is_uptrend_by_pivots(df).shift(1).fillna(False)
        gap = gap & prior_uptrend
    result = signal_series(df.index)
    result[gap.fillna(False)] = "bullish"
    return result


def window_down(
    df: pd.DataFrame,
    trend_lookback: int = 0,
) -> pd.Series:
    """Bearish continuation: falling window (downside gap).

    Current bar's high is strictly below the prior bar's low.

    Reference: Nison, ch. 13.
    """
    gap = df["high"] < df["low"].shift(1)
    if trend_lookback > 0:
        prior_downtrend = is_downtrend_by_pivots(df).shift(1).fillna(False)
        gap = gap & prior_downtrend
    result = signal_series(df.index)
    result[gap.fillna(False)] = "bearish"
    return result


# ---------------------------------------------------------------------------
# Chapter 14 — Rising and Falling Three Methods
# ---------------------------------------------------------------------------

def rising_three_methods(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    long_body_ratio: float = 0.6,
    small_body_ratio: float = 0.3,
) -> pd.Series:
    """Bullish continuation: rising three methods (five-candle pattern).

    Criteria (Nison, ch. 14):
    1. Long white candle in an established uptrend.
    2–4. Three small candles (traditionally black) that remain within the
         high–low range of the first candle — a brief consolidation.
    5. Long white candle that closes above the first candle's close.

    The signal is assigned to the index of the fifth (current) candle.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 14.
    """
    first_b     = body(df).shift(4)
    first_close = df["close"].shift(4)
    first_high  = df["high"].shift(4)
    first_low   = df["low"].shift(4)

    # All three middle candles must stay within the first candle's high/low range
    mid_in_range = (
        (df["high"].shift(3) <= first_high) & (df["low"].shift(3) >= first_low)
        & (df["high"].shift(2) <= first_high) & (df["low"].shift(2) >= first_low)
        & (df["high"].shift(1) <= first_high) & (df["low"].shift(1) >= first_low)
    )
    mid_small = (
        _small_body(df.shift(3), small_body_ratio)
        & _small_body(df.shift(2), small_body_ratio)
        & _small_body(df.shift(1), small_body_ratio)
    )

    # Trend measured at the first candle
    trend_at_first = is_uptrend_by_pivots(df).shift(4).fillna(False)

    signal = (
        trend_at_first
        & (first_b > 0) & _long_body(df.shift(4), long_body_ratio)
        & mid_in_range & mid_small
        & (body(df) > 0) & _long_body(df, long_body_ratio)
        & (df["close"] > first_close)
    ).fillna(False)

    result = signal_series(df.index)
    result[signal] = "bullish"
    return result


def falling_three_methods(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    long_body_ratio: float = 0.6,
    small_body_ratio: float = 0.3,
) -> pd.Series:
    """Bearish continuation: falling three methods (five-candle pattern).

    Mirror of rising three methods:
    1. Long black candle in an established downtrend.
    2–4. Three small candles (traditionally white) within the first candle's range.
    5. Long black candle that closes below the first candle's close.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 14.
    """
    first_b     = body(df).shift(4)
    first_close = df["close"].shift(4)
    first_high  = df["high"].shift(4)
    first_low   = df["low"].shift(4)

    mid_in_range = (
        (df["high"].shift(3) <= first_high) & (df["low"].shift(3) >= first_low)
        & (df["high"].shift(2) <= first_high) & (df["low"].shift(2) >= first_low)
        & (df["high"].shift(1) <= first_high) & (df["low"].shift(1) >= first_low)
    )
    mid_small = (
        _small_body(df.shift(3), small_body_ratio)
        & _small_body(df.shift(2), small_body_ratio)
        & _small_body(df.shift(1), small_body_ratio)
    )

    # Trend measured at the first candle
    trend_at_first = is_downtrend_by_pivots(df).shift(4).fillna(False)

    signal = (
        trend_at_first
        & (first_b < 0) & _long_body(df.shift(4), long_body_ratio)
        & mid_in_range & mid_small
        & (body(df) < 0) & _long_body(df, long_body_ratio)
        & (df["close"] < first_close)
    ).fillna(False)

    result = signal_series(df.index)
    result[signal] = "bearish"
    return result


# ---------------------------------------------------------------------------
# Chapter 14 — Three Advancing White Soldiers
# ---------------------------------------------------------------------------

def three_white_soldiers(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    long_body_ratio: float = 0.6,
    shadow_ratio: float = 0.1,
) -> pd.Series:
    """Bullish signal: three advancing white soldiers (three-candle pattern).

    Three consecutive long white candles each opening within the prior body
    and closing near its high (small upper shadow).  Appears after a downtrend
    or prolonged weakness.  Despite being listed here alongside continuation
    patterns, it also signals bullish reversal from a low base — context
    determines interpretation.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 14.
    """
    # Each opens within the prior candle's body
    opens_in_prior = lambda shift_curr, shift_prev: (
        (df["open"].shift(shift_curr) <= body_top(df).shift(shift_prev))
        & (df["open"].shift(shift_curr) >= body_bottom(df).shift(shift_prev))
    )

    # Each closes near its high (upper shadow ≤ shadow_ratio × range)
    upper_small = lambda n: upper_shadow(df).shift(n) <= shadow_ratio * candle_range(df).shift(n)

    # Trend at the first candle (t-2)
    trend_at_first = is_downtrend_by_pivots(df).shift(2).fillna(False)

    signal = (
        trend_at_first
        & (body(df).shift(2) > 0) & _long_body(df.shift(2), long_body_ratio)
        & (body(df).shift(1) > 0) & _long_body(df.shift(1), long_body_ratio)
        & (body(df) > 0)          & _long_body(df, long_body_ratio)
        & opens_in_prior(1, 2)    # 2nd opens within 1st
        & opens_in_prior(0, 1)    # 3rd opens within 2nd
        & upper_small(2) & upper_small(1) & upper_small(0)
    ).fillna(False)

    result = signal_series(df.index)
    result[signal] = "bullish"
    return result


# ---------------------------------------------------------------------------
# Chapter 14 — Separating Lines
# ---------------------------------------------------------------------------

def separating_lines(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    open_tolerance: float = 0.001,
) -> pd.Series:
    """Bullish or bearish separating lines (two-candle continuation pattern).

    Both candles open at the same price (within ``open_tolerance`` as a
    fraction of the prior open).  The first candle moves against the trend;
    the second resumes it.

    Bullish separating line (in uptrend): black candle → white candle
    opening at the same price.  The market retraced intraday but opened
    at the same level the next session, resuming the uptrend.

    Bearish separating line (in downtrend): white candle → black candle
    opening at the same price.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 14.
    """
    prev_open = df["open"].shift(1)
    open_match = (df["open"] - prev_open).abs() <= open_tolerance * prev_open

    prev_b = body(df).shift(1)
    curr_b = body(df)

    bullish = (
        open_match
        & (prev_b < 0)   # prior black (counter-trend pullback)
        & (curr_b > 0)   # current white (trend resumes)
        & is_uptrend(df, trend_lookback)
    )
    bearish = (
        open_match
        & (prev_b > 0)   # prior white (counter-trend bounce)
        & (curr_b < 0)   # current black (trend resumes)
        & is_downtrend(df, trend_lookback)
    )

    result = signal_series(df.index)
    result[bullish.fillna(False)] = "bullish"
    result[bearish.fillna(False)] = "bearish"
    return result
