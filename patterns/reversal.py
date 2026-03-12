"""Reversal candlestick patterns.

Implementations follow *Japanese Candlestick Charting Techniques* by Steve Nison.

Each public function accepts a DataFrame with columns (open, high, low, close)
and returns a pd.Series of signal strings: 'bullish', 'bearish', or None.

Trend context is determined by comparing the current close to the close
`trend_lookback` bars ago. A simple slope is used rather than a moving average
so that the signal is fully determined by the raw price series.
"""

import pandas as pd

from patterns._candle import (
    body,
    body_bottom,
    body_size,
    body_top,
    candle_range,
    is_downtrend,
    is_uptrend,
    lower_shadow,
    signal_series,
    upper_shadow,
)


# ---------------------------------------------------------------------------
# Hammer and Hanging Man
# ---------------------------------------------------------------------------

def _hammer_shape(
    df: pd.DataFrame,
    shadow_multiplier: float = 2.0,
    upper_shadow_ratio: float = 0.1,
) -> pd.Series:
    """Detect the shared shape of a hammer / hanging man.

    Criteria (Nison, ch. 3):
    - Real body at the upper end of the trading range (small body).
    - Lower shadow at least `shadow_multiplier` × body size.
    - Upper shadow no more than `upper_shadow_ratio` × total range.
    - Non-zero total range to avoid division by zero on doji-like gaps.

    Returns a boolean Series.
    """
    b     = body_size(df)
    upper = upper_shadow(df)
    lower = lower_shadow(df)
    rng   = candle_range(df)

    return (rng > 0) & (upper <= upper_shadow_ratio * rng) & (lower >= shadow_multiplier * b) & (b > 0)


def hammer(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    shadow_multiplier: float = 2.0,
    upper_shadow_ratio: float = 0.1,
) -> pd.Series:
    """Bullish reversal: hammer line appearing after a downtrend.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 3.

    Returns a Series of signal strings ('bullish' or None).
    """
    signal = (_hammer_shape(df, shadow_multiplier, upper_shadow_ratio) & is_downtrend(df, trend_lookback)).fillna(False)
    result = signal_series(df.index)
    result[signal] = "bullish"
    return result


def hanging_man(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    shadow_multiplier: float = 2.0,
    upper_shadow_ratio: float = 0.1,
) -> pd.Series:
    """Bearish reversal: hanging man line appearing after an uptrend.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 3.

    Returns a Series of signal strings ('bearish' or None).
    """
    signal = (_hammer_shape(df, shadow_multiplier, upper_shadow_ratio) & is_uptrend(df, trend_lookback)).fillna(False)
    result = signal_series(df.index)
    result[signal] = "bearish"
    return result


# ---------------------------------------------------------------------------
# Engulfing Pattern
# ---------------------------------------------------------------------------

def engulfing(
    df: pd.DataFrame,
    trend_lookback: int = 5,
) -> pd.Series:
    """Bullish or bearish engulfing pattern.

    Bullish engulfing (after downtrend): current white (bullish) body fully
    engulfs the prior black (bearish) body.

    Bearish engulfing (after uptrend): current black (bearish) body fully
    engulfs the prior white (bullish) body.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 4.

    Returns a Series of signal strings ('bullish', 'bearish', or None).
    """
    prev_body = body(df).shift(1)
    curr_body = body(df)
    curr_engulfs = (body_top(df) > body_top(df).shift(1)) & (body_bottom(df) < body_bottom(df).shift(1))

    bullish = (curr_engulfs & (curr_body > 0) & (prev_body < 0) & is_downtrend(df, trend_lookback))
    bearish = (curr_engulfs & (curr_body < 0) & (prev_body > 0) & is_uptrend(df, trend_lookback))

    result = signal_series(df.index)
    result[bullish.fillna(False)] = "bullish"
    result[bearish.fillna(False)] = "bearish"
    return result


# ---------------------------------------------------------------------------
# Dark Cloud Cover
# ---------------------------------------------------------------------------

def dark_cloud_cover(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    penetration: float = 0.5,
) -> pd.Series:
    """Bearish reversal: dark cloud cover.

    Criteria (Nison, ch. 5):
    - Prior candle: long white (bullish) body.
    - Current candle: opens above the prior close, then closes below the
      midpoint (`penetration` fraction) of the prior body.

    Returns a Series of signal strings ('bearish' or None).
    """
    prev_body  = body(df).shift(1)
    prev_open  = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_mid   = prev_open + penetration * prev_body

    signal = (
        is_uptrend(df, trend_lookback)
        & (prev_body > 0)
        & (df["open"] > prev_close)
        & (df["close"] < prev_mid)
        & (df["close"] > prev_open)
    ).fillna(False)
    result = signal_series(df.index)
    result[signal] = "bearish"
    return result


# ---------------------------------------------------------------------------
# Piercing Pattern
# ---------------------------------------------------------------------------

def piercing_pattern(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    penetration: float = 0.5,
) -> pd.Series:
    """Bullish reversal: piercing pattern.

    The mirror image of the dark cloud cover.

    Criteria (Nison, ch. 5):
    - Prior candle: long black (bearish) body.
    - Current candle: opens below the prior close, then closes above the
      midpoint (`penetration` fraction) of the prior body.

    Returns a Series of signal strings ('bullish' or None).
    """
    prev_body  = body(df).shift(1)
    prev_open  = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_mid   = prev_open + penetration * prev_body  # prev_body < 0, so mid < prev_open

    signal = (
        is_downtrend(df, trend_lookback)
        & (prev_body < 0)
        & (df["open"] < prev_close)
        & (df["close"] > prev_mid)
        & (df["close"] < prev_open)
    ).fillna(False)
    result = signal_series(df.index)
    result[signal] = "bullish"
    return result
