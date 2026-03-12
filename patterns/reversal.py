"""Reversal candlestick patterns.

Implementations follow *Japanese Candlestick Charting Techniques* by Steve Nison.

Each public function accepts a DataFrame with columns (open, high, low, close)
and returns a pd.Series of signal strings: 'bullish', 'bearish', or None.

Trend context is determined by comparing the current close to the close
`trend_lookback` bars ago. A simple slope is used rather than a moving average
so that the signal is fully determined by the raw price series.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _body(df: pd.DataFrame) -> pd.Series:
    """Signed body size: positive for bullish (white), negative for bearish (black)."""
    return df["close"] - df["open"]


def _body_size(df: pd.DataFrame) -> pd.Series:
    return _body(df).abs()


def _body_top(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].max(axis=1)


def _body_bottom(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1)


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["high"] - _body_top(df)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return _body_bottom(df) - df["low"]


def _range(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df["low"]


def _is_uptrend(df: pd.DataFrame, lookback: int) -> pd.Series:
    return df["close"] > df["close"].shift(lookback)


def _is_downtrend(df: pd.DataFrame, lookback: int) -> pd.Series:
    return df["close"] < df["close"].shift(lookback)


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
    body = _body_size(df)
    upper = _upper_shadow(df)
    lower = _lower_shadow(df)
    rng = _range(df)

    has_range = rng > 0
    small_upper = upper <= upper_shadow_ratio * rng
    long_lower = lower >= shadow_multiplier * body
    # Body must be non-trivially present (not a pure doji)
    has_body = body > 0

    return has_range & small_upper & long_lower & has_body


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
    shape = _hammer_shape(df, shadow_multiplier, upper_shadow_ratio)
    downtrend = _is_downtrend(df, trend_lookback)
    signal = (shape & downtrend).fillna(False)
    result = pd.Series([None] * len(df), index=df.index, dtype=object)
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
    shape = _hammer_shape(df, shadow_multiplier, upper_shadow_ratio)
    uptrend = _is_uptrend(df, trend_lookback)
    signal = (shape & uptrend).fillna(False)
    result = pd.Series([None] * len(df), index=df.index, dtype=object)
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
    prev_body = _body(df).shift(1)
    curr_body = _body(df)

    prev_top = _body_top(df).shift(1)
    prev_bot = _body_bottom(df).shift(1)
    curr_top = _body_top(df)
    curr_bot = _body_bottom(df)

    # Current candle must completely contain previous candle's body
    curr_engulfs = (curr_top > prev_top) & (curr_bot < prev_bot)

    bullish = (
        curr_engulfs
        & (curr_body > 0)        # current is white
        & (prev_body < 0)        # prior is black
        & _is_downtrend(df, trend_lookback)
    )
    bearish = (
        curr_engulfs
        & (curr_body < 0)        # current is black
        & (prev_body > 0)        # prior is white
        & _is_uptrend(df, trend_lookback)
    )

    result = pd.Series([None] * len(df), index=df.index, dtype=object)
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
    prev_body = _body(df).shift(1)
    prev_close = df["close"].shift(1)
    prev_open = df["open"].shift(1)

    curr_open = df["open"]
    curr_close = df["close"]

    prev_midpoint = prev_open + penetration * prev_body

    signal = (
        _is_uptrend(df, trend_lookback)
        & (prev_body > 0)                    # prior is white
        & (curr_open > prev_close)           # opens above prior close
        & (curr_close < prev_midpoint)       # closes below prior midpoint
        & (curr_close > prev_open)           # still closes within prior body
    ).fillna(False)
    result = pd.Series([None] * len(df), index=df.index, dtype=object)
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
    prev_body = _body(df).shift(1)
    prev_close = df["close"].shift(1)
    prev_open = df["open"].shift(1)

    curr_open = df["open"]
    curr_close = df["close"]

    prev_midpoint = prev_open + penetration * prev_body   # prev_body < 0, so midpoint < prev_open

    signal = (
        _is_downtrend(df, trend_lookback)
        & (prev_body < 0)                    # prior is black
        & (curr_open < prev_close)           # opens below prior close
        & (curr_close > prev_midpoint)       # closes above prior midpoint
        & (curr_close < prev_open)           # still closes within prior body
    ).fillna(False)
    result = pd.Series([None] * len(df), index=df.index, dtype=object)
    result[signal] = "bullish"
    return result
