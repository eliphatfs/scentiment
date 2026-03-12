"""Star candlestick patterns.

Implementations follow *Japanese Candlestick Charting Techniques* by Steve Nison,
chapters 6 and 9.

Each public function accepts a DataFrame with columns (open, high, low, close)
and returns a pd.Series of signal strings: 'bullish', 'bearish', or None.

Single-bar stars (inverted hammer / shooting star)
---------------------------------------------------
The inverted-hammer shape has its small body at the LOWER end of the range
with a long upper shadow — the mirror image of the hammer.

  inverted_hammer  : appears after a downtrend → bullish potential
  shooting_star    : appears after an uptrend  → bearish reversal

Three-bar stars (morning / evening)
-------------------------------------
The signal bar is the THIRD candle; the pattern spans bars t-2, t-1, t.

  morning_star      : bullish reversal after downtrend
  evening_star      : bearish reversal after uptrend
  morning_doji_star : morning star where the middle candle is a doji
  evening_doji_star : evening star where the middle candle is a doji

Gap requirement
---------------
All three-bar patterns accept ``require_gap=True`` (default).  When True,
the star's body must gap entirely away from the first candle's body — below
it for bullish patterns, above it for bearish.  Set to False to relax this
requirement for markets where overnight gaps are rare.
"""

import pandas as pd

from patterns._candle import (
    body,
    body_bottom,
    body_size,
    body_top,
    candle_range,
    is_doji,
    is_downtrend,
    is_uptrend,
    lower_shadow,
    signal_series,
    upper_shadow,
)


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _inverted_hammer_shape(
    df: pd.DataFrame,
    shadow_multiplier: float = 2.0,
    lower_shadow_ratio: float = 0.1,
) -> pd.Series:
    """Boolean mask: inverted-hammer / shooting-star shape.

    Criteria (Nison, ch. 9):
    - Small body at the lower end of the range (non-zero body).
    - Upper shadow ≥ shadow_multiplier × body size.
    - Lower shadow ≤ lower_shadow_ratio × total range.
    """
    b     = body_size(df)
    upper = upper_shadow(df)
    lower = lower_shadow(df)
    rng   = candle_range(df)

    return (rng > 0) & (b > 0) & (upper >= shadow_multiplier * b) & (lower <= lower_shadow_ratio * rng)


def _is_long_body(df: pd.DataFrame, long_body_ratio: float) -> pd.Series:
    """Body occupies at least long_body_ratio of the total range."""
    rng = candle_range(df)
    return (body_size(df) >= long_body_ratio * rng) & (rng > 0)


def _is_small_body(df: pd.DataFrame, small_body_ratio: float) -> pd.Series:
    """Body occupies at most small_body_ratio of the total range."""
    rng = candle_range(df)
    return (body_size(df) <= small_body_ratio * rng) & (rng > 0)


# ---------------------------------------------------------------------------
# Single-bar stars: inverted hammer and shooting star
# ---------------------------------------------------------------------------

def inverted_hammer(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    shadow_multiplier: float = 2.0,
    lower_shadow_ratio: float = 0.1,
) -> pd.Series:
    """Bullish reversal potential: inverted hammer after a downtrend.

    Same shape as the shooting star but appearing in a downtrend context.
    Confirmation from the next candle is recommended before acting.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 9.
    """
    signal = (
        _inverted_hammer_shape(df, shadow_multiplier, lower_shadow_ratio)
        & is_downtrend(df, trend_lookback)
    ).fillna(False)
    result = signal_series(df.index)
    result[signal] = "bullish"
    return result


def shooting_star(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    shadow_multiplier: float = 2.0,
    lower_shadow_ratio: float = 0.1,
) -> pd.Series:
    """Bearish reversal: shooting star after an uptrend.

    Same shape as the inverted hammer but appearing in an uptrend context.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 9.
    """
    signal = (
        _inverted_hammer_shape(df, shadow_multiplier, lower_shadow_ratio)
        & is_uptrend(df, trend_lookback)
    ).fillna(False)
    result = signal_series(df.index)
    result[signal] = "bearish"
    return result


# ---------------------------------------------------------------------------
# Three-bar stars: morning star and evening star
# ---------------------------------------------------------------------------

def _morning_star_signal(
    df: pd.DataFrame,
    trend_lookback: int,
    penetration: float,
    require_gap: bool,
    star_is_doji: bool,
    small_body_ratio: float,
    long_body_ratio: float,
    doji_threshold: float,
) -> pd.Series:
    """Shared logic for morning_star and morning_doji_star."""
    # First candle (t-2): long black body
    first_body = body(df).shift(2)
    first_body_bot = body_bottom(df).shift(2)
    first_body_top = body_top(df).shift(2)   # not used but kept for clarity
    first_long_black = (first_body < 0) & _is_long_body(
        df.shift(2), long_body_ratio
    )

    # Second candle / star (t-1)
    star_body_top = body_top(df).shift(1)
    if star_is_doji:
        star_qualifies = is_doji(df.shift(1), threshold=doji_threshold)
    else:
        star_qualifies = _is_small_body(df.shift(1), small_body_ratio)

    # Gap: star body entirely below first candle's body
    star_gaps_down = star_body_top < first_body_bot

    # Third candle (current, t): white, closes above midpoint of first body
    first_midpoint = first_body_bot + penetration * first_body.abs()
    third_white_penetrates = (body(df) > 0) & (df["close"] > first_midpoint)

    gap_condition = star_gaps_down if require_gap else pd.Series(True, index=df.index)

    # Measure trend at the first candle (t-2): was there a downtrend leading
    # into the pattern?  Checking at t would be misleading because the
    # pattern's own recovery candle raises the current close.
    trend_at_first = df["close"].shift(2) < df["close"].shift(2 + trend_lookback)

    return (
        trend_at_first
        & first_long_black
        & star_qualifies
        & gap_condition
        & third_white_penetrates
    ).fillna(False)


def _evening_star_signal(
    df: pd.DataFrame,
    trend_lookback: int,
    penetration: float,
    require_gap: bool,
    star_is_doji: bool,
    small_body_ratio: float,
    long_body_ratio: float,
    doji_threshold: float,
) -> pd.Series:
    """Shared logic for evening_star and evening_doji_star."""
    # First candle (t-2): long white body
    first_body = body(df).shift(2)
    first_body_top = body_top(df).shift(2)
    first_body_bot = body_bottom(df).shift(2)   # not used but kept for clarity
    first_long_white = (first_body > 0) & _is_long_body(
        df.shift(2), long_body_ratio
    )

    # Second candle / star (t-1)
    star_body_bot = body_bottom(df).shift(1)
    if star_is_doji:
        star_qualifies = is_doji(df.shift(1), threshold=doji_threshold)
    else:
        star_qualifies = _is_small_body(df.shift(1), small_body_ratio)

    # Gap: star body entirely above first candle's body
    star_gaps_up = star_body_bot > first_body_top

    # Third candle (current, t): black, closes below midpoint of first body
    first_midpoint = first_body_top - penetration * first_body.abs()
    third_black_penetrates = (body(df) < 0) & (df["close"] < first_midpoint)

    gap_condition = star_gaps_up if require_gap else pd.Series(True, index=df.index)

    # Measure trend at the first candle (t-2): was there an uptrend leading
    # into the pattern?  Checking at t would be misleading because the
    # pattern's own declining third candle lowers the current close.
    trend_at_first = df["close"].shift(2) > df["close"].shift(2 + trend_lookback)

    return (
        trend_at_first
        & first_long_white
        & star_qualifies
        & gap_condition
        & third_black_penetrates
    ).fillna(False)


def morning_star(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    penetration: float = 0.5,
    require_gap: bool = True,
    small_body_ratio: float = 0.3,
    long_body_ratio: float = 0.6,
) -> pd.Series:
    """Bullish reversal: morning star (three-candle pattern).

    Criteria (Nison, ch. 6):
    1. Long black (bearish) candle.
    2. Small-bodied star that gaps below the first candle's body.
    3. White candle that closes above the midpoint of the first candle's body.

    The signal is assigned to the index of the third (current) candle.

    Returns a Series of signal strings ('bullish' or None).
    """
    signal = _morning_star_signal(
        df, trend_lookback, penetration, require_gap,
        star_is_doji=False,
        small_body_ratio=small_body_ratio,
        long_body_ratio=long_body_ratio,
        doji_threshold=0.1,
    )
    result = signal_series(df.index)
    result[signal] = "bullish"
    return result


def morning_doji_star(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    penetration: float = 0.5,
    require_gap: bool = True,
    long_body_ratio: float = 0.6,
    doji_threshold: float = 0.1,
) -> pd.Series:
    """Bullish reversal: morning doji star (stronger than plain morning star).

    Identical to the morning star except the middle candle must be a doji
    (body ≤ doji_threshold × range).

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 8.
    """
    signal = _morning_star_signal(
        df, trend_lookback, penetration, require_gap,
        star_is_doji=True,
        small_body_ratio=0.3,
        long_body_ratio=long_body_ratio,
        doji_threshold=doji_threshold,
    )
    result = signal_series(df.index)
    result[signal] = "bullish"
    return result


def evening_star(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    penetration: float = 0.5,
    require_gap: bool = True,
    small_body_ratio: float = 0.3,
    long_body_ratio: float = 0.6,
) -> pd.Series:
    """Bearish reversal: evening star (three-candle pattern).

    Criteria (Nison, ch. 6):
    1. Long white (bullish) candle.
    2. Small-bodied star that gaps above the first candle's body.
    3. Black candle that closes below the midpoint of the first candle's body.

    The signal is assigned to the index of the third (current) candle.

    Returns a Series of signal strings ('bearish' or None).
    """
    signal = _evening_star_signal(
        df, trend_lookback, penetration, require_gap,
        star_is_doji=False,
        small_body_ratio=small_body_ratio,
        long_body_ratio=long_body_ratio,
        doji_threshold=0.1,
    )
    result = signal_series(df.index)
    result[signal] = "bearish"
    return result


def evening_doji_star(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    penetration: float = 0.5,
    require_gap: bool = True,
    long_body_ratio: float = 0.6,
    doji_threshold: float = 0.1,
) -> pd.Series:
    """Bearish reversal: evening doji star (stronger than plain evening star).

    Identical to the evening star except the middle candle must be a doji.

    Reference: Nison, *Japanese Candlestick Charting Techniques*, ch. 8.
    """
    signal = _evening_star_signal(
        df, trend_lookback, penetration, require_gap,
        star_is_doji=True,
        small_body_ratio=0.3,
        long_body_ratio=long_body_ratio,
        doji_threshold=doji_threshold,
    )
    result = signal_series(df.index)
    result[signal] = "bearish"
    return result
