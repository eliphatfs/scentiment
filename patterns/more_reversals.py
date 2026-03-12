"""More reversal candlestick patterns — Nison chapters 10–12.

Implements the patterns introduced in the latter half of
*Japanese Candlestick Charting Techniques* by Steve Nison.

Chapter 10  (two-candle / single-candle):
  harami, harami_cross, tweezers_top, tweezers_bottom, belt_hold

Chapter 11  (three-candle):
  upside_gap_two_crows, three_black_crows, counterattack_lines

Chapter 12  (multi-bar / chart patterns):
  three_mountains, three_rivers,
  dumpling_top, fry_pan_bottom,
  tower_top, tower_bottom

All functions accept a DataFrame with columns (open, high, low, close) and
return a pd.Series of signal strings: 'bullish', 'bearish', or None.

Multi-bar patterns (chapter 12) use rolling windows over past confirmed bars
only; the window passed to each helper never includes future data.
"""

import numpy as np
import pandas as pd

from patterns._candle import (
    body,
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
from patterns.pivots import pivot_highs, pivot_lows


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
# Chapter 10 — Harami
# ---------------------------------------------------------------------------

def harami(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    require_doji: bool = False,
    doji_threshold: float = 0.1,
) -> pd.Series:
    """Bullish or bearish harami.

    The second candle's body is completely contained within the first candle's
    body.  The first candle should be a long candle; the second should be small.

    Bullish harami  : large black → small body, after downtrend.
    Bearish harami  : large white → small body, after uptrend.

    Pass ``require_doji=True`` to restrict the second candle to a doji
    (equivalent to the harami cross).

    Reference: Nison, ch. 10.
    """
    prev_b = body(df).shift(1)

    # Second body contained within first body (strict)
    contained = (
        (body_top(df) < body_top(df).shift(1))
        & (body_bottom(df) > body_bottom(df).shift(1))
    )

    if require_doji:
        second_qualifies = is_doji(df, threshold=doji_threshold)
    else:
        second_qualifies = pd.Series(True, index=df.index)

    bullish = contained & (prev_b < 0) & second_qualifies & is_downtrend(df, trend_lookback)
    bearish = contained & (prev_b > 0) & second_qualifies & is_uptrend(df, trend_lookback)

    result = signal_series(df.index)
    result[bullish.fillna(False)] = "bullish"
    result[bearish.fillna(False)] = "bearish"
    return result


def harami_cross(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    doji_threshold: float = 0.1,
) -> pd.Series:
    """Harami where the second candle is a doji — stronger reversal signal.

    Reference: Nison, ch. 10.
    """
    return harami(df, trend_lookback, require_doji=True, doji_threshold=doji_threshold)


# ---------------------------------------------------------------------------
# Chapter 10 — Tweezers
# ---------------------------------------------------------------------------

def tweezers_top(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    tolerance: float = 0.001,
) -> pd.Series:
    """Bearish reversal: two consecutive bars with matching highs after an uptrend.

    The matching is within ``tolerance`` as a fraction of the prior bar's high.

    Reference: Nison, ch. 10.
    """
    high_match = (df["high"] - df["high"].shift(1)).abs() <= tolerance * df["high"].shift(1)
    signal = (high_match & is_uptrend(df, trend_lookback)).fillna(False)
    result = signal_series(df.index)
    result[signal] = "bearish"
    return result


def tweezers_bottom(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    tolerance: float = 0.001,
) -> pd.Series:
    """Bullish reversal: two consecutive bars with matching lows after a downtrend.

    Reference: Nison, ch. 10.
    """
    low_match = (df["low"] - df["low"].shift(1)).abs() <= tolerance * df["low"].shift(1)
    signal = (low_match & is_downtrend(df, trend_lookback)).fillna(False)
    result = signal_series(df.index)
    result[signal] = "bullish"
    return result


# ---------------------------------------------------------------------------
# Chapter 10 — Belt-Hold Lines
# ---------------------------------------------------------------------------

def belt_hold(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    shadow_ratio: float = 0.05,
    long_body_ratio: float = 0.6,
) -> pd.Series:
    """Bullish or bearish belt-hold line.

    Bullish belt-hold (white opening shaven bottom): white candle whose open
    is at or near the session low (no lower shadow), long body, after downtrend.

    Bearish belt-hold (black opening shaven head): black candle whose open is
    at or near the session high (no upper shadow), long body, after uptrend.

    Reference: Nison, ch. 10.
    """
    rng = candle_range(df)
    b = body(df)

    bullish = (
        is_downtrend(df, trend_lookback)
        & (b > 0)
        & (lower_shadow(df) <= shadow_ratio * rng)
        & _long_body(df, long_body_ratio)
    )
    bearish = (
        is_uptrend(df, trend_lookback)
        & (b < 0)
        & (upper_shadow(df) <= shadow_ratio * rng)
        & _long_body(df, long_body_ratio)
    )

    result = signal_series(df.index)
    result[bullish.fillna(False)] = "bullish"
    result[bearish.fillna(False)] = "bearish"
    return result


# ---------------------------------------------------------------------------
# Chapter 11 — Upside-Gap Two Crows
# ---------------------------------------------------------------------------

def upside_gap_two_crows(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    long_body_ratio: float = 0.6,
) -> pd.Series:
    """Bearish reversal: upside-gap two crows (three-candle pattern).

    Criteria (Nison, ch. 11):
    1. Long white candle.
    2. Black candle that gaps up above the first body and closes above it.
    3. Larger black candle that opens above the second open and closes into the
       first candle's body (below first close, above first open).

    The signal is assigned to the index of the third candle.
    """
    first_b     = body(df).shift(2)
    first_open  = df["open"].shift(2)
    first_close = df["close"].shift(2)

    second_b    = body(df).shift(1)
    second_open = df["open"].shift(1)
    second_close = df["close"].shift(1)

    curr_b     = body(df)
    curr_open  = df["open"]
    curr_close = df["close"]

    # Trend measured at the first candle to avoid the pattern's own decline
    trend_at_first = is_uptrend_by_pivots(df).shift(2).fillna(False)

    signal = (
        trend_at_first
        & (first_b > 0) & _long_body(df.shift(2), long_body_ratio)  # 1st long white
        & (second_b < 0)                                             # 2nd black
        & (second_open > first_close)                                # 2nd gaps up
        & (second_close > first_close)                               # 2nd doesn't fill gap
        & (curr_b < 0)                                               # 3rd black
        & (curr_open > second_open)                                  # 3rd opens higher than 2nd
        & (curr_close < first_close)                                 # 3rd closes into 1st body
        & (curr_close > first_open)
    ).fillna(False)

    result = signal_series(df.index)
    result[signal] = "bearish"
    return result


# ---------------------------------------------------------------------------
# Chapter 11 — Three Black Crows
# ---------------------------------------------------------------------------

def three_black_crows(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    long_body_ratio: float = 0.6,
    shadow_ratio: float = 0.1,
) -> pd.Series:
    """Bearish reversal: three black crows (three-candle pattern).

    Criteria (Nison, ch. 11):
    - Three consecutive long black candles.
    - Each opens within the prior candle's body.
    - Each closes near its low (small lower shadow).

    Reference: Nison, ch. 11.
    """
    # Opening within prior body
    opens_in_prior_body = lambda n: (
        (df["open"].shift(n - 1) < body_top(df).shift(n))
        & (df["open"].shift(n - 1) > body_bottom(df).shift(n))
    )

    # Lower shadow small relative to range
    low_shadow_small = lambda n: (
        lower_shadow(df).shift(n) <= shadow_ratio * candle_range(df).shift(n)
    )

    trend_at_first = is_uptrend_by_pivots(df).shift(2).fillna(False)

    signal = (
        trend_at_first
        & (body(df).shift(2) < 0) & _long_body(df.shift(2), long_body_ratio)
        & (body(df).shift(1) < 0) & _long_body(df.shift(1), long_body_ratio)
        & (body(df) < 0)          & _long_body(df, long_body_ratio)
        & opens_in_prior_body(2)   # 2nd opens within 1st
        & opens_in_prior_body(1)   # 3rd opens within 2nd
        & low_shadow_small(2) & low_shadow_small(1) & low_shadow_small(0)
    ).fillna(False)

    result = signal_series(df.index)
    result[signal] = "bearish"
    return result


# ---------------------------------------------------------------------------
# Chapter 11 — Counterattack Lines
# ---------------------------------------------------------------------------

def counterattack_lines(
    df: pd.DataFrame,
    trend_lookback: int = 5,
    close_tolerance: float = 0.001,
) -> pd.Series:
    """Bullish or bearish counterattack lines.

    The second candle opens with a gap in the direction of the trend but
    closes at (approximately) the same price as the first candle's close.
    Unlike the piercing pattern / dark cloud cover, there is no penetration
    into the prior body — the pattern is therefore a weaker reversal signal.

    Bullish counterattack: black → white, closes at same level, after downtrend.
    Bearish counterattack: white → black, closes at same level, after uptrend.

    Reference: Nison, ch. 11.
    """
    prev_close = df["close"].shift(1)
    prev_b = body(df).shift(1)

    # Closes match within tolerance
    close_match = (df["close"] - prev_close).abs() <= close_tolerance * prev_close

    bullish = (
        close_match
        & (prev_b < 0)
        & (body(df) > 0)
        & is_downtrend(df, trend_lookback)
    )
    bearish = (
        close_match
        & (prev_b > 0)
        & (body(df) < 0)
        & is_uptrend(df, trend_lookback)
    )

    result = signal_series(df.index)
    result[bullish.fillna(False)] = "bullish"
    result[bearish.fillna(False)] = "bearish"
    return result


# ---------------------------------------------------------------------------
# Chapter 12 — Three Mountains and Three Rivers
# ---------------------------------------------------------------------------

def three_mountains(
    df: pd.DataFrame,
    lookback: int = 30,
    tolerance: float = 0.003,
) -> pd.Series:
    """Bearish reversal: three mountains (triple-top chart pattern).

    Looks for three confirmed 1st-order pivot highs within ``lookback`` bars
    of each other whose high values are within ``tolerance`` (relative) of
    each other.  The signal fires on the bar that confirms the third peak
    (i.e. the bar immediately following the third pivot high).

    Reference: Nison, ch. 12.
    """
    ph = pivot_highs(df, order=1)
    ph_positions = list(np.where(ph.values)[0])

    result = signal_series(df.index)

    for j in range(2, len(ph_positions)):
        p1, p2, p3 = ph_positions[j - 2], ph_positions[j - 1], ph_positions[j]

        if p3 - p1 > lookback:
            continue

        h1 = df["high"].iloc[p1]
        h2 = df["high"].iloc[p2]
        h3 = df["high"].iloc[p3]
        mean_h = (h1 + h2 + h3) / 3

        if max(h1, h2, h3) - min(h1, h2, h3) > tolerance * mean_h:
            continue

        # Confirmed on the bar that establishes p3 as a pivot (p3 + 1)
        sig = p3 + 1
        if sig < len(df):
            result.iloc[sig] = "bearish"

    return result


def three_rivers(
    df: pd.DataFrame,
    lookback: int = 30,
    tolerance: float = 0.003,
) -> pd.Series:
    """Bullish reversal: three rivers (triple-bottom chart pattern).

    Mirror of three_mountains: three confirmed 1st-order pivot lows within
    ``lookback`` bars whose low values are within ``tolerance`` of each other.

    Reference: Nison, ch. 12.
    """
    pl = pivot_lows(df, order=1)
    pl_positions = list(np.where(pl.values)[0])

    result = signal_series(df.index)

    for j in range(2, len(pl_positions)):
        p1, p2, p3 = pl_positions[j - 2], pl_positions[j - 1], pl_positions[j]

        if p3 - p1 > lookback:
            continue

        l1 = df["low"].iloc[p1]
        l2 = df["low"].iloc[p2]
        l3 = df["low"].iloc[p3]
        mean_l = (l1 + l2 + l3) / 3

        if max(l1, l2, l3) - min(l1, l2, l3) > tolerance * mean_l:
            continue

        sig = p3 + 1
        if sig < len(df):
            result.iloc[sig] = "bullish"

    return result


# ---------------------------------------------------------------------------
# Chapter 12 — Dumpling Top and Fry Pan Bottom
# ---------------------------------------------------------------------------

def _closes_form_arc(closes: np.ndarray, is_top: bool) -> bool:
    """True if the close array forms a parabolic arc (dome or bowl shape).

    The peak (for top) or trough (for bottom) must fall in the middle third
    of the window.  We also require the first and last thirds to both move
    toward the extreme, confirming the arc structure.
    """
    n = len(closes)
    if n < 5:
        return False

    third = max(1, n // 3)
    mid_idx = np.argmax(closes) if is_top else np.argmin(closes)

    # Extreme must be in the middle third
    if not (third <= mid_idx <= 2 * third):
        return False

    # Average of first third should be below (top) or above (bottom) than average of middle
    avg_first  = closes[:third].mean()
    avg_middle = closes[third: 2 * third].mean()
    avg_last   = closes[2 * third:].mean()

    if is_top:
        return avg_first < avg_middle and avg_last < avg_middle
    else:
        return avg_first > avg_middle and avg_last > avg_middle


def dumpling_top(
    df: pd.DataFrame,
    window: int = 10,
    small_body_ratio: float = 0.4,
) -> pd.Series:
    """Bearish reversal: dumpling top.

    A series of small-bodied candles forms a rounded, dome-shaped arc in the
    close prices, confirmed by a downside gap on the current bar
    (current open < previous close).

    Reference: Nison, ch. 12.
    """
    rng = candle_range(df)
    is_small = (body_size(df) <= small_body_ratio * rng) | (rng == 0)
    gap_down = df["open"] < df["close"].shift(1)

    result = signal_series(df.index)

    for i in range(window, len(df)):
        if not gap_down.iloc[i]:
            continue
        # Window is the `window` bars immediately before bar i (no lookahead)
        w_slice = slice(i - window, i)
        if not is_small.iloc[w_slice].all():
            continue
        if _closes_form_arc(df["close"].iloc[w_slice].values, is_top=True):
            result.iloc[i] = "bearish"

    return result


def fry_pan_bottom(
    df: pd.DataFrame,
    window: int = 10,
    small_body_ratio: float = 0.4,
) -> pd.Series:
    """Bullish reversal: fry pan bottom.

    Mirror of the dumpling top: small-bodied candles form a bowl-shaped (U)
    arc in close prices, confirmed by an upside gap on the current bar
    (current open > previous close).

    Reference: Nison, ch. 12.
    """
    rng = candle_range(df)
    is_small = (body_size(df) <= small_body_ratio * rng) | (rng == 0)
    gap_up = df["open"] > df["close"].shift(1)

    result = signal_series(df.index)

    for i in range(window, len(df)):
        if not gap_up.iloc[i]:
            continue
        w_slice = slice(i - window, i)
        if not is_small.iloc[w_slice].all():
            continue
        if _closes_form_arc(df["close"].iloc[w_slice].values, is_top=False):
            result.iloc[i] = "bullish"

    return result


# ---------------------------------------------------------------------------
# Chapter 12 — Tower Tops and Bottoms
# ---------------------------------------------------------------------------

def tower_top(
    df: pd.DataFrame,
    window: int = 10,
    long_body_ratio: float = 0.6,
    small_body_ratio: float = 0.3,
) -> pd.Series:
    """Bearish reversal: tower top.

    Structure (across ``window`` bars ending at the current bar):
    - First portion : one or more long white candles.
    - Middle portion: all small-bodied candles (consolidation / stall).
    - Final portion : one or more long black candles (current bar can be one).

    Reference: Nison, ch. 12.
    """
    b = body(df)
    is_long  = _long_body(df, long_body_ratio)
    is_small = _small_body(df, small_body_ratio)

    result = signal_series(df.index)
    third = max(1, window // 3)

    for i in range(window - 1, len(df)):
        start = i - window + 1
        mid_s = start + third
        mid_e = i - third + 1

        if mid_s >= mid_e:
            continue

        # First third: at least one long white
        has_long_white = (is_long.iloc[start:mid_s] & (b.iloc[start:mid_s] > 0)).any()
        # Middle: all small bodies
        all_small_mid = is_small.iloc[mid_s:mid_e].all()
        # Last third: at least one long black
        has_long_black = (is_long.iloc[mid_e:i + 1] & (b.iloc[mid_e:i + 1] < 0)).any()

        if has_long_white and all_small_mid and has_long_black:
            result.iloc[i] = "bearish"

    return result


def tower_bottom(
    df: pd.DataFrame,
    window: int = 10,
    long_body_ratio: float = 0.6,
    small_body_ratio: float = 0.3,
) -> pd.Series:
    """Bullish reversal: tower bottom.

    Mirror of the tower top:
    - First portion : one or more long black candles.
    - Middle portion: all small-bodied candles.
    - Final portion : one or more long white candles.

    Reference: Nison, ch. 12.
    """
    b = body(df)
    is_long  = _long_body(df, long_body_ratio)
    is_small = _small_body(df, small_body_ratio)

    result = signal_series(df.index)
    third = max(1, window // 3)

    for i in range(window - 1, len(df)):
        start = i - window + 1
        mid_s = start + third
        mid_e = i - third + 1

        if mid_s >= mid_e:
            continue

        has_long_black = (is_long.iloc[start:mid_s] & (b.iloc[start:mid_s] < 0)).any()
        all_small_mid  = is_small.iloc[mid_s:mid_e].all()
        has_long_white = (is_long.iloc[mid_e:i + 1] & (b.iloc[mid_e:i + 1] > 0)).any()

        if has_long_black and all_small_mid and has_long_white:
            result.iloc[i] = "bullish"

    return result
