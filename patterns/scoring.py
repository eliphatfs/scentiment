"""Pattern strength scoring.

Assigns a 0.0–1.0 quality score to each detected pattern signal based on:

1. **Shape quality** — how close the candle geometry is to the textbook ideal
2. **Volume confirmation** — whether the signal bar has above-average volume
3. **Trend strength** — how established the preceding trend was

The scoring follows guidelines from Nison's *Japanese Candlestick Charting
Techniques* (chapters 3–14) and Grimes' *The Art and Science of Technical
Analysis* for trend measurement.

Usage
-----
Scoring functions take a DataFrame and an existing signal Series (output of
a pattern function) and return a float Series of the same length, with scores
at signal bars and NaN elsewhere.

The ``pattern_strength()`` function combines all components into a single
composite score.
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
    lower_shadow,
    upper_shadow,
    _regression_slope,
)


# ---------------------------------------------------------------------------
# Shape quality scores (0.0 – 1.0)
# ---------------------------------------------------------------------------

def score_hammer_shape(df: pd.DataFrame) -> pd.Series:
    """Shape quality for hammer / hanging man candles.

    Ideal hammer: lower shadow exactly 2× body, zero upper shadow.
    Score increases with shadow/body ratio and penalizes upper shadow.
    """
    b = body_size(df)
    ls = lower_shadow(df)
    us = upper_shadow(df)
    rng = candle_range(df)

    # Shadow ratio component: ls/body, clamped to [0, 1] for 2×–4× range
    shadow_score = ((ls / b.replace(0, np.nan)) - 2.0).clip(0, 2.0) / 2.0
    shadow_score = shadow_score.fillna(0)

    # Upper shadow penalty: 1.0 when no upper shadow, 0.0 when us >= 0.2*range
    upper_penalty = (1.0 - (us / rng.replace(0, np.nan)).clip(0, 0.2) / 0.2)
    upper_penalty = upper_penalty.fillna(0)

    return (0.7 * shadow_score + 0.3 * upper_penalty).clip(0, 1)


def score_inverted_hammer_shape(df: pd.DataFrame) -> pd.Series:
    """Shape quality for inverted hammer / shooting star candles."""
    b = body_size(df)
    us = upper_shadow(df)
    ls = lower_shadow(df)
    rng = candle_range(df)

    shadow_score = ((us / b.replace(0, np.nan)) - 2.0).clip(0, 2.0) / 2.0
    shadow_score = shadow_score.fillna(0)

    lower_penalty = (1.0 - (ls / rng.replace(0, np.nan)).clip(0, 0.2) / 0.2)
    lower_penalty = lower_penalty.fillna(0)

    return (0.7 * shadow_score + 0.3 * lower_penalty).clip(0, 1)


def score_engulfing_shape(df: pd.DataFrame) -> pd.Series:
    """Shape quality for engulfing patterns.

    Score based on how much the current body exceeds the prior body.
    A body 2× the prior scores 1.0.
    """
    curr = body_size(df)
    prev = body_size(df).shift(1)
    ratio = (curr / prev.replace(0, np.nan) - 1.0).clip(0, 1.0)
    return ratio.fillna(0)


def score_dark_cloud_piercing_shape(
    df: pd.DataFrame,
    penetration: float = 0.5,
) -> pd.Series:
    """Shape quality for dark cloud cover / piercing pattern.

    Score based on depth of penetration beyond the minimum threshold.
    """
    prev_body_abs = body_size(df).shift(1)
    prev_bt = body_top(df).shift(1)
    prev_bb = body_bottom(df).shift(1)
    curr_close = df["close"]

    # For bearish (dark cloud): how far below prev midpoint
    # For bullish (piercing): how far above prev midpoint
    # Normalize: penetration from 50% to 100% → score 0 to 1
    pen_down = (prev_bt - curr_close) / prev_body_abs.replace(0, np.nan)
    pen_up = (curr_close - prev_bb) / prev_body_abs.replace(0, np.nan)
    pen = pd.concat([pen_down, pen_up], axis=1).max(axis=1)
    score = ((pen - penetration) / (1.0 - penetration)).clip(0, 1)
    return score.fillna(0)


def score_star_shape(df: pd.DataFrame) -> pd.Series:
    """Shape quality for morning/evening star patterns.

    Based on: gap size between 1st and 2nd candle, plus penetration
    depth of 3rd candle into 1st candle's body.
    """
    # Gap between 1st body and star body
    first_bt = body_top(df).shift(2)
    first_bb = body_bottom(df).shift(2)
    first_body_abs = body_size(df).shift(2)
    star_bt = body_top(df).shift(1)
    star_bb = body_bottom(df).shift(1)

    # Gap score: how far star body is from 1st body, normalized by 1st body size
    gap_down = (first_bb - star_bt).clip(lower=0)
    gap_up = (star_bb - first_bt).clip(lower=0)
    gap = pd.concat([gap_down, gap_up], axis=1).max(axis=1)
    gap_score = (gap / first_body_abs.replace(0, np.nan)).clip(0, 1)

    # 3rd candle penetration score
    curr_close = df["close"]
    # How far into 1st body does 3rd candle close (normalized)
    pen_bull = (curr_close - first_bb) / first_body_abs.replace(0, np.nan)
    pen_bear = (first_bt - curr_close) / first_body_abs.replace(0, np.nan)
    pen = pd.concat([pen_bull, pen_bear], axis=1).max(axis=1)
    pen_score = ((pen - 0.5) / 0.5).clip(0, 1)

    score = (0.4 * gap_score + 0.6 * pen_score).clip(0, 1)
    return score.fillna(0)


def score_doji_shape(df: pd.DataFrame, doji_threshold: float = 0.1) -> pd.Series:
    """Shape quality for doji patterns.

    Score based on how close the body is to zero (ideal doji) and
    the prominence of the shadows.
    """
    b = body_size(df)
    rng = candle_range(df)
    # Body nearness to zero: 1.0 when body=0, 0.0 when body=threshold*range
    body_score = (1.0 - b / (doji_threshold * rng).replace(0, np.nan)).clip(0, 1)
    # Shadow prominence: longer shadows = more significant doji
    shadow_total = upper_shadow(df) + lower_shadow(df)
    shadow_score = (shadow_total / rng.replace(0, np.nan)).clip(0, 1)

    return (0.5 * body_score + 0.5 * shadow_score).fillna(0)


# ---------------------------------------------------------------------------
# Volume confirmation (0.0 – 1.0)
# ---------------------------------------------------------------------------

def score_volume(
    df: pd.DataFrame,
    signal: pd.Series,
    avg_window: int = 20,
) -> pd.Series:
    """Volume confirmation score.

    Above-average volume on the signal bar confirms the pattern.
    Score: 0.0 when volume ≤ 50% of average, 1.0 when volume ≥ 200% average.

    If no volume column or all-zero volume, returns 0.5 (neutral).
    """
    result = pd.Series(np.nan, index=df.index)

    if "volume" not in df.columns or df["volume"].sum() == 0:
        # No volume data — return neutral score at signal bars
        result[signal.notna()] = 0.5
        return result

    vol = df["volume"].astype(float)
    avg_vol = vol.rolling(avg_window, min_periods=1).mean()
    ratio = vol / avg_vol.replace(0, np.nan)

    # Normalize: 0.5× → 0.0, 2.0× → 1.0
    score = ((ratio - 0.5) / 1.5).clip(0, 1)
    result[signal.notna()] = score[signal.notna()]
    return result


# ---------------------------------------------------------------------------
# Trend strength (0.0 – 1.0)
# ---------------------------------------------------------------------------

def score_trend_strength(
    df: pd.DataFrame,
    signal: pd.Series,
    lookback: int = 10,
) -> pd.Series:
    """Trend strength score at signal bars.

    Uses regression slope magnitude normalized by ATR (average true range).
    Stronger preceding trend → more significant reversal pattern.
    """
    result = pd.Series(np.nan, index=df.index)

    slope = _regression_slope(df["close"], lookback)
    # Normalize by ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback, min_periods=1).mean()
    normalized_slope = (slope.abs() / atr.replace(0, np.nan))

    # Score: slope/atr from 0 to 2 maps to 0 to 1
    score = (normalized_slope / 2.0).clip(0, 1)
    result[signal.notna()] = score[signal.notna()]
    return result


# ---------------------------------------------------------------------------
# Composite strength
# ---------------------------------------------------------------------------

# Map pattern names to their shape scoring function
_SHAPE_SCORERS = {
    "hammer":           score_hammer_shape,
    "hanging_man":      score_hammer_shape,
    "inverted_hammer":  score_inverted_hammer_shape,
    "shooting_star":    score_inverted_hammer_shape,
    "engulfing":        score_engulfing_shape,
    "dark_cloud_cover": score_dark_cloud_piercing_shape,
    "piercing_pattern": score_dark_cloud_piercing_shape,
    "morning_star":     score_star_shape,
    "evening_star":     score_star_shape,
    "morning_doji_star": score_star_shape,
    "evening_doji_star": score_star_shape,
    "doji_at_top":      score_doji_shape,
    "doji_at_bottom":   score_doji_shape,
    "long_legged_doji": score_doji_shape,
    "rickshaw_man":     score_doji_shape,
    "gravestone_doji":  score_doji_shape,
    "tri_star":         score_doji_shape,
}


def pattern_strength(
    df: pd.DataFrame,
    pattern_name: str,
    signal: pd.Series,
    shape_weight: float = 0.5,
    volume_weight: float = 0.2,
    trend_weight: float = 0.3,
) -> pd.Series:
    """Composite pattern strength score (0.0 – 1.0).

    Combines shape quality, volume confirmation, and trend strength
    into a single score for each signal bar.  Non-signal bars are NaN.

    Parameters
    ----------
    df : DataFrame
        OHLCV data.
    pattern_name : str
        Name of the pattern (must be a key in _SHAPE_SCORERS, or will
        use a default neutral shape score).
    signal : Series
        Output of the pattern detection function.
    shape_weight, volume_weight, trend_weight : float
        Relative weights for each component.  Must sum to 1.0.

    Returns
    -------
    pd.Series
        Float scores at signal bars, NaN elsewhere.
    """
    result = pd.Series(np.nan, index=df.index)
    mask = signal.notna()
    if not mask.any():
        return result

    # Shape score
    scorer = _SHAPE_SCORERS.get(pattern_name)
    if scorer is not None:
        shape = scorer(df)
    else:
        shape = pd.Series(0.5, index=df.index)

    vol = score_volume(df, signal)
    trend = score_trend_strength(df, signal)

    total_weight = shape_weight + volume_weight + trend_weight
    composite = (
        shape_weight * shape
        + volume_weight * vol.fillna(0.5)
        + trend_weight * trend.fillna(0.5)
    ) / total_weight

    result[mask] = composite[mask]
    return result
