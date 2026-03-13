"""Confirmation-delayed pattern signals.

Many candlestick patterns require confirmation from the next session before
they become actionable.  Nison stresses throughout *Japanese Candlestick
Charting Techniques* that certain patterns — especially the hanging man,
shooting star, and doji — are warnings that need follow-through to confirm.

This module provides:
  - ``confirmed_signal()`` — generic confirmation engine
  - Convenience wrappers for each pattern requiring confirmation
  - ``CONFIRMATION_RULES`` — registry mapping pattern names to their
    default confirmation type

Confirmation types
------------------
  close_below_body  : next bar closes below the pattern bar's body bottom
  close_above_body  : next bar closes above the pattern bar's body top
  bearish_candle    : next bar is bearish (close < open)
  bullish_candle    : next bar is bullish (close > open)
  gap_down          : next bar opens below pattern bar's close
  gap_up            : next bar opens above pattern bar's close

No-lookahead guarantee
----------------------
The confirmed signal is emitted on the confirmation bar (T+k), not the
pattern bar (T).  At bar T+k, both the pattern and the confirmation are
fully observed.
"""

import pandas as pd

from patterns._candle import body_bottom, body_top, signal_series


# ---------------------------------------------------------------------------
# Confirmation rules registry
# ---------------------------------------------------------------------------

CONFIRMATION_RULES: dict[str, tuple[str, int]] = {
    "hanging_man":      ("close_below_body", 1),
    "shooting_star":    ("close_below_body", 1),
    "inverted_hammer":  ("close_above_body", 1),
    "doji_at_top":      ("bearish_candle", 1),
    "doji_at_bottom":   ("bullish_candle", 1),
    "gravestone_doji":  ("bearish_candle", 1),
    "long_legged_doji": ("opposite_candle", 1),
    "rickshaw_man":     ("opposite_candle", 1),
    "harami":           ("opposite_candle", 1),
    "harami_cross":     ("opposite_candle", 1),
}


# ---------------------------------------------------------------------------
# Core confirmation engine
# ---------------------------------------------------------------------------

def confirmed_signal(
    df: pd.DataFrame,
    raw_signal: pd.Series,
    confirm_type: str,
    max_wait: int = 1,
) -> pd.Series:
    """Delay a pattern signal until confirmation arrives.

    Parameters
    ----------
    df : DataFrame
        OHLCV data (must have open, high, low, close columns).
    raw_signal : Series
        Output of a pattern detection function ('bullish', 'bearish', None).
    confirm_type : str
        One of the confirmation types listed in the module docstring.
    max_wait : int
        Maximum number of bars after the pattern to wait for confirmation.
        If confirmation does not arrive within ``max_wait`` bars, the signal
        is dropped.

    Returns
    -------
    pd.Series
        Confirmed signals, emitted on the confirmation bar.
    """
    result = signal_series(df.index)
    bb = body_bottom(df)
    bt = body_top(df)

    for i in range(len(df)):
        val = raw_signal.iloc[i]
        if val is None:
            continue

        for k in range(1, max_wait + 1):
            j = i + k
            if j >= len(df):
                break

            confirmed = _check_confirmation(
                df, bb, bt, i, j, val, confirm_type,
            )
            if confirmed:
                result.iloc[j] = val
                break

    return result


def _check_confirmation(
    df: pd.DataFrame,
    bb: pd.Series,
    bt: pd.Series,
    pattern_idx: int,
    confirm_idx: int,
    signal_val: str,
    confirm_type: str,
) -> bool:
    """Check whether the bar at confirm_idx confirms the pattern at pattern_idx."""
    if confirm_type == "close_below_body":
        return df["close"].iloc[confirm_idx] < bb.iloc[pattern_idx]

    elif confirm_type == "close_above_body":
        return df["close"].iloc[confirm_idx] > bt.iloc[pattern_idx]

    elif confirm_type == "bearish_candle":
        return df["close"].iloc[confirm_idx] < df["open"].iloc[confirm_idx]

    elif confirm_type == "bullish_candle":
        return df["close"].iloc[confirm_idx] > df["open"].iloc[confirm_idx]

    elif confirm_type == "gap_down":
        return df["open"].iloc[confirm_idx] < df["close"].iloc[pattern_idx]

    elif confirm_type == "gap_up":
        return df["open"].iloc[confirm_idx] > df["close"].iloc[pattern_idx]

    elif confirm_type == "opposite_candle":
        # For bidirectional patterns: bearish signal needs bearish candle,
        # bullish signal needs bullish candle
        if signal_val == "bearish":
            return df["close"].iloc[confirm_idx] < df["open"].iloc[confirm_idx]
        elif signal_val == "bullish":
            return df["close"].iloc[confirm_idx] > df["open"].iloc[confirm_idx]
        return False

    else:
        raise ValueError(f"Unknown confirmation type: {confirm_type!r}")


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def confirmed_hanging_man(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Hanging man with confirmation: next bar closes below body."""
    from patterns.reversal import hanging_man
    raw = hanging_man(df, **kwargs)
    return confirmed_signal(df, raw, "close_below_body", max_wait=1)


def confirmed_shooting_star(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Shooting star with confirmation: next bar closes below body."""
    from patterns.stars import shooting_star
    raw = shooting_star(df, **kwargs)
    return confirmed_signal(df, raw, "close_below_body", max_wait=1)


def confirmed_inverted_hammer(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Inverted hammer with confirmation: next bar closes above body."""
    from patterns.stars import inverted_hammer
    raw = inverted_hammer(df, **kwargs)
    return confirmed_signal(df, raw, "close_above_body", max_wait=1)


def confirmed_doji_at_top(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Doji at top with confirmation: next bar is bearish."""
    from patterns.doji import doji_at_top
    raw = doji_at_top(df, **kwargs)
    return confirmed_signal(df, raw, "bearish_candle", max_wait=1)


def confirmed_doji_at_bottom(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Doji at bottom with confirmation: next bar is bullish."""
    from patterns.doji import doji_at_bottom
    raw = doji_at_bottom(df, **kwargs)
    return confirmed_signal(df, raw, "bullish_candle", max_wait=1)


def confirmed_gravestone_doji(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Gravestone doji with confirmation: next bar is bearish."""
    from patterns.doji import gravestone_doji
    raw = gravestone_doji(df, **kwargs)
    return confirmed_signal(df, raw, "bearish_candle", max_wait=1)


def confirmed_harami(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Harami with confirmation: next bar continues in reversal direction."""
    from patterns.more_reversals import harami
    raw = harami(df, **kwargs)
    return confirmed_signal(df, raw, "opposite_candle", max_wait=1)


def confirmed_harami_cross(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Harami cross with confirmation: next bar continues in reversal direction."""
    from patterns.more_reversals import harami_cross
    raw = harami_cross(df, **kwargs)
    return confirmed_signal(df, raw, "opposite_candle", max_wait=1)
