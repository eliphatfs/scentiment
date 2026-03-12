"""Shared single-bar candle geometry helpers used across pattern modules."""

import pandas as pd


def body(df: pd.DataFrame) -> pd.Series:
    """Signed body: positive for white (bullish), negative for black (bearish)."""
    return df["close"] - df["open"]


def body_size(df: pd.DataFrame) -> pd.Series:
    return body(df).abs()


def body_top(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].max(axis=1)


def body_bottom(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1)


def upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["high"] - body_top(df)


def lower_shadow(df: pd.DataFrame) -> pd.Series:
    return body_bottom(df) - df["low"]


def candle_range(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df["low"]


def is_uptrend_by_pivots(df: pd.DataFrame, order: int = 1) -> pd.Series:
    """True where confirmed pivot structure shows higher highs AND higher lows.

    Implements the Grimes trend definition (*The Art and Science of Technical
    Analysis*): an uptrend is a sequence of ascending pivot highs and ascending
    pivot lows.  At each bar we compare the most recent two confirmed pivot
    highs and the most recent two confirmed pivot lows; both pairs must be
    ascending.

    Pivots are shifted by 1 before use so the signal at bar t only relies on
    information available at the close of bar t (no lookahead).

    Returns False where fewer than two confirmed pivots of either type exist.
    """
    from patterns.pivots import pivot_highs, pivot_lows

    ph_confirmed = pivot_highs(df, order).shift(1).fillna(False).astype(bool)
    pl_confirmed = pivot_lows(df, order).shift(1).fillna(False).astype(bool)

    ph_dense = df["high"].where(ph_confirmed).dropna()
    pl_dense = df["low"].where(pl_confirmed).dropna()

    last_ph   = ph_dense.reindex(df.index).ffill()
    prev_ph   = ph_dense.shift(1).reindex(df.index).ffill()
    last_pl   = pl_dense.reindex(df.index).ffill()
    prev_pl   = pl_dense.shift(1).reindex(df.index).ffill()

    return (last_ph > prev_ph) & (last_pl > prev_pl)


def is_downtrend_by_pivots(df: pd.DataFrame, order: int = 1) -> pd.Series:
    """True where confirmed pivot structure shows lower highs AND lower lows.

    Mirror of ``is_uptrend_by_pivots``: both the most recent pair of pivot
    highs and the most recent pair of pivot lows must be descending.
    """
    from patterns.pivots import pivot_highs, pivot_lows

    ph_confirmed = pivot_highs(df, order).shift(1).fillna(False).astype(bool)
    pl_confirmed = pivot_lows(df, order).shift(1).fillna(False).astype(bool)

    ph_dense = df["high"].where(ph_confirmed).dropna()
    pl_dense = df["low"].where(pl_confirmed).dropna()

    last_ph   = ph_dense.reindex(df.index).ffill()
    prev_ph   = ph_dense.shift(1).reindex(df.index).ffill()
    last_pl   = pl_dense.reindex(df.index).ffill()
    prev_pl   = pl_dense.shift(1).reindex(df.index).ffill()

    return (last_ph < prev_ph) & (last_pl < prev_pl)


def is_uptrend(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """True where pivot structure shows an uptrend (Grimes definition).

    The ``lookback`` parameter is retained for API compatibility but is no
    longer used; trend is determined by pivot-high/low structure instead.
    """
    return is_uptrend_by_pivots(df)


def is_downtrend(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """True where pivot structure shows a downtrend (Grimes definition).

    The ``lookback`` parameter is retained for API compatibility but is no
    longer used; trend is determined by pivot-high/low structure instead.
    """
    return is_downtrend_by_pivots(df)


def is_doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """True where body_size ≤ threshold × range (or range is zero)."""
    rng = candle_range(df)
    return (body_size(df) <= threshold * rng) | (rng == 0)


def signal_series(index: pd.Index) -> pd.Series:
    """Return an all-None object Series to be filled with signal strings."""
    return pd.Series([None] * len(index), index=index, dtype=object)
