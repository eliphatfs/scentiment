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


def is_uptrend(df: pd.DataFrame, lookback: int) -> pd.Series:
    return df["close"] > df["close"].shift(lookback)


def is_downtrend(df: pd.DataFrame, lookback: int) -> pd.Series:
    return df["close"] < df["close"].shift(lookback)


def is_doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """True where body_size ≤ threshold × range (or range is zero)."""
    rng = candle_range(df)
    return (body_size(df) <= threshold * rng) | (rng == 0)


def signal_series(index: pd.Index) -> pd.Series:
    """Return an all-None object Series to be filled with signal strings."""
    return pd.Series([None] * len(index), index=index, dtype=object)
