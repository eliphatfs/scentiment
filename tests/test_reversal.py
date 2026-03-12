"""Tests for reversal candlestick patterns.

Each test constructs a minimal synthetic OHLCV DataFrame that either does or
does not satisfy the pattern criteria, then asserts the correct signal is
returned. The 'volume' column is omitted since reversal patterns don't use it.
"""

import pandas as pd
import pytest

from patterns.reversal import (
    dark_cloud_cover,
    engulfing,
    hammer,
    hanging_man,
    piercing_pattern,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of OHLC dicts."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
    return df.astype(float)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _downtrend_prefix(n: int = 5, start: float = 200.0, step: float = 2.0) -> list[dict]:
    """n bars of plain declining candles to establish a downtrend."""
    rows = []
    price = start
    for _ in range(n):
        rows.append({"open": price, "high": price + 0.5, "low": price - 0.5, "close": price - step})
        price -= step
    return rows


def _uptrend_prefix(n: int = 5, start: float = 100.0, step: float = 2.0) -> list[dict]:
    """n bars of plain rising candles to establish an uptrend."""
    rows = []
    price = start
    for _ in range(n):
        rows.append({"open": price, "high": price + 0.5, "low": price - 0.5, "close": price + step})
        price += step
    return rows


# ---------------------------------------------------------------------------
# Hammer
# ---------------------------------------------------------------------------

class TestHammer:
    def _hammer_candle(self, base: float = 100.0) -> dict:
        """Classic hammer shape: small body at top, long lower shadow, no upper shadow."""
        return {"open": base + 1.0, "high": base + 1.5, "low": base - 3.0, "close": base + 1.5}

    def test_detects_hammer_after_downtrend(self):
        rows = _downtrend_prefix() + [self._hammer_candle()]
        df = _df(rows)
        signals = hammer(df, trend_lookback=5)
        assert signals.iloc[-1] == "bullish"

    def test_no_hammer_after_uptrend(self):
        # Use base=115 so the hammer close (116.5) is above the uptrend's
        # first close (~102), ensuring the downtrend condition does not fire.
        rows = _uptrend_prefix() + [self._hammer_candle(base=115.0)]
        df = _df(rows)
        signals = hammer(df, trend_lookback=5)
        assert signals.iloc[-1] is None

    def test_no_hammer_when_upper_shadow_too_large(self):
        rows = _downtrend_prefix()
        # upper shadow = 5, body = 1, lower = 3 → fails upper-shadow filter
        rows.append({"open": 100.0, "high": 106.0, "low": 97.0, "close": 101.0})
        df = _df(rows)
        signals = hammer(df, trend_lookback=5)
        assert signals.iloc[-1] is None

    def test_no_hammer_when_lower_shadow_too_short(self):
        rows = _downtrend_prefix()
        # lower shadow = 0.5, body = 1 → fails 2× multiplier
        rows.append({"open": 100.0, "high": 100.2, "low": 99.5, "close": 101.0})
        df = _df(rows)
        signals = hammer(df, trend_lookback=5)
        assert signals.iloc[-1] is None


# ---------------------------------------------------------------------------
# Hanging Man
# ---------------------------------------------------------------------------

class TestHangingMan:
    def _hanging_man_candle(self, base: float = 120.0) -> dict:
        return {"open": base + 1.0, "high": base + 1.5, "low": base - 3.0, "close": base + 1.5}

    def test_detects_hanging_man_after_uptrend(self):
        rows = _uptrend_prefix() + [self._hanging_man_candle()]
        df = _df(rows)
        signals = hanging_man(df, trend_lookback=5)
        assert signals.iloc[-1] == "bearish"

    def test_no_hanging_man_after_downtrend(self):
        rows = _downtrend_prefix() + [self._hanging_man_candle()]
        df = _df(rows)
        signals = hanging_man(df, trend_lookback=5)
        assert signals.iloc[-1] is None


# ---------------------------------------------------------------------------
# Engulfing
# ---------------------------------------------------------------------------

class TestEngulfing:
    def test_bullish_engulfing_after_downtrend(self):
        rows = _downtrend_prefix()
        # Prior: small black candle
        rows.append({"open": 91.0, "high": 91.5, "low": 89.5, "close": 90.0})
        # Current: large white candle that engulfs prior body
        rows.append({"open": 89.5, "high": 93.0, "low": 89.0, "close": 92.0})
        df = _df(rows)
        signals = engulfing(df, trend_lookback=5)
        assert signals.iloc[-1] == "bullish"

    def test_bearish_engulfing_after_uptrend(self):
        rows = _uptrend_prefix()
        # Prior: small white candle
        rows.append({"open": 109.0, "high": 111.0, "low": 108.5, "close": 110.0})
        # Current: large black candle that engulfs prior body
        rows.append({"open": 111.0, "high": 111.5, "low": 107.5, "close": 108.0})
        df = _df(rows)
        signals = engulfing(df, trend_lookback=5)
        assert signals.iloc[-1] == "bearish"

    def test_no_engulfing_when_body_not_engulfed(self):
        rows = _downtrend_prefix()
        rows.append({"open": 91.0, "high": 91.5, "low": 89.5, "close": 90.0})
        # Current white but does NOT fully engulf prior body
        rows.append({"open": 89.5, "high": 91.0, "low": 89.0, "close": 90.5})
        df = _df(rows)
        signals = engulfing(df, trend_lookback=5)
        assert signals.iloc[-1] is None

    def test_no_engulfing_when_same_color(self):
        rows = _downtrend_prefix()
        rows.append({"open": 91.0, "high": 91.5, "low": 89.5, "close": 90.0})
        # Current is also black (same color as prior)
        rows.append({"open": 92.0, "high": 92.5, "low": 87.0, "close": 88.0})
        df = _df(rows)
        signals = engulfing(df, trend_lookback=5)
        assert signals.iloc[-1] is None


# ---------------------------------------------------------------------------
# Dark Cloud Cover
# ---------------------------------------------------------------------------

class TestDarkCloudCover:
    def test_detects_dark_cloud_after_uptrend(self):
        rows = _uptrend_prefix()
        # Long white candle
        rows.append({"open": 110.0, "high": 116.0, "low": 109.5, "close": 115.0})
        # Opens above prior close, closes below midpoint (112.5) but inside body
        rows.append({"open": 116.0, "high": 117.0, "low": 111.0, "close": 112.0})
        df = _df(rows)
        signals = dark_cloud_cover(df, trend_lookback=5)
        assert signals.iloc[-1] == "bearish"

    def test_no_dark_cloud_when_close_above_midpoint(self):
        rows = _uptrend_prefix()
        rows.append({"open": 110.0, "high": 116.0, "low": 109.5, "close": 115.0})
        # Closes above midpoint (112.5) → fails penetration
        rows.append({"open": 116.0, "high": 117.0, "low": 113.0, "close": 113.5})
        df = _df(rows)
        signals = dark_cloud_cover(df, trend_lookback=5)
        assert signals.iloc[-1] is None

    def test_no_dark_cloud_after_downtrend(self):
        rows = _downtrend_prefix()
        rows.append({"open": 180.0, "high": 182.0, "low": 179.5, "close": 181.0})
        rows.append({"open": 182.0, "high": 183.0, "low": 179.0, "close": 179.5})
        df = _df(rows)
        signals = dark_cloud_cover(df, trend_lookback=5)
        assert signals.iloc[-1] is None

    def test_no_dark_cloud_when_does_not_open_above_prior_close(self):
        rows = _uptrend_prefix()
        rows.append({"open": 110.0, "high": 116.0, "low": 109.5, "close": 115.0})
        # Opens at prior close, not above it
        rows.append({"open": 115.0, "high": 116.0, "low": 111.0, "close": 112.0})
        df = _df(rows)
        signals = dark_cloud_cover(df, trend_lookback=5)
        assert signals.iloc[-1] is None


# ---------------------------------------------------------------------------
# Piercing Pattern
# ---------------------------------------------------------------------------

class TestPiercingPattern:
    def test_detects_piercing_after_downtrend(self):
        rows = _downtrend_prefix()
        # Long black candle
        rows.append({"open": 175.0, "high": 175.5, "low": 164.5, "close": 165.0})
        # Opens below prior close, closes above midpoint (170.0) but inside body
        rows.append({"open": 164.0, "high": 171.5, "low": 163.5, "close": 171.0})
        df = _df(rows)
        signals = piercing_pattern(df, trend_lookback=5)
        assert signals.iloc[-1] == "bullish"

    def test_no_piercing_when_close_below_midpoint(self):
        rows = _downtrend_prefix()
        rows.append({"open": 175.0, "high": 175.5, "low": 164.5, "close": 165.0})
        # Closes below midpoint (170.0) → insufficient penetration
        rows.append({"open": 164.0, "high": 169.0, "low": 163.5, "close": 168.5})
        df = _df(rows)
        signals = piercing_pattern(df, trend_lookback=5)
        assert signals.iloc[-1] is None

    def test_no_piercing_after_uptrend(self):
        rows = _uptrend_prefix()
        rows.append({"open": 115.0, "high": 115.5, "low": 104.5, "close": 105.0})
        rows.append({"open": 104.0, "high": 111.5, "low": 103.5, "close": 111.0})
        df = _df(rows)
        signals = piercing_pattern(df, trend_lookback=5)
        assert signals.iloc[-1] is None

    def test_no_piercing_when_does_not_open_below_prior_close(self):
        rows = _downtrend_prefix()
        rows.append({"open": 175.0, "high": 175.5, "low": 164.5, "close": 165.0})
        # Opens at prior close, not below it
        rows.append({"open": 165.0, "high": 172.0, "low": 163.5, "close": 171.0})
        df = _df(rows)
        signals = piercing_pattern(df, trend_lookback=5)
        assert signals.iloc[-1] is None
