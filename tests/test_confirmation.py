"""Tests for confirmation-delayed pattern signals."""

import pandas as pd
import pytest

from patterns.confirmation import (
    confirmed_signal,
    confirmed_hanging_man,
    confirmed_shooting_star,
    confirmed_inverted_hammer,
    confirmed_doji_at_top,
    confirmed_doji_at_bottom,
)
from patterns._candle import signal_series


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
    return df.astype(float)


def _downtrend_prefix(n: int = 6, start: float = 200.0) -> list[dict]:
    rows, price = [], start
    for i in range(n):
        if i % 2 == 0:
            o, c = price, price - 5
            h, l = o + 0.3, c - 0.3
        else:
            o, c = price, price + 2
            h, l = c + 0.1, o - 0.1
        rows.append({"open": o, "high": h, "low": l, "close": c})
        price = c
    return rows


def _uptrend_prefix(n: int = 6, start: float = 100.0) -> list[dict]:
    rows, price = [], start
    for i in range(n):
        if i % 2 == 0:
            o, c = price, price + 5
            h, l = c + 0.3, o - 0.3
        else:
            o, c = price, price - 2
            h, l = o + 0.1, c - 0.1
        rows.append({"open": o, "high": h, "low": l, "close": c})
        price = c
    return rows


# ---------------------------------------------------------------------------
# Core confirmed_signal tests
# ---------------------------------------------------------------------------

class TestConfirmedSignal:
    def test_close_below_body_confirms(self):
        """Signal fires on confirmation bar when next close < body bottom."""
        rows = [
            {"open": 100, "high": 102, "low": 95, "close": 101},  # pattern bar
            {"open": 100, "high": 100, "low": 98, "close": 99},   # confirms (99 < 100)
        ]
        df = _df(rows)
        raw = signal_series(df.index)
        raw.iloc[0] = "bearish"

        result = confirmed_signal(df, raw, "close_below_body", max_wait=1)
        assert result.iloc[0] is None  # pattern bar: no signal
        assert result.iloc[1] == "bearish"  # confirmation bar: signal

    def test_close_below_body_not_confirmed(self):
        """Signal dropped when next close stays above body bottom."""
        rows = [
            {"open": 100, "high": 102, "low": 95, "close": 101},  # pattern bar
            {"open": 101, "high": 103, "low": 100.5, "close": 102},  # no confirm
        ]
        df = _df(rows)
        raw = signal_series(df.index)
        raw.iloc[0] = "bearish"

        result = confirmed_signal(df, raw, "close_below_body", max_wait=1)
        assert result.iloc[0] is None
        assert result.iloc[1] is None

    def test_close_above_body_confirms(self):
        rows = [
            {"open": 100, "high": 105, "low": 99, "close": 100.5},  # pattern
            {"open": 101, "high": 103, "low": 100, "close": 102},    # confirms
        ]
        df = _df(rows)
        raw = signal_series(df.index)
        raw.iloc[0] = "bullish"

        result = confirmed_signal(df, raw, "close_above_body", max_wait=1)
        assert result.iloc[0] is None
        assert result.iloc[1] == "bullish"

    def test_bearish_candle_confirms(self):
        rows = [
            {"open": 100, "high": 101, "low": 99.9, "close": 100},  # doji
            {"open": 100, "high": 100.5, "low": 98, "close": 98.5},  # bearish
        ]
        df = _df(rows)
        raw = signal_series(df.index)
        raw.iloc[0] = "bearish"

        result = confirmed_signal(df, raw, "bearish_candle", max_wait=1)
        assert result.iloc[1] == "bearish"

    def test_bullish_candle_confirms(self):
        rows = [
            {"open": 100, "high": 101, "low": 99.9, "close": 100},
            {"open": 100, "high": 102, "low": 99.5, "close": 101.5},  # bullish
        ]
        df = _df(rows)
        raw = signal_series(df.index)
        raw.iloc[0] = "bullish"

        result = confirmed_signal(df, raw, "bullish_candle", max_wait=1)
        assert result.iloc[1] == "bullish"

    def test_max_wait_extends_window(self):
        """Confirmation can arrive on the 2nd bar after pattern."""
        rows = [
            {"open": 100, "high": 102, "low": 95, "close": 101},
            {"open": 101, "high": 103, "low": 100.5, "close": 102},  # no confirm
            {"open": 101, "high": 101, "low": 98, "close": 99},      # confirms
        ]
        df = _df(rows)
        raw = signal_series(df.index)
        raw.iloc[0] = "bearish"

        result = confirmed_signal(df, raw, "close_below_body", max_wait=2)
        assert result.iloc[0] is None
        assert result.iloc[1] is None
        assert result.iloc[2] == "bearish"

    def test_pattern_at_last_bar_no_confirm(self):
        """Pattern on last bar cannot be confirmed — no future data."""
        rows = [{"open": 100, "high": 102, "low": 95, "close": 101}]
        df = _df(rows)
        raw = signal_series(df.index)
        raw.iloc[0] = "bearish"

        result = confirmed_signal(df, raw, "close_below_body", max_wait=1)
        assert result.iloc[0] is None

    def test_opposite_candle_bearish(self):
        rows = [
            {"open": 100, "high": 101, "low": 99.9, "close": 100},
            {"open": 100, "high": 100.5, "low": 98, "close": 98.5},  # bearish
        ]
        df = _df(rows)
        raw = signal_series(df.index)
        raw.iloc[0] = "bearish"

        result = confirmed_signal(df, raw, "opposite_candle", max_wait=1)
        assert result.iloc[1] == "bearish"

    def test_opposite_candle_bullish(self):
        rows = [
            {"open": 100, "high": 101, "low": 99.9, "close": 100},
            {"open": 100, "high": 102, "low": 99.5, "close": 101.5},
        ]
        df = _df(rows)
        raw = signal_series(df.index)
        raw.iloc[0] = "bullish"

        result = confirmed_signal(df, raw, "opposite_candle", max_wait=1)
        assert result.iloc[1] == "bullish"


# ---------------------------------------------------------------------------
# Convenience wrapper tests
# ---------------------------------------------------------------------------

class TestConfirmedHangingMan:
    def test_confirmed_hanging_man(self):
        """Hanging man confirmed by next close below body."""
        rows = _uptrend_prefix()
        # Hanging man shape: small body at top, long lower shadow
        rows.append({"open": 121.0, "high": 121.5, "low": 117.0, "close": 121.5})
        # Confirmation: next bar closes below 121.0 (body bottom)
        rows.append({"open": 121.0, "high": 121.5, "low": 119.0, "close": 120.0})
        df = _df(rows)

        result = confirmed_hanging_man(df, trend_lookback=5)
        # Signal should be on the confirmation bar (last), not the pattern bar
        assert result.iloc[-2] is None
        assert result.iloc[-1] == "bearish"

    def test_hanging_man_not_confirmed(self):
        """Hanging man without confirmation is dropped."""
        rows = _uptrend_prefix()
        rows.append({"open": 121.0, "high": 121.5, "low": 117.0, "close": 121.5})
        # No confirmation: next bar closes above body
        rows.append({"open": 122.0, "high": 123.0, "low": 121.5, "close": 122.5})
        df = _df(rows)

        result = confirmed_hanging_man(df, trend_lookback=5)
        assert result.iloc[-2] is None
        assert result.iloc[-1] is None


class TestConfirmedShootingStar:
    def test_confirmed_shooting_star(self):
        rows = _uptrend_prefix()
        # Shooting star shape: small body at bottom, long upper shadow
        rows.append({"open": 121.0, "high": 127.0, "low": 120.5, "close": 121.5})
        # Confirm: close below body bottom (121.0)
        rows.append({"open": 121.0, "high": 121.5, "low": 119.0, "close": 120.0})
        df = _df(rows)

        result = confirmed_shooting_star(df, trend_lookback=5)
        assert result.iloc[-1] == "bearish"


class TestConfirmedInvertedHammer:
    def test_confirmed_inverted_hammer(self):
        rows = _downtrend_prefix()
        # Inverted hammer: small body at bottom, long upper shadow
        rows.append({"open": 181.0, "high": 187.0, "low": 180.5, "close": 181.5})
        # Confirm: close above body top (181.5)
        rows.append({"open": 182.0, "high": 184.0, "low": 181.5, "close": 183.0})
        df = _df(rows)

        result = confirmed_inverted_hammer(df, trend_lookback=5)
        assert result.iloc[-1] == "bullish"


class TestConfirmedDoji:
    def test_confirmed_doji_at_top(self):
        rows = _uptrend_prefix()
        # Doji: open ≈ close, range > 0
        rows.append({"open": 121.0, "high": 123.0, "low": 119.0, "close": 121.0})
        # Confirm: bearish candle
        rows.append({"open": 121.0, "high": 121.5, "low": 118.0, "close": 119.0})
        df = _df(rows)

        result = confirmed_doji_at_top(df, trend_lookback=5)
        assert result.iloc[-1] == "bearish"

    def test_confirmed_doji_at_bottom(self):
        rows = _downtrend_prefix()
        rows.append({"open": 181.0, "high": 183.0, "low": 179.0, "close": 181.0})
        # Confirm: bullish candle
        rows.append({"open": 181.0, "high": 184.0, "low": 180.5, "close": 183.0})
        df = _df(rows)

        result = confirmed_doji_at_bottom(df, trend_lookback=5)
        assert result.iloc[-1] == "bullish"
