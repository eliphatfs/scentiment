"""Tests for patterns/doji.py."""

import pandas as pd
import pytest

from patterns.doji import (
    doji_at_bottom,
    doji_at_top,
    gravestone_doji,
    long_legged_doji,
    rickshaw_man,
    tri_star,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows):
    """Build a DataFrame from a list of (open, high, low, close) tuples."""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def _uptrend(n=6, base=100.0, step=1.0):
    """n rising candles."""
    rows = []
    for i in range(n):
        o = base + i * step
        c = o + 0.5
        rows.append((o, c + 0.1, o - 0.1, c))
    return rows


def _downtrend(n=6, base=110.0, step=1.0):
    """n falling candles."""
    rows = []
    for i in range(n):
        o = base - i * step
        c = o - 0.5
        rows.append((o, o + 0.1, c - 0.1, c))
    return rows


def _doji_candle(mid=100.0, upper=2.0, lower=2.0):
    """Open == close at mid, with specified shadow lengths."""
    return (mid, mid + upper, mid - lower, mid)


def _gravestone(mid=100.0, upper=3.0):
    """Gravestone: open == close at the low, long upper shadow."""
    return (mid, mid + upper, mid, mid)


# ---------------------------------------------------------------------------
# doji_at_top
# ---------------------------------------------------------------------------

class TestDojiAtTop:
    def test_bearish_in_uptrend(self):
        rows = _uptrend(6) + [_doji_candle(106.0)]
        df = _make_df(rows)
        sig = doji_at_top(df, trend_lookback=5)
        assert sig.iloc[-1] == "bearish"

    def test_none_in_downtrend(self):
        rows = _downtrend(6) + [_doji_candle(104.0)]
        df = _make_df(rows)
        sig = doji_at_top(df, trend_lookback=5)
        assert sig.iloc[-1] is None

    def test_none_for_non_doji(self):
        rows = _uptrend(6) + [(106.0, 108.0, 105.0, 107.5)]  # large body
        df = _make_df(rows)
        sig = doji_at_top(df, trend_lookback=5)
        assert sig.iloc[-1] is None


# ---------------------------------------------------------------------------
# doji_at_bottom
# ---------------------------------------------------------------------------

class TestDojiAtBottom:
    def test_bullish_in_downtrend(self):
        rows = _downtrend(6) + [_doji_candle(104.0)]
        df = _make_df(rows)
        sig = doji_at_bottom(df, trend_lookback=5)
        assert sig.iloc[-1] == "bullish"

    def test_none_in_uptrend(self):
        rows = _uptrend(6) + [_doji_candle(106.0)]
        df = _make_df(rows)
        sig = doji_at_bottom(df, trend_lookback=5)
        assert sig.iloc[-1] is None

    def test_none_for_non_doji(self):
        rows = _downtrend(6) + [(104.0, 106.0, 103.0, 105.5)]
        df = _make_df(rows)
        sig = doji_at_bottom(df, trend_lookback=5)
        assert sig.iloc[-1] is None


# ---------------------------------------------------------------------------
# long_legged_doji
# ---------------------------------------------------------------------------

class TestLongLeggedDoji:
    def _ll_doji(self, mid=100.0):
        # Equal shadows of 3.0 each → each is 50% of the 6.0 range; body = 0
        return _doji_candle(mid, upper=3.0, lower=3.0)

    def test_bearish_in_uptrend(self):
        rows = _uptrend(6) + [self._ll_doji(106.0)]
        df = _make_df(rows)
        sig = long_legged_doji(df, trend_lookback=5, shadow_ratio=0.3)
        assert sig.iloc[-1] == "bearish"

    def test_bullish_in_downtrend(self):
        rows = _downtrend(6) + [self._ll_doji(104.0)]
        df = _make_df(rows)
        sig = long_legged_doji(df, trend_lookback=5, shadow_ratio=0.3)
        assert sig.iloc[-1] == "bullish"

    def test_none_if_short_lower_shadow(self):
        # Very small lower shadow
        rows = _uptrend(6) + [(100.0, 106.0, 99.9, 100.0)]  # lower=0.1, upper=6.0
        df = _make_df(rows)
        sig = long_legged_doji(df, trend_lookback=5, shadow_ratio=0.3)
        assert sig.iloc[-1] is None

    def test_none_if_not_doji(self):
        rows = _uptrend(6) + [(100.0, 104.0, 97.0, 102.0)]  # body=2, range=7
        df = _make_df(rows)
        sig = long_legged_doji(df, trend_lookback=5, shadow_ratio=0.3)
        assert sig.iloc[-1] is None


# ---------------------------------------------------------------------------
# rickshaw_man
# ---------------------------------------------------------------------------

class TestRickshawMan:
    def _rickshaw(self, mid=100.0):
        # Body at exact midpoint (open==close==mid), shadows of 3.0 each
        return _doji_candle(mid, upper=3.0, lower=3.0)

    def test_bearish_in_uptrend(self):
        rows = _uptrend(6) + [self._rickshaw(106.0)]
        df = _make_df(rows)
        sig = rickshaw_man(df, trend_lookback=5, shadow_ratio=0.3, center_tolerance=0.25)
        assert sig.iloc[-1] == "bearish"

    def test_bullish_in_downtrend(self):
        rows = _downtrend(6) + [self._rickshaw(104.0)]
        df = _make_df(rows)
        sig = rickshaw_man(df, trend_lookback=5, shadow_ratio=0.3, center_tolerance=0.25)
        assert sig.iloc[-1] == "bullish"

    def test_none_if_body_not_centered(self):
        # Body near the top: open=close=103, high=103.1, low=97 → midpoint=100.05
        # body_mid=103, deviation = 2.95, range=6.1, center_tol*range = 0.25*6.1 = 1.525
        rows = _uptrend(6) + [(103.0, 103.1, 97.0, 103.0)]
        df = _make_df(rows)
        sig = rickshaw_man(df, trend_lookback=5, shadow_ratio=0.3, center_tolerance=0.25)
        assert sig.iloc[-1] is None

    def test_none_if_short_shadow(self):
        # Short lower shadow
        rows = _uptrend(6) + [(100.0, 106.0, 99.9, 100.0)]
        df = _make_df(rows)
        sig = rickshaw_man(df, trend_lookback=5, shadow_ratio=0.3, center_tolerance=0.25)
        assert sig.iloc[-1] is None


# ---------------------------------------------------------------------------
# gravestone_doji
# ---------------------------------------------------------------------------

class TestGravestoneDoji:
    def test_bearish_in_uptrend(self):
        rows = _uptrend(6) + [_gravestone(106.0, upper=3.0)]
        df = _make_df(rows)
        sig = gravestone_doji(df, trend_lookback=5)
        assert sig.iloc[-1] == "bearish"

    def test_none_in_downtrend(self):
        rows = _downtrend(6) + [_gravestone(104.0, upper=3.0)]
        df = _make_df(rows)
        sig = gravestone_doji(df, trend_lookback=5)
        assert sig.iloc[-1] is None

    def test_none_if_lower_shadow_present(self):
        # Small but non-trivial lower shadow (0.5 out of 4.0 range = 12.5%)
        rows = _uptrend(6) + [(99.5, 103.0, 99.0, 99.5)]
        df = _make_df(rows)
        sig = gravestone_doji(df, trend_lookback=5, lower_shadow_ratio=0.05)
        # lower_shadow = 99.5-99.0 = 0.5; range = 4.0; ratio = 0.125 > 0.05 → rejected
        assert sig.iloc[-1] is None

    def test_none_if_not_doji(self):
        # Large body
        rows = _uptrend(6) + [(100.0, 106.0, 100.0, 103.5)]
        df = _make_df(rows)
        sig = gravestone_doji(df, trend_lookback=5)
        assert sig.iloc[-1] is None

    def test_none_if_short_upper_shadow(self):
        # Very short upper shadow: doji but upper=0.1, range=0.1 → upper/range=1.0 ... wait
        # Let's use: open=close=100, high=100.2, low=100 → upper=0.2, range=0.2, ratio=1.0 >= 0.5 ✓
        # Actually this passes. Make it fail with a tiny upper vs range.
        # open=close=100, high=100.05, low=98.0 → upper=0.05, lower=0, range=2.05, upper/range=0.024 < 0.5
        rows = _uptrend(6) + [(100.0, 100.05, 98.0, 100.0)]
        df = _make_df(rows)
        sig = gravestone_doji(df, trend_lookback=5, upper_shadow_ratio=0.5)
        assert sig.iloc[-1] is None


# ---------------------------------------------------------------------------
# tri_star
# ---------------------------------------------------------------------------

class TestTriStar:
    def _make_tri_star(self, trend_rows, mid_prices):
        """trend_rows + three doji candles."""
        rows = list(trend_rows)
        for m in mid_prices:
            rows.append(_doji_candle(m, upper=0.5, lower=0.5))
        return _make_df(rows)

    def test_bullish_after_downtrend(self):
        trend = _downtrend(6, base=110.0)
        # doji mids go slightly down to stay in downtrend context at t-2
        df = self._make_tri_star(trend, [103.0, 102.8, 102.6])
        sig = tri_star(df, trend_lookback=5)
        assert sig.iloc[-1] == "bullish"

    def test_bearish_after_uptrend(self):
        trend = _uptrend(6, base=100.0)
        df = self._make_tri_star(trend, [106.0, 106.2, 106.4])
        sig = tri_star(df, trend_lookback=5)
        assert sig.iloc[-1] == "bearish"

    def test_none_if_middle_not_doji(self):
        trend = _uptrend(6, base=100.0)
        rows = list(trend)
        rows.append(_doji_candle(106.0, upper=0.5, lower=0.5))
        rows.append((106.0, 108.0, 104.0, 107.5))  # large body, not doji
        rows.append(_doji_candle(106.0, upper=0.5, lower=0.5))
        df = _make_df(rows)
        sig = tri_star(df, trend_lookback=5)
        assert sig.iloc[-1] is None

    def test_none_if_first_not_doji(self):
        trend = _uptrend(6, base=100.0)
        rows = list(trend)
        rows.append((106.0, 108.0, 104.0, 107.5))  # not doji
        rows.append(_doji_candle(106.0, upper=0.5, lower=0.5))
        rows.append(_doji_candle(106.0, upper=0.5, lower=0.5))
        df = _make_df(rows)
        sig = tri_star(df, trend_lookback=5)
        assert sig.iloc[-1] is None

    def test_signal_on_third_candle_only(self):
        trend = _downtrend(6, base=110.0)
        df = self._make_tri_star(trend, [103.0, 102.8, 102.6])
        sig = tri_star(df, trend_lookback=5)
        # The two intermediate doji candles should not themselves be the signal bar
        assert sig.iloc[-3] is None  # first doji (t-2)
        assert sig.iloc[-2] is None  # second doji (t-1)
        assert sig.iloc[-1] == "bullish"
