"""Tests for star candlestick patterns.

Each test builds a minimal synthetic OHLCV DataFrame.  Trend prefixes
establish the required market context; the final 1–3 bars form the pattern.
"""

import pandas as pd
import pytest

from patterns.stars import (
    evening_doji_star,
    evening_star,
    inverted_hammer,
    morning_doji_star,
    morning_star,
    shooting_star,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
    return df.astype(float)


def _downtrend(n: int = 5, start: float = 200.0, step: float = 2.0) -> list[dict]:
    rows, price = [], start
    for _ in range(n):
        rows.append({"open": price, "high": price + 0.5, "low": price - 0.5, "close": price - step})
        price -= step
    return rows


def _uptrend(n: int = 5, start: float = 100.0, step: float = 2.0) -> list[dict]:
    rows, price = [], start
    for _ in range(n):
        rows.append({"open": price, "high": price + 0.5, "low": price - 0.5, "close": price + step})
        price += step
    return rows


# ---------------------------------------------------------------------------
# Inverted Hammer
# ---------------------------------------------------------------------------

class TestInvertedHammer:
    def _candle(self, base: float = 100.0) -> dict:
        # Body at bottom: open=base, close=base+1 (white, small)
        # Long upper shadow: high=base+5
        # No lower shadow: low=base
        return {"open": base, "high": base + 5.0, "low": base, "close": base + 1.0}

    def test_detects_after_downtrend(self):
        df = _df(_downtrend() + [self._candle()])
        assert inverted_hammer(df).iloc[-1] == "bullish"

    def test_not_after_uptrend(self):
        # Candle at base=115 so close(116) > close[0](102) → not a downtrend
        df = _df(_uptrend() + [self._candle(base=115.0)])
        assert inverted_hammer(df).iloc[-1] is None

    def test_fails_when_lower_shadow_too_large(self):
        rows = _downtrend()
        # lower shadow = 3, range = 8, 3/8 = 0.375 > 0.1 → fails
        rows.append({"open": 100.0, "high": 106.0, "low": 97.0, "close": 101.0})
        df = _df(rows)
        assert inverted_hammer(df).iloc[-1] is None

    def test_fails_when_upper_shadow_too_short(self):
        rows = _downtrend()
        # upper = 0.5, body = 2 → 0.5 < 2×2 → fails
        rows.append({"open": 100.0, "high": 102.5, "low": 99.8, "close": 102.0})
        df = _df(rows)
        assert inverted_hammer(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Shooting Star
# ---------------------------------------------------------------------------

class TestShootingStar:
    def _candle(self, base: float = 115.0) -> dict:
        return {"open": base, "high": base + 5.0, "low": base, "close": base + 1.0}

    def test_detects_after_uptrend(self):
        df = _df(_uptrend() + [self._candle()])
        assert shooting_star(df).iloc[-1] == "bearish"

    def test_not_after_downtrend(self):
        # base=100; close=101 < close[0]=198 in downtrend → not an uptrend
        df = _df(_downtrend() + [self._candle(base=100.0)])
        assert shooting_star(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Morning Star
# ---------------------------------------------------------------------------

class TestMorningStar:
    def _pattern(self, base: float = 170.0) -> list[dict]:
        """Three-candle morning star.

        1st: long black body  (open=base+10, close=base, range covers body)
        2nd: small body that gaps below 1st body (open & close < base)
              body=0.5, range=4.0 → ratio 0.125 < 0.3 threshold ✓
        3rd: white body that closes above midpoint of 1st (midpoint = base+5)
        """
        first  = {"open": base + 10, "high": base + 11, "low": base - 1,  "close": base}
        star   = {"open": base - 3,  "high": base - 1,  "low": base - 5,  "close": base - 3.5}
        third  = {"open": base - 1,  "high": base + 8,  "low": base - 2,  "close": base + 7}
        return [first, star, third]

    def test_detects_morning_star(self):
        df = _df(_downtrend() + self._pattern())
        assert morning_star(df).iloc[-1] == "bullish"

    def test_signal_on_third_candle_only(self):
        df = _df(_downtrend() + self._pattern())
        result = morning_star(df)
        # Only the last bar should be 'bullish'
        assert result.iloc[-2] is None
        assert result.iloc[-3] is None

    def test_fails_when_no_gap(self):
        rows = _downtrend()
        base = 170.0
        first = {"open": base + 10, "high": base + 11, "low": base - 1, "close": base}
        # Star does NOT gap below first body (star body_top = base+2 > first body_bottom = base)
        star  = {"open": base + 1,  "high": base + 3,  "low": base - 1, "close": base + 2}
        third = {"open": base - 1,  "high": base + 8,  "low": base - 2, "close": base + 7}
        df = _df(rows + [first, star, third])
        assert morning_star(df).iloc[-1] is None

    def test_no_gap_allowed_when_require_gap_false(self):
        rows = _downtrend()
        base = 170.0
        first = {"open": base + 10, "high": base + 11, "low": base - 1, "close": base}
        star  = {"open": base + 1,  "high": base + 3,  "low": base - 1, "close": base + 2}
        third = {"open": base - 1,  "high": base + 8,  "low": base - 2, "close": base + 7}
        df = _df(rows + [first, star, third])
        assert morning_star(df, require_gap=False).iloc[-1] == "bullish"

    def test_fails_when_third_does_not_penetrate(self):
        rows = _downtrend()
        base = 170.0
        first = {"open": base + 10, "high": base + 11, "low": base - 1,  "close": base}
        star  = {"open": base - 3,  "high": base - 2,  "low": base - 5,  "close": base - 2}
        # Third closes at base+4, but midpoint is base+5 → fails penetration
        third = {"open": base - 1,  "high": base + 5,  "low": base - 2,  "close": base + 4}
        df = _df(rows + [first, star, third])
        assert morning_star(df).iloc[-1] is None

    def test_fails_after_uptrend(self):
        df = _df(_uptrend() + self._pattern())
        assert morning_star(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Evening Star
# ---------------------------------------------------------------------------

class TestEveningStar:
    def _pattern(self, base: float = 100.0) -> list[dict]:
        """Three-candle evening star.

        1st: long white body  (open=base, close=base+10)
        2nd: small body gaps above 1st body (open & close > base+10)
              body=0.5, range=4.0 → ratio 0.125 < 0.3 threshold ✓
        3rd: black body closes below midpoint of 1st (midpoint = base+5)
        """
        first  = {"open": base,      "high": base + 11, "low": base - 1,  "close": base + 10}
        star   = {"open": base + 12, "high": base + 14, "low": base + 11, "close": base + 12.5}
        third  = {"open": base + 11, "high": base + 12, "low": base + 2,  "close": base + 3}
        return [first, star, third]

    def test_detects_evening_star(self):
        df = _df(_uptrend() + self._pattern())
        assert evening_star(df).iloc[-1] == "bearish"

    def test_fails_when_no_gap(self):
        rows = _uptrend()
        base = 100.0
        first = {"open": base,      "high": base + 11, "low": base - 1,  "close": base + 10}
        # Star body_bottom = base+9 < first body_top = base+10 → no gap
        star  = {"open": base + 9,  "high": base + 12, "low": base + 8,  "close": base + 11}
        third = {"open": base + 11, "high": base + 12, "low": base + 2,  "close": base + 3}
        df = _df(rows + [first, star, third])
        assert evening_star(df).iloc[-1] is None

    def test_fails_when_third_does_not_penetrate(self):
        rows = _uptrend()
        base = 100.0
        first  = {"open": base,      "high": base + 11, "low": base - 1,  "close": base + 10}
        star   = {"open": base + 12, "high": base + 14, "low": base + 11, "close": base + 13}
        # Closes at base+6, midpoint is base+5 → fails (close must be BELOW midpoint)
        third  = {"open": base + 11, "high": base + 12, "low": base + 5,  "close": base + 6}
        df = _df(rows + [first, star, third])
        assert evening_star(df).iloc[-1] is None

    def test_fails_after_downtrend(self):
        df = _df(_downtrend() + self._pattern())
        assert evening_star(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Morning Doji Star
# ---------------------------------------------------------------------------

class TestMorningDojiStar:
    def _pattern(self, base: float = 170.0) -> list[dict]:
        first = {"open": base + 10, "high": base + 11, "low": base - 1,  "close": base}
        # Doji: open == close, gaps below first body
        doji  = {"open": base - 3,  "high": base - 2,  "low": base - 4,  "close": base - 3}
        third = {"open": base - 2,  "high": base + 8,  "low": base - 3,  "close": base + 7}
        return [first, doji, third]

    def test_detects_morning_doji_star(self):
        df = _df(_downtrend() + self._pattern())
        assert morning_doji_star(df).iloc[-1] == "bullish"

    def test_fails_when_star_not_doji(self):
        rows = _downtrend()
        base = 170.0
        first = {"open": base + 10, "high": base + 11, "low": base - 1,  "close": base}
        # Large body → not a doji
        star  = {"open": base - 3,  "high": base - 2,  "low": base - 9,  "close": base - 8}
        third = {"open": base - 2,  "high": base + 8,  "low": base - 3,  "close": base + 7}
        df = _df(rows + [first, star, third])
        assert morning_doji_star(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Evening Doji Star
# ---------------------------------------------------------------------------

class TestEveningDojiStar:
    def _pattern(self, base: float = 100.0) -> list[dict]:
        first = {"open": base,      "high": base + 11, "low": base - 1,  "close": base + 10}
        # Doji: open == close, gaps above first body
        doji  = {"open": base + 12, "high": base + 13, "low": base + 11, "close": base + 12}
        third = {"open": base + 11, "high": base + 12, "low": base + 2,  "close": base + 3}
        return [first, doji, third]

    def test_detects_evening_doji_star(self):
        df = _df(_uptrend() + self._pattern())
        assert evening_doji_star(df).iloc[-1] == "bearish"

    def test_fails_when_star_not_doji(self):
        rows = _uptrend()
        base = 100.0
        first = {"open": base,      "high": base + 11, "low": base - 1,  "close": base + 10}
        # Large body → not a doji
        star  = {"open": base + 11, "high": base + 18, "low": base + 10, "close": base + 17}
        third = {"open": base + 11, "high": base + 12, "low": base + 2,  "close": base + 3}
        df = _df(rows + [first, star, third])
        assert evening_doji_star(df).iloc[-1] is None
