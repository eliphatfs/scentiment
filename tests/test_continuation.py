"""Tests for continuation candlestick patterns (Nison chapters 13–14)."""

import pandas as pd
import pytest

from patterns.continuation import (
    falling_three_methods,
    rising_three_methods,
    separating_lines,
    three_white_soldiers,
    window_down,
    window_up,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
    return df.astype(float)


def _uptrend(n: int = 6, start: float = 100.0) -> list[dict]:
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


def _downtrend(n: int = 6, start: float = 200.0) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

class TestWindowUp:
    def test_detects_gap_up(self):
        rows = [
            {"open": 100.0, "high": 105.0, "low": 99.0,  "close": 104.0},
            {"open": 107.0, "high": 110.0, "low": 106.0, "close": 109.0},  # low > prior high
        ]
        df = _df(rows)
        assert window_up(df).iloc[-1] == "bullish"

    def test_no_gap_when_low_touches_prior_high(self):
        rows = [
            {"open": 100.0, "high": 105.0, "low": 99.0,  "close": 104.0},
            {"open": 106.0, "high": 108.0, "low": 105.0, "close": 107.0},  # low == prior high
        ]
        df = _df(rows)
        assert window_up(df).iloc[-1] is None

    def test_no_gap_when_bars_overlap(self):
        rows = [
            {"open": 100.0, "high": 105.0, "low": 99.0,  "close": 104.0},
            {"open": 103.0, "high": 107.0, "low": 102.0, "close": 106.0},
        ]
        df = _df(rows)
        assert window_up(df).iloc[-1] is None

    def test_trend_filter_respected(self):
        # Gap up during a downtrend: trend_lookback=5 requires a prior uptrend
        # (pivot HH+HL), which does not exist in a downtrend prefix → None.
        rows = _downtrend()
        last_high = max(r["high"] for r in rows[-1:])
        rows.append({"open": last_high + 2, "high": last_high + 5,
                     "low": last_high + 1, "close": last_high + 4})  # gap up
        df = _df(rows)
        assert window_up(df, trend_lookback=5).iloc[-1] is None

    def test_first_bar_is_never_a_gap(self):
        df = _df([{"open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0}])
        assert window_up(df).iloc[0] is None


class TestWindowDown:
    def test_detects_gap_down(self):
        rows = [
            {"open": 110.0, "high": 112.0, "low": 105.0, "close": 106.0},
            {"open": 102.0, "high": 104.0, "low": 101.0, "close": 102.0},  # high < prior low
        ]
        df = _df(rows)
        assert window_down(df).iloc[-1] == "bearish"

    def test_no_gap_when_high_touches_prior_low(self):
        rows = [
            {"open": 110.0, "high": 112.0, "low": 105.0, "close": 106.0},
            {"open": 104.0, "high": 105.0, "low": 103.0, "close": 104.0},  # high == prior low
        ]
        df = _df(rows)
        assert window_down(df).iloc[-1] is None

    def test_trend_filter_respected(self):
        # Gap down during an uptrend: trend_lookback=5 requires a prior downtrend
        # (pivot LH+LL), which does not exist in an uptrend prefix → None.
        rows = _uptrend()
        last_low = min(r["low"] for r in rows[-1:])
        rows.append({"open": last_low - 2, "high": last_low - 1,
                     "low": last_low - 5, "close": last_low - 4})  # gap down
        df = _df(rows)
        assert window_down(df, trend_lookback=5).iloc[-1] is None


# ---------------------------------------------------------------------------
# Rising Three Methods
# ---------------------------------------------------------------------------

class TestRisingThreeMethods:
    def _pattern(self, base: float = 110.0) -> list[dict]:
        """Classic five-candle rising three methods.

        Candle layout (base = close of first white):
          1st: long white (open=base-8, close=base, high=base+0.5, low=base-8.5)
          2nd–4th: small blacks inside the 1st range
          5th: long white closing above 1st close
        """
        first = {"open": base - 8, "high": base + 0.5, "low": base - 8.5, "close": base}
        # Three small blacks within [base-8.5, base+0.5]
        # body=0.2, range=3.8 → ratio 0.053 < 0.3 threshold ✓
        mid = [
            {"open": base - 1.0, "high": base - 0.3, "low": base - 4.1, "close": base - 1.2},
            {"open": base - 2.0, "high": base - 1.3, "low": base - 5.1, "close": base - 2.2},
            {"open": base - 3.0, "high": base - 2.3, "low": base - 6.1, "close": base - 3.2},
        ]
        fifth = {"open": base - 4.0, "high": base + 2.5, "low": base - 4.5, "close": base + 2.0}
        return [first] + mid + [fifth]

    def test_detects_rising_three_methods(self):
        df = _df(_uptrend() + self._pattern())
        assert rising_three_methods(df).iloc[-1] == "bullish"

    def test_signal_on_fifth_candle_only(self):
        df = _df(_uptrend() + self._pattern())
        result = rising_three_methods(df)
        # The five pattern candles are at positions -5..-1; only -1 should fire.
        for i in range(-5, -1):
            assert result.iloc[i] is None

    def test_fails_when_middle_exceeds_first_range(self):
        rows = _uptrend()
        base = 110.0
        first = {"open": base - 8, "high": base + 0.5, "low": base - 8.5, "close": base}
        # One middle candle pokes above the first high
        mid = [
            {"open": base - 1.0, "high": base + 2.0, "low": base - 2.5, "close": base - 2.0},
            {"open": base - 2.5, "high": base - 2.0, "low": base - 4.0, "close": base - 3.5},
            {"open": base - 4.0, "high": base - 3.5, "low": base - 5.5, "close": base - 5.0},
        ]
        fifth = {"open": base - 4.0, "high": base + 2.5, "low": base - 4.5, "close": base + 2.0}
        df = _df(rows + [first] + mid + [fifth])
        assert rising_three_methods(df).iloc[-1] is None

    def test_fails_when_fifth_does_not_close_above_first(self):
        rows = _uptrend()
        base = 110.0
        first = {"open": base - 8, "high": base + 0.5, "low": base - 8.5, "close": base}
        mid = [
            {"open": base - 1.0, "high": base - 0.5, "low": base - 2.5, "close": base - 2.0},
            {"open": base - 2.5, "high": base - 2.0, "low": base - 4.0, "close": base - 3.5},
            {"open": base - 4.0, "high": base - 3.5, "low": base - 5.5, "close": base - 5.0},
        ]
        # Fifth closes at base - 1 (below first close = base)
        fifth = {"open": base - 4.0, "high": base + 0.4, "low": base - 4.5, "close": base - 1.0}
        df = _df(rows + [first] + mid + [fifth])
        assert rising_three_methods(df).iloc[-1] is None

    def test_fails_in_downtrend(self):
        df = _df(_downtrend() + self._pattern())
        assert rising_three_methods(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Falling Three Methods
# ---------------------------------------------------------------------------

class TestFallingThreeMethods:
    def _pattern(self, base: float = 180.0) -> list[dict]:
        """base = close (= open - 8) of first black candle."""
        first = {"open": base + 8, "high": base + 8.5, "low": base - 0.5, "close": base}
        # Three small whites inside [base-0.5, base+8.5]
        # body=0.2, range=3.8 → ratio 0.053 < 0.3 threshold ✓
        mid = [
            {"open": base + 1.0, "high": base + 4.8, "low": base + 0.7, "close": base + 1.2},
            {"open": base + 2.0, "high": base + 5.8, "low": base + 1.7, "close": base + 2.2},
            {"open": base + 3.0, "high": base + 6.8, "low": base + 2.7, "close": base + 3.2},
        ]
        fifth = {"open": base + 4.0, "high": base + 4.5, "low": base - 2.5, "close": base - 2.0}
        return [first] + mid + [fifth]

    def test_detects_falling_three_methods(self):
        df = _df(_downtrend() + self._pattern())
        assert falling_three_methods(df).iloc[-1] == "bearish"

    def test_fails_when_fifth_does_not_close_below_first(self):
        rows = _downtrend()
        base = 180.0
        first = {"open": base + 8, "high": base + 8.5, "low": base - 0.5, "close": base}
        mid = [
            {"open": base + 1.0, "high": base + 2.5, "low": base + 0.5, "close": base + 2.0},
            {"open": base + 2.5, "high": base + 4.0, "low": base + 2.0, "close": base + 3.5},
            {"open": base + 4.0, "high": base + 5.5, "low": base + 3.5, "close": base + 5.0},
        ]
        # Fifth closes at base+1 (above first close = base)
        fifth = {"open": base + 4.0, "high": base + 4.5, "low": base + 0.5, "close": base + 1.0}
        df = _df(rows + [first] + mid + [fifth])
        assert falling_three_methods(df).iloc[-1] is None

    def test_fails_in_uptrend(self):
        df = _df(_uptrend() + self._pattern())
        assert falling_three_methods(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Three White Soldiers
# ---------------------------------------------------------------------------

class TestThreeWhiteSoldiers:
    def _soldiers(self, base: float = 170.0) -> list[dict]:
        """Three long whites, each opening within the prior body, small upper shadow."""
        c1 = {"open": base,       "high": base + 7.5, "low": base - 0.5, "close": base + 7.5}
        c2 = {"open": base + 5.0, "high": base + 13.0,"low": base + 4.5, "close": base + 12.5}
        c3 = {"open": base + 10.0,"high": base + 18.0,"low": base + 9.5, "close": base + 17.5}
        return [c1, c2, c3]

    def test_detects_three_white_soldiers(self):
        df = _df(_downtrend() + self._soldiers())
        assert three_white_soldiers(df).iloc[-1] == "bullish"

    def test_fails_when_one_candle_not_white(self):
        rows = _downtrend()
        base = 170.0
        c1 = {"open": base,       "high": base + 7.5, "low": base - 0.5, "close": base + 7.5}
        # Second candle is black
        c2 = {"open": base + 8.0, "high": base + 8.5, "low": base + 1.0, "close": base + 2.0}
        c3 = {"open": base + 4.0, "high": base + 12.0,"low": base + 3.5, "close": base + 11.5}
        df = _df(rows + [c1, c2, c3])
        assert three_white_soldiers(df).iloc[-1] is None

    def test_fails_when_upper_shadow_too_large(self):
        rows = _downtrend()
        base = 170.0
        # Third candle has a large upper shadow (close much below high)
        c1 = {"open": base,       "high": base + 7.5, "low": base - 0.5, "close": base + 7.5}
        c2 = {"open": base + 5.0, "high": base + 13.0,"low": base + 4.5, "close": base + 12.5}
        c3 = {"open": base + 10.0,"high": base + 20.0,"low": base + 9.5, "close": base + 11.0}
        df = _df(rows + [c1, c2, c3])
        assert three_white_soldiers(df).iloc[-1] is None

    def test_fails_in_uptrend(self):
        # Three white soldiers require a prior downtrend
        df = _df(_uptrend() + self._soldiers(base=115.0))
        assert three_white_soldiers(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Separating Lines
# ---------------------------------------------------------------------------

class TestSeparatingLines:
    def test_bullish_separating_line(self):
        rows = _uptrend()
        # Black candle (counter-trend), then white candle opening at same price
        rows.append({"open": 112.0, "high": 113.0, "low": 108.0, "close": 109.0})
        rows.append({"open": 112.0, "high": 118.0, "low": 111.5, "close": 117.0})
        df = _df(rows)
        assert separating_lines(df).iloc[-1] == "bullish"

    def test_bearish_separating_line(self):
        rows = _downtrend()
        # White candle (counter-trend), then black candle opening at same price
        rows.append({"open": 188.0, "high": 192.0, "low": 187.5, "close": 191.0})
        rows.append({"open": 188.0, "high": 188.5, "low": 182.0, "close": 183.0})
        df = _df(rows)
        assert separating_lines(df).iloc[-1] == "bearish"

    def test_fails_when_opens_differ(self):
        rows = _uptrend()
        rows.append({"open": 112.0, "high": 113.0, "low": 108.0, "close": 109.0})
        # Second candle opens at a different price
        rows.append({"open": 114.0, "high": 118.0, "low": 113.5, "close": 117.0})
        df = _df(rows)
        assert separating_lines(df).iloc[-1] is None

    def test_fails_when_colors_match_trend(self):
        rows = _uptrend()
        # Both candles white in uptrend — not a separating line (first should be counter-trend)
        rows.append({"open": 110.0, "high": 113.0, "low": 109.5, "close": 112.0})
        rows.append({"open": 110.0, "high": 118.0, "low": 109.5, "close": 117.0})
        df = _df(rows)
        assert separating_lines(df).iloc[-1] is None

    def test_fails_in_wrong_trend(self):
        rows = _downtrend()
        # Bullish separating shape but market is in downtrend
        rows.append({"open": 188.0, "high": 189.0, "low": 184.0, "close": 185.0})
        rows.append({"open": 188.0, "high": 194.0, "low": 187.5, "close": 193.0})
        df = _df(rows)
        assert separating_lines(df).iloc[-1] is None
