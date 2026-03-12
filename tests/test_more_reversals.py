"""Tests for more reversal candlestick patterns (Nison chapters 10–12)."""

import pandas as pd
import pytest

from patterns.more_reversals import (
    belt_hold,
    counterattack_lines,
    dumpling_top,
    fry_pan_bottom,
    harami,
    harami_cross,
    three_black_crows,
    three_mountains,
    three_rivers,
    tower_bottom,
    tower_top,
    tweezers_bottom,
    tweezers_top,
    upside_gap_two_crows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
    return df.astype(float)


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


def _neutral(n: int, price: float = 100.0, body: float = 0.5) -> list[dict]:
    """Flat small-bodied candles — used as 'consolidation' filler."""
    return [{"open": price, "high": price + 1.0, "low": price - 1.0, "close": price + body}
            for _ in range(n)]


# ---------------------------------------------------------------------------
# Harami
# ---------------------------------------------------------------------------

class TestHarami:
    def test_bullish_harami_after_downtrend(self):
        rows = _downtrend()
        # Large black candle
        rows.append({"open": 190.0, "high": 191.0, "low": 179.0, "close": 180.0})
        # Small white candle fully inside prior body
        rows.append({"open": 182.0, "high": 185.0, "low": 181.0, "close": 184.0})
        df = _df(rows)
        assert harami(df).iloc[-1] == "bullish"

    def test_bearish_harami_after_uptrend(self):
        rows = _uptrend()  # ends near 109; keep pattern prices above prior pivot lows
        # Large white candle — low stays above prior pivot lows (~106) to maintain HL
        rows.append({"open": 109.0, "high": 121.0, "low": 108.5, "close": 120.0})
        # Small black candle inside prior body
        rows.append({"open": 117.0, "high": 118.0, "low": 113.0, "close": 114.0})
        df = _df(rows)
        assert harami(df).iloc[-1] == "bearish"

    def test_fails_when_body_not_contained(self):
        rows = _downtrend()
        rows.append({"open": 190.0, "high": 191.0, "low": 179.0, "close": 180.0})
        # Second body_top exceeds first body_top
        rows.append({"open": 182.0, "high": 194.0, "low": 181.0, "close": 193.0})
        df = _df(rows)
        assert harami(df).iloc[-1] is None

    def test_fails_wrong_trend(self):
        rows = _uptrend()
        rows.append({"open": 190.0, "high": 191.0, "low": 179.0, "close": 180.0})
        rows.append({"open": 182.0, "high": 185.0, "low": 181.0, "close": 184.0})
        df = _df(rows)
        # Context is uptrend but we have a black first candle → bullish harami not valid
        assert harami(df).iloc[-1] is None


class TestHaramiCross:
    def test_bullish_harami_cross(self):
        rows = _downtrend()
        rows.append({"open": 190.0, "high": 191.0, "low": 179.0, "close": 180.0})
        # Doji inside prior body (open == close)
        rows.append({"open": 184.0, "high": 186.0, "low": 182.0, "close": 184.0})
        df = _df(rows)
        assert harami_cross(df).iloc[-1] == "bullish"

    def test_fails_when_second_is_not_doji(self):
        rows = _downtrend()
        rows.append({"open": 190.0, "high": 191.0, "low": 179.0, "close": 180.0})
        # Small but NOT a doji (body = 2, range = 4 → 0.5 > 0.1 threshold)
        rows.append({"open": 182.0, "high": 186.0, "low": 182.0, "close": 184.0})
        df = _df(rows)
        assert harami_cross(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Tweezers
# ---------------------------------------------------------------------------

class TestTweezersTop:
    def test_detects_tweezers_top(self):
        rows = _uptrend()
        rows.append({"open": 108.0, "high": 113.0, "low": 107.0, "close": 112.0})
        rows.append({"open": 112.0, "high": 113.0, "low": 110.0, "close": 111.0})
        df = _df(rows)
        assert tweezers_top(df).iloc[-1] == "bearish"

    def test_fails_when_highs_differ(self):
        rows = _uptrend()
        rows.append({"open": 108.0, "high": 113.0, "low": 107.0, "close": 112.0})
        rows.append({"open": 112.0, "high": 115.0, "low": 110.0, "close": 111.0})
        df = _df(rows)
        assert tweezers_top(df).iloc[-1] is None

    def test_fails_after_downtrend(self):
        rows = _downtrend()
        rows.append({"open": 188.0, "high": 192.0, "low": 187.0, "close": 190.0})
        rows.append({"open": 190.0, "high": 192.0, "low": 189.0, "close": 191.0})
        df = _df(rows)
        assert tweezers_top(df).iloc[-1] is None


class TestTweezersBottom:
    def test_detects_tweezers_bottom(self):
        rows = _downtrend()
        rows.append({"open": 192.0, "high": 193.0, "low": 186.0, "close": 187.0})
        rows.append({"open": 187.0, "high": 189.0, "low": 186.0, "close": 188.0})
        df = _df(rows)
        assert tweezers_bottom(df).iloc[-1] == "bullish"

    def test_fails_when_lows_differ(self):
        rows = _downtrend()
        rows.append({"open": 192.0, "high": 193.0, "low": 186.0, "close": 187.0})
        rows.append({"open": 187.0, "high": 189.0, "low": 183.0, "close": 188.0})
        df = _df(rows)
        assert tweezers_bottom(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Belt-Hold Lines
# ---------------------------------------------------------------------------

class TestBeltHold:
    def test_bullish_belt_hold(self):
        rows = _downtrend()
        # White, opens at low (no lower shadow), long body
        rows.append({"open": 186.0, "high": 194.0, "low": 186.0, "close": 193.0})
        df = _df(rows)
        assert belt_hold(df).iloc[-1] == "bullish"

    def test_bearish_belt_hold(self):
        rows = _uptrend()
        # Black, opens at high (no upper shadow), long body
        rows.append({"open": 114.0, "high": 114.0, "low": 106.0, "close": 107.0})
        df = _df(rows)
        assert belt_hold(df).iloc[-1] == "bearish"

    def test_fails_when_lower_shadow_too_large(self):
        rows = _downtrend()
        # Long lower shadow disqualifies bullish belt-hold
        rows.append({"open": 186.0, "high": 194.0, "low": 180.0, "close": 193.0})
        df = _df(rows)
        assert belt_hold(df).iloc[-1] is None

    def test_fails_wrong_trend(self):
        rows = _uptrend()
        # Bullish belt-hold shape but in uptrend → not a bullish belt-hold
        rows.append({"open": 114.0, "high": 122.0, "low": 114.0, "close": 121.0})
        df = _df(rows)
        # No bullish signal expected
        assert belt_hold(df).iloc[-1] != "bullish"


# ---------------------------------------------------------------------------
# Upside-Gap Two Crows
# ---------------------------------------------------------------------------

class TestUpsideGapTwoCrows:
    def _pattern(self, base: float = 100.0) -> list[dict]:
        # 1st: long white
        first  = {"open": base,       "high": base + 11, "low": base - 1,  "close": base + 10}
        # 2nd: black, gaps above 1st close, closes above 1st close
        second = {"open": base + 13,  "high": base + 14, "low": base + 10.5, "close": base + 11.5}
        # 3rd: black, opens above 2nd open, closes into 1st body (below base+10, above base)
        third  = {"open": base + 14,  "high": base + 15, "low": base + 3,  "close": base + 5}
        return [first, second, third]

    def test_detects_upside_gap_two_crows(self):
        df = _df(_uptrend() + self._pattern())
        assert upside_gap_two_crows(df).iloc[-1] == "bearish"

    def test_fails_when_third_does_not_close_into_first_body(self):
        rows = _uptrend()
        base = 100.0
        first  = {"open": base,       "high": base + 11, "low": base - 1,  "close": base + 10}
        second = {"open": base + 13,  "high": base + 14, "low": base + 10.5, "close": base + 11.5}
        # Third closes above first close → not inside first body properly
        third  = {"open": base + 14,  "high": base + 15, "low": base + 10, "close": base + 11}
        df = _df(rows + [first, second, third])
        assert upside_gap_two_crows(df).iloc[-1] is None

    def test_fails_after_downtrend(self):
        df = _df(_downtrend() + self._pattern())
        assert upside_gap_two_crows(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Three Black Crows
# ---------------------------------------------------------------------------

class TestThreeBlackCrows:
    def _three_crows(self, base: float = 110.0) -> list[dict]:
        # Three long black candles, each opening within the prior body,
        # each closing near its low.
        c1 = {"open": base,       "high": base + 0.5, "low": base - 8.0, "close": base - 7.5}
        c2 = {"open": base - 5.0, "high": base - 4.5, "low": base - 13.0, "close": base - 12.5}
        c3 = {"open": base - 10.0,"high": base - 9.5, "low": base - 18.0, "close": base - 17.5}
        return [c1, c2, c3]

    def test_detects_three_black_crows(self):
        df = _df(_uptrend() + self._three_crows())
        assert three_black_crows(df).iloc[-1] == "bearish"

    def test_fails_when_one_candle_not_black(self):
        rows = _uptrend()
        base = 110.0
        c1 = {"open": base,       "high": base + 0.5, "low": base - 8.0, "close": base - 7.5}
        # White candle in position 2
        c2 = {"open": base - 9.0, "high": base - 4.0, "low": base - 10.0, "close": base - 5.0}
        c3 = {"open": base - 7.0, "high": base - 6.5, "low": base - 15.0, "close": base - 14.5}
        df = _df(rows + [c1, c2, c3])
        assert three_black_crows(df).iloc[-1] is None

    def test_fails_after_downtrend(self):
        df = _df(_downtrend() + self._three_crows(base=170.0))
        assert three_black_crows(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Counterattack Lines
# ---------------------------------------------------------------------------

class TestCounterattackLines:
    def test_bullish_counterattack(self):
        rows = _downtrend()
        # Black candle closes at 183
        rows.append({"open": 192.0, "high": 193.0, "low": 182.5, "close": 183.0})
        # White candle opens lower, closes at same level (183)
        rows.append({"open": 180.0, "high": 184.0, "low": 179.0, "close": 183.0})
        df = _df(rows)
        assert counterattack_lines(df).iloc[-1] == "bullish"

    def test_bearish_counterattack(self):
        rows = _uptrend()
        # White candle closes at 113
        rows.append({"open": 104.0, "high": 113.5, "low": 103.5, "close": 113.0})
        # Black candle opens higher, closes at same level (113)
        rows.append({"open": 116.0, "high": 117.0, "low": 112.5, "close": 113.0})
        df = _df(rows)
        assert counterattack_lines(df).iloc[-1] == "bearish"

    def test_fails_when_closes_differ(self):
        rows = _downtrend()
        rows.append({"open": 192.0, "high": 193.0, "low": 182.5, "close": 183.0})
        # Close differs by more than tolerance
        rows.append({"open": 180.0, "high": 186.0, "low": 179.0, "close": 185.0})
        df = _df(rows)
        assert counterattack_lines(df).iloc[-1] is None


# ---------------------------------------------------------------------------
# Three Mountains
# ---------------------------------------------------------------------------

class TestThreeMountains:
    def _three_peaks_df(self) -> pd.DataFrame:
        # Build: down → up → down → up → down → up → down
        # Three peaks at roughly the same high (~50)
        # Need enough bars before the pattern for pivot_highs to work
        highs = [
            30, 40, 30,   # 1st peak at ~40 (index 1)
            20, 40, 20,   # 2nd peak at ~40 (index 4)
            20, 41, 20,   # 3rd peak at ~41 (index 7) — within 3% tolerance
            15,           # confirms 3rd peak as pivot (index 8)
        ]
        lows = [h - 1 for h in highs]
        rows = [{"open": h - 2, "high": h, "low": l, "close": h - 2} for h, l in zip(highs, lows)]
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
        return df.astype(float)

    def test_detects_three_mountains(self):
        df = self._three_peaks_df()
        result = three_mountains(df, lookback=20, tolerance=0.05)
        # Signal fires at index 8 (bar after 3rd peak is confirmed)
        assert result.iloc[8] == "bearish"

    def test_no_signal_when_peaks_differ_too_much(self):
        df = self._three_peaks_df()
        # Use tight tolerance → peaks at 40, 40, 41 spread = 1/40.3 ≈ 2.5% → fails at 2%
        result = three_mountains(df, lookback=20, tolerance=0.02)
        assert result.iloc[8] is None


# ---------------------------------------------------------------------------
# Three Rivers
# ---------------------------------------------------------------------------

class TestThreeRivers:
    def _three_troughs_df(self) -> pd.DataFrame:
        lows_seq = [
            70, 60, 70,   # 1st trough at 60 (index 1)
            80, 60, 80,   # 2nd trough at 60 (index 4)
            80, 59, 80,   # 3rd trough at 59 (index 7) — within 3% tolerance
            85,           # confirms 3rd trough (index 8)
        ]
        highs = [l + 1 for l in lows_seq]
        rows = [{"open": l + 2, "high": h, "low": l, "close": l + 2}
                for l, h in zip(lows_seq, highs)]
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
        return df.astype(float)

    def test_detects_three_rivers(self):
        df = self._three_troughs_df()
        result = three_rivers(df, lookback=20, tolerance=0.05)
        assert result.iloc[8] == "bullish"

    def test_no_signal_when_troughs_differ_too_much(self):
        df = self._three_troughs_df()
        result = three_rivers(df, lookback=20, tolerance=0.005)
        assert result.iloc[8] is None


# ---------------------------------------------------------------------------
# Dumpling Top
# ---------------------------------------------------------------------------

class TestDumplingTop:
    def _arc_df(self) -> pd.DataFrame:
        # Rising small-body candles, peak in middle, falling, then gap down
        closes = [100, 101, 102, 103, 102, 101, 100, 99, 98, 97]
        rows = [{"open": c, "high": c + 0.4, "low": c - 0.4, "close": c + 0.1}
                for c in closes]
        # Gap-down confirmation bar: opens below previous close
        rows.append({"open": 96.0, "high": 96.5, "low": 95.5, "close": 96.0})
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
        return df.astype(float)

    def test_detects_dumpling_top(self):
        df = self._arc_df()
        result = dumpling_top(df, window=10)
        assert result.iloc[-1] == "bearish"

    def test_fails_without_gap_down(self):
        df = self._arc_df()
        # Remove the gap: last bar opens above previous close
        df.iloc[-1, df.columns.get_loc("open")] = 97.2  # above prev close 97.1
        result = dumpling_top(df, window=10)
        assert result.iloc[-1] is None

    def test_fails_when_body_too_large(self):
        closes = [100, 101, 102, 103, 102, 101, 100, 99, 98, 97]
        # body=2.0, range=4.0 → ratio 0.5 > 0.4 threshold → not small bodies
        rows = [{"open": c, "high": c + 2.0, "low": c - 2.0, "close": c + 2.0}
                for c in closes]
        rows.append({"open": 96.0, "high": 96.5, "low": 95.5, "close": 96.0})
        df = _df(rows)
        assert dumpling_top(df, window=10).iloc[-1] is None


# ---------------------------------------------------------------------------
# Fry Pan Bottom
# ---------------------------------------------------------------------------

class TestFryPanBottom:
    def _bowl_df(self) -> pd.DataFrame:
        closes = [100, 99, 98, 97, 98, 99, 100, 101, 102, 103]
        rows = [{"open": c, "high": c + 0.4, "low": c - 0.4, "close": c + 0.1}
                for c in closes]
        # Gap-up confirmation bar
        rows.append({"open": 104.2, "high": 105.0, "low": 104.0, "close": 104.5})
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
        return df.astype(float)

    def test_detects_fry_pan_bottom(self):
        df = self._bowl_df()
        result = fry_pan_bottom(df, window=10)
        assert result.iloc[-1] == "bullish"

    def test_fails_without_gap_up(self):
        df = self._bowl_df()
        df.iloc[-1, df.columns.get_loc("open")] = 103.0  # below prev close (103.1)
        result = fry_pan_bottom(df, window=10)
        assert result.iloc[-1] is None


# ---------------------------------------------------------------------------
# Tower Top
# ---------------------------------------------------------------------------

class TestTowerTop:
    def _tower_df(self) -> pd.DataFrame:
        # First third (3 bars): long white candles
        first = [
            {"open": 100.0, "high": 108.5, "low": 99.5, "close": 108.0},
            {"open": 108.0, "high": 116.5, "low": 107.5, "close": 116.0},
            {"open": 116.0, "high": 124.5, "low": 115.5, "close": 124.0},
        ]
        # Middle (4 bars): small bodies
        mid = [{"open": 124.0, "high": 124.8, "low": 123.2, "close": 124.3}] * 4
        # Last third (3 bars): long black candles
        last = [
            {"open": 124.0, "high": 124.5, "low": 115.5, "close": 116.0},
            {"open": 116.0, "high": 116.5, "low": 107.5, "close": 108.0},
            {"open": 108.0, "high": 108.5, "low":  99.5, "close": 100.0},
        ]
        rows = first + mid + last
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
        return df.astype(float)

    def test_detects_tower_top(self):
        df = self._tower_df()
        result = tower_top(df, window=10)
        assert result.iloc[-1] == "bearish"

    def test_fails_when_middle_has_large_bodies(self):
        df = self._tower_df()
        # Replace middle with large-body candles
        for i in range(3, 7):
            df.iloc[i, df.columns.get_loc("open")]  = 120.0
            df.iloc[i, df.columns.get_loc("close")] = 110.0
            df.iloc[i, df.columns.get_loc("high")]  = 121.0
            df.iloc[i, df.columns.get_loc("low")]   = 109.0
        result = tower_top(df, window=10)
        assert result.iloc[-1] is None


# ---------------------------------------------------------------------------
# Tower Bottom
# ---------------------------------------------------------------------------

class TestTowerBottom:
    def _tower_df(self) -> pd.DataFrame:
        # Mirror of tower top
        first = [
            {"open": 124.0, "high": 124.5, "low": 115.5, "close": 116.0},
            {"open": 116.0, "high": 116.5, "low": 107.5, "close": 108.0},
            {"open": 108.0, "high": 108.5, "low":  99.5, "close": 100.0},
        ]
        mid = [{"open": 100.0, "high": 100.8, "low": 99.2, "close": 100.3}] * 4
        last = [
            {"open": 100.0, "high": 108.5, "low": 99.5, "close": 108.0},
            {"open": 108.0, "high": 116.5, "low": 107.5, "close": 116.0},
            {"open": 116.0, "high": 124.5, "low": 115.5, "close": 124.0},
        ]
        rows = first + mid + last
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
        return df.astype(float)

    def test_detects_tower_bottom(self):
        df = self._tower_df()
        result = tower_bottom(df, window=10)
        assert result.iloc[-1] == "bullish"
