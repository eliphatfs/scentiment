"""Tests for pivot high/low detection.

Covers:
- 1st-order pivot highs and lows on hand-crafted sequences
- No-lookahead: last bar is always False for all orders
- 2nd-order: only confirmed once the next 1st-order pivot has appeared
- 3rd-order: only confirmed once the next 2nd-order pivot has appeared
- Edge cases: fewer bars than required to confirm any pivot
"""

import pandas as pd
import pytest

from patterns.pivots import pivot_highs, pivot_lows


def _df(highs, lows=None) -> pd.DataFrame:
    """Build a minimal DataFrame from lists of high (and optionally low) values."""
    if lows is None:
        # Mirror lows just below highs so they don't interfere
        lows = [h - 0.5 for h in highs]
    df = pd.DataFrame({"high": highs, "low": lows})
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
    return df


# ---------------------------------------------------------------------------
# 1st-order pivot highs
# ---------------------------------------------------------------------------

class TestFirstOrderPivotHighs:
    def test_simple_peak(self):
        # Sequence: 1 3 2 → index 1 is a pivot high
        df = _df([1, 3, 2])
        result = pivot_highs(df, order=1)
        assert result.iloc[1] is True or result.iloc[1] == True
        assert result.iloc[0] == False
        assert result.iloc[2] == False   # last bar always False

    def test_multiple_peaks(self):
        # 1 4 2 5 1 3 2  → pivots at index 1 (4) and 3 (5)
        # Index 5 (3) would be a pivot but index 6 is last → False
        df = _df([1, 4, 2, 5, 1, 3, 2])
        result = pivot_highs(df, order=1)
        assert result.iloc[1] == True
        assert result.iloc[3] == True
        assert result.iloc[5] == True   # confirmed by index 6
        assert result.iloc[6] == False  # last bar

    def test_last_bar_always_false(self):
        # Even if the last bar would be a pivot (higher than its predecessor),
        # it must be False because the confirming next bar is absent.
        df = _df([1, 2, 3])  # strictly rising — no pivots except potentially last
        result = pivot_highs(df, order=1)
        assert result.iloc[-1] == False

    def test_no_pivot_on_plateau(self):
        # Equal adjacent bars should not be pivots (strict inequality)
        df = _df([1, 3, 3, 2])
        result = pivot_highs(df, order=1)
        assert result.iloc[1] == False
        assert result.iloc[2] == False

    def test_monotone_rising_has_no_pivot(self):
        df = _df([1, 2, 3, 4, 5])
        assert pivot_highs(df, order=1).any() == False

    def test_monotone_falling_has_no_pivot(self):
        df = _df([5, 4, 3, 2, 1])
        assert pivot_highs(df, order=1).any() == False


# ---------------------------------------------------------------------------
# 1st-order pivot lows
# ---------------------------------------------------------------------------

class TestFirstOrderPivotLows:
    def test_simple_trough(self):
        # highs don't matter; lows: 3 1 2 → index 1 is a pivot low
        df = _df(highs=[4, 2, 3], lows=[3, 1, 2])
        result = pivot_lows(df, order=1)
        assert result.iloc[1] == True
        assert result.iloc[2] == False  # last bar

    def test_last_bar_always_false(self):
        df = _df(highs=[3, 2, 1], lows=[3, 2, 1])  # falling lows
        assert pivot_lows(df, order=1).iloc[-1] == False


# ---------------------------------------------------------------------------
# 2nd-order pivot highs
# ---------------------------------------------------------------------------

class TestSecondOrderPivotHighs:
    def _wave_df(self):
        # Construct a clear wave: small peak, big peak, small peak
        # Highs:  1  3  2  5  2  4  1
        # Indices:0  1  2  3  4  5  6
        # 1st-order pivot highs at 1 (3), 3 (5), 5 (4)
        # Among those: 3 (val 3), 5 (val 5), 4 (val 4)
        # → 5 is a 2nd-order pivot high (5 > 3 and 5 > 4), confirmed at index 5
        return _df([1, 3, 2, 5, 2, 4, 1])

    def test_detects_second_order_peak(self):
        df = self._wave_df()
        result = pivot_highs(df, order=2)
        # Index 3 (value 5) should be the sole 2nd-order pivot high
        assert result.iloc[3] == True

    def test_non_peaks_are_false(self):
        df = self._wave_df()
        result = pivot_highs(df, order=2)
        for i in [0, 1, 2, 4, 5, 6]:
            assert result.iloc[i] == False, f"unexpected 2nd-order pivot at index {i}"

    def test_last_first_order_pivot_is_unconfirmed(self):
        # Add a 4th 1st-order pivot at the end that would be a 2nd-order pivot
        # but cannot be confirmed yet because no subsequent 1st-order pivot exists.
        # Highs: 1 3 2 4 1 6 1   → 1st-order pivots at 1(3), 3(4), 5(6)
        # 6 is the biggest and last 1st-order pivot → unconfirmed as 2nd-order
        df = _df([1, 3, 2, 4, 1, 6, 1])
        result = pivot_highs(df, order=2)
        assert result.iloc[5] == False

    def test_fewer_than_three_first_order_pivots_gives_no_second_order(self):
        # Only 2 first-order pivots → impossible to confirm any 2nd-order
        df = _df([1, 3, 2, 5, 1])
        result = pivot_highs(df, order=2)
        assert result.any() == False


# ---------------------------------------------------------------------------
# 3rd-order pivot highs
# ---------------------------------------------------------------------------

class TestThirdOrderPivotHighs:
    def _three_wave_df(self):
        # Build a sequence with 3 clear 2nd-order pivot highs so that one
        # 3rd-order pivot can be confirmed.
        #
        # Pattern (highs only):
        #  small wiggle → medium peak → small wiggle → big peak → small wiggle → medium peak → small wiggle
        #
        # Highs: 1 3 2 4 2 3 1 6 1 3 1 4 1
        # 0 1 2 3 4 5 6 7 8 9 10 11 12
        #
        # 1st-order pivot highs: 1(3),3(4),5(3),7(6),9(3),11(4)
        # 2nd-order among those: 3(4 > 3, 4 < 6) → no; 7(6 > 4, 6 > 4) → yes;
        #   actually let me re-check:
        # Among 1st-order pivot values [3,4,3,6,3,4]:
        #   idx 1 val 3: 3 > prev? no prev → NaN → False
        #   idx 3 val 4: 4 > 3 and 4 > 3 → True (2nd-order)
        #   idx 5 val 3: 3 < 4 → False
        #   idx 7 val 6: 6 > 3 and 6 > 3 → True (2nd-order)
        #   idx 9 val 3: 3 < 6 → False
        #   idx 11 val 4: last 1st-order pivot → False (unconfirmed)
        # 2nd-order pivots: index 3 (val 4) and index 7 (val 6)
        # Only 2 confirmed 2nd-order pivots → no 3rd-order yet
        #
        # Extend: add another 2nd-order pivot to confirm index 7 as 3rd-order
        # Highs: 1 3 2 4 2 3 1 6 1 3 1 4 1 3 1 5 1
        #         0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        # 1st-order: 1,3,5,7,9,11,13,15
        # Pivot values: 3,4,3,6,3,4,3,5
        # 2nd-order: among [3,4,3,6,3,4,3,5]:
        #   idx 3 (4): 4>3, 4>3 → True
        #   idx 7 (6): 6>3, 6>4 → True
        #   idx 11 (4): 4<6 → False
        #   idx 15 (5): last 1st-order → False (unconfirmed)
        # 2nd-order confirmed: indices 3 (val 4) and 7 (val 6)
        # Only 2 confirmed 2nd-order → still no 3rd-order
        #
        # Need 3 confirmed 2nd-order pivots:
        # Add: ... 1 3 1 8 1 3 1 5 1 3 1 7 1 3 1
        #       0  1 2 3 4 5 6 7 8 9 10 11 12 13 14
        # 1st-order pivots at 1,3,5,7,9,11,13 (values: 3,8,3,5,3,7,3)
        # 2nd-order:
        #   idx 1 (3): no prev → False
        #   idx 3 (8): 8>3, 8>3 → True
        #   idx 5 (3): 3<8 → False
        #   idx 7 (5): 5<8 → False
        #   idx 9 (3): 3<5 → False
        #   idx 11 (7): 7>3, 7>3 → True
        #   idx 13 (3): last → False
        # 2nd-order confirmed: indices 3 (val 8) and 11 (val 7)
        # Still only 2 confirmed 2nd-order → no 3rd-order
        #
        # I need 3 confirmed 2nd-order and a 4th to close them.
        # Simpler: highs = [1, 3, 1, 5, 1, 3, 1, 8, 1, 3, 1, 5, 1, 3, 1, 6, 1, 3, 1]
        # 1st-order pivots (odd indices 1..17): vals [3,5,3,8,3,5,3,6,3]
        # 2nd-order among [3,5,3,8,3,5,3,6,3]:
        #   idx 1(3): no prev → F
        #   idx 3(5): 5>3,5>3 → T  ← 2nd-order
        #   idx 5(3): 3<5 → F
        #   idx 7(8): 8>3,8>3 → T  ← 2nd-order
        #   idx 9(3): 3<8 → F
        #   idx 11(5): 5<8 → F
        #   idx 13(3): 3<5 → F
        #   idx 15(6): 6>3,6>3 → T  ← 2nd-order
        #   idx 17(3): last → F
        # 2nd-order confirmed at indices 3(val5), 7(val8), 15(val6)
        # 3rd-order among [5,8,6]:
        #   idx 3(5): no prev → F
        #   idx 7(8): 8>5, 8>6 → T  ← 3rd-order pivot!
        #   idx 15(6): last 2nd-order → F
        return _df([1, 3, 1, 5, 1, 3, 1, 8, 1, 3, 1, 5, 1, 3, 1, 6, 1, 3, 1])

    def test_detects_third_order_peak(self):
        df = self._three_wave_df()
        result = pivot_highs(df, order=3)
        # Index 7 (value 8) is the sole 3rd-order pivot high
        assert result.iloc[7] == True

    def test_only_one_third_order_peak(self):
        df = self._three_wave_df()
        result = pivot_highs(df, order=3)
        assert result.sum() == 1

    def test_last_second_order_pivot_is_unconfirmed_as_third(self):
        df = self._three_wave_df()
        result = pivot_highs(df, order=3)
        # Index 15 (2nd-order pivot val 6) must not be a confirmed 3rd-order pivot
        assert result.iloc[15] == False


# ---------------------------------------------------------------------------
# 2nd-order pivot lows (symmetry check)
# ---------------------------------------------------------------------------

class TestSecondOrderPivotLows:
    def test_detects_second_order_trough(self):
        # Lows:   5 3 4 1 4 3 5
        # 0 1 2 3 4 5 6
        # 1st-order pivot lows at 1(3), 3(1), 5(3)
        # Among [3,1,3]: index 3(val 1) → 1<3 and 1<3 → 2nd-order
        df = _df(
            highs=[6, 4, 5, 2, 5, 4, 6],
            lows=[5, 3, 4, 1, 4, 3, 5],
        )
        result = pivot_lows(df, order=2)
        assert result.iloc[3] == True

    def test_last_first_order_low_unconfirmed(self):
        # Lows: 4 2 3 1 3  → 1st-order lows at 1(2), 3(1); index 3 is last → unconfirmed
        df = _df(highs=[5, 3, 4, 2, 4], lows=[4, 2, 3, 1, 3])
        result = pivot_lows(df, order=2)
        assert result.iloc[3] == False


# ---------------------------------------------------------------------------
# Invalid order
# ---------------------------------------------------------------------------

def test_invalid_order_raises():
    df = _df([1, 2, 3])
    with pytest.raises(ValueError):
        pivot_highs(df, order=0)
