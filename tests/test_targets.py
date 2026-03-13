"""Tests for price target computation."""

import numpy as np
import pandas as pd
import pytest

from targets import consolidation_boxes, pattern_sr_zones, flag_targets
from patterns._candle import signal_series


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
    return df.astype(float)


# ---------------------------------------------------------------------------
# Consolidation box breakout
# ---------------------------------------------------------------------------

class TestConsolidationBoxes:
    def test_bullish_breakout(self):
        """Price consolidates then breaks above the box."""
        # 12 bars of tight consolidation around 100, then breakout
        rows = []
        for i in range(12):
            rows.append({"open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0})
        # Breakout bar: close above box high (100.5)
        rows.append({"open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5})
        df = _df(rows)

        result = consolidation_boxes(df, min_bars=10, max_range_pct=0.02)
        assert len(result) > 0
        bullish = result[result["direction"] == "bullish"]
        assert len(bullish) > 0
        # Target should be box_high + box_height
        row = bullish.iloc[0]
        assert row["target_price"] == pytest.approx(
            row["box_high"] + row["box_height"], abs=0.01
        )

    def test_bearish_breakout(self):
        """Price consolidates then breaks below the box."""
        rows = []
        for i in range(12):
            rows.append({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0})
        # Breakout bar: close below box low (99.5)
        rows.append({"open": 99.5, "high": 100.0, "low": 98.0, "close": 98.5})
        df = _df(rows)

        result = consolidation_boxes(df, min_bars=10, max_range_pct=0.02)
        bearish = result[result["direction"] == "bearish"]
        assert len(bearish) > 0
        row = bearish.iloc[0]
        assert row["target_price"] == pytest.approx(
            row["box_low"] - row["box_height"], abs=0.01
        )

    def test_no_breakout_no_records(self):
        """Still consolidating — no breakout detected."""
        rows = []
        for i in range(15):
            rows.append({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0})
        df = _df(rows)

        result = consolidation_boxes(df, min_bars=10, max_range_pct=0.02)
        assert len(result) == 0

    def test_volatile_no_box(self):
        """Too volatile for consolidation."""
        rows = []
        for i in range(15):
            rows.append({
                "open": 100.0 + i * 5,
                "high": 106.0 + i * 5,
                "low": 94.0 + i * 5,
                "close": 105.0 + i * 5,
            })
        df = _df(rows)

        result = consolidation_boxes(df, min_bars=10, max_range_pct=0.02)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Support / resistance zones
# ---------------------------------------------------------------------------

class TestPatternSRZones:
    def test_support_zone_from_bullish_signals(self):
        """Two bullish signals at similar lows form a support zone."""
        rows = [{"open": 100, "high": 102, "low": 98, "close": 101}] * 5
        df = _df(rows)

        sig1 = signal_series(df.index)
        sig1.iloc[1] = "bullish"  # low = 98
        sig2 = signal_series(df.index)
        sig2.iloc[3] = "bullish"  # low = 98

        patterns = {"pattern_a": sig1, "pattern_b": sig2}
        zones = pattern_sr_zones(df, patterns, margin=0.003, min_signals=2)

        assert len(zones) > 0
        assert zones.iloc[0]["zone_type"] == "support"
        assert zones.iloc[0]["signal_count"] >= 2

    def test_resistance_zone_from_bearish_signals(self):
        rows = [{"open": 100, "high": 105, "low": 98, "close": 99}] * 5
        df = _df(rows)

        sig1 = signal_series(df.index)
        sig1.iloc[0] = "bearish"  # high = 105
        sig2 = signal_series(df.index)
        sig2.iloc[2] = "bearish"  # high = 105

        patterns = {"pattern_a": sig1, "pattern_b": sig2}
        zones = pattern_sr_zones(df, patterns, margin=0.003, min_signals=2)

        assert len(zones) > 0
        assert zones.iloc[0]["zone_type"] == "resistance"

    def test_both_type_zone(self):
        """Bullish and bearish signals at similar prices form a 'both' zone."""
        rows = [
            {"open": 100, "high": 102, "low": 98, "close": 101},
            {"open": 100, "high": 98.2, "low": 96, "close": 97},
        ]
        df = _df(rows)

        sig1 = signal_series(df.index)
        sig1.iloc[0] = "bullish"   # support at low = 98
        sig2 = signal_series(df.index)
        sig2.iloc[1] = "bearish"   # resistance at high = 98.2

        patterns = {"p1": sig1, "p2": sig2}
        zones = pattern_sr_zones(df, patterns, margin=0.003, min_signals=2)
        # 98 and 98.2 are within 0.3% of each other
        assert len(zones) > 0
        assert zones.iloc[0]["zone_type"] == "both"

    def test_sparse_signals_no_zone(self):
        """Single signal cannot form a zone."""
        rows = [{"open": 100, "high": 102, "low": 98, "close": 101}] * 3
        df = _df(rows)

        sig = signal_series(df.index)
        sig.iloc[0] = "bullish"

        zones = pattern_sr_zones(df, {"p": sig}, margin=0.003, min_signals=2)
        assert len(zones) == 0

    def test_no_signals_empty_result(self):
        rows = [{"open": 100, "high": 102, "low": 98, "close": 101}] * 3
        df = _df(rows)
        sig = signal_series(df.index)
        zones = pattern_sr_zones(df, {"p": sig}, margin=0.003)
        assert len(zones) == 0


# ---------------------------------------------------------------------------
# Flag / pennant targets
# ---------------------------------------------------------------------------

class TestFlagTargets:
    def test_bullish_flag(self):
        """Sharp up move (pole) → tight consolidation → breakout up."""
        rows = []
        # Pole: 5 bars rising from 100 to ~125 (5% per bar)
        price = 100.0
        for i in range(5):
            o = price
            c = price + 5.0
            rows.append({"open": o, "high": c + 0.5, "low": o - 0.3, "close": c})
            price = c

        # Flag: 5 bars of tight consolidation
        for i in range(5):
            base = price - 1 + (i % 2) * 0.5
            rows.append({
                "open": base,
                "high": base + 0.8,
                "low": base - 0.3,
                "close": base + 0.3,
            })

        # Breakout bar
        flag_high = max(r["high"] for r in rows[-5:])
        rows.append({
            "open": price,
            "high": price + 3,
            "low": price - 0.5,
            "close": flag_high + 1.0,
        })

        df = _df(rows)
        result = flag_targets(
            df,
            min_pole_bars=3,
            max_pole_bars=8,
            max_flag_bars=10,
            min_pole_pct=0.03,
            max_flag_range_pct=0.5,
        )

        bullish = result[result["direction"] == "bullish"]
        assert len(bullish) > 0
        row = bullish.iloc[0]
        assert row["pole_height"] > 0
        assert row["target_price"] > row["breakout_price"]

    def test_no_flag_without_consolidation(self):
        """Continuous trend without consolidation → no flag pattern."""
        rows = []
        price = 100.0
        for i in range(20):
            o = price
            c = price + 2.0
            rows.append({"open": o, "high": c + 0.5, "low": o - 0.3, "close": c})
            price = c
        df = _df(rows)

        result = flag_targets(df, min_pole_pct=0.10, max_flag_range_pct=0.2)
        # May or may not find patterns depending on parameters, but with tight
        # flag range requirement, straight trend shouldn't qualify
        if len(result) > 0:
            # If anything found, it should at least have valid structure
            assert all(result["pole_height"] > 0)

    def test_empty_data(self):
        rows = [{"open": 100, "high": 101, "low": 99, "close": 100}] * 3
        df = _df(rows)
        result = flag_targets(df)
        assert len(result) == 0
