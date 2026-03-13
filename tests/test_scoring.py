"""Tests for pattern strength scoring."""

import numpy as np
import pandas as pd
import pytest

from patterns.scoring import (
    score_hammer_shape,
    score_inverted_hammer_shape,
    score_engulfing_shape,
    score_doji_shape,
    score_volume,
    score_trend_strength,
    pattern_strength,
)
from patterns._candle import signal_series


def _df(rows: list[dict], volumes: list[float] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    if volumes is not None:
        df["volume"] = volumes
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")
    return df.astype(float)


# ---------------------------------------------------------------------------
# Shape scores
# ---------------------------------------------------------------------------

class TestHammerShapeScore:
    def test_perfect_hammer_high_score(self):
        """Ideal hammer: zero upper shadow, lower shadow = 4× body."""
        rows = [{"open": 100, "high": 101, "low": 93, "close": 101}]
        df = _df(rows)
        score = score_hammer_shape(df)
        # lower = 7, body = 1, upper = 0 → shadow ratio = 7x > 2x → high score
        assert score.iloc[0] > 0.7

    def test_weak_hammer_low_score(self):
        """Hammer with barely qualifying proportions."""
        rows = [{"open": 100, "high": 101, "low": 98, "close": 101}]
        df = _df(rows)
        score = score_hammer_shape(df)
        # lower = 2, body = 1, upper = 0 → shadow ratio exactly 2x → low score
        assert score.iloc[0] < 0.5

    def test_score_between_0_and_1(self):
        rows = [{"open": 100, "high": 102, "low": 94, "close": 101}]
        df = _df(rows)
        score = score_hammer_shape(df)
        assert 0.0 <= score.iloc[0] <= 1.0


class TestInvertedHammerShapeScore:
    def test_strong_inverted_hammer(self):
        rows = [{"open": 100, "high": 108, "low": 99.5, "close": 100.5}]
        df = _df(rows)
        score = score_inverted_hammer_shape(df)
        # upper = 7.5, body = 0.5, lower = 0.5 → very high ratio
        assert score.iloc[0] > 0.7


class TestEngulfingShapeScore:
    def test_large_engulfing_high_score(self):
        rows = [
            {"open": 101, "high": 102, "low": 99, "close": 100},   # small body
            {"open": 99, "high": 105, "low": 98, "close": 104},    # large engulfing
        ]
        df = _df(rows)
        score = score_engulfing_shape(df)
        # curr body = 5, prev body = 1, ratio = 4x → high
        assert score.iloc[1] > 0.5

    def test_barely_engulfing_low_score(self):
        rows = [
            {"open": 101, "high": 102, "low": 99, "close": 100},   # body = 1
            {"open": 99, "high": 102, "low": 98, "close": 100.2},  # body = 1.2 vs 1
        ]
        df = _df(rows)
        score = score_engulfing_shape(df)
        assert score.iloc[1] > 0
        assert score.iloc[1] < 0.5  # only 20% larger


class TestDojiShapeScore:
    def test_perfect_doji_high_score(self):
        rows = [{"open": 100, "high": 105, "low": 95, "close": 100}]
        df = _df(rows)
        score = score_doji_shape(df)
        # body = 0, range = 10, shadows = 10 → perfect
        assert score.iloc[0] > 0.8

    def test_wide_body_low_score(self):
        rows = [{"open": 100, "high": 105, "low": 95, "close": 100.9}]
        df = _df(rows)
        score = score_doji_shape(df)
        # body = 0.9, range = 10, threshold = 0.1 → body = 0.9*range*threshold
        assert score.iloc[0] < 1.0


# ---------------------------------------------------------------------------
# Volume score
# ---------------------------------------------------------------------------

class TestVolumeScore:
    def test_high_volume_high_score(self):
        rows = [{"open": 100, "high": 102, "low": 99, "close": 101}] * 5
        vols = [100, 100, 100, 100, 300]  # last bar 3× average
        df = _df(rows, volumes=vols)
        sig = signal_series(df.index)
        sig.iloc[-1] = "bullish"

        score = score_volume(df, sig)
        assert score.iloc[-1] > 0.7

    def test_low_volume_low_score(self):
        rows = [{"open": 100, "high": 102, "low": 99, "close": 101}] * 5
        vols = [100, 100, 100, 100, 30]  # last bar 0.3× average
        df = _df(rows, volumes=vols)
        sig = signal_series(df.index)
        sig.iloc[-1] = "bullish"

        score = score_volume(df, sig)
        assert score.iloc[-1] < 0.3

    def test_no_volume_returns_neutral(self):
        rows = [{"open": 100, "high": 102, "low": 99, "close": 101}] * 3
        df = _df(rows)  # no volume column
        sig = signal_series(df.index)
        sig.iloc[-1] = "bearish"

        score = score_volume(df, sig)
        assert score.iloc[-1] == 0.5


# ---------------------------------------------------------------------------
# Trend strength score
# ---------------------------------------------------------------------------

class TestTrendStrengthScore:
    def test_strong_trend_high_score(self):
        # Strong downtrend
        rows = []
        for i in range(15):
            base = 200 - i * 5
            rows.append({"open": base, "high": base + 1, "low": base - 6, "close": base - 5})
        df = _df(rows)
        sig = signal_series(df.index)
        sig.iloc[-1] = "bullish"

        score = score_trend_strength(df, sig)
        assert score.iloc[-1] > 0.3  # strong consistent slope

    def test_flat_trend_low_score(self):
        rows = [{"open": 100, "high": 101, "low": 99, "close": 100}] * 15
        df = _df(rows)
        sig = signal_series(df.index)
        sig.iloc[-1] = "bullish"

        score = score_trend_strength(df, sig)
        assert score.iloc[-1] < 0.2


# ---------------------------------------------------------------------------
# Composite pattern_strength
# ---------------------------------------------------------------------------

class TestPatternStrength:
    def test_returns_score_at_signal_bars_only(self):
        rows = [{"open": 100, "high": 102, "low": 95, "close": 101}] * 5
        df = _df(rows)
        sig = signal_series(df.index)
        sig.iloc[2] = "bullish"

        strength = pattern_strength(df, "hammer", sig)
        assert np.isnan(strength.iloc[0])
        assert np.isnan(strength.iloc[1])
        assert not np.isnan(strength.iloc[2])
        assert np.isnan(strength.iloc[3])

    def test_score_between_0_and_1(self):
        rows = [{"open": 100, "high": 102, "low": 95, "close": 101}] * 5
        df = _df(rows)
        sig = signal_series(df.index)
        sig.iloc[-1] = "bullish"

        strength = pattern_strength(df, "hammer", sig)
        assert 0.0 <= strength.iloc[-1] <= 1.0

    def test_unknown_pattern_uses_neutral_shape(self):
        rows = [{"open": 100, "high": 102, "low": 99, "close": 101}] * 5
        df = _df(rows)
        sig = signal_series(df.index)
        sig.iloc[-1] = "bullish"

        # Unknown pattern name → neutral shape score of 0.5
        strength = pattern_strength(df, "some_unknown_pattern", sig)
        assert not np.isnan(strength.iloc[-1])

    def test_no_signals_returns_all_nan(self):
        rows = [{"open": 100, "high": 102, "low": 99, "close": 101}] * 3
        df = _df(rows)
        sig = signal_series(df.index)

        strength = pattern_strength(df, "hammer", sig)
        assert strength.isna().all()
