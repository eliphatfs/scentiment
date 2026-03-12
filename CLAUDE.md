# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repository implements and backtests **technical sentiment analysis** strategies derived from two primary references:
- *Japanese Candlestick Charting Techniques* by Steve Nison — candlestick pattern recognition and interpretation
- *The Art and Science of Technical Analysis* by Adam Grimes — broader technical analysis theory and backtesting methodology

The goal is to identify candlestick/chart patterns programmatically, apply sentiment signals, and backtest their predictive value on market data.

## Data

- `data/spy.json` — sample daily OHLCV data for SPY (S&P 500 ETF), starting 2021-06-29
- `data/exhibit_9_2.csv` — TOCOM crude oil (JPY/kl), Jan–Apr 1990, transcribed from Nison exhibit 9.2
- Fields: `datetime`/`Date` (YYYY-MM-DD), `open`, `high`, `low`, `close`, `volume` (all strings in JSON; CSV is floats without volume)

## Architecture

The project is Python-based (`ruff` for linting, `pytest` for testing). Planned structure:

- **Pattern detection** — functions that take OHLCV data and return a `pd.Series` of `'bullish'`/`'bearish'`/`None` signals
- **Backtesting engine** — applies detected signals to historical data and computes performance metrics (win rate, expectancy, Sharpe, drawdown)
- **Data loading** — `data.py` loads and normalizes `data/*.json` into pandas DataFrames with numeric columns

### Pattern modules (all in `patterns/`)

| Module | Nison chapter(s) | Patterns |
|---|---|---|
| `_candle.py` | — | Shared geometry helpers (`body`, `upper_shadow`, `is_doji`, `signal_series`, …), trend detection (`is_uptrend`, `is_downtrend`, `is_uptrend_by_pivots`, `is_downtrend_by_pivots`) |
| `reversal.py` | 4–5 | `hammer`, `hanging_man`, `engulfing`, `dark_cloud_cover`, `piercing_pattern` |
| `pivots.py` | Grimes | `pivot_highs`, `pivot_lows` (orders 1–3, strict no-lookahead) |
| `stars.py` | 6, 9 | `inverted_hammer`, `shooting_star`, `morning_star`, `evening_star`, `morning_doji_star`, `evening_doji_star` |
| `doji.py` | 7–8 | `doji_at_top`, `doji_at_bottom`, `long_legged_doji`, `rickshaw_man`, `gravestone_doji`, `tri_star` |
| `more_reversals.py` | 10–12 | `harami`, `harami_cross`, `tweezers_top`, `tweezers_bottom`, `belt_hold`, `upside_gap_two_crows`, `three_black_crows`, `counterattack_lines`, `three_mountains`, `three_rivers`, `dumpling_top`, `fry_pan_bottom`, `tower_top`, `tower_bottom` |
| `continuation.py` | 13–14 | `window_up`, `window_down`, `rising_three_methods`, `falling_three_methods`, `three_white_soldiers`, `separating_lines` |

### Trend analysis

| Module | Source | Purpose |
|---|---|---|
| `_candle.py` | Grimes | `is_uptrend_by_pivots` / `is_downtrend_by_pivots` — confirmed HH+HL / LH+LL via pivot structure; `is_uptrend` / `is_downtrend` — hybrid (pivot when confirmed, short-term close-comparison fallback) |
| `trend.py` | Grimes | Multi-scale trend: `short_trend` (close-vs-close), `pivot_trend` (order-1/2 pivots), `multi_scale_trend`, `effective_trend` (pivot-preferred with fallback), `trend_terminations` (reversal patterns firing against active trends) |

### Plotting

- `plot_exhibit_9_2.py` — runs all pattern detections on exhibit 9.2 data, overlays multi-scale trend background and trend-termination signals, saves `exhibit_9_2_patterns.png`

## Development Commands

```bash
# Install dependencies
pip install pandas pytest

# Run all tests
pytest

# Run a single test file
pytest tests/test_doji.py
```

## Key Conventions

- OHLCV columns must be converted to `float` on load (they are strings in the JSON)
- Pattern functions accept a pandas DataFrame with columns `open`, `high`, `low`, `close` and return a `pd.Series` of signal strings
- Pattern logic must reference the specific book chapter in docstrings
- **Trend detection** uses Grimes pivot structure (HH+HL = uptrend, LH+LL = downtrend) with a short-term close-comparison fallback when pivot data is insufficient. `is_uptrend_by_pivots` / `is_downtrend_by_pivots` are the strict pivot-only versions; `is_uptrend` / `is_downtrend` add the fallback. The `trend_lookback` parameter on pattern functions is retained for API compatibility but ignored by the pivot-based implementation.
- Trend is always measured at the **first candle of the pattern** (not the last) via `is_uptrend_by_pivots(df).shift(N)` to avoid contamination from the pattern's own bars. For window gaps, check trend at `t-1` (the bar before the gap).
- **Never** initialize a signal Series with `pd.Series(None, index=..., dtype=object)` — this stores `float NaN`. Use `signal_series(df.index)` from `patterns._candle` instead, which returns `pd.Series([None] * len(df), index=df.index, dtype=object)`. Assign with `result[mask.fillna(False)] = "signal"`.
