# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repository implements and backtests **technical sentiment analysis** strategies derived from two primary references:
- *Japanese Candlestick Charting Techniques* by Steve Nison — candlestick pattern recognition and interpretation
- *The Art and Science of Technical Analysis* by Adam Grimes — broader technical analysis theory and backtesting methodology

The goal is to identify candlestick/chart patterns programmatically, apply sentiment signals, and backtest their predictive value on market data.

## Data

- `data/spy.json` — sample daily OHLCV data for SPY (S&P 500 ETF), starting 2021-06-29
- Fields: `datetime` (YYYY-MM-DD), `open`, `high`, `low`, `close`, `volume` (all strings)

## Architecture

The project is Python-based (`ruff` for linting, `pytest` for testing). Planned structure:

- **Pattern detection** — functions that take OHLCV data and return a `pd.Series` of `'bullish'`/`'bearish'`/`None` signals
- **Backtesting engine** — applies detected signals to historical data and computes performance metrics (win rate, expectancy, Sharpe, drawdown)
- **Data loading** — `data.py` loads and normalizes `data/*.json` into pandas DataFrames with numeric columns

### Pattern modules (all in `patterns/`)

| Module | Nison chapter(s) | Patterns |
|---|---|---|
| `_candle.py` | — | Shared geometry helpers (`body`, `upper_shadow`, `is_doji`, `signal_series`, …) |
| `reversal.py` | 4–5 | `hammer`, `hanging_man`, `engulfing`, `dark_cloud_cover`, `piercing_pattern` |
| `pivots.py` | Grimes | `pivot_highs`, `pivot_lows` (orders 1–3, strict no-lookahead) |
| `stars.py` | 6, 9 | `inverted_hammer`, `shooting_star`, `morning_star`, `evening_star`, `morning_doji_star`, `evening_doji_star` |
| `doji.py` | 7–8 | `doji_at_top`, `doji_at_bottom`, `long_legged_doji`, `rickshaw_man`, `gravestone_doji`, `tri_star` |
| `more_reversals.py` | 10–12 | `harami`, `harami_cross`, `tweezers_top`, `tweezers_bottom`, `belt_hold`, `upside_gap_two_crows`, `three_black_crows`, `counterattack_lines`, `three_mountains`, `three_rivers`, `dumpling_top`, `fry_pan_bottom`, `tower_top`, `tower_bottom` |
| `continuation.py` | 13–14 | `window_up`, `window_down`, `rising_three_methods`, `falling_three_methods`, `three_white_soldiers`, `separating_lines` |

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
- Trend is always measured at the **first candle of the pattern** (not the last) via `df["close"].shift(N)` to avoid contamination from the pattern's own bars. For window gaps, check trend at `t-1` (the bar before the gap) because a large gap can flip the close-vs-lookback comparison.
- **Never** initialize a signal Series with `pd.Series(None, index=..., dtype=object)` — this stores `float NaN`. Use `signal_series(df.index)` from `patterns._candle` instead, which returns `pd.Series([None] * len(df), index=df.index, dtype=object)`. Assign with `result[mask.fillna(False)] = "signal"`.
