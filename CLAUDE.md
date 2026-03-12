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

## Expected Architecture (as code is added)

The project is Python-based (standard `.gitignore` configured for Python, with `ruff` for linting and `pytest` for testing). Structure to follow:

- **Pattern detection** — functions/classes that take OHLCV data and return boolean or scored signals for each candle/bar
- **Backtesting engine** — applies detected signals to historical data and computes performance metrics (win rate, expectancy, Sharpe, drawdown)
- **Data loading** — utilities to load and normalize `data/*.json` into a consistent structure (e.g., pandas DataFrames with numeric columns)

## Development Commands

```bash
# Install dependencies
pip install pandas pytest

# Run tests
pytest

# Run a single test file
pytest tests/test_reversal.py
```

## Key Conventions

- OHLCV columns should be converted to `float` on load (they are strings in the JSON)
- Candlestick pattern functions should accept a pandas DataFrame with columns `open`, `high`, `low`, `close`, `volume` and return a Series of signals
- Backtest results should be reproducible — no randomness without a fixed seed
- Pattern logic should reference the specific book and pattern name in docstrings
- **Never** initialize a signal Series with `pd.Series(None, index=..., dtype=object)` — this stores `float NaN`, not Python `None`. Use `pd.Series([None] * len(df), index=df.index, dtype=object)` instead, then assign signal rows with `result[mask.fillna(False)] = "signal"`
