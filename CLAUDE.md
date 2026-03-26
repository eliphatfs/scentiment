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
| `_candle.py` | Grimes | `is_uptrend_by_pivots` / `is_downtrend_by_pivots` — confirmed HH+HL / LH+LL via pivot structure; `is_uptrend` / `is_downtrend` — hybrid (pivot when confirmed, regression-slope fallback); `_regression_slope` — rolling OLS slope of closes |
| `trend.py` | Grimes + Nison | Multi-scale trend with four time-scales: `body_run_trend` (Nison-style micro-trend from consecutive same-color bodies), `regression_slope_trend` / `short_trend` (rolling OLS slope), `pivot_trend` (order-1/2 pivots); `multi_scale_trend` (micro/short/medium/long), `effective_trend` (pivot → regression → body-run fallback), `trend_terminations` (reversal patterns firing against active trends) |

### Pattern scoring and confirmation (in `patterns/`)

| Module | Purpose |
|---|---|
| `scoring.py` | Shape-quality scoring (0.0–1.0) for each pattern family, volume confirmation score, trend strength score, and composite `pattern_strength()` combiner |
| `confirmation.py` | Confirmation-delayed signals: `confirmed_signal()` engine + convenience wrappers (`confirmed_hanging_man`, `confirmed_shooting_star`, `confirmed_inverted_hammer`, `confirmed_doji_at_top`, `confirmed_doji_at_bottom`, `confirmed_gravestone_doji`); `CONFIRMATION_RULES` registry. Note: harami/harami_cross do NOT need confirmation — they are complete two-bar patterns where the second candle already demonstrates loss of momentum. |

### Price targets

| Module | Source | Purpose |
|---|---|---|
| `targets.py` | Nison ch. 16 | `consolidation_boxes` — box/range breakout detection with projected targets; `pattern_sr_zones` — support/resistance zones from clustered pattern signals; `flag_targets` — flag/pennant continuation targets (pole height projection) |

### Backtesting

| Module | Purpose |
|---|---|
| `backtest.py` | Streaming candlestick-pattern backtest engine: processes data bar-by-bar (no lookahead), runs pattern detection via `_analyze_bar()` (shared compute for signals + S/R zones), Kelly criterion for position sizing (cap 0.40, min R:R 1.0), exits on reversal signals. `close_hack` execution mode. Includes `compute_alpha_beta()` for CAPM decomposition vs buy-and-hold, and equity curve plotting. |
| `supertrend.py` | Supertrend(ATR period, multiplier) indicator + long-only backtest. Computes trailing ATR bands, trend direction, buy/sell signals on trend flips. Plots price with Supertrend bands and equity curve vs buy-and-hold. |

### Plotting

| Module | Purpose |
|---|---|
| `plot_common.py` | Shared helpers for real-time plotting: `load_api_key()` (reads from `.env` or env var), `fetch_ohlcv(symbol, interval, outputsize, api_key)` (Twelve Data API), `run_all_patterns(df, match_tolerance)`, `plot_chart()` (candlesticks + pattern annotations + multi-scale trend panel). Parameterized for different intervals and tick formats. |
| `plot_realtime.py` | 5-min intraday chart via Twelve Data API. Uses `match_tolerance=0.001` (0.1%). Usage: `python plot_realtime.py [SYMBOL] [DAYS]` |
| `plot_daily.py` | Daily chart via Twelve Data API, default 6 months. Uses daily default tolerances. Usage: `python plot_daily.py [SYMBOL] [MONTHS]` |
| `plot_exhibit_9_2.py` | Runs all pattern detections on exhibit 9.2 data (static CSV), overlays multi-scale trend background and trend-termination signals, saves `exhibit_9_2_patterns.png` |

## Development Commands

```bash
# Install dependencies
pip install pandas numpy pytest matplotlib

# Run all tests
pytest

# Run a single test file
pytest tests/test_doji.py

# Real-time plots (requires TWELVEDATA_API_KEY in .env)
python plot_realtime.py SPY 3       # 5-min bars, 3 trading days
python plot_daily.py SPY 6          # daily bars, 6 months

# Backtests
python backtest.py                  # candlestick pattern strategy on SPY 2021-2022
python supertrend.py                # supertrend strategy on SPY 2021-2022
```

## Key Conventions

- OHLCV columns must be converted to `float` on load (they are strings in the JSON)
- Pattern functions accept a pandas DataFrame with columns `open`, `high`, `low`, `close` and return a `pd.Series` of signal strings
- Pattern logic must reference the specific book chapter in docstrings
- **Trend detection** uses Grimes pivot structure (HH+HL = uptrend, LH+LL = downtrend) with a rolling linear regression slope fallback when pivot data is insufficient. `is_uptrend_by_pivots` / `is_downtrend_by_pivots` are the strict pivot-only versions; `is_uptrend` / `is_downtrend` add the regression fallback. The `trend_lookback` parameter on pattern functions controls the regression window size when pivots are unavailable.
- **Multi-scale trend** (`trend.py`) provides four time-scales: micro (body runs, Nison-style 2+ consecutive same-color candles), short (regression slope), medium (order-1 pivots), long (order-2 pivots). `trend_terminations()` flags where reversal patterns fire against an active trend at any scale.
- Trend is always measured at the **first candle of the pattern** (not the last) via `is_uptrend_by_pivots(df).shift(N)` to avoid contamination from the pattern's own bars. For window gaps, check trend at `t-1` (the bar before the gap).
- **Never** initialize a signal Series with `pd.Series(None, index=..., dtype=object)` — this stores `float NaN`. Use `signal_series(df.index)` from `patterns._candle` instead, which returns `pd.Series([None] * len(df), index=df.index, dtype=object)`. Assign with `result[mask.fillna(False)] = "signal"`.
- **Confirmation-delayed signals** (`patterns/confirmation.py`): Patterns like hanging man, shooting star, and doji require confirmation from the next session. The `confirmed_signal()` engine delays the signal to the confirmation bar (no lookahead). Confirmation types: `close_below_body`, `close_above_body`, `bearish_candle`, `bullish_candle`, `gap_down`, `gap_up`, `opposite_candle`.
- **Pattern scoring** (`patterns/scoring.py`): Each pattern signal gets a 0.0–1.0 strength score combining shape quality (50%), volume confirmation (20%), and trend strength (30%). Shape scorers are registered in `_SHAPE_SCORERS`; use `pattern_strength(df, name, signal)` for the composite score.
- **Price targets** (`targets.py`): Three methods — (1) consolidation box breakouts (target = breakout ± box height), (2) S/R zones from pattern clusters within 0.3% margin, (3) flag/pennant continuation (target = breakout + pole height). All are causal.
- **Match tolerance** for "similar price" patterns (tweezers, counterattack, three mountains/rivers, separating lines) is parameterized via `match_tolerance` in `run_all_patterns()`. Intraday (5-min) uses 0.001 (0.1%); daily charts use function defaults (0.001 for most, 0.003 for three_mountains/rivers).
- **Twelve Data API key** stored in `.env` (gitignored): `TWELVEDATA_API_KEY=your_key`. Read by `plot_common.load_api_key()`.
