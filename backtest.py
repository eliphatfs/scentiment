"""Streaming backtest engine for candlestick pattern strategies.

Processes data bar-by-bar to guarantee no future information leaks.
At each timestep, only data up to the current bar is visible.

Strategy
--------
- Long trades only.
- Entry on bullish pattern signals (confirmed where required).
- TP/SL from nearest resistance/support zones built from historical patterns.
- Position size via Kelly criterion: f = (b*p - q) / b
  where b = reward/risk, p = win probability (linear in signal strength).
- Exit on trend-reversal signals (bearish pending or confirmed) even if TP/SL
  not yet triggered.
- Bid/ask spread and commission modelled as transaction costs.

Two execution modes
-------------------
  "next_open"  — buy at the next bar's open after the signal bar
  "close_hack" — buy at the signal bar's close (same-bar execution)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from data import load_json
from patterns.reversal import (
    hammer, engulfing, dark_cloud_cover, piercing_pattern, hanging_man,
)
from patterns.stars import (
    inverted_hammer, shooting_star, morning_star, evening_star,
    morning_doji_star, evening_doji_star,
)
from patterns.doji import (
    doji_at_top, doji_at_bottom, long_legged_doji, rickshaw_man,
    gravestone_doji, tri_star,
)
from patterns.more_reversals import (
    harami, harami_cross, tweezers_top, tweezers_bottom, belt_hold,
    upside_gap_two_crows, three_black_crows, counterattack_lines,
    three_mountains, three_rivers, dumpling_top, fry_pan_bottom,
    tower_top, tower_bottom,
)
from patterns.continuation import (
    window_up, window_down, rising_three_methods, falling_three_methods,
    three_white_soldiers, separating_lines,
)
from patterns.confirmation import (
    confirmed_hanging_man, confirmed_shooting_star, confirmed_inverted_hammer,
    confirmed_doji_at_top, confirmed_doji_at_bottom, confirmed_gravestone_doji,
    CONFIRMATION_RULES,
)
from patterns.scoring import pattern_strength
from targets import pattern_sr_zones


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    spread: float = 0.01          # $0.01 half-spread (SPY typical)
    commission_rate: float = 1e-4  # 0.01% per trade
    min_win_rate: float = 0.45
    max_win_rate: float = 0.85
    sr_margin: float = 0.003       # 0.3% for S/R zone clustering
    sr_min_signals: int = 2
    max_kelly_fraction: float = 0.40  # cap Kelly
    min_reward_risk: float = 1.0   # require at least 1:1 R:R
    execution: str = "next_open"   # "next_open" or "close_hack"


# ---------------------------------------------------------------------------
# Pattern detection (streaming)
# ---------------------------------------------------------------------------

# Patterns that produce bullish signals (potential long entries)
BULLISH_PATTERNS = {
    "hammer":              hammer,
    "engulfing":           engulfing,
    "piercing_pattern":    piercing_pattern,
    "inverted_hammer":     inverted_hammer,
    "morning_star":        morning_star,
    "morning_doji_star":   morning_doji_star,
    "doji_at_bottom":      doji_at_bottom,
    "harami":              harami,
    "harami_cross":        harami_cross,
    "tweezers_bottom":     tweezers_bottom,
    "belt_hold":           belt_hold,
    "three_rivers":        three_rivers,
    "fry_pan_bottom":      fry_pan_bottom,
    "tower_bottom":        tower_bottom,
    "window_up":           window_up,
    "rising_three_methods": rising_three_methods,
    "three_white_soldiers": three_white_soldiers,
    "separating_lines":    separating_lines,
    "counterattack_lines": counterattack_lines,
}

# Patterns that produce bearish signals (exit triggers for longs)
BEARISH_PATTERNS = {
    "hanging_man":         hanging_man,
    "shooting_star":       shooting_star,
    "dark_cloud_cover":    dark_cloud_cover,
    "evening_star":        evening_star,
    "evening_doji_star":   evening_doji_star,
    "doji_at_top":         doji_at_top,
    "gravestone_doji":     gravestone_doji,
    "long_legged_doji":    long_legged_doji,
    "rickshaw_man":        rickshaw_man,
    "tri_star":            tri_star,
    "tweezers_top":        tweezers_top,
    "three_mountains":     three_mountains,
    "dumpling_top":        dumpling_top,
    "tower_top":           tower_top,
    "upside_gap_two_crows": upside_gap_two_crows,
    "three_black_crows":   three_black_crows,
    "engulfing":           engulfing,
    "harami":              harami,
    "harami_cross":        harami_cross,
    "belt_hold":           belt_hold,
    "counterattack_lines": counterattack_lines,
    "separating_lines":    separating_lines,
}

# Confirmed versions for patterns needing confirmation
CONFIRMED_BEARISH = {
    "hanging_man":    confirmed_hanging_man,
    "shooting_star":  confirmed_shooting_star,
    "doji_at_top":    confirmed_doji_at_top,
    "gravestone_doji": confirmed_gravestone_doji,
}

CONFIRMED_BULLISH = {
    "inverted_hammer": confirmed_inverted_hammer,
    "doji_at_bottom":  confirmed_doji_at_bottom,
}


def _run_patterns_on_slice(df_slice: pd.DataFrame) -> dict[str, pd.Series]:
    """Run all pattern detectors on a data slice. Returns raw signals."""
    all_patterns = {}
    all_fns = set()
    for name, fn in {**BULLISH_PATTERNS, **BEARISH_PATTERNS}.items():
        if fn not in all_fns:
            all_fns.add(fn)
            all_patterns[name] = fn(df_slice)
    return all_patterns


@dataclass
class BarAnalysis:
    """Combined analysis results for a single bar, sharing pattern runs."""
    bullish_signals: list[tuple[str, str, float]]
    bearish_signals: list[tuple[str, str, float]]
    sr_zones: pd.DataFrame


def _analyze_bar(
    df_slice: pd.DataFrame,
    config: BacktestConfig,
) -> BarAnalysis:
    """Run patterns once on the visible slice and return signals + S/R zones."""
    if len(df_slice) < 3:
        return BarAnalysis([], [], pd.DataFrame())

    all_raw = _run_patterns_on_slice(df_slice)

    # --- Extract last-bar signals ---
    bullish_signals = []
    bearish_signals = []

    for name in BULLISH_PATTERNS:
        raw_sig = all_raw.get(name)
        if raw_sig is None:
            continue
        if name in CONFIRMED_BULLISH:
            conf_sig = CONFIRMED_BULLISH[name](df_slice)
            val = conf_sig.iloc[-1]
        else:
            val = raw_sig.iloc[-1]
        if val == "bullish":
            s = pattern_strength(df_slice, name, raw_sig)
            score = s.iloc[-1] if not np.isnan(s.iloc[-1]) else 0.5
            bullish_signals.append((name, val, score))

    for name in BEARISH_PATTERNS:
        raw_sig = all_raw.get(name)
        if raw_sig is None:
            continue
        raw_val = raw_sig.iloc[-1]
        if name in CONFIRMED_BEARISH:
            conf_sig = CONFIRMED_BEARISH[name](df_slice)
            conf_val = conf_sig.iloc[-1]
            if raw_val == "bearish" or conf_val == "bearish":
                s = pattern_strength(df_slice, name, raw_sig)
                score = s.iloc[-1] if not np.isnan(s.iloc[-1]) else 0.5
                bearish_signals.append((name, "bearish", score))
        else:
            if raw_val == "bearish":
                s = pattern_strength(df_slice, name, raw_sig)
                score = s.iloc[-1] if not np.isnan(s.iloc[-1]) else 0.5
                bearish_signals.append((name, "bearish", score))

    # --- S/R zones from the same pattern run ---
    zones = pattern_sr_zones(
        df_slice, all_raw,
        margin=config.sr_margin,
        min_signals=config.sr_min_signals,
    )

    return BarAnalysis(bullish_signals, bearish_signals, zones)


def _find_nearest_levels(
    price: float,
    zones: pd.DataFrame,
) -> tuple[float | None, float | None]:
    """Find nearest support below and resistance above the given price.

    Returns (support_price, resistance_price) or None if not found.
    """
    support = None
    resistance = None

    if zones.empty:
        return support, resistance

    for _, zone in zones.iterrows():
        zone_mid = zone["zone_mid"]
        if zone_mid < price:
            if zone["zone_type"] in ("support", "both"):
                if support is None or zone_mid > support:
                    support = zone_mid
        elif zone_mid > price:
            if zone["zone_type"] in ("resistance", "both"):
                if resistance is None or zone_mid < resistance:
                    resistance = zone_mid

    return support, resistance


# ---------------------------------------------------------------------------
# Trade tracking
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    tp_price: float | None
    sl_price: float | None
    kelly_fraction: float
    signal_strength: float
    pattern_name: str
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl: float = 0.0


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> tuple[list[Trade], pd.DataFrame]:
    """Run streaming backtest on OHLCV data.

    Parameters
    ----------
    df : DataFrame
        Full OHLCV data for the backtest period.
    config : BacktestConfig
        Strategy and cost parameters.

    Returns
    -------
    trades : list[Trade]
        Completed trades with entry/exit details.
    equity : DataFrame
        Daily equity curve with columns: date, equity, drawdown.
    """
    if config is None:
        config = BacktestConfig()

    capital = config.initial_capital
    position: Trade | None = None
    trades: list[Trade] = []
    equity_records = []
    pending_entry: dict | None = None  # for next_open execution

    for t in range(len(df)):
        current_bar = df.iloc[t]
        current_date = df.index[t]
        df_visible = df.iloc[:t + 1]

        # -- Handle pending entry (next_open mode) --
        if pending_entry is not None and position is None:
            entry_price = current_bar["open"] + config.spread
            cost = entry_price * config.commission_rate
            entry_price += cost

            shares = int(
                capital * pending_entry["kelly_fraction"] / entry_price
            )
            if shares > 0:
                position = Trade(
                    entry_date=current_date,
                    entry_price=entry_price,
                    shares=shares,
                    tp_price=pending_entry["tp"],
                    sl_price=pending_entry["sl"],
                    kelly_fraction=pending_entry["kelly_fraction"],
                    signal_strength=pending_entry["strength"],
                    pattern_name=pending_entry["pattern"],
                )
                capital -= shares * entry_price
            pending_entry = None

        # -- Check TP/SL on current bar if in position --
        if position is not None:
            exited = False
            # SL check first (same-day SL takes priority per requirements)
            if position.sl_price is not None and current_bar["low"] <= position.sl_price:
                exit_price = position.sl_price - config.spread
                exit_price -= exit_price * config.commission_rate
                position.exit_date = current_date
                position.exit_price = exit_price
                position.exit_reason = "stop_loss"
                position.pnl = (exit_price - position.entry_price) * position.shares
                capital += position.shares * exit_price
                trades.append(position)
                position = None
                exited = True
            elif position.tp_price is not None and current_bar["high"] >= position.tp_price:
                exit_price = position.tp_price - config.spread
                exit_price -= exit_price * config.commission_rate
                position.exit_date = current_date
                position.exit_price = exit_price
                position.exit_reason = "take_profit"
                position.pnl = (exit_price - position.entry_price) * position.shares
                capital += position.shares * exit_price
                trades.append(position)
                position = None
                exited = True

            # -- Run combined analysis (patterns + S/R) once per bar --
        analysis = None
        if len(df_visible) >= 3:
            analysis = _analyze_bar(df_visible, config)

            # -- Check for bearish reversal signals (exit trigger) --
            if not exited and position is not None and analysis.bearish_signals:
                    exit_price = current_bar["close"] - config.spread
                    exit_price -= exit_price * config.commission_rate
                    names = ", ".join(s[0] for s in analysis.bearish_signals)
                    position.exit_date = current_date
                    position.exit_price = exit_price
                    position.exit_reason = f"reversal: {names}"
                    position.pnl = (exit_price - position.entry_price) * position.shares
                    capital += position.shares * exit_price
                    trades.append(position)
                    position = None

        # -- Look for entry signals if no position --
        if position is None and pending_entry is None and analysis is not None and len(df_visible) >= 6:
            bullish_sigs = analysis.bullish_signals

            if bullish_sigs:
                # Use the strongest signal
                best = max(bullish_sigs, key=lambda s: s[2])
                pattern_name, _, strength = best

                # S/R zones from the same analysis
                zones = analysis.sr_zones
                ref_price = current_bar["close"]
                support, resistance = _find_nearest_levels(ref_price, zones)

                # Fallback: use ATR-based levels if no zones found
                if len(df_visible) >= 14:
                    tr = pd.concat([
                        df_visible["high"] - df_visible["low"],
                        (df_visible["high"] - df_visible["close"].shift(1)).abs(),
                        (df_visible["low"] - df_visible["close"].shift(1)).abs(),
                    ], axis=1).max(axis=1)
                    atr = tr.iloc[-14:].mean()
                else:
                    atr = current_bar["high"] - current_bar["low"]

                if support is None:
                    support = ref_price - 2 * atr
                if resistance is None:
                    resistance = ref_price + 2 * atr

                # Compute reward/risk
                entry_est = ref_price + config.spread  # estimated entry cost
                reward = resistance - entry_est
                risk = entry_est - support

                if risk <= 0 or reward <= 0:
                    # Record equity and skip
                    mark = capital
                    if position is not None:
                        mark += position.shares * current_bar["close"]
                    equity_records.append({
                        "date": current_date, "equity": mark,
                    })
                    continue

                b = reward / risk  # reward/risk ratio
                if b < config.min_reward_risk:
                    mark = capital
                    if position is not None:
                        mark += position.shares * current_bar["close"]
                    equity_records.append({
                        "date": current_date, "equity": mark,
                    })
                    continue

                # Win rate from strength
                p = config.min_win_rate + (
                    config.max_win_rate - config.min_win_rate
                ) * strength
                p = min(p, config.max_win_rate)
                q = 1 - p

                # Kelly fraction
                kelly = (b * p - q) / b
                kelly = max(0, min(kelly, config.max_kelly_fraction))

                if kelly > 0:
                    if config.execution == "next_open":
                        pending_entry = {
                            "kelly_fraction": kelly,
                            "tp": resistance,
                            "sl": support,
                            "strength": strength,
                            "pattern": pattern_name,
                        }
                    else:  # close_hack
                        entry_price = current_bar["close"] + config.spread
                        entry_price += entry_price * config.commission_rate
                        shares = int(capital * kelly / entry_price)
                        if shares > 0:
                            position = Trade(
                                entry_date=current_date,
                                entry_price=entry_price,
                                shares=shares,
                                tp_price=resistance,
                                sl_price=support,
                                kelly_fraction=kelly,
                                signal_strength=strength,
                                pattern_name=pattern_name,
                            )
                            capital -= shares * entry_price

        # -- Record equity --
        mark = capital
        if position is not None:
            mark += position.shares * current_bar["close"]
        equity_records.append({"date": current_date, "equity": mark})

    # Close any remaining position at the last bar's close
    if position is not None:
        last_bar = df.iloc[-1]
        exit_price = last_bar["close"] - config.spread
        exit_price -= exit_price * config.commission_rate
        position.exit_date = df.index[-1]
        position.exit_price = exit_price
        position.exit_reason = "end_of_data"
        position.pnl = (exit_price - position.entry_price) * position.shares
        capital += position.shares * exit_price
        trades.append(position)

    equity_df = pd.DataFrame(equity_records)
    if len(equity_df) > 0:
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]

    return trades, equity_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    trades: list[Trade],
    equity: pd.DataFrame,
    config: BacktestConfig,
    label: str = "",
):
    """Print backtest summary statistics."""
    if label:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

    print(f"\n  Execution mode : {config.execution}")
    print(f"  Initial capital: ${config.initial_capital:,.2f}")
    print(f"  Spread (half)  : ${config.spread}")
    print(f"  Commission     : {config.commission_rate:.4%}")

    if not trades:
        print("\n  No trades executed.")
        return

    n = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    win_rate = len(wins) / n if n else 0

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

    final_equity = equity["equity"].iloc[-1] if len(equity) > 0 else config.initial_capital
    total_return = (final_equity - config.initial_capital) / config.initial_capital
    max_dd = equity["drawdown"].min() if len(equity) > 0 else 0

    # Annualized metrics
    n_days = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days if len(equity) > 1 else 1
    annual_return = (1 + total_return) ** (365 / max(n_days, 1)) - 1

    # Daily returns for Sharpe
    if len(equity) > 1:
        daily_ret = equity["equity"].pct_change().dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    else:
        sharpe = 0

    print(f"\n  Trades         : {n}")
    print(f"  Wins           : {len(wins)} ({win_rate:.1%})")
    print(f"  Losses         : {len(losses)}")
    print(f"  Avg win P&L    : ${avg_win:,.2f}")
    print(f"  Avg loss P&L   : ${avg_loss:,.2f}")
    print(f"  Total P&L      : ${total_pnl:,.2f}")
    print(f"  Final equity   : ${final_equity:,.2f}")
    print(f"  Total return   : {total_return:.2%}")
    print(f"  Annual return  : {annual_return:.2%}")
    print(f"  Max drawdown   : {max_dd:.2%}")
    print(f"  Sharpe ratio   : {sharpe:.2f}")

    # Profit factor
    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    print(f"  Profit factor  : {profit_factor:.2f}")

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"\n  Exit reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:30s} {count}")

    # Trade details
    print(f"\n  {'Date':12s} {'Pattern':25s} {'Entry':>8s} {'Exit':>8s} "
          f"{'P&L':>10s} {'Kelly':>6s} {'Str':>5s} {'Reason'}")
    print(f"  {'-'*100}")
    for t in trades:
        print(
            f"  {t.entry_date.date()!s:12s} {t.pattern_name:25s} "
            f"${t.entry_price:>7.2f} ${t.exit_price:>7.2f} "
            f"${t.pnl:>9.2f} {t.kelly_fraction:>5.1%} {t.signal_strength:>5.2f} "
            f"{t.exit_reason}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_alpha_beta(
    equity: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> dict:
    """Compute alpha, beta, and comparison metrics vs buy-and-hold benchmark.

    Parameters
    ----------
    equity : DataFrame with columns date, equity
    benchmark : DataFrame with columns date, equity (buy-and-hold)

    Returns dict with alpha, beta, correlation, tracking_error, info_ratio,
    strategy/benchmark returns and Sharpe ratios.
    """
    merged = equity[["date", "equity"]].merge(
        benchmark[["date", "equity"]],
        on="date", suffixes=("_strat", "_bench"),
    )
    if len(merged) < 2:
        return {}

    strat_ret = merged["equity_strat"].pct_change().dropna()
    bench_ret = merged["equity_bench"].pct_change().dropna()

    # Beta = Cov(strat, bench) / Var(bench)
    cov = np.cov(strat_ret, bench_ret)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
    # Alpha (annualized) = annualized(strat excess over beta * bench)
    alpha_daily = strat_ret.mean() - beta * bench_ret.mean()
    alpha_annual = alpha_daily * 252
    correlation = np.corrcoef(strat_ret, bench_ret)[0, 1]

    # Tracking error and information ratio
    active_ret = strat_ret.values - bench_ret.values
    tracking_error = np.std(active_ret) * np.sqrt(252)
    info_ratio = (np.mean(active_ret) * 252) / tracking_error if tracking_error > 0 else 0

    # Individual Sharpe ratios
    strat_sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    bench_sharpe = bench_ret.mean() / bench_ret.std() * np.sqrt(252) if bench_ret.std() > 0 else 0

    # Total returns
    strat_total = (merged["equity_strat"].iloc[-1] / merged["equity_strat"].iloc[0]) - 1
    bench_total = (merged["equity_bench"].iloc[-1] / merged["equity_bench"].iloc[0]) - 1

    # Max drawdowns
    strat_peak = merged["equity_strat"].cummax()
    strat_dd = ((merged["equity_strat"] - strat_peak) / strat_peak).min()
    bench_peak = merged["equity_bench"].cummax()
    bench_dd = ((merged["equity_bench"] - bench_peak) / bench_peak).min()

    return {
        "alpha_annual": alpha_annual,
        "beta": beta,
        "correlation": correlation,
        "tracking_error": tracking_error,
        "info_ratio": info_ratio,
        "strat_sharpe": strat_sharpe,
        "bench_sharpe": bench_sharpe,
        "strat_return": strat_total,
        "bench_return": bench_total,
        "strat_max_dd": strat_dd,
        "bench_max_dd": bench_dd,
    }


def print_comparison(metrics: dict, label: str = ""):
    """Print alpha/beta comparison table."""
    if label:
        print(f"\n  --- {label} vs Buy & Hold ---")
    print(f"                        Strategy    Benchmark")
    print(f"  Total return      : {metrics['strat_return']:>10.2%}   {metrics['bench_return']:>10.2%}")
    print(f"  Sharpe ratio      : {metrics['strat_sharpe']:>10.2f}   {metrics['bench_sharpe']:>10.2f}")
    print(f"  Max drawdown      : {metrics['strat_max_dd']:>10.2%}   {metrics['bench_max_dd']:>10.2%}")
    print(f"")
    print(f"  Alpha (annualized): {metrics['alpha_annual']:>10.4f}")
    print(f"  Beta              : {metrics['beta']:>10.4f}")
    print(f"  Correlation       : {metrics['correlation']:>10.4f}")
    print(f"  Tracking error    : {metrics['tracking_error']:>10.4f}")
    print(f"  Information ratio : {metrics['info_ratio']:>10.4f}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load SPY data, filter to 2021-2022
    df_full = load_json("data/spy.json")
    df = df_full["2021":"2022"].copy()
    print(f"SPY data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} bars)")

    # --- Buy-and-hold benchmark ---
    initial_capital = 100_000.0
    bh_shares = int(initial_capital / df.iloc[0]["close"])
    bh_cash = initial_capital - bh_shares * df.iloc[0]["close"]
    benchmark = pd.DataFrame({
        "date": df.index,
        "equity": bh_cash + bh_shares * df["close"].values,
    })
    bh_return = (benchmark["equity"].iloc[-1] / benchmark["equity"].iloc[0]) - 1
    print(f"\nBuy & Hold: {bh_shares} shares @ ${df.iloc[0]['close']:.2f}"
          f" → ${benchmark['equity'].iloc[-1]:,.2f} ({bh_return:.2%})")

    # --- Run close_hack mode ---
    cfg = BacktestConfig(execution="close_hack")
    trades, equity = run_backtest(df, cfg)
    print_report(trades, equity, cfg, label="SPY 2021-2022 — close_hack")

    # Alpha/beta comparison
    metrics = compute_alpha_beta(equity, benchmark)
    if metrics:
        print_comparison(metrics, label="close_hack")

    # --- Plot equity curve with buy-and-hold overlay ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    strat_pct = (equity["equity"] / equity["equity"].iloc[0] - 1) * 100
    bench_pct = (benchmark["equity"] / benchmark["equity"].iloc[0] - 1) * 100

    ax.plot(equity["date"], strat_pct.values, linewidth=1.2,
            color="#2980b9", label="Strategy (close_hack)")
    ax.plot(benchmark["date"], bench_pct.values, linewidth=1,
            color="#95a5a6", linestyle="--", label="Buy & Hold (SPY)")
    ax.fill_between(equity["date"], strat_pct.values, bench_pct.values,
                    where=strat_pct.values > bench_pct.values,
                    alpha=0.15, color="#27ae60", label="Outperformance")
    ax.fill_between(equity["date"], strat_pct.values, bench_pct.values,
                    where=strat_pct.values <= bench_pct.values,
                    alpha=0.15, color="#e74c3c", label="Underperformance")

    for t in trades:
        color = "#27ae60" if t.pnl > 0 else "#e74c3c"
        ax.axvline(t.entry_date, color=color, alpha=0.2, linewidth=0.5)

    ax.set_ylabel("Return (%)")
    ax.set_title(
        f"close_hack | "
        f"alpha={metrics.get('alpha_annual', 0):.3f}  "
        f"beta={metrics.get('beta', 0):.3f}  "
        f"Sharpe={metrics.get('strat_sharpe', 0):.2f} vs {metrics.get('bench_sharpe', 0):.2f}  "
        f"IR={metrics.get('info_ratio', 0):.2f}",
        fontsize=10,
    )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Date")

    plt.tight_layout()
    out = "backtest_spy_2021_2022.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved → {out}")
