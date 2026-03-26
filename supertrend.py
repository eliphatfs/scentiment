"""Supertrend indicator and backtest.

Implements the classic Supertrend indicator (ATR-based trailing stop that
flips between uptrend and downtrend) and backtests long/short signals on
SPY 2021-2022.

Pine Script reference: @version=4 Supertrend by KivancOzbilgic.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_json


# ---------------------------------------------------------------------------
# Supertrend computation
# ---------------------------------------------------------------------------

def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    use_true_atr: bool = True,
) -> pd.DataFrame:
    """Compute the Supertrend indicator.

    Parameters
    ----------
    df : DataFrame
        OHLCV data with columns: open, high, low, close.
    period : int
        ATR lookback period.
    multiplier : float
        ATR multiplier for band width.
    use_true_atr : bool
        If True, use Wilder's ATR (EMA of TR). If False, use SMA of TR.

    Returns
    -------
    DataFrame with columns:
        trend      : 1 (up) or -1 (down)
        upper_band : trailing support (valid when trend == 1)
        lower_band : trailing resistance (valid when trend == -1)
        buy_signal : True on bars where trend flips to 1
        sell_signal: True on bars where trend flips to -1
    """
    hl2 = (df["high"] + df["low"]) / 2

    # True range
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    if use_true_atr:
        atr = tr.ewm(span=period, adjust=False).mean()
    else:
        atr = tr.rolling(period).mean()

    # Raw bands
    basic_up = hl2 - multiplier * atr
    basic_dn = hl2 + multiplier * atr

    n = len(df)
    up = np.empty(n)
    dn = np.empty(n)
    trend = np.ones(n, dtype=int)

    up[0] = basic_up.iloc[0]
    dn[0] = basic_dn.iloc[0]

    close = df["close"].values

    for i in range(1, n):
        # Upper band (support): ratchet up if price stays above
        up[i] = basic_up.iloc[i]
        if close[i - 1] > up[i - 1]:
            up[i] = max(up[i], up[i - 1])

        # Lower band (resistance): ratchet down if price stays below
        dn[i] = basic_dn.iloc[i]
        if close[i - 1] < dn[i - 1]:
            dn[i] = min(dn[i], dn[i - 1])

        # Trend direction
        if trend[i - 1] == -1 and close[i] > dn[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and close[i] < up[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    result = pd.DataFrame({
        "trend": trend,
        "upper_band": up,
        "lower_band": dn,
    }, index=df.index)

    result["buy_signal"] = (result["trend"] == 1) & (result["trend"].shift(1) == -1)
    result["sell_signal"] = (result["trend"] == -1) & (result["trend"].shift(1) == 1)

    return result


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest_supertrend(
    df: pd.DataFrame,
    st: pd.DataFrame,
    initial_capital: float = 100_000.0,
    spread: float = 0.01,
    commission_rate: float = 1e-4,
) -> tuple[list[dict], pd.DataFrame]:
    """Backtest long-only Supertrend strategy.

    Entry on buy_signal (trend flips up), exit on sell_signal (trend flips
    down).  Executes at the signal bar's close.

    Returns (trades, equity_df).
    """
    capital = initial_capital
    position = None  # {entry_date, entry_price, shares}
    trades = []
    equity_records = []

    for i in range(len(df)):
        date = df.index[i]
        bar = df.iloc[i]
        row = st.iloc[i]

        # Exit on sell signal
        if position is not None and row["sell_signal"]:
            exit_price = bar["close"] - spread
            exit_price -= exit_price * commission_rate
            pnl = (exit_price - position["entry_price"]) * position["shares"]
            capital += position["shares"] * exit_price
            trades.append({
                "entry_date": position["entry_date"],
                "entry_price": position["entry_price"],
                "exit_date": date,
                "exit_price": exit_price,
                "shares": position["shares"],
                "pnl": pnl,
                "exit_reason": "sell_signal",
            })
            position = None

        # Enter on buy signal
        if position is None and row["buy_signal"]:
            entry_price = bar["close"] + spread
            entry_price += entry_price * commission_rate
            shares = int(capital / entry_price)
            if shares > 0:
                position = {
                    "entry_date": date,
                    "entry_price": entry_price,
                    "shares": shares,
                }
                capital -= shares * entry_price

        # Mark to market
        mark = capital
        if position is not None:
            mark += position["shares"] * bar["close"]
        equity_records.append({"date": date, "equity": mark})

    # Close open position at end
    if position is not None:
        last = df.iloc[-1]
        exit_price = last["close"] - spread
        exit_price -= exit_price * commission_rate
        pnl = (exit_price - position["entry_price"]) * position["shares"]
        capital += position["shares"] * exit_price
        trades.append({
            "entry_date": position["entry_date"],
            "entry_price": position["entry_price"],
            "exit_date": df.index[-1],
            "exit_price": exit_price,
            "shares": position["shares"],
            "pnl": pnl,
            "exit_reason": "end_of_data",
        })

    equity_df = pd.DataFrame(equity_records)
    if len(equity_df):
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]

    return trades, equity_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(trades, equity_df, initial_capital):
    if not trades:
        print("  No trades.")
        return

    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / n

    final_eq = equity_df["equity"].iloc[-1]
    total_ret = (final_eq - initial_capital) / initial_capital
    max_dd = equity_df["drawdown"].min()

    n_days = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days
    annual_ret = (1 + total_ret) ** (365 / max(n_days, 1)) - 1

    daily_ret = equity_df["equity"].pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print(f"  Trades         : {n}")
    print(f"  Wins           : {len(wins)} ({win_rate:.1%})")
    print(f"  Losses         : {len(losses)}")
    print(f"  Avg win P&L    : ${avg_win:,.2f}")
    print(f"  Avg loss P&L   : ${avg_loss:,.2f}")
    print(f"  Total P&L      : ${total_pnl:,.2f}")
    print(f"  Final equity   : ${final_eq:,.2f}")
    print(f"  Total return   : {total_ret:.2%}")
    print(f"  Annual return  : {annual_ret:.2%}")
    print(f"  Max drawdown   : {max_dd:.2%}")
    print(f"  Sharpe ratio   : {sharpe:.2f}")
    print(f"  Profit factor  : {pf:.2f}")

    print(f"\n  {'Entry':12s} {'Exit':12s} {'Entry$':>8s} {'Exit$':>8s} "
          f"{'P&L':>10s} {'Reason'}")
    print(f"  {'-'*70}")
    for t in trades:
        print(f"  {str(t['entry_date'].date()):12s} {str(t['exit_date'].date()):12s} "
              f"${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"${t['pnl']:>9.2f} {t['exit_reason']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_full = load_json("data/spy.json")
    df = df_full["2021":"2022"].copy()
    print(f"SPY data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} bars)")

    # Compute Supertrend
    st = supertrend(df, period=10, multiplier=3.0)
    n_buys = st["buy_signal"].sum()
    n_sells = st["sell_signal"].sum()
    print(f"Supertrend(10, 3.0): {n_buys} buy signals, {n_sells} sell signals")

    # Backtest
    initial_capital = 100_000.0
    trades, equity = backtest_supertrend(df, st, initial_capital=initial_capital)

    print(f"\n{'='*60}")
    print(f"  Supertrend Backtest — SPY 2021-2022")
    print(f"{'='*60}")
    print_report(trades, equity, initial_capital)

    # Buy-and-hold benchmark
    bh_shares = int(initial_capital / df.iloc[0]["close"])
    bh_cash = initial_capital - bh_shares * df.iloc[0]["close"]
    benchmark = pd.DataFrame({
        "date": df.index,
        "equity": bh_cash + bh_shares * df["close"].values,
    })
    bh_ret = (benchmark["equity"].iloc[-1] / benchmark["equity"].iloc[0]) - 1

    # Alpha / beta
    merged = equity[["date", "equity"]].merge(
        benchmark[["date", "equity"]], on="date", suffixes=("_st", "_bh"),
    )
    strat_ret = merged["equity_st"].pct_change().dropna()
    bench_ret = merged["equity_bh"].pct_change().dropna()
    cov = np.cov(strat_ret, bench_ret)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
    alpha_daily = strat_ret.mean() - beta * bench_ret.mean()
    alpha_annual = alpha_daily * 252
    strat_sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    bench_sharpe = bench_ret.mean() / bench_ret.std() * np.sqrt(252) if bench_ret.std() > 0 else 0
    strat_total = (merged["equity_st"].iloc[-1] / merged["equity_st"].iloc[0]) - 1
    bench_total = (merged["equity_bh"].iloc[-1] / merged["equity_bh"].iloc[0]) - 1

    print(f"\n  --- vs Buy & Hold ---")
    print(f"                        Strategy    Benchmark")
    print(f"  Total return      : {strat_total:>10.2%}   {bench_total:>10.2%}")
    print(f"  Sharpe ratio      : {strat_sharpe:>10.2f}   {bench_sharpe:>10.2f}")
    print(f"  Alpha (annualized): {alpha_annual:>10.4f}")
    print(f"  Beta              : {beta:>10.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.08})

    # Top panel: price + supertrend bands
    ax1.plot(df.index, df["close"], color="#555", linewidth=0.8, label="Close")

    # Uptrend band (green)
    up_band = pd.Series(st["upper_band"].values, index=df.index)
    up_band[st["trend"] != 1] = np.nan
    ax1.plot(df.index, up_band, color="#27ae60", linewidth=1.5, label="Supertrend (up)")

    # Downtrend band (red)
    dn_band = pd.Series(st["lower_band"].values, index=df.index)
    dn_band[st["trend"] != -1] = np.nan
    ax1.plot(df.index, dn_band, color="#e74c3c", linewidth=1.5, label="Supertrend (down)")

    # Fill between price and band
    ax1.fill_between(df.index, df["close"], up_band,
                     where=st["trend"].values == 1,
                     alpha=0.08, color="#27ae60")
    ax1.fill_between(df.index, df["close"], dn_band,
                     where=st["trend"].values == -1,
                     alpha=0.08, color="#e74c3c")

    # Buy/sell markers
    buy_dates = df.index[st["buy_signal"]]
    sell_dates = df.index[st["sell_signal"]]
    ax1.scatter(buy_dates, df.loc[buy_dates, "close"], marker="^",
                color="#27ae60", s=80, zorder=5, label="Buy")
    ax1.scatter(sell_dates, df.loc[sell_dates, "close"], marker="v",
                color="#e74c3c", s=80, zorder=5, label="Sell")

    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"Supertrend(10, 3.0) — SPY 2021-2022", fontsize=12)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # Bottom panel: equity curves
    strat_pct = (equity["equity"] / equity["equity"].iloc[0] - 1) * 100
    bench_pct = (benchmark["equity"] / benchmark["equity"].iloc[0] - 1) * 100

    ax2.plot(equity["date"], strat_pct.values, linewidth=1.2,
             color="#2980b9", label="Supertrend strategy")
    ax2.plot(benchmark["date"], bench_pct.values, linewidth=1,
             color="#95a5a6", linestyle="--", label="Buy & Hold")
    ax2.fill_between(equity["date"], strat_pct.values, bench_pct.values,
                     where=strat_pct.values > bench_pct.values,
                     alpha=0.15, color="#27ae60")
    ax2.fill_between(equity["date"], strat_pct.values, bench_pct.values,
                     where=strat_pct.values <= bench_pct.values,
                     alpha=0.15, color="#e74c3c")
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax2.set_ylabel("Return (%)")
    ax2.set_xlabel("Date")
    ax2.set_title(
        f"alpha={alpha_annual:.3f}  beta={beta:.3f}  "
        f"Sharpe={strat_sharpe:.2f} vs {bench_sharpe:.2f}",
        fontsize=10,
    )
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = "supertrend_spy_2021_2022.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved → {out}")
