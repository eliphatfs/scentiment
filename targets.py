"""Price target computation — Nison chapter 16 techniques.

Provides three kinds of price targets:

1. **Consolidation / box breakout** — Nison ch. 16: when price breaks out of a
   sideways range (box), the target is the breakout level ± the box height.

2. **Support / resistance zones** from pattern clusters — multiple candlestick
   patterns firing at similar price levels (within a margin) form S/R zones.
   Confluence of several patterns strengthens the zone.

3. **Flag / pennant continuation** — after a sharp move (pole), a narrow
   counter-trend consolidation (flag) resolves with a target equal to the
   pole height projected from the breakout point.

All functions are causal: signals at bar T only use data available at T's close.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Consolidation box breakout targets
# ---------------------------------------------------------------------------

def consolidation_boxes(
    df: pd.DataFrame,
    min_bars: int = 10,
    max_range_pct: float = 0.03,
) -> pd.DataFrame:
    """Detect consolidation (box) regions and their breakout targets.

    A consolidation box exists where the previous ``min_bars`` bars have
    a total range (high−low) / mean(close) ≤ ``max_range_pct``.

    When the current bar's close breaks above the box high or below the
    box low, a breakout is recorded with target = breakout ± box_height.

    Parameters
    ----------
    df : DataFrame
        OHLCV data.
    min_bars : int
        Minimum number of bars in the consolidation region.
    max_range_pct : float
        Maximum range as fraction of mean close for the region to qualify.

    Returns
    -------
    DataFrame with columns:
        date, direction ('bullish'/'bearish'), breakout_price, box_high,
        box_low, box_height, target_price, box_start
    """
    records = []
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    dates = df.index

    for i in range(min_bars, len(df)):
        # Check if bars [i-min_bars, i-1] form a consolidation box
        window_high = highs[i - min_bars:i].max()
        window_low = lows[i - min_bars:i].min()
        window_mean = closes[i - min_bars:i].mean()

        if window_mean == 0:
            continue

        box_range_pct = (window_high - window_low) / window_mean
        if box_range_pct > max_range_pct:
            continue

        box_height = window_high - window_low

        # Check for breakout at bar i
        if closes[i] > window_high:
            records.append({
                "date": dates[i],
                "direction": "bullish",
                "breakout_price": closes[i],
                "box_high": window_high,
                "box_low": window_low,
                "box_height": box_height,
                "target_price": window_high + box_height,
                "box_start": dates[i - min_bars],
            })
        elif closes[i] < window_low:
            records.append({
                "date": dates[i],
                "direction": "bearish",
                "breakout_price": closes[i],
                "box_high": window_high,
                "box_low": window_low,
                "box_height": box_height,
                "target_price": window_low - box_height,
                "box_start": dates[i - min_bars],
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Support / resistance zones from pattern clusters
# ---------------------------------------------------------------------------

def pattern_sr_zones(
    df: pd.DataFrame,
    patterns: dict[str, pd.Series],
    margin: float = 0.003,
    min_signals: int = 2,
) -> pd.DataFrame:
    """Identify support/resistance zones from clusters of pattern signals.

    For each pattern signal, records the relevant price level:
      - Bullish signal → support zone at the bar's low
      - Bearish signal → resistance zone at the bar's high

    Signals within ``margin`` (relative) of each other are clustered into
    a single zone.  Zones with fewer than ``min_signals`` are dropped.

    Parameters
    ----------
    df : DataFrame
        OHLCV data.
    patterns : dict[str, Series]
        Output of pattern detection functions keyed by name.
    margin : float
        Maximum relative price difference to cluster signals (0.003 = 0.3%).
    min_signals : int
        Minimum number of signals to form a zone.

    Returns
    -------
    DataFrame with columns:
        zone_low, zone_high, zone_mid, zone_type ('support'/'resistance'/'both'),
        signal_count, pattern_names, first_date, last_date
    """
    # Collect all signal price levels
    entries = []
    for name, sig in patterns.items():
        for dt, val in sig.dropna().items():
            idx = df.index.get_loc(dt)
            if val == "bullish":
                entries.append({
                    "price": df["low"].iloc[idx],
                    "type": "support",
                    "pattern": name,
                    "date": dt,
                })
            elif val == "bearish":
                entries.append({
                    "price": df["high"].iloc[idx],
                    "type": "resistance",
                    "pattern": name,
                    "date": dt,
                })

    if not entries:
        return pd.DataFrame(columns=[
            "zone_low", "zone_high", "zone_mid", "zone_type",
            "signal_count", "pattern_names", "first_date", "last_date",
        ])

    entries.sort(key=lambda e: e["price"])

    # Greedy clustering
    clusters = []
    current_cluster = [entries[0]]

    for entry in entries[1:]:
        ref_price = current_cluster[-1]["price"]
        if ref_price == 0:
            current_cluster.append(entry)
            continue
        if abs(entry["price"] - ref_price) / ref_price <= margin:
            current_cluster.append(entry)
        else:
            clusters.append(current_cluster)
            current_cluster = [entry]
    clusters.append(current_cluster)

    # Build zone records
    records = []
    for cluster in clusters:
        if len(cluster) < min_signals:
            continue

        prices = [e["price"] for e in cluster]
        types = set(e["type"] for e in cluster)
        names = sorted(set(e["pattern"] for e in cluster))
        dates = [e["date"] for e in cluster]

        if types == {"support"}:
            zone_type = "support"
        elif types == {"resistance"}:
            zone_type = "resistance"
        else:
            zone_type = "both"

        records.append({
            "zone_low": min(prices),
            "zone_high": max(prices),
            "zone_mid": np.mean(prices),
            "zone_type": zone_type,
            "signal_count": len(cluster),
            "pattern_names": names,
            "first_date": min(dates),
            "last_date": max(dates),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Flag / pennant continuation targets
# ---------------------------------------------------------------------------

def flag_targets(
    df: pd.DataFrame,
    min_pole_bars: int = 3,
    max_pole_bars: int = 8,
    max_flag_bars: int = 15,
    min_pole_pct: float = 0.03,
    max_flag_range_pct: float = 0.5,
) -> pd.DataFrame:
    """Detect flag/pennant continuation patterns and compute price targets.

    Algorithm:
    1. Identify sharp directional moves ("poles") of ``min_pole_bars`` to
       ``max_pole_bars`` where the close changes by ≥ ``min_pole_pct``.
    2. After each pole, look for a narrow consolidation ("flag") of up to
       ``max_flag_bars`` where the range is ≤ ``max_flag_range_pct`` of the
       pole height.
    3. On breakout from the flag in the pole direction, target =
       breakout point + pole height.

    Parameters
    ----------
    df : DataFrame
        OHLCV data.
    min_pole_bars : int
        Minimum bars for the pole.
    max_pole_bars : int
        Maximum bars for the pole.
    max_flag_bars : int
        Maximum bars for the flag consolidation.
    min_pole_pct : float
        Minimum absolute % move for the pole.
    max_flag_range_pct : float
        Maximum flag range as fraction of pole height.

    Returns
    -------
    DataFrame with columns:
        date, direction ('bullish'/'bearish'), pole_height,
        breakout_price, target_price, pole_start, flag_end
    """
    records = []
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    dates = df.index
    n = len(df)

    # Scan for poles
    used_bars = set()  # avoid overlapping patterns

    for pole_len in range(min_pole_bars, max_pole_bars + 1):
        for start in range(n - pole_len):
            end = start + pole_len - 1

            if any(b in used_bars for b in range(start, end + 1)):
                continue

            move = closes[end] - closes[start]
            if closes[start] == 0:
                continue
            move_pct = abs(move) / closes[start]

            if move_pct < min_pole_pct:
                continue

            pole_height = abs(move)
            direction = "bullish" if move > 0 else "bearish"

            # Look for flag after the pole
            flag_start = end + 1
            if flag_start >= n:
                continue

            best_breakout = None
            for flag_len in range(2, max_flag_bars + 1):
                flag_end_idx = flag_start + flag_len - 1
                if flag_end_idx >= n:
                    break

                flag_high = highs[flag_start:flag_end_idx + 1].max()
                flag_low = lows[flag_start:flag_end_idx + 1].min()
                flag_range = flag_high - flag_low

                if flag_range > max_flag_range_pct * pole_height:
                    break  # flag too wide, stop looking

                # Check for breakout at the bar after the flag
                breakout_idx = flag_end_idx + 1
                if breakout_idx >= n:
                    continue

                if direction == "bullish" and closes[breakout_idx] > flag_high:
                    best_breakout = {
                        "date": dates[breakout_idx],
                        "direction": direction,
                        "pole_height": pole_height,
                        "breakout_price": closes[breakout_idx],
                        "target_price": flag_high + pole_height,
                        "pole_start": dates[start],
                        "flag_end": dates[flag_end_idx],
                    }
                elif direction == "bearish" and closes[breakout_idx] < flag_low:
                    best_breakout = {
                        "date": dates[breakout_idx],
                        "direction": direction,
                        "pole_height": pole_height,
                        "breakout_price": closes[breakout_idx],
                        "target_price": flag_low - pole_height,
                        "pole_start": dates[start],
                        "flag_end": dates[flag_end_idx],
                    }

            if best_breakout is not None:
                records.append(best_breakout)
                for b in range(start, flag_start):
                    used_bars.add(b)

    # Deduplicate by date (keep the one with the largest pole)
    if records:
        result_df = pd.DataFrame(records)
        result_df = result_df.sort_values("pole_height", ascending=False)
        result_df = result_df.drop_duplicates(subset="date", keep="first")
        result_df = result_df.sort_values("date").reset_index(drop=True)
        return result_df

    return pd.DataFrame(columns=[
        "date", "direction", "pole_height", "breakout_price",
        "target_price", "pole_start", "flag_end",
    ])
