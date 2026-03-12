"""Utilities for loading and normalizing OHLCV data."""

import json
import pandas as pd


def load_json(path: str) -> pd.DataFrame:
    """Load an OHLCV JSON file into a DataFrame.

    Returns a DataFrame with a DatetimeIndex and float columns:
    open, high, low, close, volume.
    """
    with open(path) as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    return df
