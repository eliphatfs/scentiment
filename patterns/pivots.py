"""Pivot high/low detection following Grimes, *The Art and Science of Technical Analysis*.

Definitions
-----------
1st-order pivot high : a bar whose high is strictly higher than the bar
    immediately before it and the bar immediately after it.
1st-order pivot low  : a bar whose low is strictly lower than the bar
    immediately before and after it.
nth-order pivot high : among all (n-1)th-order pivot highs, one whose high is
    strictly higher than the previous and next (n-1)th-order pivot high
    (intervening pivot lows are ignored).
nth-order pivot low  : symmetric definition using lows.

No-lookahead guarantee
----------------------
A 1st-order pivot at bar *i* can only be confirmed once bar *i+1* closes.
A 2nd-order pivot can only be confirmed once the next 1st-order pivot appears,
and so on.  Each returned Series reflects this: the last bar(s) that cannot yet
be confirmed are always False.  Callers building real-time signals should apply
``series.shift(1)`` to read only the pivots that are confirmed as of each bar.
"""

import pandas as pd


def _first_order(values: pd.Series, is_high: bool) -> pd.Series:
    """Compute 1st-order pivots for a raw price series.

    Uses ``shift(-1)`` to peek at the next bar, which naturally produces NaN
    on the last element.  After ``fillna(False)`` the last bar is always False,
    satisfying the no-lookahead guarantee.
    """
    if is_high:
        mask = (values > values.shift(1)) & (values > values.shift(-1))
    else:
        mask = (values < values.shift(1)) & (values < values.shift(-1))
    return mask.fillna(False)


def _higher_order(values: pd.Series, prev_mask: pd.Series, is_high: bool) -> pd.Series:
    """Compute next-order pivots given the previous-order pivot mask.

    Among the subset of bars flagged by ``prev_mask``, a bar is an nth-order
    pivot if its value is strictly greater (or less) than the adjacent
    (n-1)th-order pivot values.

    The *last* (n-1)th-order pivot is always set to False because the next
    (n-1)th-order pivot — needed to confirm it — has not yet appeared.
    """
    pivot_vals = values[prev_mask]

    if len(pivot_vals) < 3:
        # Need at least three (n-1)th-order pivots to confirm any nth-order pivot.
        return pd.Series(False, index=values.index, dtype=bool)

    if is_high:
        is_nth = (pivot_vals > pivot_vals.shift(1)) & (pivot_vals > pivot_vals.shift(-1))
    else:
        is_nth = (pivot_vals < pivot_vals.shift(1)) & (pivot_vals < pivot_vals.shift(-1))

    is_nth = is_nth.fillna(False)
    # The last (n-1)th-order pivot cannot be confirmed until a future pivot arrives.
    is_nth.iloc[-1] = False

    result = pd.Series(False, index=values.index, dtype=bool)
    result[is_nth[is_nth].index] = True
    return result


def _pivots(values: pd.Series, order: int, is_high: bool) -> pd.Series:
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    mask = _first_order(values, is_high)
    for _ in range(order - 1):
        mask = _higher_order(values, mask, is_high)
    return mask


def pivot_highs(df: pd.DataFrame, order: int = 1) -> pd.Series:
    """Return a boolean Series marking nth-order pivot highs.

    ``result[i] is True`` means bar *i* is an nth-order pivot high.
    Bars that cannot yet be confirmed are False.

    Parameters
    ----------
    df : DataFrame with a ``high`` column.
    order : 1 (default), 2, or 3.
    """
    return _pivots(df["high"], order, is_high=True)


def pivot_lows(df: pd.DataFrame, order: int = 1) -> pd.Series:
    """Return a boolean Series marking nth-order pivot lows.

    ``result[i] is True`` means bar *i* is an nth-order pivot low.
    Bars that cannot yet be confirmed are False.

    Parameters
    ----------
    df : DataFrame with a ``low`` column.
    order : 1 (default), 2, or 3.
    """
    return _pivots(df["low"], order, is_high=False)
