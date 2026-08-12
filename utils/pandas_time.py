"""Warning-free calendar period labels for pandas research aggregates."""

from __future__ import annotations

import pandas as pd


def sunday_week_end_labels(index: pd.Index) -> pd.DatetimeIndex:
    """Return the Sunday closing label for every timestamp in ``index``."""

    selected = pd.DatetimeIndex(index)
    return selected.normalize() + pd.to_timedelta(
        (6 - selected.dayofweek) % 7,
        unit="D",
    )
