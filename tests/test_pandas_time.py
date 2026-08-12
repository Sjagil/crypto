from __future__ import annotations

import pandas as pd

from utils.pandas_time import sunday_week_end_labels


def test_sunday_week_end_labels_preserve_timezone_and_calendar_week() -> None:
    index = pd.DatetimeIndex(
        ["2026-08-09T12:00:00Z", "2026-08-10T12:00:00Z"]
    )

    labels = sunday_week_end_labels(index)

    assert str(labels.tz) == "UTC"
    assert labels.tolist() == [
        pd.Timestamp("2026-08-09T00:00:00Z"),
        pd.Timestamp("2026-08-16T00:00:00Z"),
    ]


def test_sunday_week_end_labels_support_naive_indices() -> None:
    index = pd.date_range("2026-08-10", periods=7, freq="D")

    labels = sunday_week_end_labels(index)

    assert labels.nunique() == 1
    assert labels[0] == pd.Timestamp("2026-08-16")
