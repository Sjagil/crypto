from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from research.generated_pair_lab import _rotation_targets, pair_compatibility
from research.simple_strategy_lab import registry_driven_signal_blocks


def test_close_derived_strategy_is_pair_compatible() -> None:
    registry = registry_driven_signal_blocks()

    result = pair_compatibility(("rsi_recovery",), registry)

    assert result["compatible"] is True
    assert result["contract"] == "CLOSE_DERIVED_SYNTHETIC_RATIO"


def test_native_volume_strategy_remains_normal_market_only() -> None:
    registry = registry_driven_signal_blocks()

    result = pair_compatibility(("relative_volume_expansion",), registry)

    assert result["compatible"] is False
    assert result["contract"] == "NORMAL_MARKETS_ONLY"
    assert any("VOLUME" in reason.upper() for reason in result["reasons"])


def test_rotation_targets_never_create_short_state() -> None:
    index = pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC")
    base = SimpleNamespace(
        entry=pd.Series(
            [False, True, True, False, False, False, False, False, False, False, False, False],
            index=index,
        ),
        exit=pd.Series(
            [False, False, False, False, False, True, False, False, False, False, False, False],
            index=index,
        ),
    )
    inverse = SimpleNamespace(
        entry=pd.Series(
            [False, False, False, False, False, False, False, False, False, True, True, False],
            index=index,
        ),
        exit=pd.Series(False, index=index),
    )

    targets = _rotation_targets(base, inverse)

    assert set(targets.unique()) <= {"BASE", "BENCHMARK", "CASH"}
    assert "SHORT" not in set(targets.unique())
    assert "BASE" in set(targets.unique())
