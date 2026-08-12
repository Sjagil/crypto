from __future__ import annotations

from research.combinatorial_lab import (
    NORMAL_SPOT_SWING_MTF_TEMPLATES,
    BlockRole,
    SignalOperator,
    signal_block_registry,
)


def test_normal_spot_swing_templates_obey_bounded_grammar() -> None:
    registry = signal_block_registry()

    assert len(NORMAL_SPOT_SWING_MTF_TEMPLATES) == 7
    for membership in NORMAL_SPOT_SWING_MTF_TEMPLATES.values():
        blocks = [registry[block_id] for block_id in membership]
        assert sum(block.role is BlockRole.ENTRY_TRIGGER for block in blocks) == 1
        assert sum(block.role is BlockRole.REGIME_FILTER for block in blocks) <= 1
        assert sum(block.role is BlockRole.CONFIRMATION for block in blocks) <= 2


def test_bearish_daily_context_is_explicit_inverse_not_a_short_entry() -> None:
    block = signal_block_registry()["htf_1d_regime_bearish"]

    assert block.role is BlockRole.REGIME_FILTER
    assert block.operator is SignalOperator.BOOLEAN_FALSE
    assert block.feature == "htf_1d_regime_bullish"


def test_every_normal_swing_template_has_closed_higher_timeframe_context() -> None:
    for name, membership in NORMAL_SPOT_SWING_MTF_TEMPLATES.items():
        if name == "NORMAL_SWING_15M_RANGE_REVERSION":
            continue
        assert any(block_id.startswith("htf_1d_") for block_id in membership)
        assert "htf_4h_trend_bullish" in membership
