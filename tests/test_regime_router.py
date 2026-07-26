from __future__ import annotations

import numpy as np
import pandas as pd

from research.portfolio_selection import RotationPortfolioPolicy
from research.regime_router import (
    GENESIS_HASH,
    MarketRegime,
    RegimeRouterPolicy,
    RouterMode,
    SleeveStyle,
    StrategySleeve,
    append_router_decision,
    apply_regime_hysteresis,
    audit_router_decision_chain,
    classify_latest_regime,
    route_approved_sleeves,
)


def _frames(
    *,
    rows: int = 700,
    daily_log_return: float = 0.004,
) -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2021-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )
    result: dict[str, pd.DataFrame] = {}
    for offset, market in enumerate(
        ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    ):
        returns = np.full(
            rows,
            daily_log_return * (1.0 + offset * 0.02),
        )
        close = (100.0 + offset * 10.0) * np.exp(
            np.cumsum(returns)
        )
        open_ = np.r_[close[0], close[:-1]]
        result[market] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.0005,
                "low": np.minimum(open_, close) * 0.9995,
                "close": close,
                "volume": np.full(rows, 10_000.0),
            },
            index=index,
        )
    return result


def _portfolio_policy() -> RotationPortfolioPolicy:
    return RotationPortfolioPolicy(
        allowed_markets=(
            "BTC-EUR",
            "ETH-EUR",
            "SOL-EUR",
            "LINK-EUR",
        ),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=200,
    )


def _sleeve(
    strategy_id: str,
    *,
    style: SleeveStyle = SleeveStyle.TREND,
    live_ready: bool = False,
) -> StrategySleeve:
    return StrategySleeve(
        strategy_id=strategy_id,
        family=f"{strategy_id}_FAMILY",
        style=style,
        strategy_dna_hash=(strategy_id[0].lower() * 64),
        research_pass=True,
        forward_pass=True,
        shadow_candidate_permitted=True,
        paper_candidate_permitted=True,
        live_ready=live_ready,
    )


def test_latest_regime_uses_closed_causal_trend_features() -> None:
    result = classify_latest_regime(
        _frames(),
        portfolio_policy=_portfolio_policy(),
        router_policy=RegimeRouterPolicy(),
    )
    assert result["raw_regime"] == MarketRegime.TREND_RISK_ON
    assert result["features"]["btc_close"] > result["features"][
        "btc_ema200"
    ]
    assert result["feature_causality"].endswith("NEXT_OPEN")


def test_risk_off_switch_is_immediate() -> None:
    result = classify_latest_regime(
        _frames(daily_log_return=-0.004),
        portfolio_policy=_portfolio_policy(),
        router_policy=RegimeRouterPolicy(),
    )
    assert result["raw_regime"] == MarketRegime.RISK_OFF
    state = apply_regime_hysteresis(
        MarketRegime.RISK_OFF,
        previous={
            "active_regime": MarketRegime.TREND_RISK_ON,
            "pending_regime": None,
            "pending_count": 0,
        },
        policy=RegimeRouterPolicy(),
    )
    assert state["active_regime"] == MarketRegime.RISK_OFF
    assert state["transition"] == "IMMEDIATE_RISK_OFF"


def test_risk_on_needs_three_closed_confirmations() -> None:
    policy = RegimeRouterPolicy()
    state: dict[str, object] = {}
    for expected_count in (1, 2):
        state = apply_regime_hysteresis(
            MarketRegime.TREND_RISK_ON,
            previous=state,
            policy=policy,
        )
        assert state["active_regime"] == MarketRegime.UNCERTAIN
        assert state["pending_count"] == expected_count
    state = apply_regime_hysteresis(
        MarketRegime.TREND_RISK_ON,
        previous=state,
        policy=policy,
    )
    assert state["active_regime"] == MarketRegime.TREND_RISK_ON
    assert state["transition"] == "RISK_ON_HYSTERESIS_CONFIRMED"


def test_unapproved_sleeves_route_fully_to_cash() -> None:
    rejected = StrategySleeve(
        strategy_id="REJECTED",
        family="REJECTED_FAMILY",
        style=SleeveStyle.TREND,
        strategy_dna_hash="a" * 64,
        research_pass=False,
        forward_pass=False,
        shadow_candidate_permitted=False,
        paper_candidate_permitted=False,
        live_ready=False,
    )
    route = route_approved_sleeves(
        [rejected],
        active_regime=MarketRegime.TREND_RISK_ON,
        mode=RouterMode.RESEARCH_OBSERVER,
        policy=RegimeRouterPolicy(),
    )
    assert route["status"] == "CASH_ONLY_NO_APPROVED_STRATEGIES"
    assert route["allocations"] == {}
    assert route["cash_fraction"] == 1.0
    assert route["orders_generated"] == 0


def test_approved_sleeves_respect_40_20_60_caps() -> None:
    route = route_approved_sleeves(
        [_sleeve("ALPHA"), _sleeve("BETA")],
        active_regime=MarketRegime.TREND_RISK_ON,
        mode=RouterMode.PAPER,
        policy=RegimeRouterPolicy(),
    )
    assert route["status"] == "APPROVED_STRATEGIES_ROUTED_ORDERLESS"
    assert route["allocations"] == {"ALPHA": 0.20, "BETA": 0.20}
    assert route["total_exposure"] == 0.40
    assert route["cash_fraction"] == 0.60
    assert route["orders_generated"] == 0


def test_live_mode_fails_closed_without_live_approval() -> None:
    route = route_approved_sleeves(
        [_sleeve("ALPHA", live_ready=False)],
        active_regime=MarketRegime.TREND_RISK_ON,
        mode=RouterMode.LIVE,
        policy=RegimeRouterPolicy(),
    )
    assert route["allocations"] == {}
    assert route["cash_fraction"] == 1.0
    assert "live_ready" in route["eligibility_audit"][0][
        "failed_checks"
    ]


def test_router_decision_chain_is_append_only_and_deduplicated() -> None:
    first = append_router_decision(
        None,
        {
            "decision_at": "2026-07-25T00:00:00+00:00",
            "active_regime": "RISK_OFF",
            "allocations": {},
            "cash_fraction": 1.0,
        },
    )
    assert first["decisions"][0]["previous_hash"] == GENESIS_HASH
    duplicate = append_router_decision(
        first,
        {
            "decision_at": "2026-07-25T00:00:00+00:00",
            "active_regime": "RISK_OFF",
            "allocations": {},
            "cash_fraction": 1.0,
        },
    )
    assert duplicate["deduplicated"]
    assert duplicate["decision_count"] == 1
    second = append_router_decision(
        first,
        {
            "decision_at": "2026-07-26T00:00:00+00:00",
            "active_regime": "UNCERTAIN",
            "allocations": {},
            "cash_fraction": 1.0,
        },
    )
    assert second["decision_count"] == 2
    assert (
        second["decisions"][1]["previous_hash"]
        == first["chain_root_hash"]
    )
    audit = audit_router_decision_chain(second["decisions"])
    assert audit["status"] == "PASSED"
    tampered = [dict(row) for row in second["decisions"]]
    tampered[0]["cash_fraction"] = 0.5
    with np.testing.assert_raises_regex(
        RuntimeError,
        "REGIME_ROUTER_RECORD_HASH_BREAK",
    ):
        audit_router_decision_chain(tampered)
