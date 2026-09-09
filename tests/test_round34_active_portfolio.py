from __future__ import annotations

from config.settings import Settings
from portfolio.active_allocator import (
    WalletState,
    build_active_portfolio_plan,
)


def _packet(*, qualified: bool = False, survivor: int = 0):
    def market(score, h1, h4):
        return {
            "mtf": {"alignment_score_01": 0.8},
            "timeframes": {
                "15m": {
                    "close": 100,
                    "technical_score": 0.2,
                    "data_quality": {
                        "fresh": True,
                        "provenance": {"status": "READY"},
                    },
                },
                "1h": {
                    "close": 100,
                    "technical_score": h1,
                    "data_quality": {
                        "fresh": True,
                        "provenance": {"status": "READY"},
                    },
                },
                "4h": {
                    "close": 100,
                    "technical_score": h4,
                    "data_quality": {
                        "fresh": True,
                        "provenance": {"status": "READY"},
                    },
                },
                "1d": {
                    "close": 100,
                    "technical_score": 0.4,
                    "data_quality": {
                        "fresh": True,
                        "provenance": {"status": "READY"},
                    },
                },
            },
        }
    return {
        "markets": ["BTC-EUR", "ETH-EUR"],
        "canonical_context": {
            "market_context": {
                "BTC-EUR": market(0.0, 0.5, 0.6),
                "ETH-EUR": market(0.0, 0.4, 0.5),
            },
            "research_evidence": {"passed_gate_count": survivor},
        },
        "swing_advisory": {
            "selector_policy": {
                "qualified": qualified,
                "mean_stressed_net_bps": 20 if qualified else None,
            },
            "market_advisory": {
                "BTC-EUR": {
                    "descriptive_advisory": {"score": 0.6},
                    "fundamentals_cmc": {"status": "READY"},
                    "news": {"status": "READY", "confidence": 0.5},
                    "prospective_selector": {
                        "status": "READY" if qualified else "COLLECTING",
                        "passes": qualified,
                        "action": "SELECT_24H" if qualified else "ABSTAIN",
                    },
                },
                "ETH-EUR": {
                    "descriptive_advisory": {"score": 0.4},
                    "fundamentals_cmc": {"status": "READY"},
                    "news": {"status": "READY", "confidence": 0.2},
                    "prospective_selector": {
                        "status": "READY" if qualified else "COLLECTING",
                        "passes": qualified,
                        "action": "SELECT_24H" if qualified else "ABSTAIN",
                    },
                },
            },
        },
    }


def test_paper_allocator_actively_allocates_without_live_authority():
    settings = Settings.load(create_directories=False)
    wallet = WalletState(2000.0, 2000.0, ())
    plan = build_active_portfolio_plan(_packet(), wallet, settings, mode="paper")
    assert plan["target_exposure_eur"] > 0
    assert any(row["action"] == "ENTRY" for row in plan["decisions"])
    assert plan["canonical_live_targets"] == []
    assert plan["authority"]["live_orders_submitted"] == 0


def test_live_plan_requires_selector_and_research_survivor():
    settings = Settings.load(create_directories=False)
    wallet = WalletState(2000.0, 2000.0, ())
    blocked = build_active_portfolio_plan(_packet(), wallet, settings, mode="live-plan")
    assert blocked["canonical_live_targets"] == []
    assert any(
        "SELECTOR_POLICY_NOT_QUALIFIED" in row["live_blockers"]
        for row in blocked["decisions"]
        if row["action"] in {"ENTRY", "ADD"}
    )


def test_live_plan_uses_level2_caps_when_evidence_is_ready():
    settings = Settings.load(create_directories=False)
    wallet = WalletState(2000.0, 2000.0, ())
    plan = build_active_portfolio_plan(
        _packet(qualified=True, survivor=1),
        wallet,
        settings,
        mode="live-plan",
    )
    assert plan["limits"]["target_exposure_cap_eur"] == 75.0
    assert plan["limits"]["target_position_cap_eur"] == 25.0
    assert plan["canonical_live_targets"]


def test_live_plan_can_construct_risk_reducing_target():
    settings = Settings.load(create_directories=False)
    from portfolio.active_allocator import WalletPosition

    packet = _packet(qualified=True, survivor=1)
    packet["swing_advisory"]["market_advisory"]["BTC-EUR"]["descriptive_advisory"]["score"] = -0.5
    packet["canonical_context"]["market_context"]["BTC-EUR"]["timeframes"]["1h"]["technical_score"] = -0.6
    packet["canonical_context"]["market_context"]["BTC-EUR"]["timeframes"]["4h"]["technical_score"] = -0.6
    wallet = WalletState(
        2000.0,
        1900.0,
        (WalletPosition("BTC-EUR", 1.0, 100.0),),
    )
    plan = build_active_portfolio_plan(
        packet,
        wallet,
        settings,
        mode="live-plan",
    )
    btc = next(row for row in plan["decisions"] if row["market"] == "BTC-EUR")
    assert btc["action"] in {"REDUCE", "FULL_EXIT"}
    assert btc["live_plan_eligible"] is True
    assert any(
        target["portfolio_decision"]["target"]["market"] == "BTC-EUR"
        for target in plan["canonical_live_targets"]
    )
