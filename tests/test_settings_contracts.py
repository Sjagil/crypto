from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import Settings
from core.contracts import (
    CandidateArtifact,
    CandidateLifecycle,
    EligibilityStatus,
    HistoricalCoverage,
    IntelligenceRecord,
    TimestampQuality,
)


def test_default_allowlist_is_fail_closed(isolated_settings: Settings) -> None:
    assert isolated_settings.shariah.eligibility("BTC-EUR").status is EligibilityStatus.ALLOWED
    assert (
        isolated_settings.shariah.eligibility("DOGE-EUR").status
        is EligibilityStatus.REVIEW_REQUIRED
    )
    isolated_settings.shariah.require_allowed("ETH-EUR")
    with pytest.raises(PermissionError):
        isolated_settings.shariah.require_allowed("UNKNOWN-EUR")


def test_secret_aliases_are_loaded_and_redacted(tmp_path) -> None:
    env = tmp_path / "aliases.env"
    env.write_text(
        "VENUE_A_API_KEY=unit-key-value\n"
        "VENUE_A_API_SECRET=unit-secret-value\n"
        "WM_OPERATOR_ID=1234\n",
        encoding="utf-8",
    )
    settings = Settings.load(env_file=env, create_directories=False)
    rendered = repr(settings) + settings.model_dump_json() + repr(settings.redacted_dict())
    assert settings.providers.bitvavo_operator_id == 1234
    assert "unit-key-value" not in rendered
    assert "unit-secret-value" not in rendered
    assert settings.redacted_dict()["providers"]["bitvavo_trade_api_key"] == "***REDACTED***"


def test_live_is_blocked_by_default(isolated_settings: Settings) -> None:
    failures = isolated_settings.static_live_preflight_failures()
    assert "LIVE_BLOCKED_NOT_PRODUCTION" in failures
    assert "LIVE_BLOCKED_MODE_NOT_LIVE" in failures
    assert "LIVE_BLOCKED_MISSING_ORDER_CAP" in failures


def test_default_market_data_includes_requested_intraday_timeframes(
    isolated_settings: Settings,
) -> None:
    assert {"5m", "15m", "1h", "4h", "1d"} <= set(
        isolated_settings.market_data.timeframes
    )
    assert isolated_settings.lab.deep_history_mode == "common_full_history"


def test_practical_profile_and_candidate_manifest_are_fail_closed(
    isolated_settings: Settings,
) -> None:
    profile = isolated_settings.operational
    assert profile.profile_name == "practical_spot_v1"
    assert profile.risk_per_trade == 0.0025
    assert profile.maximum_risk_per_trade == 0.005
    assert profile.maximum_total_open_risk == 0.01
    assert profile.maximum_positions == 2
    assert profile.maximum_portfolio_exposure == 0.40
    now = datetime.now(UTC)
    candidate = CandidateArtifact.create(
        candidate_id="candidate-v1",
        strategy_dna_hash="dna",
        software_version="1.0.0",
        signal_blocks=({"id": "ema", "version": "1"},),
        parameters={"fast": 20, "slow": 50},
        logic_mode="ALL",
        logic_weights={},
        exit_profile={"kind": "atr"},
        risk_profile={"base_risk": 0.0025},
        eligible_markets=("BTC-EUR",),
        supported_timeframes=("1h",),
        required_providers=("bitvavo",),
        required_context_datasets=(),
        data_hashes={"BTC-EUR:1h": "data"},
        train_period={"start": "2020", "end": "2023"},
        validation_periods=({"start": "2023", "end": "2024"},),
        final_holdout_period={"start": "2024", "end": "2025"},
        normal_cost_metrics={},
        stressed_cost_metrics={},
        double_cost_metrics={},
        walk_forward_metrics={},
        cpcv_diagnostics=None,
        monte_carlo_metrics={},
        parameter_stability_metrics={},
        asset_generalization_metrics={},
        lifecycle_state=CandidateLifecycle.SHADOW_CANDIDATE,
        created_at=now,
        expires_at=now + timedelta(days=30),
    )
    assert candidate.verify_manifest()
    assert candidate.permits_transition(CandidateLifecycle.SHADOW_ACTIVE)
    tampered = candidate.model_dump(mode="json")
    tampered["parameters"]["fast"] = 5
    with pytest.raises(ValueError, match="parameter hash mismatch"):
        CandidateArtifact.model_validate(tampered)


def test_forward_only_intelligence_cannot_be_backdated() -> None:
    observed = datetime(2025, 1, 2, tzinfo=UTC)
    with pytest.raises(ValueError):
        IntelligenceRecord(
            event_id="event",
            source="unit",
            url="https://example.test/event",
            title="Bitcoin event",
            published_at=datetime(2025, 1, 1, tzinfo=UTC),
            observed_at=observed,
            timestamp_quality=TimestampQuality.OBSERVED_ONLY,
            relevance_score=1.0,
            historical_coverage=HistoricalCoverage.FORWARD_ONLY,
            raw_hash="hash",
        )


def test_strict_research_defaults_match_formal_campaign_specification(
    isolated_settings: Settings,
) -> None:
    research = isolated_settings.research
    assert research.minimum_positive_folds == 5
    assert research.maximum_probability_of_backtest_overfitting == 0.10
    assert research.maximum_white_reality_check_pvalue == 0.10
    assert research.maximum_hansen_spa_pvalue == 0.05
    assert research.multiple_testing_bootstrap_samples >= 2_000
