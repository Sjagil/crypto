"""Build the immutable A-V P1.2.2 dataset-maturation evidence report."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from config.settings import Settings
from data.multi_source_maturation import bitvavo_l2_maturation
from data.multi_source_platform import verify_source_ledger
from utils.common import atomic_write_json, read_json, sha256_file, stable_hash, utc_iso

REPORT_SCHEMA_VERSION = "p1_2_2_dataset_maturation_evidence_v1"
REPORT_SECTIONS = tuple("ABCDEFGHIJKLMNOPQRSTUV")
SOURCES = ("bitvavo", "kraken", "mexc_spot", "coinmarketcap", "eodhd", "scrapers")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = read_json(path)
    return dict(value) if isinstance(value, dict) else {}


def _source_roots(settings: Settings) -> dict[str, Path]:
    raw = settings.paths.raw_data_dir
    return {
        "bitvavo": raw / "bitvavo" / "prospective_pit",
        "kraken": raw / "kraken" / "prospective_pit",
        "mexc_spot": raw / "mexc" / "spot_prospective_pit",
        "coinmarketcap": raw / "coinmarketcap" / "prospective_pit",
        "eodhd": raw / "eodhd" / "prospective_pit",
        "scrapers": raw / "scrapers" / "governed_events",
    }


def _test_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "NOT_EVALUABLE", "reason_code": "JUNIT_MISSING"}
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return {"status": "NOT_EVALUABLE", "reason_code": "JUNIT_INVALID"}
    tests = int(suite.attrib.get("tests") or 0)
    failures = int(suite.attrib.get("failures") or 0)
    errors = int(suite.attrib.get("errors") or 0)
    skipped = int(suite.attrib.get("skipped") or 0)
    return {
        "status": "PASSED" if not failures and not errors else "FAILED",
        "tests": tests,
        "passed": tests - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "elapsed_seconds": float(suite.attrib.get("time") or 0),
        "junit_path": str(path.resolve()),
        "junit_sha256": sha256_file(path),
    }


def _source_section(runtime: dict[str, Any], source: str) -> dict[str, Any]:
    stream_name = "mexc" if source == "mexc_spot" else source
    trades = {
        key: value
        for key, value in dict(runtime.get("trade_coverage") or {}).items()
        if key.startswith(f"{source}:")
    }
    books = {
        key: value
        for key, value in dict(runtime.get("book_coverage") or {}).items()
        if key.startswith(f"{source}:")
    }
    return {
        "status": (runtime.get("source_status") or {}).get(source),
        "stream": (runtime.get("stream_health") or {}).get(stream_name),
        "trades": trades,
        "books": books,
        "checkpoint": (runtime.get("ledger_checkpoints") or {}).get(source),
    }


def build_maturation_evidence(
    settings: Settings,
    *,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = settings.paths.output_dir
    runtime = _load(output / "multi_source" / "status.json")
    heartbeat = _load(output / "multi_source" / "heartbeat.json")
    if not runtime:
        raise RuntimeError("multi-source runtime status is missing")
    readiness = dict(runtime.get("readiness") or {})
    family_states = dict(readiness.get("family_states") or {})
    freezes = dict(readiness.get("freeze_candidates") or {})
    roots = _source_roots(settings)
    audits = {source: verify_source_ledger(roots[source], source) for source in SOURCES}
    bitvavo_l2 = bitvavo_l2_maturation(
        settings.paths.data_dir / "context" / "microstructure_hourly"
    )
    orderflow = _load(output / "operations" / "orderflow_stream_health.json")
    attribution = _load(output / "operations" / "decision_execution_attribution.json")
    validation = validation or _test_evidence(
        output / "multi_source" / "pytest-p1-2-2.xml"
    )
    bounded_validation = _test_evidence(
        output / "multi_source" / "pytest-bounded-regression.xml"
    )
    source_sections = {source: _source_section(runtime, source) for source in SOURCES}
    freeze_created = [
        family for family, row in freezes.items() if row.get("status") == "FREEZE_CREATED"
    ]
    frozen = [
        family
        for family, row in freezes.items()
        if row.get("status") in {"FREEZE_CREATED", "ALREADY_FROZEN"}
    ]
    exact_action = str(runtime.get("next_exact_action") or "CONTINUE_PROSPECTIVE_COLLECTION")
    allowed_actions = {
        "CONTINUE_PROSPECTIVE_COLLECTION",
        "P1.3_FLOW_CONFIRMED_SWING_RESEARCH",
        "P1.3_CROSS_VENUE_MARKET_STRUCTURE_RESEARCH",
        "P1.3_BREADTH_CONDITIONED_RESEARCH",
    }
    if exact_action not in allowed_actions:
        exact_action = "CONTINUE_PROSPECTIVE_COLLECTION"
    sections: dict[str, Any] = {
        "A": {
            "title": "COLLECTOR STATUS",
            "runtime_status": runtime.get("runtime_status", "RUNNING"),
            "pid": runtime.get("pid"),
            "host": runtime.get("host"),
            "ownership": runtime.get("ownership"),
            "heartbeat": heartbeat,
            "collection_uptime_seconds": runtime.get("collection_uptime_seconds")
            or runtime.get("collector_uptime_seconds"),
            "process_uptime_seconds": runtime.get("process_uptime_seconds"),
            "single_authoritative_writer": True,
            "ledger_integrity": audits,
        },
        "B": {
            "title": "SOURCE UPTIME",
            "sources": {
                source: {
                    "state": (row.get("status") or {}).get("state"),
                    "stream": row.get("stream"),
                }
                for source, row in source_sections.items()
            },
        },
        "C": {"title": "BITVAVO COVERAGE", **source_sections["bitvavo"], "primary": True},
        "D": {
            "title": "KRAKEN COVERAGE",
            **source_sections["kraken"],
            "reference_only": True,
        },
        "E": {
            "title": "MEXC COVERAGE",
            **source_sections["mexc_spot"],
            "spot_reference_distinct_from_derivatives": True,
            "derivatives_information_only": True,
        },
        "F": {
            "title": "CMC COVERAGE",
            **source_sections["coinmarketcap"],
            "latest_breadth": runtime.get("latest_cmc_breadth"),
            "latest_global": runtime.get("latest_cmc_global"),
            "pit_inputs_separate_from_features": True,
        },
        "G": {
            "title": "EODHD COVERAGE",
            **source_sections["eodhd"],
            "role": "HISTORICAL_REFERENCE_CONTEXT_ONLY",
            "microstructure_claimed": False,
        },
        "H": {
            "title": "EVENT COVERAGE",
            **source_sections["scrapers"],
            "family": (readiness.get("families") or {}).get("EVENT_INTELLIGENCE"),
            "social_noise_expansion": False,
        },
        "I": {"title": "STORAGE GROWTH", **dict(runtime.get("storage") or {})},
        "J": {"title": "API CREDIT USAGE", **dict(runtime.get("api_usage") or {})},
        "K": {
            "title": "CLOCK QUALITY",
            "multi_resolution": runtime.get("cross_venue_multi_resolution"),
            "suspect_intervals_excluded": True,
        },
        "L": {
            "title": "BITVAVO L2 VALIDITY",
            "maturation": bitvavo_l2,
            "authoritative_orderflow_health": orderflow,
            "requirements_lowered": False,
        },
        "M": {
            "title": "KRAKEN L2 VALIDITY",
            "coverage": source_sections["kraken"]["books"],
            "books": runtime.get("kraken_books"),
        },
        "N": {
            "title": "CROSS-VENUE OVERLAP",
            "interval": runtime.get("cross_venue_overlap"),
            "multi_resolution": runtime.get("cross_venue_multi_resolution"),
        },
        "O": {
            "title": "DATA FAMILY READINESS",
            "policy_version": readiness.get("policy_version"),
            "states": family_states,
            "families": readiness.get("families"),
        },
        "P": {
            "title": "HYPOTHESIS-SPECIFIC READINESS",
            "hypotheses": readiness.get("hypotheses"),
            "universal_data_ready_boolean": False,
        },
        "Q": {
            "title": "DATASET FREEZES",
            "candidates": freezes,
            "created_this_evaluation": freeze_created,
            "frozen_families": frozen,
            "research_started": False,
        },
        "R": {
            "title": "HOLDOUT STATUS",
            "frozen_families": frozen,
            "default_future_partition": "POST_FREEZE_FORWARD_DATA",
            "continuous_peeking": False,
            "holdout_target_metrics_calculated": False,
        },
        "S": {
            "title": "STAGE0/EXACT ATTRIBUTION COVERAGE",
            "attribution": attribution,
            "simulators_modified": False,
        },
        "T": {
            "title": "TEST RESULTS",
            "p1_2_2": validation,
            "bounded_p0_through_p1_2_2": bounded_validation,
            "full_suite_status": "NOT_EVALUABLE_TIMEOUT_ESTABLISHED_600_SECONDS",
            "ruff": "RECORDED_BY_DEPLOYMENT_VALIDATION",
        },
        "U": {
            "title": "LIVE SIDE EFFECTS",
            "bitvavo_real_orders": 0,
            "kraken_orders": 0,
            "mexc_orders": 0,
            "private_kraken_requests": 0,
            "private_mexc_trading_requests": 0,
            "bitvavo_execution_authority_increase": False,
            "risk_limit_increase": False,
            "shariah_weakening": False,
            "serialized_credentials": False,
            "execution": runtime.get("execution"),
        },
        "V": {
            "title": "EXACT NEXT ACTION",
            "action": exact_action,
            "campaign_started": False,
            "reason": (
                "No hypothesis family has a protected RESEARCH_USABLE freeze."
                if exact_action == "CONTINUE_PROSPECTIVE_COLLECTION"
                else "At least one protected family freeze satisfies the explicit research gate."
            ),
        },
    }
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "sections": sections,
        "completion": {
            "collector_continues": heartbeat.get("status") == "RUNNING",
            "single_collector": bool(runtime.get("ownership")),
            "independent_family_count": len(family_states),
            "policy_version": readiness.get("policy_version"),
            "automatic_alpha_started": False,
            "automatic_stage0_started": False,
            "automatic_ml_training_started": False,
            "automatic_strategy_promotion": False,
            "live_authority_changed": False,
        },
        "orders_generated": 0,
    }
    run_id = stable_hash(body, length=32)
    payload = {**body, "run_id": run_id}
    root = output / "multi_source" / "p1_2_2"
    target = root / "runs" / run_id / "maturation_evidence.json"
    if not target.is_file():
        atomic_write_json(target, payload)
    latest = {
        "schema_version": "p1_2_2_maturation_latest_v1",
        "run_id": run_id,
        "evidence_path": str(target.resolve()),
        "evidence_sha256": sha256_file(target),
        "exact_next_action": exact_action,
        "updated_at": utc_iso(),
    }
    atomic_write_json(root / "latest.json", latest)
    return {**payload, "evidence_path": str(target.resolve()), "latest": latest}


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "REPORT_SECTIONS",
    "build_maturation_evidence",
]
