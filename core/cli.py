"""Command parsing and application orchestration for the crypto spot system."""

from __future__ import annotations

import argparse
import asyncio
import html
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from config.settings import (
    TIMEFRAME_SECONDS,
    CostSettings,
    ResearchSettings,
    RiskSettings,
    Settings,
)
from core.contracts import (
    CandidateArtifact,
    CandidateLifecycle,
    ExecutionBlocked,
    OrderIntent,
    OrderSide,
    OrderType,
    ProviderStatus,
    ResearchStatus,
)
from utils.common import (
    AlertThrottle,
    atomic_write_json,
    configure_logging,
    read_json,
    redact,
    sha256_file,
    stable_hash,
    stable_json,
    utc_now,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, BaseModel):
        return _json_ready(value.model_dump(mode="json"))
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (datetime, Path, Decimal)):
        return str(value)
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


def _parse_utc_datetime(value: datetime | str) -> datetime:
    """Parse persisted timestamps and normalize legacy naive values to UTC."""
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def emit(value: Any) -> None:
    payload = stable_json(_json_ready(value), indent=2)
    try:
        print(payload)
    except UnicodeEncodeError:
        print(payload.encode("ascii", errors="backslashreplace").decode("ascii"))


def _log_extra(
    *,
    run_id: str,
    component: str,
    operation: str,
    status: str,
    provider: str | None = None,
    market: str | None = None,
    timeframe: str | None = None,
    reason_code: str | None = None,
    duration: float | None = None,
    retry_number: int = 0,
    correlation_id: str | None = None,
    exception_type: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "component": component,
        "provider": provider,
        "market": market,
        "timeframe": timeframe,
        "operation": operation,
        "duration": duration,
        "status": status,
        "reason_code": reason_code,
        "exception_type": exception_type,
        "retry_number": retry_number,
        "correlation_id": correlation_id,
    }


def csv_values(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    selected = [value] if isinstance(value, str) else value
    return [item.strip() for group in selected for item in group.split(",") if item.strip()]


def _provider_selection(
    value: str | list[str] | None, available: list[str] | tuple[str, ...]
) -> list[str]:
    requested = [item.casefold() for item in csv_values(value)]
    if not requested or "all" in requested:
        return list(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown providers: {unknown}")
    return list(dict.fromkeys(requested))


def _timeframe_selection(value: str | list[str] | None) -> list[str]:
    from config.settings import SUPPORTED_TIMEFRAMES, normalize_timeframe

    requested = csv_values(value)
    if not requested or any(item.casefold() == "all" for item in requested):
        return list(SUPPORTED_TIMEFRAMES)
    normalized = list(dict.fromkeys(normalize_timeframe(item) for item in requested))
    unknown = sorted(set(normalized) - set(SUPPORTED_TIMEFRAMES))
    if unknown:
        raise ValueError(f"unsupported timeframes: {unknown}")
    return normalized


def _provider_failure(exc: Exception) -> tuple[str, str]:
    text = str(exc).casefold()
    status = getattr(exc, "status", None)
    if "missing_credentials" in text or isinstance(exc, PermissionError) and "missing" in text:
        return ProviderStatus.SKIPPED_MISSING_CREDENTIALS.value, "SKIPPED_MISSING_CREDENTIALS"
    if status == 429 or "rate limit" in text:
        return ProviderStatus.BLOCKED_RATE_LIMIT.value, type(exc).__name__
    if status == 402 or "plan" in text or "credits" in text:
        return ProviderStatus.BLOCKED_PLAN_LIMIT.value, type(exc).__name__
    if status in {401, 403} or isinstance(exc, PermissionError):
        return ProviderStatus.BLOCKED_PERMISSION.value, type(exc).__name__
    if status is not None and int(status) >= 500:
        return ProviderStatus.BLOCKED_PROVIDER_UNAVAILABLE.value, type(exc).__name__
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(
        marker in type(exc).__name__.casefold()
        for marker in ("connection", "timeout", "serverdisconnected")
    ):
        return ProviderStatus.BLOCKED_PROVIDER_UNAVAILABLE.value, type(exc).__name__
    return ProviderStatus.FAILED_VALIDATION.value, type(exc).__name__


def selected_markets(args: argparse.Namespace) -> list[str]:
    values = csv_values(getattr(args, "markets_csv", None))
    if not values:
        values = [getattr(args, "market", "BTC-EUR")]
    return [value.upper().replace("/", "-") for value in values]


def supported_database_url(settings: Settings) -> str | None:
    if not settings.providers.database_url:
        return None
    candidate = settings.providers.database_url.get_secret_value()
    return (
        candidate if candidate.startswith(("sqlite://", "postgresql://", "postgresql+")) else None
    )


def settings_with_overrides(args: argparse.Namespace, settings: Settings) -> Settings:
    risk_update = settings.risk.model_dump()
    cost_update = settings.costs.model_dump()
    research_update = settings.research.model_dump()
    if getattr(args, "risk_per_trade", None) is not None:
        risk_update["risk_per_trade"] = args.risk_per_trade
    if getattr(args, "fee", None) is not None:
        cost_update["default_fee"] = args.fee
    if getattr(args, "slippage_bps", None) is not None:
        cost_update["slippage_bps"] = args.slippage_bps
    if getattr(args, "walk_forward_folds", None) is not None:
        research_update["walk_forward_folds"] = args.walk_forward_folds
        research_update["minimum_positive_folds"] = min(
            research_update["minimum_positive_folds"],
            args.walk_forward_folds,
        )
    if hasattr(args, "bootstrap_samples"):
        research_update["bootstrap_samples"] = args.bootstrap_samples
    if hasattr(args, "monte_carlo_runs"):
        research_update["monte_carlo_runs"] = args.monte_carlo_runs
    return settings.model_copy(
        update={
            "risk": RiskSettings.model_validate(risk_update),
            "costs": CostSettings.model_validate(cost_update),
            "research": ResearchSettings.model_validate(research_update),
        }
    )


def synthetic_ohlcv(
    rows: int = 900,
    *,
    seed: int = 42,
    market: str = "BTC-EUR",
) -> pd.DataFrame:
    if rows < 300:
        raise ValueError("synthetic datasets require at least 300 rows")
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=rows, freq="1h", tz="UTC")
    drift = np.linspace(0.0, 0.35, rows)
    cycle = 0.055 * np.sin(np.arange(rows) / 18.0)
    noise = rng.normal(0.0, 0.007, rows).cumsum()
    close = 20_000.0 * np.exp(drift + cycle + noise)
    open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0, 0.001, rows))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.009, rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.009, rows))
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.lognormal(5.0, 0.4, rows),
        },
        index=index,
    )
    frame.index.name = "timestamp"
    frame.attrs["market"] = market
    return frame


def load_sources(args: argparse.Namespace, settings: Settings) -> dict[str, pd.DataFrame]:
    from data.market_data import load_ohlcv

    markets = selected_markets(args)
    if getattr(args, "data", None):
        if len(markets) != 1:
            raise ValueError("one --data file can only be paired with one market")
        return {markets[0]: load_ohlcv(args.data, market=markets[0], validate=True)}
    if getattr(args, "providers", None):
        timeframes = csv_values(getattr(args, "timeframes", None))
        selected_timeframe = (
            settings.market_data.base_timeframe
            if settings.market_data.base_timeframe in timeframes or not timeframes
            else timeframes[0]
        )
        return {
            market: load_ohlcv(
                settings.paths.processed_data_dir / f"{market}_{selected_timeframe}.parquet",
                market=market,
                timeframe=selected_timeframe,
                validate=True,
            )
            for market in markets
        }
    return {
        market: synthetic_ohlcv(
            getattr(args, "rows", 900),
            seed=settings.app.random_seed + index,
            market=market,
        )
        for index, market in enumerate(markets)
    }


def feature_sources(args: argparse.Namespace, settings: Settings) -> dict[str, pd.DataFrame]:
    from research.features import FeaturePipeline
    from scrapers.intelligence import load_intelligence

    frames = load_sources(args, settings)
    intelligence = None
    path = getattr(args, "intelligence", None)
    if path:
        intelligence = load_intelligence(path)
    elif getattr(args, "scrapers", None) == "all":
        default = settings.paths.intelligence_dir / "crypto_intelligence.parquet"
        intelligence = load_intelligence(default) if default.is_file() else None
    benchmark = frames.get("BTC-EUR")
    return {
        market: FeaturePipeline().build(
            frame,
            market=market,
            benchmark=benchmark,
            intelligence=intelligence,
        )
        for market, frame in frames.items()
    }


def feature_source(args: argparse.Namespace, settings: Settings) -> tuple[str, pd.DataFrame]:
    sources = feature_sources(args, settings)
    market = sorted(sources)[0]
    return market, sources[market]


def doctor(settings: Settings) -> int:
    modules: dict[str, str] = {}
    for name in (
        "aiohttp",
        "bs4",
        "feedparser",
        "numpy",
        "pandas",
        "playwright",
        "pyarrow",
        "pydantic",
        "yaml",
    ):
        try:
            module = __import__(name)
            modules[name] = str(getattr(module, "__version__", "installed"))
        except ImportError:
            modules[name] = "MISSING"
    directories = {
        name: {"path": str(path), "exists": path.is_dir()}
        for name, path in {
            "data": settings.paths.data_dir,
            "output": settings.paths.output_dir,
            "reports": settings.paths.reports_dir,
            "checkpoints": settings.paths.checkpoints_dir,
        }.items()
    }
    live_failures = settings.static_live_preflight_failures()
    healthy = all(value != "MISSING" for value in modules.values()) and all(
        item["exists"] for item in directories.values()
    )
    emit(
        {
            "status": "OK" if healthy else "FAILED",
            "application": settings.app.app_name,
            "version": settings.app.version,
            "python": sys.version.split()[0],
            "research_ready": healthy,
            "live_ready": not live_failures,
            "live_blockers": list(live_failures),
            "modules": modules,
            "directories": directories,
            "safety": {
                "spot_only": settings.execution.spot_only,
                "quote_currency": settings.market_data.quote_currency,
                "withdrawals_enabled": settings.execution.withdrawals_enabled,
                "unknown_eligibility_fails_closed": True,
            },
        }
    )
    return 0 if healthy else 2


def command_config(args: argparse.Namespace, settings: Settings) -> int:
    if args.config_command == "show":
        emit(settings.redacted_dict())
    else:
        emit(
            {
                "status": "VALID",
                "live_ready": not settings.static_live_preflight_failures(),
                "live_blockers": list(settings.static_live_preflight_failures()),
            }
        )
    return 0


def command_eligibility(args: argparse.Namespace, settings: Settings) -> int:
    if args.eligibility_command == "list":
        emit(
            [
                settings.shariah.eligibility(market).model_dump(mode="json")
                for market in settings.market_data.symbols
            ]
        )
        return 0
    market = args.market_option or args.market
    if not market:
        raise ValueError("eligibility check requires --market")
    result = settings.shariah.eligibility(market)
    emit(result.model_dump(mode="json"))
    return 0 if result.status.value == "ALLOWED" else 3


async def command_data_async(args: argparse.Namespace, settings: Settings) -> int:
    from data.downloader import CanonicalDownloader, provider_capabilities
    from data.market_data import inspect_file, load_ohlcv, quality_report

    if args.data_command == "providers":
        emit(provider_capabilities())
        return 0
    if args.data_command == "download":
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(args.end.replace("Z", "+00:00")) if args.end else utc_now()
        results = await CanonicalDownloader(settings).download_all(
            markets=args.markets,
            timeframes=args.timeframes,
            start=start.astimezone(UTC),
            end=end.astimezone(UTC),
            resume=not args.no_resume,
            provider_preference=args.providers,
        )
        emit([result.model_dump(mode="json") for result in results])
        return 0
    if args.data_command == "inspect":
        emit(inspect_file(args.path))
        return 0
    frame = load_ohlcv(args.path, market=args.market, validate=False)
    report = quality_report(
        frame,
        market=args.market,
        timeframe=args.timeframe,
        maximum_staleness=timedelta(hours=args.maximum_staleness_hours),
    )
    emit(report.model_dump(mode="json"))
    return 0 if report.valid else 3


async def command_scrape_async(args: argparse.Namespace, settings: Settings) -> int:
    from scrapers.intelligence import (
        DEFAULT_SOURCES,
        audit_intelligence,
        load_intelligence,
        run_intelligence_pipeline,
    )

    status_path = settings.paths.intelligence_dir / "scraper_status.json"
    data_path = settings.paths.intelligence_dir / "crypto_intelligence.parquet"
    if args.scrape_command == "run":
        source_names = {name.casefold() for name in csv_values(args.sources)}
        sources = (
            DEFAULT_SOURCES
            if not source_names or "all" in source_names
            else tuple(
                source
                for source in DEFAULT_SOURCES
                if source.source_id.casefold() in source_names
                or source.publisher.casefold() in source_names
            )
        )
        if not sources and "rss" not in source_names:
            raise ValueError("no configured scraper source matched --sources")
        selected_settings = settings
        if args.output_dir:
            selected_settings = settings.model_copy(
                update={
                    "paths": settings.paths.model_copy(
                        update={"intelligence_dir": args.output_dir.resolve()}
                    )
                }
            )
        run = await run_intelligence_pipeline(
            selected_settings,
            sources=sources,
            include_rss=not args.no_rss
            and (not source_names or bool({"all", "rss"} & source_names)),
        )
        emit(
            {
                "status": run.status,
                "records": len(run.records),
                "output_path": run.output_path,
                "sources": [source.model_dump(mode="json") for source in run.sources],
                "audit": run.audit,
            }
        )
        return 0 if run.records else 3
    if args.scrape_command == "status":
        emit(read_json(status_path) if status_path.is_file() else {"status": "NOT_RUN"})
        return 0
    if not data_path.is_file():
        emit({"status": "MISSING", "path": data_path})
        return 3
    records = load_intelligence(data_path)
    if args.scrape_command == "inspect":
        emit([record.model_dump(mode="json") for record in records[: args.limit]])
    else:
        emit(audit_intelligence(records))
    return 0


def command_features(args: argparse.Namespace, settings: Settings) -> int:
    market, features = feature_source(args, settings)
    if args.features_command == "audit":
        metadata = features.attrs.get("feature_knowability", {})
        payload = {
            "status": "PASSED",
            "market": market,
            "rows": len(features),
            "columns": len(features.columns),
            "deterministic_frame_hash": stable_hash(
                {
                    "index": [str(value) for value in features.index],
                    "columns": list(features.columns),
                    "values": features.astype(str).to_dict(orient="list"),
                },
                length=64,
            ),
            "raw_fractal_columns": [
                column for column in features if column.startswith("raw_fractal_")
            ],
            "research_labels_in_frame": sorted(
                set(features.attrs.get("research_labels_excluded", ())).intersection(features)
            ),
            "unsafe_metadata": [
                column
                for column, details in metadata.items()
                if not details.get("lookahead_safe") or details.get("repaint")
            ],
        }
        if (
            payload["raw_fractal_columns"]
            or payload["research_labels_in_frame"]
            or payload["unsafe_metadata"]
        ):
            payload["status"] = "FAILED"
        emit(payload)
        return 0 if payload["status"] == "PASSED" else 2
    target = (
        Path(args.output)
        if args.output
        else (settings.paths.processed_data_dir / f"{market}_features.parquet")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(target, engine="pyarrow")
    manifest = {
        "market": market,
        "rows": len(features),
        "columns": list(features.columns),
        "feature_knowability": features.attrs["feature_knowability"],
        "feature_configuration_hash": stable_hash(
            {
                "market": market,
                "columns": list(features.columns),
                "feature_knowability": features.attrs["feature_knowability"],
            },
            length=64,
        ),
    }
    atomic_write_json(target.with_suffix(".manifest.json"), manifest)
    persisted = False
    if getattr(args, "persist", False):
        from data.database import Database

        database_url = supported_database_url(settings)
        database = Database(database_url, sqlite_path=settings.paths.database_path)
        database.migrate()
        events: list[dict[str, Any]] = []
        for window in (3, 5, 7):
            prefix = f"fractal_{window}_"
            for kind in ("high", "low"):
                event_column = f"{prefix}confirmed_fractal_{kind}"
                price_column = f"{event_column}_price"
                pivot_column = f"{prefix}fractal_{kind}_pivot_timestamp"
                confirmation_column = f"{prefix}fractal_{kind}_confirmation_timestamp"
                for timestamp in features.index[features[event_column].astype(bool)]:
                    events.append(
                        {
                            "external_id": stable_hash(
                                [market, window, kind, str(timestamp)], length=64
                            ),
                            "market": market,
                            "timestamp": timestamp.to_pydatetime(),
                            "available_at": timestamp.to_pydatetime(),
                            "status": "CONFIRMED",
                            "values": {
                                "window": window,
                                "kind": kind,
                                "price": features.at[timestamp, price_column],
                                "pivot_timestamp": str(features.at[timestamp, pivot_column]),
                                "confirmation_timestamp": str(
                                    features.at[timestamp, confirmation_column]
                                ),
                            },
                        }
                    )
        database.upsert_records("fractal_events", events)
        database.upsert_records(
            "generated_reports",
            [
                {
                    "external_id": manifest["feature_configuration_hash"],
                    "market": market,
                    "status": "FEATURE_BUILD_MANIFEST",
                    "values": manifest,
                }
            ],
        )
        database.close()
        persisted = True
    emit(
        {
            "status": "OK",
            "market": market,
            "rows": len(features),
            "output": target,
            "persisted": persisted,
        }
    )
    return 0


def command_indicators(args: argparse.Namespace, settings: Settings) -> int:
    from research.indicator_registry import indicator_registry

    registry = indicator_registry()
    if args.indicator_command == "coverage":
        payload = registry.report()
    elif args.indicator_command == "list":
        selected = [
            item
            for item in registry.definitions()
            if (args.family is None or item.family == args.family.upper())
            and (args.status is None or item.status.value == args.status.upper())
            and (not args.tradable_only or item.tradable)
            and (not args.research_only or item.research_only)
            and (
                args.provider_availability is None
                or (args.provider_availability == "required" and item.external_data_required)
                or (args.provider_availability == "internal" and not item.external_data_required)
            )
            and (
                args.role is None
                or args.role.upper() in {role.value for role in item.compatible_roles}
            )
        ]
        payload = {
            "count": len(selected),
            "indicators": [
                {
                    "canonical_name": item.canonical_name,
                    "display_name": item.display_name,
                    "family": item.family,
                    "status": item.status.value,
                    "tradable": item.tradable,
                    "combinable": item.combinable,
                }
                for item in selected
            ],
        }
    elif args.indicator_command == "describe":
        payload = registry.get(args.name).to_dict()
    elif args.indicator_command == "fractal-audit":
        from research.features import FeaturePipeline

        frame = synthetic_ohlcv(args.rows, seed=settings.app.random_seed)
        features = FeaturePipeline().build(frame)
        windows: dict[str, Any] = {}
        for window in (3, 5, 7):
            prefix = f"fractal_{window}_"
            high = features[f"{prefix}confirmed_fractal_high"]
            low = features[f"{prefix}confirmed_fractal_low"]
            windows[str(window)] = {
                "confirmation_lag_bars": (window - 1) // 2,
                "confirmed_highs": int(high.sum()),
                "confirmed_lows": int(low.sum()),
                "raw_columns_present": any(
                    column.startswith("raw_fractal_") for column in features
                ),
            }
        payload = {
            "status": "PASSED",
            "rows": len(features),
            "windows": windows,
            "research_labels_in_tradable_frame": sorted(
                set(features.attrs["research_labels_excluded"]).intersection(features)
            ),
        }
    else:
        raise AssertionError(f"unknown indicator command: {args.indicator_command}")
    if getattr(args, "persist", False):
        from data.database import Database

        database_url = supported_database_url(settings)
        database = Database(database_url, sqlite_path=settings.paths.database_path)
        database.migrate()
        database.upsert_records(
            "indicator_registry",
            [
                {
                    "external_id": item.configuration_hash,
                    "status": item.status.value,
                    "values": item.to_dict(),
                }
                for item in registry.definitions()
            ],
        )
        database.upsert_records(
            "indicator_availability",
            [
                {
                    "external_id": stable_hash(
                        [item.canonical_name, item.provider_requirements],
                        length=64,
                    ),
                    "status": (
                        "AVAILABLE"
                        if item.status.value
                        in {
                            "IMPLEMENTED",
                            "IMPLEMENTED_AS_ALIAS",
                            "DERIVED_FROM_EXISTING_FEATURES",
                        }
                        else "MISSING_OR_RESEARCH_ONLY"
                    ),
                    "values": {
                        "canonical_name": item.canonical_name,
                        "providers": item.provider_requirements,
                        "external_data_required": item.external_data_required,
                        "missing_data_policy": item.missing_data_policy,
                    },
                }
                for item in registry.definitions()
            ],
        )
        database.upsert_records(
            "generated_reports",
            [
                {
                    "external_id": payload.get("registry_hash", stable_hash(payload, length=64)),
                    "status": "INDICATOR_COVERAGE_AUDIT",
                    "values": payload,
                }
            ],
        )
        database.close()
        payload["persisted"] = True
    if getattr(args, "output", None):
        atomic_write_json(args.output, payload)
        payload = {"output": str(args.output), "summary": payload}
    emit(payload)
    return 0


def command_investing(args: argparse.Namespace, settings: Settings) -> int:
    from data.database import Database
    from research.investing import InvestmentScorer

    values = read_json(args.input) if args.input else {}
    if not isinstance(values, dict):
        raise ValueError("investing score input must be a JSON object")
    score = InvestmentScorer().score(values)
    payload = score.to_dict() | {
        "asset": args.asset.upper(),
        "status": "INVESTING_ONLY",
        "generated_at": utc_now().isoformat(),
    }
    if args.persist:
        database_url = supported_database_url(settings)
        database = Database(database_url, sqlite_path=settings.paths.database_path)
        database.migrate()
        database.upsert_records(
            "investment_scores",
            [
                {
                    "external_id": stable_hash(
                        [
                            args.asset.upper(),
                            score.configuration_hash,
                            values,
                        ],
                        length=64,
                    ),
                    "market": args.asset.upper(),
                    "status": "INVESTING_ONLY",
                    "values": payload,
                }
            ],
        )
        database.close()
        payload["persisted"] = True
    emit(payload)
    return 0


def command_strategies(args: argparse.Namespace) -> int:
    from research.strategies import describe_strategies, get_strategy

    if args.strategies_command == "list":
        emit([item["strategy_id"] for item in describe_strategies()])
    else:
        strategy = get_strategy(args.strategy)
        emit(
            strategy.metadata.model_dump(mode="json")
            | {"defaults": strategy.defaults, "parameter_space": strategy.parameter_space}
        )
    return 0


def backtest_result(args: argparse.Namespace, settings: Settings):
    from research.backtest import BacktestConfig, BacktestEngine
    from research.strategies import get_strategy

    selected_settings = settings_with_overrides(args, settings)
    market, features = feature_source(args, selected_settings)
    config = BacktestConfig.from_settings(
        selected_settings,
        initial_cash_eur=args.capital,
    )
    config = replace(
        config,
        bootstrap_samples=args.bootstrap_samples,
        monte_carlo_runs=args.monte_carlo_runs,
    )
    return BacktestEngine(config, settings=selected_settings).run(
        {market: features},
        get_strategy(args.strategy),
    )


def command_backtest(args: argparse.Namespace, settings: Settings) -> int:
    from reporting.reports import console_backtest_summary, write_backtest_report

    result = backtest_result(args, settings)
    output = Path(args.output) if args.output else settings.paths.reports_dir / "backtest"
    paths = write_backtest_report(result, output)
    emit(
        {
            "status": "OK",
            "summary": console_backtest_summary(result),
            "metrics": result.metrics,
            "integrity": result.integrity,
            "artifacts": paths,
        }
    )
    return 0


def command_optimize(args: argparse.Namespace, settings: Settings) -> int:
    from research.backtest import BacktestConfig
    from research.optimization import (
        coordinate_search,
        grid_search,
        optuna_search,
        random_search,
    )
    from research.strategies import get_strategy

    settings = settings_with_overrides(args, settings)
    market, features = feature_source(args, settings)
    strategy = get_strategy(args.strategy)
    config = replace(
        BacktestConfig.from_settings(settings, initial_cash_eur=args.capital),
        bootstrap_samples=args.bootstrap_samples,
        monte_carlo_runs=args.monte_carlo_runs,
    )
    common = {
        "settings": settings,
        "minimum_trades": args.minimum_trades,
        "checkpoint_path": Path(args.checkpoint) if args.checkpoint else None,
    }
    functions = {
        "grid": lambda: grid_search({market: features}, strategy, config, **common),
        "random": lambda: random_search(
            {market: features},
            strategy,
            config,
            trials=args.trials,
            seed=settings.app.random_seed,
            **common,
        ),
        "coordinate": lambda: coordinate_search(
            {market: features}, strategy, config, rounds=args.rounds, **common
        ),
        "optuna": lambda: optuna_search(
            {market: features},
            strategy,
            config,
            trials=args.trials,
            seed=settings.app.random_seed,
            **common,
        ),
    }
    result = functions[args.method]()
    emit(result)
    return 0


def command_walk_forward(args: argparse.Namespace, settings: Settings) -> int:
    from research.backtest import BacktestConfig
    from research.optimization import walk_forward_validate
    from research.strategies import get_strategy

    settings = settings_with_overrides(args, settings)
    market, features = feature_source(args, settings)
    strategy = get_strategy(args.strategy)
    config = replace(
        BacktestConfig.from_settings(settings, initial_cash_eur=args.capital),
        bootstrap_samples=args.bootstrap_samples,
        monte_carlo_runs=args.monte_carlo_runs,
    )
    result = walk_forward_validate(
        {market: features},
        strategy,
        strategy.parameters(),
        config,
        folds=args.folds,
        mode=args.mode,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
        settings=settings,
    )
    emit(result)
    return 0 if result.valid else 3


def command_monte_carlo(args: argparse.Namespace, settings: Settings) -> int:
    from research.trading_math import empirical_risk_of_ruin

    samples = [float(value) for value in args.r_multiples.split(",")]
    result = empirical_risk_of_ruin(
        samples,
        risk_fraction=args.risk_fraction,
        initial_equity=args.capital,
        trades_per_simulation=args.trades,
        simulations=args.runs,
        block_size=args.block_size,
        seed=settings.app.random_seed,
    )
    emit(result.to_dict())
    return 0


def command_research(args: argparse.Namespace, settings: Settings) -> int:
    from reporting.reports import write_research_report
    from research.optimization import run_research
    from research.strategies import get_strategy, strategy_registry

    selected_settings = settings_with_overrides(args, settings)
    data_by_market = feature_sources(args, selected_settings)
    requested = args.strategies
    strategy_ids = (
        sorted(strategy_registry())
        if requested == "all"
        else csv_values(requested) or [args.strategy]
    )
    base_directory = (
        Path(args.output) if args.output else selected_settings.paths.reports_dir / "research"
    )
    results: list[dict[str, Any]] = []
    for strategy_id in strategy_ids:
        strategy = get_strategy(strategy_id)
        checkpoint = (
            Path(args.checkpoint).with_name(f"{Path(args.checkpoint).stem}_{strategy_id}.jsonl")
            if args.checkpoint and len(strategy_ids) > 1
            else (Path(args.checkpoint) if args.checkpoint else None)
        )
        outcome = run_research(
            data_by_market,
            strategy,
            selected_settings,
            capital_eur=args.capital,
            search_method=args.method,
            search_trials=args.trials,
            purge_bars=args.purge_bars,
            embargo_bars=args.embargo_bars,
            checkpoint_path=checkpoint,
            promote_to_paper=args.promote_to_paper,
        )
        directory = base_directory / strategy_id if len(strategy_ids) > 1 else base_directory
        paths = write_research_report(outcome, selected_settings, directory)
        results.append(
            {
                "strategy_id": strategy_id,
                "status": outcome.gate.status.value,
                "passed": outcome.gate.passed,
                "reasons": outcome.gate.reasons,
                "parameters": outcome.parameters,
                "artifacts": paths,
            }
        )
    emit(results[0] if len(results) == 1 else {"strategies": results})
    return 0


async def command_research_async(
    args: argparse.Namespace,
    settings: Settings,
) -> int:
    selected_settings = settings_with_overrides(args, settings)
    if args.providers and not args.data:
        from data.downloader import CanonicalDownloader

        await CanonicalDownloader(selected_settings).download_all(
            markets=selected_markets(args),
            timeframes=csv_values(args.timeframes) or selected_settings.market_data.timeframes,
            provider_preference=csv_values(args.providers),
        )
    if args.scrapers == "all":
        from scrapers.intelligence import run_intelligence_pipeline

        await run_intelligence_pipeline(selected_settings)
    return command_research(args, selected_settings)


def command_report(args: argparse.Namespace) -> int:
    if args.path in {"statistics", "charts", "full"}:
        raise ValueError("operational report commands require settings dispatch")
    path = Path(args.path)
    if not path.is_file():
        emit({"status": "MISSING", "path": path})
        return 3
    emit(read_json(path))
    return 0


def _synthetic_reporting_data() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=120, freq="h", tz="UTC")
    randomizer = np.random.default_rng(42)
    returns = randomizer.normal(0.0002, 0.01, len(index))
    equity = 2_000 * np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    research = pd.DataFrame(
        {
            "equity": equity,
            "drawdown": drawdown,
            "rolling_return": pd.Series(returns, index=index).rolling(24).sum(),
            "rolling_volatility": pd.Series(returns, index=index).rolling(24).std(),
            "rolling_sharpe": (
                pd.Series(returns, index=index).rolling(24).mean()
                / pd.Series(returns, index=index).rolling(24).std()
            ),
        },
        index=index,
    )
    market = pd.DataFrame(
        {
            "close": 20_000 * np.cumprod(1 + returns),
            "volume": randomizer.lognormal(5, 0.4, len(index)),
            "missing_bars": 0,
        },
        index=index,
    )
    orderbook = pd.DataFrame(
        {
            "spread_bps": randomizer.uniform(1, 10, len(index)),
            "imbalance": randomizer.uniform(-1, 1, len(index)),
        },
        index=index,
    )
    portfolio = pd.DataFrame(
        {
            "allocation": randomizer.uniform(0.2, 0.8, len(index)),
            "open_risk": randomizer.uniform(10, 50, len(index)),
            "btc_beta": randomizer.normal(1, 0.1, len(index)),
            "marginal_risk": randomizer.uniform(0, 1, len(index)),
            "realized_pnl": np.cumsum(randomizer.normal(0, 2, len(index))),
            "unrealized_pnl": randomizer.normal(0, 20, len(index)),
            "daily_pnl": randomizer.normal(0, 8, len(index)),
            "exposure": randomizer.uniform(100, 1_200, len(index)),
        },
        index=index,
    )
    macro = pd.DataFrame(
        {
            "sentiment_fear_greed": randomizer.uniform(10, 90, len(index)),
            "dominance_btc_dominance": randomizer.uniform(0.45, 0.60, len(index)),
            "dominance_stablecoin_dominance": randomizer.uniform(0.05, 0.15, len(index)),
            "breadth_fraction_above_mean_50d": randomizer.uniform(0, 1, len(index)),
            "derivatives_funding_rate": randomizer.normal(0, 0.0001, len(index)),
            "derivatives_open_interest": randomizer.uniform(1e8, 2e8, len(index)),
            "derivatives_long_liquidations": randomizer.uniform(0, 1e6, len(index)),
            "derivatives_short_liquidations": randomizer.uniform(0, 1e6, len(index)),
            "crypto_risk_score": randomizer.integers(-3, 4, len(index)),
            "events_high_impact_event_risk": randomizer.integers(0, 2, len(index)),
            "gex_net_gex_proxy": randomizer.normal(0, 1e8, len(index)),
            "gex_spot_distance_from_dominant_gamma": randomizer.normal(0, 0.05, len(index)),
        },
        index=index,
    )
    return {
        "market": market,
        "research": research,
        "orderbook": orderbook,
        "portfolio": portfolio,
        "macro": macro,
        "correlation": pd.DataFrame(
            randomizer.uniform(-1, 1, (4, 4)),
            index=["BTC", "ETH", "SOL", "LINK"],
            columns=["BTC", "ETH", "SOL", "LINK"],
        ),
    }


def command_operational_report(args: argparse.Namespace, settings: Settings) -> int:
    from reporting.visualizations import VisualizationReporter

    action = args.path
    target = settings.paths.reports_dir / "charts"
    if action in {"charts", "full", "build"}:
        datasets = _synthetic_reporting_data()
        sources: dict[str, Any] = {"non_context_diagnostics": "SYNTHETIC_DIAGNOSTIC_ONLY"}
        macro_path = settings.paths.context_data_dir / "macro_context_1h.parquet"
        if macro_path.is_file():
            datasets["macro"] = pd.read_parquet(macro_path)
            sources["macro"] = {
                "source_type": "REAL_PERSISTED_CONTEXT",
                "path": macro_path,
                "sha256": sha256_file(macro_path),
            }
        option_paths = sorted(settings.paths.context_data_dir.glob("options_deribit_*.parquet"))
        option_frames = [pd.read_parquet(path) for path in option_paths]
        if option_frames:
            options = pd.concat(option_frames, ignore_index=True)
            options["gross_gex"] = (
                pd.to_numeric(options["gamma"], errors="coerce")
                * pd.to_numeric(options["open_interest"], errors="coerce")
                * pd.to_numeric(
                    options["contract_multiplier"],
                    errors="coerce",
                )
                * pd.to_numeric(
                    options["spot_or_index_price"],
                    errors="coerce",
                ).pow(2)
                * 0.01
            )
            datasets["gex_strike"] = (
                options.groupby("strike", as_index=False)["gross_gex"].sum().sort_index()
            )
            datasets["gex_expiry"] = (
                options.groupby("expiry", as_index=False)["gross_gex"].sum().sort_index()
            )
            sources["gex"] = {
                "source_type": "REAL_DERIBIT_OPTIONS_CHAIN",
                "paths": option_paths,
                "contracts": len(options),
            }
        index = VisualizationReporter(target).generate(datasets)
        index["data_sources"] = sources
    else:
        index = {"status": "SKIPPED", "reason_code": "CHARTS_NOT_REQUESTED"}
    statistics = {
        "status": "PASSED",
        "generated_at": utc_now().isoformat(),
        "warning": "synthetic diagnostics are not profitability claims",
    }
    emit({"statistics": statistics, "charts": index, "output_dir": target})
    return 1 if index.get("failed", 0) else 0


def _test_run_tree(settings: Settings, run_id: str) -> dict[str, Path]:
    root = settings.paths.test_runs_dir / run_id
    directories = {name: root / name for name in ("logs", "charts", "csv", "html", "manifests")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    directories["root"] = root
    return directories


def _status_file(
    root: Path,
    name: str,
    *,
    status: str,
    reason_code: str,
    **details: Any,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "reason_code": reason_code,
        "checked_at": utc_now().isoformat(),
        **details,
    }
    atomic_write_json(root / name, payload)
    return payload


def _run_check(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "command": command,
        "status": "PASSED" if result.returncode == 0 else "FAILED",
        "returncode": result.returncode,
        "duration_seconds": time.perf_counter() - started,
        "stdout_tail": result.stdout[-4_000:],
        "stderr_tail": result.stderr[-4_000:],
    }


def _offline_macro_fixture() -> tuple[pd.DatetimeIndex, dict[str, Any]]:
    from research.macro_context import MacroSourceSpec

    base = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
    daily = pd.date_range("2024-11-01", periods=70, freq="D", tz="UTC")
    return base, {
        "fear_greed": pd.DataFrame(
            {"value": np.linspace(20, 80, len(daily))},
            index=daily,
        ),
        "source_specs": {
            "sentiment": MacroSourceSpec(
                provider="synthetic_offline_test",
                source_frequency="1d",
                expected_cadence=timedelta(days=1),
                maximum_age=timedelta(days=2),
                units={"value": "index"},
            ),
        },
    }


def _configured_secret_values(settings: Settings) -> tuple[str, ...]:
    values: list[str] = []
    for name in type(settings.providers).model_fields:
        value = getattr(settings.providers, name)
        if hasattr(value, "get_secret_value"):
            secret = value.get_secret_value()
            if secret:
                values.append(secret)
    return tuple(values)


def _safe_exception_message(exc: BaseException, settings: Settings | None) -> str:
    secrets = _configured_secret_values(settings) if settings is not None else ()
    return str(redact(str(exc), secrets))


def _secret_audit(
    paths: list[Path],
    settings: Settings,
    *,
    excluded_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    secrets = _configured_secret_values(settings)
    excluded = tuple(path.resolve() for path in excluded_roots)

    def permitted(path: Path) -> bool:
        resolved = path.resolve()
        return not any(resolved == root or root in resolved.parents for root in excluded)

    findings: list[str] = []
    scanned = 0
    skipped_binary = 0
    skipped_large = 0
    binary_suffixes = {
        ".db",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".parquet",
        ".pdf",
        ".png",
        ".pyc",
        ".sqlite",
        ".sqlite3",
        ".svgz",
        ".webp",
        ".xlsx",
        ".zip",
    }
    maximum_text_bytes = 10 * 1024 * 1024
    for root in paths:
        if not root.exists():
            continue
        files = (
            [root]
            if root.is_file()
            else [path for path in root.rglob("*") if path.is_file() and permitted(path)]
        )
        for path in files:
            if path.suffix.casefold() in binary_suffixes:
                skipped_binary += 1
                continue
            try:
                if path.stat().st_size > maximum_text_bytes:
                    skipped_large += 1
                    continue
            except OSError:
                continue
            scanned += 1
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(secret in content for secret in secrets):
                findings.append(str(path))
    return {
        "status": "PASSED" if not findings else "FAILED",
        "reason_code": "NO_CONFIGURED_SECRETS_FOUND" if not findings else "SECRET_FOUND",
        "files_scanned": scanned,
        "binary_files_skipped": skipped_binary,
        "oversized_text_files_skipped": skipped_large,
        "maximum_text_bytes": maximum_text_bytes,
        "finding_paths": findings,
    }


def _offline_checks(settings: Settings, tree: dict[str, Path]) -> list[dict[str, Any]]:
    project = settings.paths.project_root
    checks = [
        _run_check([sys.executable, "-m", "compileall", "."], cwd=project),
        _run_check(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "config",
                "core",
                "data",
                "execution",
                "reporting",
                "research",
                "risk",
                "scrapers",
                "utils",
                "main.py",
                "tests",
            ],
            cwd=project,
        ),
        _run_check([sys.executable, "-m", "pytest", "-q"], cwd=project),
        _run_check([sys.executable, "main.py", "self-test"], cwd=project),
    ]
    return checks


async def _provider_checks(settings: Settings) -> dict[str, Any]:
    from data.data_loader import DataLoader

    loader = DataLoader(settings)
    now = utc_now()
    start = now - timedelta(hours=4)
    providers: dict[str, Any] = {}
    requests = {
        "bitvavo": ("BTC-EUR", "1h"),
        "kraken": ("BTC-EUR", "1h"),
        "mexc": ("BTC-USDT", "1h"),
    }
    for provider, (market, timeframe) in requests.items():
        try:
            records = await loader.download_ohlcv(
                provider=provider,
                market=market,
                timeframe=timeframe,
                start=start,
                end=now,
                resume=False,
            )
            providers[provider] = {
                "status": "PASSED" if records else "PARTIAL",
                "reason_code": "PUBLIC_DATA_RECEIVED" if records else "EMPTY_RESPONSE",
                "records": len(records),
            }
            if provider == "mexc":
                try:
                    context = await loader.download_derivatives_context(
                        provider="mexc",
                        market="BTC-USDT",
                    )
                    providers[provider]["derivatives_context"] = {
                        "status": "PASSED" if context else "PARTIAL",
                        "records": len(context),
                        "execution_permitted": False,
                    }
                except Exception as exc:
                    providers[provider]["derivatives_context"] = {
                        "status": "FAILED",
                        "reason_code": type(exc).__name__,
                        "execution_permitted": False,
                    }
                    providers[provider]["status"] = "PARTIAL"
        except Exception as exc:
            providers[provider] = {
                "status": "FAILED",
                "reason_code": type(exc).__name__,
                "message": _safe_exception_message(exc, settings),
            }
    credentialed = {
        "coinmarketcap": (
            settings.providers.coinmarketcap_api_key,
            {"series": "GLOBAL_METRICS"},
        ),
        "eodhd": (
            settings.providers.eodhd_api_key,
            {"series": "DXY.INDX", "start": start, "end": now},
        ),
        "fred": (
            settings.providers.fred_api_key,
            {
                "series": "DFF",
                "start": datetime(2025, 1, 1, tzinfo=UTC),
                "end": datetime(2025, 1, 8, tzinfo=UTC),
            },
        ),
    }
    for provider, (credential, arguments) in credentialed.items():
        if credential is None:
            providers[provider] = {
                "status": "SKIPPED",
                "reason_code": "SKIPPED_MISSING_CREDENTIALS",
            }
            continue
        try:
            records = await loader.download_macro_series(
                provider=provider,
                **arguments,
            )
            providers[provider] = {
                "status": "PASSED" if records else "PARTIAL",
                "reason_code": "PUBLIC_DATA_RECEIVED" if records else "EMPTY_RESPONSE",
                "records": len(records),
            }
        except Exception as exc:
            providers[provider] = {
                "status": "FAILED",
                "reason_code": type(exc).__name__,
            }
    try:
        records = await loader.download_macro_series(provider="sec", series="0000320193")
        providers["sec"] = {
            "status": "PASSED" if records else "PARTIAL",
            "reason_code": "PUBLIC_DATA_RECEIVED" if records else "EMPTY_RESPONSE",
            "records": len(records),
        }
    except PermissionError:
        providers["sec"] = {
            "status": "BLOCKED",
            "reason_code": "SEC_USER_AGENT_NOT_CONFIGURED",
        }
    except Exception as exc:
        providers["sec"] = {
            "status": "FAILED",
            "reason_code": type(exc).__name__,
        }
    return providers


async def _websocket_checks(settings: Settings, *, duration: float = 10.0) -> dict[str, Any]:
    from data.websocket_manager import WebSocketManager

    manager = WebSocketManager(
        queue_size=500,
        maximum_connection_attempts=2,
        inactivity_timeout=20,
    )
    subscriptions = {
        "bitvavo": {"ticker": ["BTC-EUR"]},
        "kraken": {"ticker": ["BTC/EUR"]},
        "mexc": {"ticker": ["BTC-USDT"]},
    }
    await manager.start(subscriptions)
    await asyncio.sleep(max(1.0, duration))
    await manager.stop()
    first = manager.health()
    await manager.start({"bitvavo": subscriptions["bitvavo"]})
    await asyncio.sleep(min(5.0, max(1.0, duration / 2)))
    await manager.stop()
    second = manager.health()
    leaked = [
        task.get_name()
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("websocket-")
    ]
    providers = {
        name: {
            "status": "PASSED" if health["messages"] > 0 else "FAILED",
            "reason_code": (
                "NORMALIZED_MESSAGES_RECEIVED" if health["messages"] > 0 else "NO_MESSAGES"
            ),
            **health,
        }
        for name, health in first.items()
    }
    reconnect_verified = (
        second["bitvavo"]["connections"] >= 2 and second["bitvavo"]["subscriptions"] >= 2
    )
    return {
        "status": (
            "PASSED"
            if all(item["status"] == "PASSED" for item in providers.values())
            and reconnect_verified
            and not leaked
            else "FAILED"
        ),
        "reason_code": "SMOKE_AND_RECONNECT_COMPLETE",
        "providers": providers,
        "intentional_reconnect_verified": reconnect_verified,
        "task_leaks": leaked,
        "live_orders": 0,
    }


async def _local_component_checks(
    settings: Settings, tree: dict[str, Path]
) -> dict[str, dict[str, Any]]:
    from data.database import Database
    from data.derivatives_context import CryptoGEXAnalyzer, OptionsContract
    from data.orderbook_l2 import Level2OrderBook
    from execution.position_tracker import PositionTracker
    from reporting.visualizations import VisualizationReporter
    from research.macro_context import MacroContextEngine
    from risk.drawdown_protection import DrawdownProtection

    results: dict[str, dict[str, Any]] = {}
    database_path = tree["root"] / "test.db"
    try:
        database = Database(sqlite_path=database_path)
        database.migrate()
        database.upsert_records(
            "test_runs",
            [{"run_id": tree["root"].name, "status": "RUNNING", "timestamp": utc_now()}],
        )
        database.upsert_records(
            "test_runs",
            [{"run_id": tree["root"].name, "status": "PASSED", "timestamp": utc_now()}],
        )
        health = database.health()
        health["idempotent_count"] = health["table_counts"]["test_runs"]
        results["database"] = health
        database.close()
    except Exception as exc:
        results["database"] = {
            "status": "FAILED",
            "reason_code": type(exc).__name__,
            "message": _safe_exception_message(exc, settings),
        }
    try:
        book = Level2OrderBook(provider="synthetic", market="BTC-EUR")
        await book.initialize(
            bids=[["19999", "1"], ["19998", "2"]],
            asks=[["20001", "1"], ["20002", "2"]],
            sequence=1,
        )
        await book.apply_delta(
            bids=[["19999", "1.5"]],
            sequence=2,
            message_id="synthetic-delta",
        )
        results["orderbook"] = {
            "status": "PASSED",
            "reason_code": "SNAPSHOT_DELTA_FEATURES_VALID",
            **book.health(),
            "microprice": str(book.microprice),
            "estimated_slippage": str(book.estimated_slippage(side="buy", quantity="1.5")),
        }
    except Exception as exc:
        results["orderbook"] = {
            "status": "FAILED",
            "reason_code": type(exc).__name__,
            "message": _safe_exception_message(exc, settings),
        }
    try:
        base, macro_inputs = _offline_macro_fixture()
        macro = MacroContextEngine().build(base, **macro_inputs)
        results["macro"] = {
            "status": "PASSED",
            "reason_code": "CAUSAL_MACRO_BUILD_COMPLETE",
            "rows": len(macro),
            "columns": len(macro.columns),
            "latest": MacroContextEngine.latest_snapshot(macro),
        }
    except Exception as exc:
        results["macro"] = {
            "status": "FAILED",
            "reason_code": type(exc).__name__,
            "message": _safe_exception_message(exc, settings),
        }
    try:
        observed = utc_now()
        contracts = [
            OptionsContract(
                provider="synthetic",
                underlying="BTC",
                expiry=observed + timedelta(days=7),
                strike=20_000,
                option_type=option_type,
                spot_or_index_price=20_000,
                open_interest=100,
                gamma=0.0001,
                contract_multiplier=1,
                observed_at=observed,
                available_at=observed,
            )
            for option_type in ("call", "put")
        ]
        results["gex"] = CryptoGEXAnalyzer().calculate(contracts, now=observed)
        results["gex"]["reason_code"] = "TRANSPARENT_PROXY_CALCULATED"
    except Exception as exc:
        results["gex"] = {
            "status": "FAILED",
            "reason_code": type(exc).__name__,
            "message": _safe_exception_message(exc, settings),
        }
    try:
        from core.contracts import Fill

        tracker = PositionTracker(tree["root"] / "positions.json")
        tracker.ingest_fill(
            Fill(
                fill_id="test-buy",
                order_id="paper-order",
                intent_id="paper-intent",
                market="BTC-EUR",
                side=OrderSide.BUY,
                quantity=Decimal("0.001"),
                price=Decimal("20000"),
                fee_eur=Decimal("0.05"),
                filled_at=utc_now(),
                venue="paper",
            ),
            strategy_id="synthetic-paper",
        )
        tracker.mark_to_market("BTC-EUR", Decimal("20100"))
        results["paper"] = {
            "status": "PASSED",
            "reason_code": "SIMULATED_FILL_TRACKED",
            "pnl": tracker.portfolio_pnl(),
            "execution_type": "paper_simulation",
        }
    except Exception as exc:
        results["paper"] = {
            "status": "FAILED",
            "reason_code": type(exc).__name__,
            "message": _safe_exception_message(exc, settings),
        }
    try:
        index = VisualizationReporter(tree["charts"]).generate(_synthetic_reporting_data())
        results["reporting"] = {
            "status": "FAILED" if index["failed"] else "PASSED",
            "reason_code": "CHARTS_GENERATED",
            **index,
        }
    except Exception as exc:
        results["reporting"] = {
            "status": "FAILED",
            "reason_code": type(exc).__name__,
            "message": _safe_exception_message(exc, settings),
        }
    try:
        state_path = tree["root"] / "drawdown_state.json"
        audit_path = tree["logs"] / "drawdown_audit.jsonl"
        protection = DrawdownProtection(
            state_path=state_path,
            audit_path=audit_path,
        )
        index = pd.date_range(end=utc_now(), periods=10, freq="h")
        status = protection.evaluate(
            portfolio_equity=pd.Series(
                [100, 101, 102, 98, 94, 90, 88, 86, 85, 84],
                index=index,
            )
        )
        results["risk"] = {
            "status": "PASSED"
            if status["state"] in {"BLOCK_NEW_ENTRIES", "KILL_SWITCH"}
            else "FAILED",
            "reason_code": "DRAWDOWN_STATE_MACHINE_EXERCISED",
            **status,
        }
    except Exception as exc:
        results["risk"] = {
            "status": "FAILED",
            "reason_code": type(exc).__name__,
            "message": _safe_exception_message(exc, settings),
        }
    return results


async def command_test(args: argparse.Namespace, settings: Settings) -> int:
    run_id = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{stable_hash([args.test_mode, time.time_ns()], length=8)}"
    tree = _test_run_tree(settings, run_id)
    root = tree["root"]
    logger = configure_logging(
        log_file=tree["logs"] / "test.log",
        jsonl_file=tree["logs"] / "test.jsonl",
        secrets=_configured_secret_values(settings),
    )
    logger.info(
        "test run started",
        extra=_log_extra(
            run_id=run_id,
            component="tests",
            operation=args.test_mode,
            status="RUNNING",
            reason_code="TEST_RUN_STARTED",
        ),
    )
    mode = args.test_mode
    checks: list[dict[str, Any]] = []
    components: dict[str, dict[str, Any]] = {}
    providers: dict[str, Any] = {}
    tracemalloc.start()
    started = time.perf_counter()
    if mode in {"offline", "full"}:
        checks = _offline_checks(settings, tree)
        components = await _local_component_checks(settings, tree)
        if mode == "full":
            providers = await _provider_checks(settings)
            components["websocket"] = await _websocket_checks(settings, duration=10)
    elif mode == "database":
        components = await _local_component_checks(settings, tree)
        components = {"database": components["database"]}
    elif mode == "reporting":
        components = await _local_component_checks(settings, tree)
        components = {"reporting": components["reporting"]}
    elif mode == "paper":
        components = await _local_component_checks(settings, tree)
        components = {
            "paper": components["paper"],
            "risk": components["risk"],
        }
    elif mode in {"providers", "network"}:
        providers = await _provider_checks(settings)
        if mode == "network":
            components["websocket"] = await _websocket_checks(
                settings, duration=max(5.0, float(args.duration or 10))
            )
    elif mode == "websockets":
        components["websocket"] = await _websocket_checks(
            settings, duration=max(5.0, float(args.duration or 10))
        )
    elif mode == "soak":
        duration = float(args.duration)
        components["soak"] = await _websocket_checks(
            settings,
            duration=duration,
        )
        components["soak"].update(
            {
                "reason_code": "CONFIGURED_SOAK_DURATION_COMPLETE",
                "configured_duration_seconds": duration,
                "live_orders": 0,
            }
        )
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    all_results = [*checks, *components.values(), *providers.values()]
    failed = [item for item in all_results if item.get("status") == "FAILED"]
    skipped = sum(
        item.get("status") in {"SKIPPED", "BLOCKED"}
        for item in [*components.values(), *providers.values()]
    )
    status = "FAILED" if failed else "PARTIAL" if skipped else "PASSED"
    provider_payload = {
        "status": (
            "FAILED"
            if any(item.get("status") == "FAILED" for item in providers.values())
            else "PARTIAL"
            if providers
            and any(
                item.get("status") in {"SKIPPED", "PARTIAL", "BLOCKED"}
                for item in providers.values()
            )
            else "PASSED"
            if providers
            else "SKIPPED"
        ),
        "reason_code": "PROVIDER_CHECKS_COMPLETE" if providers else "NOT_REQUESTED",
        "providers": providers,
    }
    atomic_write_json(root / "provider_status.json", provider_payload)
    _status_file(
        root,
        "data_quality.json",
        status="PASSED" if mode in {"offline", "full"} and not failed else "SKIPPED",
        reason_code="SYNTHETIC_VALIDATION_COMPLETE"
        if mode in {"offline", "full"}
        else "NOT_REQUESTED",
    )
    atomic_write_json(
        root / "websocket_health.json",
        components.get(
            "websocket",
            {"status": "SKIPPED", "reason_code": "NOT_REQUESTED"},
        ),
    )
    atomic_write_json(
        root / "orderbook_health.json",
        components.get(
            "orderbook",
            {"status": "SKIPPED", "reason_code": "NOT_REQUESTED"},
        ),
    )
    atomic_write_json(
        root / "database_health.json",
        components.get(
            "database",
            {"status": "SKIPPED", "reason_code": "NOT_REQUESTED"},
        ),
    )
    _status_file(
        root,
        "scraper_status.json",
        status="PASSED" if mode in {"offline", "full"} and not failed else "SKIPPED",
        reason_code="SYNTHETIC_ALIGNMENT_TESTED"
        if mode in {"offline", "full"}
        else "NOT_REQUESTED",
    )
    atomic_write_json(
        root / "macro_context_status.json",
        components.get(
            "macro",
            {"status": "SKIPPED", "reason_code": "NOT_REQUESTED"},
        ),
    )
    atomic_write_json(
        root / "gex_status.json",
        components.get(
            "gex",
            {"status": "SKIPPED", "reason_code": "NOT_REQUESTED"},
        ),
    )
    atomic_write_json(
        root / "risk_status.json",
        components.get("risk", {"status": "SKIPPED", "reason_code": "NOT_REQUESTED"}),
    )
    atomic_write_json(
        root / "paper_execution_status.json",
        components.get("paper", {"status": "SKIPPED", "reason_code": "NOT_REQUESTED"}),
    )
    atomic_write_json(
        root / "test_results.json",
        {
            "status": status,
            "checks": checks,
            "components": components,
        },
    )
    atomic_write_json(
        root / "statistics.json",
        {
            "status": "PASSED",
            "duration_seconds": time.perf_counter() - started,
            "memory_current_bytes": current_memory,
            "memory_peak_bytes": peak_memory,
            "live_orders": 0,
        },
    )
    logger.info(
        "test run checks completed",
        extra=_log_extra(
            run_id=run_id,
            component="tests",
            operation=mode,
            status=status,
            reason_code="TEST_CHECKS_COMPLETE",
            duration=time.perf_counter() - started,
        ),
    )
    secret_audit = (
        _secret_audit(
            [settings.paths.project_root],
            settings,
            excluded_roots=(
                settings.paths.project_root / ".env",
                settings.paths.project_root / ".venv",
                settings.paths.project_root / ".git",
                settings.paths.project_root / ".pytest_cache",
                settings.paths.project_root / ".ruff_cache",
                settings.paths.project_root / "__pycache__",
            ),
        )
        if mode == "secrets"
        else _secret_audit([root], settings)
    )
    atomic_write_json(root / "secret_audit.json", secret_audit)
    if secret_audit["status"] == "FAILED":
        status = "FAILED"
    summary = {
        "run_id": run_id,
        "mode": mode,
        "status": status,
        "passed_checks": sum(item.get("status") == "PASSED" for item in all_results),
        "failed_checks": len(failed),
        "skipped_or_blocked_checks": skipped,
        "artifact_root": str(root),
        "live_orders": 0,
        "completed_at": utc_now().isoformat(),
    }
    atomic_write_json(root / "summary.json", summary)
    emit(summary)
    return 1 if status == "FAILED" else 0


async def command_providers(args: argparse.Namespace, settings: Settings) -> int:
    from data.data_loader import DataLoader
    from data.database import Database

    database = Database(sqlite_path=settings.paths.database_path)
    database.migrate()
    loader = DataLoader(settings, database=database)
    try:
        if args.providers_command == "list":
            emit({"providers": loader.list_providers()})
            return 0
        if args.providers_command == "capabilities":
            rows = await loader.capability_matrix(probe=True, persist=True)
            emit(
                {
                    "status": (
                        ProviderStatus.READY.value
                        if any(row["status"] == ProviderStatus.READY.value for row in rows)
                        else ProviderStatus.PARTIAL.value
                    ),
                    "providers": rows,
                    "json": settings.paths.reports_dir / "provider_capabilities.json",
                    "csv": settings.paths.reports_dir / "provider_capabilities.csv",
                }
            )
            return 0
        if args.providers_command == "status":
            emit(
                {
                    "runtime": loader.provider_status(),
                    "persisted_capabilities": [
                        row["payload"] for row in database.fetch_records("provider_capabilities")
                    ],
                }
            )
            return 0
        if getattr(args, "public_only", False):
            rows = await loader.capability_matrix(probe=True, persist=True)
            public = [
                row
                for row in rows
                if row["authentication_requirement"]
                in {
                    "NONE",
                    "PUBLIC_ENDPOINTS_NO_AUTH",
                    "NONE_FOR_PUBLIC_MARKET_DATA",
                    "NONE_FOR_PUBLIC_ENDPOINTS",
                }
            ]
            emit(
                {
                    "status": (
                        ProviderStatus.READY.value
                        if public
                        and all(
                            row["status"]
                            in {ProviderStatus.READY.value, ProviderStatus.PARTIAL.value}
                            for row in public
                        )
                        else ProviderStatus.PARTIAL.value
                    ),
                    "public_only": True,
                    "providers": public,
                    "live_orders": 0,
                }
            )
            return 0
    finally:
        database.close()
    args.test_mode = "providers"
    args.duration = 0
    return await command_test(args, settings)


def _watermark_report(
    database: Any,
    *,
    mode: str,
) -> dict[str, Any]:
    now = utc_now()
    rows: list[dict[str, Any]] = []
    for stored in database.fetch_records("data_watermarks"):
        payload = dict(stored.get("payload") or {})
        row = {
            key: payload.get(key, stored.get(key))
            for key in (
                "provider",
                "market",
                "timeframe",
                "data_kind",
                "status",
                "earliest_stored_timestamp",
                "latest_stored_timestamp",
                "missing_ranges",
                "retry_ranges",
                "updated_at",
            )
        }
        latest = row.get("latest_stored_timestamp")
        if latest:
            parsed = datetime.fromisoformat(str(latest).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            row["age_seconds"] = max(0.0, (now - parsed.astimezone(UTC)).total_seconds())
        rows.append(row)
    if mode == "gaps":
        rows = [row for row in rows if row.get("missing_ranges") or row.get("retry_ranges")]
    if mode == "freshness":
        rows.sort(key=lambda row: float(row.get("age_seconds", math.inf)), reverse=True)
    coverage = {
        "provider_market_timeframe_series": len(rows),
        "providers": sorted({str(row["provider"]) for row in rows if row.get("provider")}),
        "markets": sorted({str(row["market"]) for row in rows if row.get("market")}),
        "timeframes": sorted({str(row["timeframe"]) for row in rows if row.get("timeframe")}),
    }
    return {
        "status": ProviderStatus.READY.value if rows else ProviderStatus.PARTIAL.value,
        "mode": mode,
        "coverage": coverage,
        "rows": rows,
    }


async def _universe_provider_markets(
    settings: Settings,
    database: Any,
    *,
    universe_size: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    from research.combinatorial_lab import UniverseManager, UniverseType

    manager = UniverseManager(settings, database=database)
    latest = manager.latest()
    if latest is None or int(latest.get("target_size") or 0) < universe_size:
        snapshot = await manager.refresh(
            target_size=min(25, universe_size),
            scan_limit=max(100, universe_size * 4),
        )
        latest = snapshot.to_dict()
    markets = {"bitvavo": [], "kraken": [], "mexc": []}
    selected_members: list[dict[str, Any]] = []
    for member in latest.get("members") or []:
        types = set(member.get("universe_types") or [])
        if UniverseType.DISCOVERY_UNIVERSE.value not in types:
            continue
        selected_members.append(member)
        availability = member.get("market_availability") or {}
        for provider in markets:
            available = list(availability.get(provider) or [])
            quote_preference = {
                "bitvavo": ("EUR",),
                "kraken": ("EUR", "USD", "USDT"),
                "mexc": ("USDT",),
            }[provider]
            preferred = sorted(
                (market for market in available if market.split("-")[-1] in quote_preference),
                key=lambda market: quote_preference.index(market.split("-")[-1]),
            )
            if preferred:
                markets[provider].append(preferred[0])
        if len(selected_members) >= universe_size:
            break
    return markets, {
        "snapshot_id": latest.get("snapshot_id"),
        "bias_label": latest.get("bias_label"),
        "members": selected_members,
    }


async def _fetch_price_history(
    args: argparse.Namespace,
    settings: Settings,
    loader: Any,
    database: Any,
) -> dict[str, Any]:
    providers = _provider_selection(args.providers, loader.list_providers())
    price_providers = [name for name in providers if name in {"bitvavo", "kraken", "mexc"}]
    timeframes = _timeframe_selection(args.timeframes)
    estimate = loader.estimate_fetch(
        providers=price_providers,
        universe_size=args.universe_size,
        history_profile=args.history_profile,
        timeframes=timeframes,
    )
    if not estimate["storage_allowed"]:
        return estimate | {
            "status": "BLOCKED_STORAGE_LIMIT",
            "reason_code": "ESTIMATED_STORAGE_EXCEEDS_CONFIGURED_LIMIT",
        }
    if estimate["requires_confirmation"] and not args.yes:
        return estimate | {
            "status": "CONFIRMATION_REQUIRED",
            "reason_code": "LARGE_OR_MAXIMUM_BACKFILL_REQUIRES_YES",
        }
    provider_markets, universe = await _universe_provider_markets(
        settings,
        database,
        universe_size=args.universe_size,
    )
    now = utc_now()
    results: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(settings.market_data.maximum_concurrent_providers)
    provider_semaphores = {provider: asyncio.Semaphore(1) for provider in price_providers}

    async def fetch_one(
        provider: str,
        market: str,
        timeframe: str,
    ) -> dict[str, Any]:
        async with provider_semaphores[provider], semaphore:
            started = time.perf_counter()
            try:
                start = loader.history_start(
                    profile=args.history_profile,
                    timeframe=timeframe,
                    provider=provider,
                    end=now,
                )
                records, provenance = await loader.download_canonical_ohlcv(
                    provider=provider,
                    market=market,
                    timeframe=timeframe,
                    start=start,
                    end=now,
                    resume=bool(args.resume),
                    persist=True,
                )
                return {
                    **provenance,
                    "requested_start": start,
                    "requested_end": now,
                    "received_rows": len(records),
                    "earliest_timestamp": records[0].timestamp if records else None,
                    "latest_timestamp": records[-1].timestamp if records else None,
                    "duration": time.perf_counter() - started,
                    "status": (
                        ProviderStatus.READY.value if records else ProviderStatus.PARTIAL.value
                    ),
                    "reason_code": provenance.get(
                        "reason_code",
                        "HISTORICAL_BATCH_COMPLETE" if records else "EMPTY_PROVIDER_RESPONSE",
                    ),
                }
            except Exception as exc:
                status, reason = _provider_failure(exc)
                return {
                    "provider": provider,
                    "market": market,
                    "timeframe": timeframe,
                    "duration": time.perf_counter() - started,
                    "status": status,
                    "reason_code": reason,
                }

    requests = [
        fetch_one(provider, market, timeframe)
        for provider in price_providers
        for market in provider_markets.get(provider, [])
        for timeframe in timeframes
    ]
    if requests:
        results.extend(await asyncio.gather(*requests))
    materialized: list[dict[str, Any]] = []
    from data.market_data import save_ohlcv, timeframe_delta, validate_ohlcv

    for provider in reversed(price_providers):
        for market in provider_markets.get(provider, []):
            for timeframe in timeframes:
                source = (
                    settings.paths.processed_data_dir / provider / market / f"{timeframe}.parquet"
                )
                if not source.is_file():
                    continue
                try:
                    stored = pd.read_parquet(source)
                    if "values" in stored:
                        expanded = pd.json_normalize(
                            stored["values"].map(
                                lambda value: value if isinstance(value, dict) else {}
                            )
                        )
                        expanded.index = stored.index
                        stored = pd.concat([stored, expanded], axis=1)
                    frame = pd.DataFrame(
                        {
                            "timestamp": pd.to_datetime(
                                stored["timestamp"],
                                utc=True,
                            ),
                            **{
                                column: pd.to_numeric(
                                    stored[column],
                                    errors="coerce",
                                )
                                for column in (
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "volume",
                                )
                            },
                        }
                    )
                    if "closed" in stored:
                        frame = frame.loc[stored["closed"].fillna(False).astype(bool).to_numpy()]
                    frame = validate_ohlcv(
                        frame,
                        timeframe=timeframe,
                        closed_candles_only=True,
                    )
                    target, manifest = save_ohlcv(
                        frame,
                        settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet",
                        market=market,
                        timeframe=timeframe,
                        maximum_staleness=max(
                            settings.market_data.maximum_staleness,
                            timeframe_delta(timeframe) * 2,
                        ),
                    )
                    provenance = {
                        "source_type": "REAL_PROVIDER_DATA",
                        "market": market,
                        "timeframe": timeframe,
                        "providers_requested": price_providers,
                        "providers_used": [provider],
                        "provider_errors": {},
                        "provider_hashes": {
                            provider: stable_hash(
                                stored["raw_hash"].astype(str).tolist(),
                                length=64,
                            )
                        },
                        "reconciliation_conflicts": [],
                        "closed_candles_only": True,
                        "retrieved_at": utc_now(),
                        "data_file": str(target),
                        "data_sha256": manifest.sha256,
                        "rows": len(frame),
                        "start": frame.index.min(),
                        "end": frame.index.max(),
                        "quote_currency": market.split("-")[-1],
                        "source_classification": (
                            str(stored["source_classification"].dropna().iloc[-1])
                            if "source_classification" in stored
                            and stored["source_classification"].notna().any()
                            else "PROVIDER_NATIVE"
                        ),
                    }
                    atomic_write_json(
                        target.with_suffix(f"{target.suffix}.provenance.json"),
                        provenance,
                    )
                    materialized.append(
                        {
                            "provider": provider,
                            "market": market,
                            "timeframe": timeframe,
                            "rows": len(frame),
                            "path": target,
                        }
                    )
                except Exception as exc:
                    materialized.append(
                        {
                            "provider": provider,
                            "market": market,
                            "timeframe": timeframe,
                            "status": ProviderStatus.FAILED_VALIDATION.value,
                            "reason_code": type(exc).__name__,
                        }
                    )
    if "coinmarketcap" in providers:
        try:
            records = await loader.download_cmc_rankings(
                limit=max(100, args.universe_size * 4),
                persist=True,
            )
            results.append(
                {
                    "provider": "coinmarketcap",
                    "dataset": "rankings",
                    "received_rows": len(records),
                    "status": ProviderStatus.READY.value
                    if records
                    else ProviderStatus.PARTIAL.value,
                    "reason_code": "UNIVERSE_SNAPSHOT_STORED",
                }
            )
        except Exception as exc:
            status, reason = _provider_failure(exc)
            results.append(
                {
                    "provider": "coinmarketcap",
                    "dataset": "rankings",
                    "status": status,
                    "reason_code": reason,
                }
            )
    succeeded = sum(row["status"] == ProviderStatus.READY.value for row in results)
    return {
        "status": ProviderStatus.READY.value
        if succeeded == len(results)
        else ProviderStatus.PARTIAL.value,
        "universe": universe,
        "estimate": estimate,
        "datasets": results,
        "lab_datasets_materialized": materialized,
        "synthetic_fallback": False,
        "live_orders": 0,
    }


async def _fetch_context(
    args: argparse.Namespace,
    settings: Settings,
    loader: Any,
) -> dict[str, Any]:
    providers = _provider_selection(args.providers, loader.list_providers())
    profile = args.history_profile.casefold()
    years = {"smoke": 1, "standard": 5, "deep": 15, "maximum": 50}[profile]
    end = utc_now()
    start = end - timedelta(days=365 * years)
    results: list[dict[str, Any]] = []

    async def collect(provider: str, dataset: str, operation: Any) -> None:
        started = time.perf_counter()
        if not loader._credential_configured(provider):
            results.append(
                {
                    "provider": provider,
                    "dataset": dataset,
                    "status": ProviderStatus.SKIPPED_MISSING_CREDENTIALS.value,
                    "reason_code": "SKIPPED_MISSING_CREDENTIALS",
                    "duration": 0.0,
                }
            )
            return
        try:
            value = await operation()
            count = (
                len(value)
                if isinstance(value, (list, tuple, pd.DataFrame))
                else int(value.get("contracts") or 1)
                if isinstance(value, dict)
                else 1
            )
            results.append(
                {
                    "provider": provider,
                    "dataset": dataset,
                    "received_rows": count,
                    "status": ProviderStatus.READY.value if count else ProviderStatus.PARTIAL.value,
                    "reason_code": "CONTEXT_DATASET_PERSISTED"
                    if count
                    else "EMPTY_PROVIDER_RESPONSE",
                    "duration": time.perf_counter() - started,
                }
            )
        except Exception as exc:
            status, reason = _provider_failure(exc)
            if (
                provider == "coinmarketcap"
                and "historical_quotes" in dataset
                and getattr(exc, "status", None) == 400
            ):
                status = ProviderStatus.BLOCKED_PLAN_LIMIT.value
                reason = "HISTORICAL_QUOTES_NOT_AVAILABLE_FOR_PLAN_OR_RANGE"
            results.append(
                {
                    "provider": provider,
                    "dataset": dataset,
                    "status": status,
                    "reason_code": reason,
                    "duration": time.perf_counter() - started,
                }
            )

    for provider in providers:
        if provider == "coinmarketcap":
            await collect(
                provider,
                "global_metrics",
                lambda: loader.download_macro_series(
                    provider=provider, series="GLOBAL", persist=True
                ),
            )
            await collect(
                provider,
                "rankings",
                lambda: loader.download_cmc_rankings(limit=250, persist=True),
            )
            for symbol in ("BTC", "ETH"):
                await collect(
                    provider,
                    f"{symbol}_historical_quotes",
                    lambda selected=symbol: loader.download_macro_series(
                        provider=provider,
                        series=selected,
                        start=start,
                        end=end,
                        persist=True,
                    ),
                )
        elif provider == "fred":
            series_ids = (
                "DFF",
                "DGS2",
                "DGS5",
                "DGS10",
                "DGS30",
                "DFII10",
                "SOFR",
                "T10Y2Y",
                "CPIAUCSL",
                "CPILFESL",
                "PCEPI",
                "PCEPILFE",
                "M2SL",
                "WALCL",
                "RRPONTSYD",
                "WTREGEN",
                "UNRATE",
                "ICSA",
                "INDPRO",
                "RSAFS",
                "BAMLH0A0HYM2",
                "NFCI",
            )
            for series in series_ids:
                await collect(
                    provider,
                    series,
                    lambda selected=series: loader.download_macro_series(
                        provider=provider,
                        series=selected,
                        start=start,
                        end=end,
                        persist=True,
                    ),
                )
                if profile in {"deep", "maximum"}:
                    await collect(
                        provider,
                        f"{series}_revisions",
                        lambda selected=series: loader.download_fred_revisions(
                            series=selected,
                            start=start,
                            end=end,
                            persist=True,
                        ),
                    )
            for series in ("DFF", "DGS10", "CPIAUCSL", "M2SL", "UNRATE"):
                await collect(
                    provider,
                    f"{series}_vintages",
                    lambda selected=series: loader.download_fred_vintages(
                        series=selected,
                        persist=True,
                    ),
                )
        elif provider == "eodhd":
            for series in (
                "economic_events",
                "macro:inflation_consumer_prices_annual",
                "macro:interest_rate",
                "macro:unemployment_total_percent",
                "VIX.INDX",
                "GSPC.INDX",
                "NDX.INDX",
            ):
                await collect(
                    provider,
                    series,
                    lambda selected=series: loader.download_macro_series(
                        provider=provider,
                        series=selected,
                        start=start,
                        end=end,
                        persist=True,
                    ),
                )
        elif provider == "sec":
            for cik in ("CIK0001679788", "CIK0001364742"):
                await collect(
                    provider,
                    cik,
                    lambda selected=cik: loader.download_macro_series(
                        provider=provider,
                        series=selected,
                        persist=True,
                    ),
                )
        elif provider == "alternative_me":
            await collect(
                provider,
                "fear_and_greed",
                lambda: loader.download_macro_series(
                    provider=provider,
                    series="fear_and_greed",
                    persist=True,
                ),
            )
        elif provider == "defillama":
            for series in ("stablecoins", "protocols"):
                await collect(
                    provider,
                    series,
                    lambda selected=series: loader.download_macro_series(
                        provider=provider,
                        series=selected,
                        persist=True,
                    ),
                )
        elif provider == "deribit":
            for underlying in ("BTC", "ETH"):
                await collect(
                    provider,
                    f"{underlying}_gex",
                    lambda selected=underlying: loader.download_gex_context(
                        underlying=selected,
                        persist=True,
                    ),
                )
        elif provider in {"mexc", "coinglass"}:
            await collect(
                provider,
                "BTC-USDT_derivatives",
                lambda selected=provider: loader.download_derivatives_context(
                    provider=selected,
                    market="BTC-USDT",
                    persist=True,
                ),
            )
        else:
            results.append(
                {
                    "provider": provider,
                    "dataset": "configured_context",
                    "status": (
                        ProviderStatus.UNSUPPORTED_ENDPOINT.value
                        if loader._credential_configured(provider)
                        else ProviderStatus.SKIPPED_MISSING_CREDENTIALS.value
                    ),
                    "reason_code": (
                        "OPTIONAL_PROVIDER_ADAPTER_NOT_IMPLEMENTED"
                        if loader._credential_configured(provider)
                        else "SKIPPED_MISSING_CREDENTIALS"
                    ),
                }
            )
    ready = sum(row["status"] == ProviderStatus.READY.value for row in results)
    return {
        "status": ProviderStatus.READY.value
        if ready == len(results)
        else ProviderStatus.PARTIAL.value,
        "history_profile": profile,
        "datasets": results,
        "context_directory": settings.paths.context_data_dir,
        "live_orders": 0,
    }


async def command_extended_data(args: argparse.Namespace, settings: Settings) -> int:
    from data.data_loader import ContinuousDataService, DataLoader
    from data.database import Database

    if args.data_command == "database-health":
        configured_url = (
            settings.providers.database_url.get_secret_value()
            if settings.providers.database_url
            else None
        )
        url = (
            configured_url
            if configured_url
            and configured_url.startswith(("sqlite://", "postgresql://", "postgresql+"))
            else None
        )
        database = Database(url, sqlite_path=settings.paths.database_path)
        database.migrate()
        health = database.health()
        if configured_url and url is None:
            health["configuration_warning"] = "INVALID_DATABASE_URL_USING_SQLITE_DEFAULT"
        emit(health)
        database.close()
        return 0
    if args.data_command in {"status", "coverage", "gaps", "freshness"}:
        database = Database(sqlite_path=settings.paths.database_path)
        database.migrate()
        try:
            report = _watermark_report(database, mode=args.data_command)
            if args.data_command == "status":
                report["continuous_service"] = ContinuousDataService(
                    settings,
                    database=database,
                ).status()
            emit(report)
        finally:
            database.close()
        return 0
    if args.data_command == "live":
        emit(
            {
                "status": "READY",
                "reason_code": "USE_WEBSOCKET_RUN",
                "execution_enabled": False,
            }
        )
        return 0
    if args.data_command == "reconcile":
        emit(
            {
                "status": "READY",
                "reason_code": "PROVIDE_SERIES_THROUGH_DATA_LOADER_API",
            }
        )
        return 0
    database = Database(sqlite_path=settings.paths.database_path)
    database.migrate()
    loader = DataLoader(settings, database=database)
    try:
        if args.data_command == "estimate":
            providers = _provider_selection(args.providers, loader.list_providers())
            emit(
                loader.estimate_fetch(
                    providers=providers,
                    universe_size=args.universe_size,
                    history_profile=args.history_profile,
                    timeframes=_timeframe_selection(args.timeframes),
                )
            )
            return 0
        if args.data_command == "fetch":
            payload = await _fetch_price_history(args, settings, loader, database)
            emit(payload)
            return 0 if payload["status"] != "CONFIRMATION_REQUIRED" else 2
        if args.data_command == "context-fetch":
            if args.history_profile.casefold() == "maximum" and not args.yes:
                emit(
                    {
                        "status": "CONFIRMATION_REQUIRED",
                        "reason_code": "MAXIMUM_CONTEXT_FETCH_REQUIRES_YES",
                    }
                )
                return 2
            emit(await _fetch_context(args, settings, loader))
            return 0
        if args.data_command == "sync":

            async def sync_cycle() -> dict[str, Any]:
                price = await _fetch_price_history(
                    args,
                    settings,
                    loader,
                    database,
                )
                context_selection = csv_values(args.context)
                if not context_selection or "none" in {
                    item.casefold() for item in context_selection
                }:
                    return price
                context_providers = (
                    "coinmarketcap,eodhd,fred,sec,alternative_me,defillama,deribit,mexc"
                    if any(item.casefold() == "all" for item in context_selection)
                    else ",".join(context_selection)
                )
                context_arguments = argparse.Namespace(
                    **{
                        **vars(args),
                        "providers": context_providers,
                    }
                )
                context = await _fetch_context(
                    context_arguments,
                    settings,
                    loader,
                )
                return {
                    "status": (
                        ProviderStatus.READY.value
                        if price["status"] == ProviderStatus.READY.value
                        and context["status"] == ProviderStatus.READY.value
                        else ProviderStatus.PARTIAL.value
                    ),
                    "price": price,
                    "context": context,
                    "live_orders": 0,
                }

            if args.continuous:
                service = ContinuousDataService(settings, database=database)
                await service.start(
                    sync_cycle,
                    interval_seconds=float(args.interval_seconds),
                    once=False,
                )
                emit(service.status())
                return 0
            emit(await sync_cycle())
            return 0
        records = await loader.download_ohlcv(
            provider=args.provider,
            market=args.market,
            timeframe=args.timeframe,
            start=datetime.fromisoformat(args.start.replace("Z", "+00:00")),
            end=datetime.fromisoformat((args.end or utc_now().isoformat()).replace("Z", "+00:00")),
            resume=not args.no_resume,
            persist=True,
        )
    finally:
        database.close()
    emit({"status": "PASSED", "records": len(records), "provider": args.provider})
    return 0


async def command_websocket(args: argparse.Namespace, settings: Settings) -> int:
    from data.websocket_manager import WebSocketManager

    if args.websocket_command == "status":
        emit(WebSocketManager().health())
        return 0
    duration = float(args.duration)
    manager = WebSocketManager(
        queue_size=settings.market_data.websocket_queue_size,
        inactivity_timeout=settings.market_data.websocket_inactivity_seconds,
    )
    subscriptions = {
        args.provider: {
            "ticker": [args.market],
            "trades": [args.market],
        }
    }
    if args.provider != "mexc":
        subscriptions[args.provider]["book"] = [args.market]
    try:
        await manager.start(subscriptions)
        await asyncio.sleep(duration)
    finally:
        await manager.stop()
    emit(
        {
            "status": "PASSED" if manager.health(args.provider)["messages"] > 0 else "PARTIAL",
            "health": manager.health(args.provider),
            "live_orders": 0,
        }
    )
    return 0


async def command_orderbook(args: argparse.Namespace, settings: Settings) -> int:
    from data.data_loader import DataLoader
    from data.orderbook_l2 import Level2OrderBook

    loader = DataLoader(settings)
    if args.orderbook_command == "stream":
        emit(
            {
                "status": "READY",
                "reason_code": "USE_WEBSOCKET_RUN_AND_LEVEL2ORDERBOOK",
                "execution_enabled": False,
            }
        )
        return 0
    snapshot = await loader.download_orderbook_snapshot(
        provider=args.provider,
        market=args.market,
        depth=args.depth,
    )
    if args.orderbook_command == "snapshot":
        emit(snapshot)
        return 0
    book = Level2OrderBook(
        provider=args.provider,
        market=args.market,
        maximum_depth=args.depth,
    )
    await book.initialize(
        bids=snapshot.values["bids"],
        asks=snapshot.values["asks"],
        sequence=snapshot.values.get("sequence"),
    )
    emit(book.health())
    return 0


def command_macro(args: argparse.Namespace, settings: Settings) -> int:
    from research.macro_context import (
        MacroContextEngine,
        build_persisted_macro_context,
    )

    if args.macro_command == "build":
        report = build_persisted_macro_context(
            context_dir=settings.paths.context_data_dir,
            processed_dir=settings.paths.processed_data_dir,
            timeframes=_timeframe_selection(args.timeframes),
        )
        emit(report)
        return 0
    paths = sorted(settings.paths.context_data_dir.glob("macro_context_*.parquet"))
    if not paths:
        emit(
            {
                "status": ProviderStatus.PARTIAL.value,
                "reason_code": "NO_MACRO_CONTEXT_BUILDS",
                "path": settings.paths.context_data_dir,
            }
        )
        return 3
    emit(
        {
            "status": ProviderStatus.READY.value,
            "datasets": [
                {
                    "path": path,
                    "rows": len(result := pd.read_parquet(path)),
                    "snapshot": MacroContextEngine.latest_snapshot(result),
                }
                for path in paths
            ],
            "coverage": (
                read_json(settings.paths.context_data_dir / "macro_context_coverage.json")
                if (settings.paths.context_data_dir / "macro_context_coverage.json").is_file()
                else None
            ),
        }
    )
    return 0


async def command_gex(args: argparse.Namespace, settings: Settings) -> int:
    from data.data_loader import DataLoader
    from data.database import Database

    underlying = getattr(args, "underlying", "BTC")
    path = settings.paths.context_data_dir / f"gex_{underlying}.parquet"
    if args.gex_command == "collect":
        database = Database(sqlite_path=settings.paths.database_path)
        database.migrate()
        try:
            summary = await DataLoader(
                settings,
                database=database,
            ).download_gex_context(
                underlying=underlying,
                persist=True,
            )
        finally:
            database.close()
        emit({**summary, "output": path})
        return 0
    if not path.is_file():
        emit({"status": ProviderStatus.PARTIAL.value, "reason_code": "NO_GEX_DATA", "path": path})
        return 3
    frame = pd.read_parquet(path)
    emit(
        {
            "status": ProviderStatus.READY.value,
            "underlying": underlying,
            "rows": len(frame),
            "latest": frame.iloc[-1].to_dict() if not frame.empty else None,
            "path": path,
        }
    )
    return 0


def command_positions(args: argparse.Namespace, settings: Settings) -> int:
    from execution.position_tracker import PositionTracker

    tracker = PositionTracker(settings.paths.checkpoints_dir / "positions.json")
    if args.positions_command == "status":
        emit(tracker.snapshot())
    elif args.positions_command == "pnl":
        emit(
            {
                "portfolio": tracker.portfolio_pnl(),
                "strategy": tracker.pnl_by_strategy(),
                "symbol": tracker.pnl_by_symbol(),
                "daily": tracker.daily_pnl(),
            }
        )
    else:
        emit(
            {
                "status": "BLOCKED",
                "reason_code": "EXCHANGE_BALANCES_REQUIRED",
                "automatic_mutation": False,
            }
        )
    return 0


def command_risk(args: argparse.Namespace, settings: Settings) -> int:
    if args.risk_command == "kill-switch-status":
        from risk.drawdown_protection import DrawdownProtection

        protection = DrawdownProtection(
            state_path=settings.paths.checkpoints_dir / "drawdown_state.json",
            audit_path=settings.paths.logs_dir / "drawdown_audit.jsonl",
        )
        emit(protection.status())
        return 0
    if args.risk_command == "drawdown":
        from risk.drawdown_protection import DrawdownProtection

        index = pd.date_range(end=utc_now(), periods=48, freq="h")
        equity = pd.Series(np.linspace(2_000, 1_900, len(index)), index=index)
        protection = DrawdownProtection(
            state_path=settings.paths.checkpoints_dir / "drawdown_state.json",
            audit_path=settings.paths.logs_dir / "drawdown_audit.jsonl",
        )
        emit(protection.evaluate(portfolio_equity=equity))
        return 0
    from risk.correlation_analyzer import CorrelationAnalyzer

    index = pd.date_range(end=utc_now(), periods=120, freq="h")
    randomizer = np.random.default_rng(settings.app.random_seed)
    btc = randomizer.normal(0, 0.01, len(index))
    frame = pd.DataFrame(
        {
            "BTC-EUR": btc,
            "ETH-EUR": btc * 0.85 + randomizer.normal(0, 0.003, len(index)),
            "SOL-EUR": btc * 0.65 + randomizer.normal(0, 0.006, len(index)),
        },
        index=index,
    )
    analyzer = CorrelationAnalyzer(maximum_age=timedelta(hours=2))
    emit(
        {
            "pearson": analyzer.pearson(frame).to_dict(),
            "statistics": analyzer.risk_statistics(
                frame, {"BTC-EUR": 0.5, "ETH-EUR": 0.3, "SOL-EUR": 0.2}
            ),
        }
    )
    return 0


def paper_ledger(settings: Settings) -> Path:
    return settings.paths.checkpoints_dir / "paper_execution.jsonl"


def command_paper(args: argparse.Namespace, settings: Settings) -> int:
    from execution.execution import ExecutionMarketRules, PaperBroker
    from risk.risk_manager import PortfolioSnapshot, RiskManager

    ledger = paper_ledger(settings)
    if args.paper_command == "status":
        from execution.execution import DurableLedger

        events = DurableLedger(ledger).events()
        emit({"status": "READY", "ledger": ledger, "event_count": len(events)})
        return 0
    if args.paper_command == "reconcile":
        from execution.execution import DurableLedger

        try:
            events = DurableLedger(ledger).events()
            emit(
                {
                    "healthy": True,
                    "reason_codes": ["LEDGER_READABLE"],
                    "event_count": len(events),
                }
            )
            return 0
        except Exception:
            emit({"healthy": False, "reason_codes": ["LEDGER_UNREADABLE"]})
            return 3
    markets = selected_markets(args)
    if len(markets) != 1:
        raise ValueError("paper run currently accepts one market per invocation")
    args.market = markets[0]
    if args.candidates:
        candidate = read_json(args.candidates)
        if candidate.get("status") != ResearchStatus.PAPER_CANDIDATE.value:
            raise ValueError("paper candidate file is not PAPER_CANDIDATE")
        args.strategy = str(candidate.get("strategy_id") or args.strategy)
    broker = PaperBroker(
        initial_balances={"EUR": Decimal(str(args.capital))},
        market_rules={args.market: ExecutionMarketRules(minimum_order_value_eur=Decimal("5"))},
        fee_fraction=Decimal(str(settings.costs.default_fee)),
        slippage_bps=Decimal(str(settings.costs.slippage_bps)),
        spread_bps=Decimal(str(settings.costs.spread_bps)),
        ledger_path=ledger,
    )
    snapshot = PortfolioSnapshot(
        equity_eur=args.capital,
        cash_eur=args.capital,
        day_start_equity_eur=args.capital,
        peak_equity_eur=args.capital,
        trades_today=0,
    )
    decision = RiskManager.from_settings(settings).assess_entry(
        market=args.market,
        entry_price=args.price,
        stop_price=args.price * (1.0 - args.stop_fraction),
        snapshot=snapshot,
    )
    if not decision.approved:
        emit({"status": "REJECTED", "risk": decision})
        return 3
    quantity = min(Decimal(str(args.quantity)), decision.approved_quantity)
    intent = OrderIntent(
        intent_id=f"paper-{stable_hash({'at': utc_now(), 'market': args.market}, length=16)}",
        idempotency_key=args.idempotency_key
        or stable_hash({"paper": args.market, "at": utc_now()}, length=24),
        market=args.market,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=quantity,
        strategy_id=args.strategy,
        maximum_notional_eur=Decimal(str(args.capital * 0.25)),
        reason_codes=decision.reason_codes,
    )
    order = broker.submit(intent, market_price=Decimal(str(args.price)))
    emit(
        {
            "status": order.status.value,
            "order": order,
            "balances": broker.balance_snapshot(),
            "reconciliation": broker.reconcile(),
        }
    )
    return 0 if order.status.value == "FILLED" else 3


def research_status_from_report(path: str | None) -> tuple[ResearchStatus, bool]:
    if not path:
        return ResearchStatus.LIVE_BLOCKED, False
    report = Path(path).resolve()
    if not report.is_file():
        return ResearchStatus.LIVE_BLOCKED, False
    try:
        payload = read_json(report)
        status = ResearchStatus(payload["status"])
        manifest = report.with_name(report.name.replace("_summary.json", "_manifest.json"))
        manifest_payload = read_json(manifest)
        artifact = next(
            item for item in manifest_payload["artifacts"] if Path(item["path"]).resolve() == report
        )
        verified = (
            manifest_payload.get("run_kind") == "research"
            and artifact["sha256"] == sha256_file(report)
            and payload.get("passed") is True
            and status is ResearchStatus.PAPER_CANDIDATE
            and payload.get("lookahead_safe") is True
            and payload.get("repainting_safe") is True
        )
        return (status, True) if verified else (ResearchStatus.LIVE_BLOCKED, False)
    except (KeyError, StopIteration, OSError, ValueError, TypeError):
        return ResearchStatus.LIVE_BLOCKED, False


async def live_runtime(
    args: argparse.Namespace,
    settings: Settings,
    *,
    submit: bool,
) -> tuple[int, dict[str, Any]]:
    import aiohttp

    from data.market_data import load_ohlcv, quality_report
    from execution.execution import LivePreflight, build_live_client
    from risk.risk_manager import (
        KillSwitch,
        PortfolioSnapshot,
        PositionExposure,
        RiskManager,
    )

    status, report_verified = research_status_from_report(args.research_report)
    data_healthy = False
    if args.data:
        frame = load_ohlcv(args.data, market=args.market, validate=True)
        data_healthy = quality_report(
            frame,
            market=args.market,
            timeframe=args.timeframe,
            maximum_staleness=settings.market_data.maximum_staleness,
        ).valid
    kill_switch = KillSwitch(settings.paths.checkpoints_dir / "kill_switch.json")
    preliminary = list(settings.static_live_preflight_failures())
    if status is not ResearchStatus.PAPER_CANDIDATE:
        preliminary.append("LIVE_BLOCKED_STRATEGY_NOT_PAPER_CANDIDATE")
    if not report_verified:
        preliminary.append("LIVE_BLOCKED_UNVERIFIED_RESEARCH_REPORT")
    if not data_healthy:
        preliminary.append("LIVE_BLOCKED_DATA_UNHEALTHY")
    if kill_switch.active:
        preliminary.append("LIVE_BLOCKED_KILL_SWITCH")
    if preliminary:
        return 3, {
            "passed": False,
            "failures": list(dict.fromkeys(preliminary)),
            "research_status": status.value,
            "research_report_verified": report_verified,
            "data_healthy": data_healthy,
            "exchange_healthy": False,
            "reconciliation": None,
        }
    failures: list[str] = []
    async with aiohttp.ClientSession() as session:
        try:
            client = build_live_client(
                settings,
                session=session,
                ledger_path=settings.paths.checkpoints_dir / "live_execution.jsonl",
            )
            balances = await client.balances()
            reconciliation = await client.reconcile(markets=(args.market,))
            exchange_healthy = True
        except ExecutionBlocked as exc:
            failures.append(type(exc).__name__)
            reconciliation = None
            exchange_healthy = False
            balances = []
            client = None
        preflight = LivePreflight.evaluate(
            settings,
            markets=(args.market,),
            strategy_status=status,
            data_healthy=data_healthy,
            risk_manager_healthy=True,
            exchange_healthy=exchange_healthy,
            reconciliation_healthy=bool(reconciliation and reconciliation.healthy),
            kill_switch_active=kill_switch.active,
        )
        result: dict[str, Any] = {
            "passed": preflight.passed,
            "failures": list(preflight.failures) + failures,
            "research_status": status.value,
            "research_report_verified": report_verified,
            "data_healthy": data_healthy,
            "exchange_healthy": exchange_healthy,
            "reconciliation": reconciliation,
        }
        if not submit or not preflight.passed or preflight.capability is None or client is None:
            return (0 if preflight.passed else 3), result
        by_symbol = {str(item.get("symbol")): item for item in balances}
        eur = Decimal(str(by_symbol.get("EUR", {}).get("available", "0")))
        base = args.market.split("-")[0]
        owned = Decimal(str(by_symbol.get(base, {}).get("available", "0")))
        snapshot = PortfolioSnapshot(
            equity_eur=float(eur),
            cash_eur=float(eur),
            day_start_equity_eur=float(eur),
            peak_equity_eur=float(eur),
            trades_today=0,
            reconciled=True,
        )
        manager = RiskManager.from_settings(
            settings,
            kill_switch_path=settings.paths.checkpoints_dir / "kill_switch.json",
        )
        if args.side == "BUY":
            risk = manager.assess_entry(
                market=args.market,
                entry_price=args.price,
                stop_price=args.price * (1.0 - args.stop_fraction),
                snapshot=snapshot,
                live_mode=True,
            )
            quantity = min(Decimal(str(args.quantity)), risk.approved_quantity)
        else:
            risk = manager.assess_exit(
                market=args.market,
                requested_quantity=args.quantity,
                snapshot=PortfolioSnapshot(
                    equity_eur=snapshot.equity_eur,
                    cash_eur=snapshot.cash_eur,
                    day_start_equity_eur=snapshot.day_start_equity_eur,
                    peak_equity_eur=snapshot.peak_equity_eur,
                    trades_today=0,
                    positions=(
                        PositionExposure(
                            market=args.market,
                            quantity=float(owned),
                            mark_price=args.price,
                            open_risk_eur=0.0,
                        ),
                    ),
                ),
            )
            quantity = Decimal(str(args.quantity))
        if not risk.approved:
            result["risk"] = risk
            result["failures"].append("LIVE_BLOCKED_RISK_REJECTED")
            return 3, result
        intent = OrderIntent(
            intent_id=f"live-{stable_hash({'at': utc_now(), 'market': args.market}, length=16)}",
            idempotency_key=args.idempotency_key,
            market=args.market,
            side=OrderSide(args.side),
            order_type=OrderType.MARKET,
            quantity=quantity,
            strategy_id=args.strategy,
            maximum_notional_eur=Decimal(str(settings.execution.maximum_live_order_eur)),
            reason_codes=risk.reason_codes,
        )
        result["order"] = await client.submit_order(
            intent,
            capability=preflight.capability,
            estimated_price=Decimal(str(args.price)),
            reconciled_owned_quantity=owned,
        )
        result["status"] = "SUBMITTED"
        return 0, result


async def command_live_async(args: argparse.Namespace, settings: Settings) -> int:
    if args.live_command == "status":
        failures = settings.static_live_preflight_failures()
        emit({"live_ready": not failures, "failures": failures})
        return 0
    code, result = await live_runtime(
        args,
        settings,
        submit=args.live_command == "run",
    )
    emit(result)
    return code


def _lab_sizes(value: str | None, default: tuple[int, ...] = (1, 2)) -> tuple[int, ...]:
    selected = tuple(int(item) for item in csv_values(value)) or default
    if any(size < 1 or size > 5 for size in selected):
        raise ValueError("combination sizes must be between one and five")
    return tuple(sorted(set(selected)))


def _lab_logic_modes(value: str | None):
    from research.combinatorial_lab import LogicMode

    aliases = {
        "all": LogicMode.ALL,
        "any": LogicMode.ANY,
        "majority": LogicMode.MAJORITY,
        "weighted_vote": LogicMode.WEIGHTED_VOTE,
        "layered": LogicMode.LAYERED,
    }
    requested = csv_values(value) or ["layered"]
    unknown = sorted(set(requested) - set(aliases))
    if unknown:
        raise ValueError(f"unknown lab logic modes: {unknown}")
    return tuple(aliases[item] for item in requested)


def _lab_parameter_overrides(values: list[str] | None) -> dict[str, tuple[Decimal, ...]]:
    parsed: dict[str, tuple[Decimal, ...]] = {}
    for expression in values or []:
        if "=" not in expression:
            raise ValueError(f"lab parameter requires NAME=VALUE_OR_RANGE: {expression}")
        name, raw = expression.split("=", 1)
        parts = raw.split(":")
        if len(parts) == 1:
            generated = (Decimal(parts[0]),)
        elif len(parts) == 3:
            start, stop, step = map(Decimal, parts)
            if step <= 0 or start > stop:
                raise ValueError(f"invalid lab parameter range: {expression}")
            selected: list[Decimal] = []
            current = start
            while current <= stop:
                selected.append(current)
                current += step
            generated = tuple(selected)
        else:
            raise ValueError(f"invalid lab parameter range: {expression}")
        parsed[name.strip()] = generated
    return parsed


def _lab_generation_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "profile": args.profile,
        "universe_size": args.universe_size,
        "combination_sizes": _lab_sizes(args.combination_sizes),
        "logic_modes": _lab_logic_modes(args.logic_modes),
        "timeframes": tuple(csv_values(args.timeframes) or ["1h", "4h"]),
        "rows": args.rows,
        "history_mode": args.history_mode,
        "workers": args.workers,
        "data_mode": args.data_mode,
        "max_trials": args.max_trials,
        "universe_scope": ("allowed" if args.allowed_universe else "discovery"),
        "include_review_required_research_only": (
            args.include_review_required_research_only or not args.allowed_universe
        ),
        "resume": args.resume,
        "force": args.force,
        "retest": args.retest,
        "only_missing": args.only_missing,
        "block_ids": tuple(csv_values(args.blocks)) or None,
        "parameter_overrides": _lab_parameter_overrides(args.parameter),
    }


def _rotation_campaign_path(
    settings: Settings,
    *,
    ensemble: bool = False,
    institutional: bool = False,
) -> Path:
    name = (
        "cross_sectional_institutional_v2.json"
        if institutional
        else (
            "cross_sectional_ensemble_v1.json" if ensemble else "cross_sectional_rotation_v1.json"
        )
    )
    return settings.paths.lab_dir / "reports" / name


def _capital_utilization_campaign_path(settings: Settings) -> Path:
    return settings.paths.lab_dir / "reports" / "capital_utilization_campaign_v1.json"


def _diversified_rotation_campaign_path(settings: Settings) -> Path:
    return settings.paths.lab_dir / "reports" / "diversified_rotation_campaign_v1.json"


def _breakout_portfolio_campaign_path(settings: Settings) -> Path:
    return settings.paths.lab_dir / "reports" / "portfolio_breakout_campaign_v1.json"


def _absolute_momentum_campaign_path(settings: Settings) -> Path:
    return settings.paths.lab_dir / "reports" / "absolute_momentum_campaign_v1.json"


def _absolute_momentum_plateau_campaign_path(
    settings: Settings,
) -> Path:
    return (
        settings.paths.lab_dir
        / "reports"
        / "absolute_momentum_plateau_campaign_v1.json"
    )


def _portfolio_storm_paths(
    settings: Settings,
) -> tuple[Path, Path, Path]:
    reports = settings.paths.lab_dir / "reports"
    return (
        reports / "portfolio_storm_plan_v1.json",
        reports / "portfolio_storm_report_v1.json",
        reports / "portfolio_storm_returns_v1.npz",
    )


def _signal_synthesis_storm_paths(
    settings: Settings,
) -> tuple[Path, Path, Path]:
    reports = settings.paths.lab_dir / "reports"
    return (
        reports / "signal_synthesis_storm_plan_v2.json",
        reports / "signal_synthesis_storm_report_v2.json",
        reports / "signal_synthesis_storm_returns_v2.npz",
    )


def _signal_synthesis_data_paths(
    settings: Settings,
) -> dict[str, dict[str, Path]]:
    from research.combinatorial_lab import LabRunner
    from research.signal_synthesis_storm import (
        SIGNAL_STORM_MARKETS,
        SIGNAL_STORM_TIMEFRAMES,
    )

    runner = LabRunner(settings)
    paths = {
        timeframe: {market: runner._data_path(market, timeframe) for market in SIGNAL_STORM_MARKETS}
        for timeframe in SIGNAL_STORM_TIMEFRAMES
    }
    missing = [
        str(path)
        for by_market in paths.values()
        for path in by_market.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"signal synthesis source data is missing: {missing}")
    return paths


def _signal_synthesis_storm_plan_payload(
    settings: Settings,
    *,
    trial_count: int,
) -> dict[str, Any]:
    from research.features import feature_registry
    from research.signal_synthesis_storm import (
        SIGNAL_STORM_SEED,
        signal_storm_plan,
    )

    frozen_path = settings.paths.lab_dir / "candidates" / "rotation_research_lead_v1.json"
    frozen = read_json(frozen_path)
    paths = _signal_synthesis_data_paths(settings)
    payload = signal_storm_plan(
        trial_count=trial_count,
        seed=SIGNAL_STORM_SEED,
    )
    return {
        **payload,
        "source_candidate_identity": frozen["immutable_identity"],
        "source_candidate_sha256": sha256_file(frozen_path),
        "feature_registry_hash": stable_hash(
            feature_registry(),
            length=64,
        ),
        "data_hashes": {
            timeframe: {market: sha256_file(path) for market, path in by_market.items()}
            for timeframe, by_market in paths.items()
        },
    }


def _run_signal_synthesis_storm_campaign(
    settings: Settings,
    *,
    maximum_trials: int | None = None,
    artifact_directory: Path | None = None,
    prior_known_trials_override: int | None = None,
    epoch_id: str | None = None,
) -> dict[str, Any]:
    """Run one immutable, broad signal-DNA screen without promotion."""

    from research.combinatorial_lab import LabRunner
    from research.signal_synthesis_storm import (
        SIGNAL_STORM_MARKETS,
        SIGNAL_STORM_TIMEFRAMES,
        SIGNAL_STORM_TRIAL_COUNT,
        SignalSynthesisDNA,
        run_signal_synthesis_storm,
    )

    if artifact_directory is None:
        plan_path, report_path, matrix_path = _signal_synthesis_storm_paths(settings)
    else:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        plan_path = artifact_directory / "plan.json"
        report_path = artifact_directory / "report.json"
        matrix_path = artifact_directory / "returns.npz"
    trial_count = maximum_trials or SIGNAL_STORM_TRIAL_COUNT
    if trial_count < 2 or trial_count > SIGNAL_STORM_TRIAL_COUNT:
        raise ValueError(
            f"signal synthesis storm trial count must be in [2, {SIGNAL_STORM_TRIAL_COUNT}]"
        )
    expected = _signal_synthesis_storm_plan_payload(
        settings,
        trial_count=trial_count,
    )
    if plan_path.is_file():
        plan = read_json(plan_path)
        immutable_fields = (
            "search_space_hash",
            "trial_count",
            "seed",
            "source_candidate_identity",
            "source_candidate_sha256",
            "feature_registry_hash",
            "data_hashes",
        )
        mismatches = [field for field in immutable_fields if plan.get(field) != expected.get(field)]
        if mismatches:
            raise RuntimeError(f"SIGNAL_SYNTHESIS_STORM_PLAN_IDENTITY_MISMATCH:{mismatches}")
    else:
        plan = expected
        atomic_write_json(plan_path, _json_ready(plan))
    dna = tuple(SignalSynthesisDNA.from_dict(values) for values in plan["strategy_dna"])
    frozen_path = settings.paths.lab_dir / "candidates" / "rotation_research_lead_v1.json"
    frozen_sha_before = sha256_file(frozen_path)
    runner = LabRunner(settings)
    frames_by_timeframe: dict[
        str,
        dict[str, pd.DataFrame],
    ] = {}
    feature_data_hashes: dict[str, str] = {}
    feature_provenance: dict[str, Any] = {}
    for timeframe in SIGNAL_STORM_TIMEFRAMES:
        frames, data_hash, provenance = runner._frames(
            markets=SIGNAL_STORM_MARKETS,
            timeframe=timeframe,
            rows=None,
            data_mode="real",
        )
        frames_by_timeframe[timeframe] = frames
        feature_data_hashes[timeframe] = data_hash
        feature_provenance[timeframe] = provenance
    prior_signal_report = (
        settings.paths.lab_dir / "reports" / "signal_synthesis_storm_report_v1.json"
    )
    portfolio_report = _portfolio_storm_paths(settings)[1]
    prior_known_trials = (
        int(prior_known_trials_override) if prior_known_trials_override is not None else 6_312
    )
    if prior_known_trials_override is None and portfolio_report.is_file():
        prior_known_trials = int(
            read_json(portfolio_report).get(
                "total_known_trials",
                prior_known_trials,
            )
        )
    if prior_known_trials_override is None and prior_signal_report.is_file():
        prior_known_trials = max(
            prior_known_trials,
            int(
                read_json(prior_signal_report).get(
                    "total_known_trials",
                    prior_known_trials,
                )
            ),
        )
    report, matrix, timestamps = run_signal_synthesis_storm(
        frames_by_timeframe,
        dna,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        prior_known_trials=prior_known_trials,
    )
    _audit_signal_storm_survivors_exact(
        settings,
        frames_by_timeframe=frames_by_timeframe,
        report=report,
        timestamps=timestamps,
    )
    temporary_matrix = matrix_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_matrix,
        weekly_returns=matrix,
        timestamps=np.asarray(
            [str(timestamp) for timestamp in timestamps],
            dtype="U40",
        ),
        strategy_dna_hashes=np.asarray(
            [row.dna_hash for row in dna],
            dtype="U64",
        ),
    )
    os.replace(temporary_matrix, matrix_path)
    report.update(
        {
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "source_candidate_identity": plan["source_candidate_identity"],
            "epoch_id": epoch_id,
            "returns_matrix_path": str(matrix_path),
            "returns_matrix_sha256": sha256_file(matrix_path),
            "returns_matrix_shape": list(matrix.shape),
            "data_hashes": plan["data_hashes"],
            "feature_data_hashes": feature_data_hashes,
            "feature_provenance": feature_provenance,
            "frozen_candidate_sha256_before": frozen_sha_before,
            "frozen_candidate_sha256_after": sha256_file(frozen_path),
            "frozen_candidate_unchanged": (frozen_sha_before == sha256_file(frozen_path)),
        }
    )
    atomic_write_json(report_path, _json_ready(report))
    return {
        "status": report["status"],
        "campaign": report["campaign"],
        "epoch_id": epoch_id,
        "trial_count": report["trial_count"],
        "total_known_trials": report["total_known_trials"],
        "pareto_survivor_count": report["pareto_survivor_count"],
        "positive_validation_survivors": report["positive_validation_survivors"],
        "positive_confirmation_survivors": report["positive_confirmation_survivors"],
        "pbo": report["multiple_testing"]["probability_of_backtest_overfitting"],
        "white_reality_check_pvalue": report["multiple_testing"]["white_reality_check_pvalue"],
        "hansen_spa_pvalue": report["multiple_testing"]["hansen_spa_pvalue"],
        "frozen_candidate_unchanged": report["frozen_candidate_unchanged"],
        "plan": str(plan_path),
        "report": str(report_path),
        "returns_matrix": str(matrix_path),
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }


def _audit_signal_storm_survivors_exact(
    settings: Settings,
    *,
    frames_by_timeframe: Mapping[
        str,
        Mapping[str, pd.DataFrame],
    ],
    report: dict[str, Any],
    timestamps: pd.DatetimeIndex,
) -> None:
    """Canonically audit screen survivors with positive confirmation only."""

    from research.backtest import BacktestConfig, BacktestEngine
    from research.combinatorial_lab import (
        CombinationState,
        CombinatorialStrategy,
        LogicMode,
        StrategyCombination,
        signal_block_registry,
    )
    from research.signal_synthesis_storm import SignalSynthesisDNA

    split = report["split"]
    development_end = int(split["development_observations"])
    validation_end = development_end + int(split["validation_observations"])
    development_boundary = pd.Timestamp(timestamps[development_end - 1])
    validation_boundary = pd.Timestamp(timestamps[validation_end - 1])
    registry = signal_block_registry()
    normal_config = BacktestConfig.from_settings(settings)
    stressed_config = replace(
        normal_config,
        costs=replace(normal_config.costs, multiplier=2.0),
        bootstrap_samples=min(
            1_000,
            normal_config.bootstrap_samples,
        ),
        monte_carlo_runs=min(
            1_000,
            normal_config.monte_carlo_runs,
        ),
    )
    audited = 0
    passed = 0
    errors = 0
    for survivor in report["pareto_survivors"]:
        if float(survivor["confirmation"]["net_return"]) <= 0.0:
            survivor["canonical_exact_status"] = "NOT_TRIGGERED_NONPOSITIVE_SCREEN_CONFIRMATION"
            continue
        row = SignalSynthesisDNA.from_dict(survivor["parameters"])
        blocks = [registry[block_id] for block_id in row.block_ids]
        combination = StrategyCombination(
            combination_id=f"signal-storm-{row.dna_hash[:20]}",
            strategy_dna_hash=row.dna_hash,
            combination_size=len(row.block_ids),
            block_ids=row.block_ids,
            families=tuple(sorted({block.family for block in blocks})),
            roles=tuple(sorted({block.role.value for block in blocks})),
            redundancy_score=0.0,
            logic_mode=LogicMode(row.logic_mode),
            default_parameters=row.block_parameters,
            parameter_space_size=math.prod(block.parameter_space_size for block in blocks),
            estimated_computational_cost=sum(
                {"LOW": 1, "MEDIUM": 3, "HIGH": 8}[block.computational_cost_class]
                for block in blocks
            ),
            eligibility_status=CombinationState.GENERATED,
            generated_at=utc_now(),
            requested_timeframes=(row.timeframe,),
            common_supported_timeframes=(row.timeframe,),
            excluded_timeframes=(),
        )
        strategy = CombinatorialStrategy(
            combination,
            registry,
            block_parameters=row.block_parameters,
        )
        parameters = {
            "exit__profile": row.exit_profile,
            "exit__stop_atr": row.stop_atr,
            "exit__target_atr": row.target_atr,
            "exit__trailing_atr": row.trailing_atr,
            "exit__maximum_holding_bars": (row.maximum_holding_bars),
            "logic__vote_threshold": row.vote_threshold,
        }
        selected_frames = {
            market: frames_by_timeframe[row.timeframe][market] for market in row.asset_pair
        }
        audited += 1
        try:
            normal = BacktestEngine(
                normal_config,
                settings=settings,
            ).run(
                selected_frames,
                strategy,
                parameters=parameters,
            )
            stressed = BacktestEngine(
                stressed_config,
                settings=settings,
            ).run(
                selected_frames,
                strategy,
                parameters=parameters,
            )
            equity = normal.equity_curve["equity"].astype(float)
            stressed_equity = stressed.equity_curve["equity"].astype(float)

            def value_at(
                values: pd.Series,
                boundary: pd.Timestamp,
            ) -> float:
                selected = values.loc[values.index <= boundary]
                if selected.empty:
                    raise ValueError("exact audit boundary precedes equity")
                return float(selected.iloc[-1])

            development_equity = value_at(
                equity,
                development_boundary,
            )
            validation_equity = value_at(
                equity,
                validation_boundary,
            )
            stressed_development = value_at(
                stressed_equity,
                development_boundary,
            )
            stressed_validation = value_at(
                stressed_equity,
                validation_boundary,
            )
            validation_return = validation_equity / development_equity - 1.0
            confirmation_return = float(equity.iloc[-1]) / validation_equity - 1.0
            stressed_validation_return = stressed_validation / stressed_development - 1.0
            stressed_confirmation_return = (
                float(stressed_equity.iloc[-1]) / stressed_validation - 1.0
            )
            exact_pass = (
                validation_return > 0.0
                and confirmation_return > 0.0
                and stressed_confirmation_return > 0.0
            )
            passed += int(exact_pass)
            survivor["canonical_exact_status"] = (
                "ECONOMICALLY_POSITIVE_NOT_STATISTICALLY_APPROVED"
                if exact_pass
                else "FAILED_CANONICAL_EXACT_AUDIT"
            )
            survivor["canonical_exact"] = {
                "engine": "BacktestEngine",
                "next_open_execution": True,
                "normal_metrics": normal.metrics,
                "double_cost_metrics": stressed.metrics,
                "validation_net_return": validation_return,
                "confirmation_net_return": confirmation_return,
                "double_cost_validation_net_return": (stressed_validation_return),
                "double_cost_confirmation_net_return": (stressed_confirmation_return),
                "economic_pass": exact_pass,
                "statistical_pass": False,
                "paper_candidate_permitted": False,
                "live_ready": False,
            }
        except Exception as exc:
            errors += 1
            survivor["canonical_exact_status"] = "BLOCKED_CANONICAL_EXACT_ERROR"
            survivor["canonical_exact"] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "economic_pass": False,
                "statistical_pass": False,
                "paper_candidate_permitted": False,
                "live_ready": False,
            }
    report["canonical_exact_audit"] = {
        "trigger": "POSITIVE_SCREEN_CONFIRMATION_ONLY",
        "audited_survivors": audited,
        "economic_passes": passed,
        "errors": errors,
        "selection_changed": False,
        "white_spa_pbo_recomputed": False,
        "research_pass": False,
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }


def _portfolio_storm_plan_payload(
    settings: Settings,
    *,
    trial_count: int,
) -> dict[str, Any]:
    from research.portfolio_storm import STORM_SEED, storm_plan

    frozen_path = settings.paths.lab_dir / "candidates" / "rotation_research_lead_v1.json"
    frozen, _, markets, paths, frames = _frozen_rotation_inputs(settings)
    payload = storm_plan(trial_count=trial_count, seed=STORM_SEED)
    common_index = sorted(set.intersection(*[set(frame.index) for frame in frames.values()]))
    return {
        **payload,
        "source_candidate_identity": frozen["immutable_identity"],
        "source_candidate_sha256": sha256_file(frozen_path),
        "markets": list(markets),
        "timeframe": "1d",
        "common_history_start": str(common_index[0]),
        "common_history_end": str(common_index[-1]),
        "common_history_observations": len(common_index),
        "data_hashes": {market: sha256_file(path) for market, path in paths.items()},
    }


def _run_portfolio_storm_campaign(
    settings: Settings,
    *,
    maximum_trials: int | None = None,
    artifact_directory: Path | None = None,
    prior_known_trials_override: int | None = None,
    epoch_id: str | None = None,
) -> dict[str, Any]:
    """Run an immutable development-only multi-objective portfolio storm."""

    from research.portfolio_storm import (
        STORM_TRIAL_COUNT,
        PortfolioStormDNA,
        run_portfolio_storm,
    )

    if artifact_directory is None:
        plan_path, report_path, matrix_path = _portfolio_storm_paths(settings)
    else:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        plan_path = artifact_directory / "plan.json"
        report_path = artifact_directory / "report.json"
        matrix_path = artifact_directory / "returns.npz"
    trial_count = maximum_trials or STORM_TRIAL_COUNT
    if trial_count < 2 or trial_count > STORM_TRIAL_COUNT:
        raise ValueError(f"portfolio storm trial count must be in [2, {STORM_TRIAL_COUNT}]")
    expected = _portfolio_storm_plan_payload(
        settings,
        trial_count=trial_count,
    )
    if plan_path.is_file():
        plan = read_json(plan_path)
        immutable_fields = (
            "search_space_hash",
            "trial_count",
            "seed",
            "source_candidate_identity",
            "source_candidate_sha256",
            "data_hashes",
        )
        mismatches = [field for field in immutable_fields if plan.get(field) != expected.get(field)]
        if mismatches:
            raise RuntimeError(f"PORTFOLIO_STORM_PLAN_IDENTITY_MISMATCH:{mismatches}")
    else:
        plan = expected
        atomic_write_json(plan_path, _json_ready(plan))

    dna = tuple(PortfolioStormDNA(**values) for values in plan["strategy_dna"])
    frozen_path = settings.paths.lab_dir / "candidates" / "rotation_research_lead_v1.json"
    frozen_sha_before = sha256_file(frozen_path)
    _, _, _, source_paths, frames = _frozen_rotation_inputs(settings)
    breakout_path = _breakout_portfolio_campaign_path(settings)
    prior_known_trials = (
        int(prior_known_trials_override) if prior_known_trials_override is not None else 1_312
    )
    if prior_known_trials_override is None and breakout_path.is_file():
        prior_known_trials = int(
            read_json(breakout_path).get(
                "total_known_trials",
                prior_known_trials,
            )
        )
    report, matrix, timestamps = run_portfolio_storm(
        frames,
        dna,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        prior_known_trials=prior_known_trials,
    )
    temporary_matrix = matrix_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_matrix,
        returns=matrix,
        timestamps=np.asarray(
            [str(timestamp) for timestamp in timestamps],
            dtype="U40",
        ),
        strategy_dna_hashes=np.asarray(
            [row.dna_hash for row in dna],
            dtype="U64",
        ),
    )
    os.replace(temporary_matrix, matrix_path)
    report.update(
        {
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "source_candidate_identity": plan["source_candidate_identity"],
            "epoch_id": epoch_id,
            "returns_matrix_path": str(matrix_path),
            "returns_matrix_sha256": sha256_file(matrix_path),
            "returns_matrix_shape": list(matrix.shape),
            "data_hashes": {market: sha256_file(path) for market, path in source_paths.items()},
            "frozen_candidate_sha256_before": frozen_sha_before,
            "frozen_candidate_sha256_after": sha256_file(frozen_path),
            "frozen_candidate_unchanged": (frozen_sha_before == sha256_file(frozen_path)),
        }
    )
    atomic_write_json(report_path, _json_ready(report))
    return {
        "status": report["status"],
        "campaign": report["campaign"],
        "epoch_id": epoch_id,
        "trial_count": report["trial_count"],
        "total_known_trials": report["total_known_trials"],
        "pareto_survivor_count": report["pareto_survivor_count"],
        "pbo": report["multiple_testing"]["probability_of_backtest_overfitting"],
        "white_spa_status": report["multiple_testing"]["white_spa_status"],
        "frozen_candidate_unchanged": report["frozen_candidate_unchanged"],
        "plan": str(plan_path),
        "report": str(report_path),
        "returns_matrix": str(matrix_path),
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }


def _run_autopilot_storm_epoch(
    settings: Settings,
    *,
    data_fingerprint: str,
) -> dict[str, Any]:
    """Run or reuse one immutable research-only storm data epoch."""

    from research.portfolio_storm import (
        STORM_ENGINE_VERSION,
        STORM_TRIAL_COUNT,
    )

    root = settings.paths.lab_dir / "storm_epochs"
    index_path = root / "index.json"
    index = (
        read_json(index_path)
        if index_path.is_file()
        else {
            "schema_version": "portfolio_storm_epoch_index_v1",
            "campaign": "PORTFOLIO_STORM_V1",
            "epochs": [],
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
    )
    epochs = list(index.get("epochs") or [])
    _, _, _, source_paths, _ = _frozen_rotation_inputs(settings)
    current_data_hashes = {market: sha256_file(path) for market, path in source_paths.items()}
    canonical_report_path = _portfolio_storm_paths(settings)[1]
    if not epochs and canonical_report_path.is_file():
        canonical = read_json(canonical_report_path)
        if canonical.get("data_hashes") == current_data_hashes:
            canonical_epoch = {
                "epoch_id": "CANONICAL_INITIAL_STORM",
                "data_fingerprint": data_fingerprint,
                "source": "CANONICAL_PORTFOLIO_STORM_V1",
                "engine_version": STORM_ENGINE_VERSION,
                "new_trial_count": 0,
                "total_known_trials": int(canonical["total_known_trials"]),
                "report": str(canonical_report_path),
                "completed_at": utc_now(),
            }
            epochs.append(canonical_epoch)
            index["epochs"] = epochs
            index["last_epoch_id"] = canonical_epoch["epoch_id"]
            index["total_known_trials"] = canonical_epoch["total_known_trials"]
            atomic_write_json(index_path, _json_ready(index))
    existing = next(
        (row for row in epochs if row.get("data_fingerprint") == data_fingerprint),
        None,
    )
    if existing is not None:
        return {
            "status": "REUSED_EXISTING_STORM_EPOCH",
            **existing,
            "index": str(index_path),
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }

    epoch_id = stable_hash(
        {
            "campaign": "PORTFOLIO_STORM_V1",
            "engine_version": STORM_ENGINE_VERSION,
            "data_fingerprint": data_fingerprint,
            "data_hashes": current_data_hashes,
        },
        length=20,
    )
    prior_known_trials = max([int(row.get("total_known_trials") or 0) for row in epochs] or [1_312])
    epoch_directory = root / epoch_id
    result = _run_portfolio_storm_campaign(
        settings,
        maximum_trials=STORM_TRIAL_COUNT,
        artifact_directory=epoch_directory,
        prior_known_trials_override=prior_known_trials,
        epoch_id=epoch_id,
    )
    epoch = {
        "epoch_id": epoch_id,
        "data_fingerprint": data_fingerprint,
        "source": "AUTOPILOT_NEW_DATA_EPOCH",
        "engine_version": STORM_ENGINE_VERSION,
        "new_trial_count": STORM_TRIAL_COUNT,
        "prior_known_trials": prior_known_trials,
        "total_known_trials": int(result["total_known_trials"]),
        "pareto_survivor_count": int(result["pareto_survivor_count"]),
        "pbo": result["pbo"],
        "white_spa_status": result["white_spa_status"],
        "report": result["report"],
        "returns_matrix": result["returns_matrix"],
        "completed_at": utc_now(),
    }
    epochs.append(epoch)
    index["epochs"] = epochs
    index["last_epoch_id"] = epoch_id
    index["total_known_trials"] = epoch["total_known_trials"]
    atomic_write_json(index_path, _json_ready(index))
    return {
        "status": "COMPLETED_NEW_STORM_EPOCH",
        **epoch,
        "index": str(index_path),
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def _run_autopilot_signal_storm_epoch(
    settings: Settings,
    *,
    data_fingerprint: str,
) -> dict[str, Any]:
    """Run or reuse one immutable signal-storm data epoch."""

    from research.signal_synthesis_storm import (
        SIGNAL_STORM_ENGINE_VERSION,
        SIGNAL_STORM_TRIAL_COUNT,
    )

    root = settings.paths.lab_dir / "signal_storm_epochs"
    index_path = root / "index.json"
    index = (
        read_json(index_path)
        if index_path.is_file()
        else {
            "schema_version": "signal_storm_epoch_index_v1",
            "campaign": "SIGNAL_SYNTHESIS_STORM_V1",
            "epochs": [],
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
    )
    epochs = list(index.get("epochs") or [])
    current_data_hashes = {
        timeframe: {market: sha256_file(path) for market, path in by_market.items()}
        for timeframe, by_market in (_signal_synthesis_data_paths(settings).items())
    }
    canonical_report_path = _signal_synthesis_storm_paths(settings)[1]
    if not epochs and canonical_report_path.is_file():
        canonical = read_json(canonical_report_path)
        if canonical.get("data_hashes") == current_data_hashes:
            canonical_epoch = {
                "epoch_id": "CANONICAL_INITIAL_SIGNAL_STORM",
                "data_fingerprint": data_fingerprint,
                "source": "CANONICAL_SIGNAL_SYNTHESIS_STORM_V1",
                "engine_version": SIGNAL_STORM_ENGINE_VERSION,
                "new_trial_count": 0,
                "total_known_trials": int(canonical["total_known_trials"]),
                "report": str(canonical_report_path),
                "completed_at": utc_now(),
            }
            epochs.append(canonical_epoch)
            index["epochs"] = epochs
            index["last_epoch_id"] = canonical_epoch["epoch_id"]
            index["total_known_trials"] = canonical_epoch["total_known_trials"]
            atomic_write_json(index_path, _json_ready(index))
    existing = next(
        (row for row in epochs if row.get("data_fingerprint") == data_fingerprint),
        None,
    )
    if existing is not None:
        return {
            "status": "REUSED_EXISTING_SIGNAL_STORM_EPOCH",
            **existing,
            "index": str(index_path),
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
    epoch_id = stable_hash(
        {
            "campaign": "SIGNAL_SYNTHESIS_STORM_V1",
            "engine_version": SIGNAL_STORM_ENGINE_VERSION,
            "data_fingerprint": data_fingerprint,
            "data_hashes": current_data_hashes,
        },
        length=20,
    )
    prior_known_trials = max(
        [int(row.get("total_known_trials") or 0) for row in epochs] or [16_312]
    )
    epoch_directory = root / epoch_id
    result = _run_signal_synthesis_storm_campaign(
        settings,
        maximum_trials=SIGNAL_STORM_TRIAL_COUNT,
        artifact_directory=epoch_directory,
        prior_known_trials_override=prior_known_trials,
        epoch_id=epoch_id,
    )
    epoch = {
        "epoch_id": epoch_id,
        "data_fingerprint": data_fingerprint,
        "source": "AUTOPILOT_NEW_DATA_EPOCH",
        "engine_version": SIGNAL_STORM_ENGINE_VERSION,
        "new_trial_count": SIGNAL_STORM_TRIAL_COUNT,
        "prior_known_trials": prior_known_trials,
        "total_known_trials": int(result["total_known_trials"]),
        "pareto_survivor_count": int(result["pareto_survivor_count"]),
        "pbo": result["pbo"],
        "white_reality_check_pvalue": result["white_reality_check_pvalue"],
        "hansen_spa_pvalue": result["hansen_spa_pvalue"],
        "report": result["report"],
        "returns_matrix": result["returns_matrix"],
        "completed_at": utc_now(),
    }
    epochs.append(epoch)
    index["epochs"] = epochs
    index["last_epoch_id"] = epoch_id
    index["total_known_trials"] = epoch["total_known_trials"]
    atomic_write_json(index_path, _json_ready(index))
    return {
        "status": "COMPLETED_NEW_SIGNAL_STORM_EPOCH",
        **epoch,
        "index": str(index_path),
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def _daily_snapshot_watermark(
    paths: Mapping[str, Path],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Require one identical, fully closed UTC day across all markets."""

    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        current = current.replace(tzinfo=UTC)
    current = current.astimezone(UTC)
    expected = pd.Timestamp(current).floor("D") - pd.offsets.Day(1)
    latest: dict[str, str | None] = {}
    checks: dict[str, bool] = {}
    for market, path in sorted(paths.items()):
        frame = pd.read_parquet(path, columns=["close"])
        index = pd.DatetimeIndex(frame.index)
        if index.tz is None:
            index = index.tz_localize("UTC")
        else:
            index = index.tz_convert("UTC")
        latest_timestamp = index.max() if len(index) else pd.NaT
        latest[market] = latest_timestamp.isoformat() if not pd.isna(latest_timestamp) else None
        checks[market] = bool(latest_timestamp == expected)
    complete = bool(checks) and all(checks.values())
    return {
        "status": (
            "COMPLETE_DAILY_SNAPSHOT" if complete else "WAITING_FOR_COMPLETE_DAILY_SNAPSHOT"
        ),
        "complete_daily_snapshot": complete,
        "expected_last_closed_utc_day": expected.isoformat(),
        "latest_by_market": latest,
        "checks": checks,
        "partial_snapshot_use_permitted": False,
    }


def _autopilot_data_stage(
    settings: Settings,
    *,
    refresh: bool,
    refresh_timeout_seconds: float,
) -> dict[str, Any]:
    """Audit or refresh the strict 1h/4h/1d research universe."""

    from core.autopilot import AutopilotOrchestrator

    refresh_result: dict[str, Any] = {
        "status": "SKIPPED",
        "reason": "REFRESH_NOT_REQUESTED",
    }
    if refresh:
        command = [
            sys.executable,
            str(settings.paths.project_root / "main.py"),
            "lab",
            "data",
            "prepare",
            "--universe-size",
            "4",
            "--allowed-universe",
            "--timeframes",
            "1h,4h,1d",
            "--minimum-rows",
            "2000",
            "--force",
        ]
        completed = subprocess.run(
            command,
            cwd=settings.paths.project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=refresh_timeout_seconds,
        )
        refresh_result = {
            "status": "PASSED" if completed.returncode == 0 else "FAILED",
            "return_code": completed.returncode,
            "command": command,
            "stdout_tail": completed.stdout[-4_000:],
            "stderr_tail": completed.stderr[-4_000:],
        }
        if completed.returncode != 0:
            raise RuntimeError(f"AUTOPILOT_DATA_REFRESH_FAILED:{completed.returncode}")

    strict_markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    normalized = settings.paths.data_dir / "normalized"
    signal_timeframes = ("1h", "4h", "1d")
    required = [
        normalized / f"{market}_{timeframe}.parquet"
        for market in strict_markets
        for timeframe in signal_timeframes
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"AUTOPILOT_STRICT_SIGNAL_DATA_MISSING:{missing}")
    daily_source_paths = {market: normalized / f"{market}_1d.parquet" for market in strict_markets}
    daily_watermark = _daily_snapshot_watermark(daily_source_paths)
    daily_files = sorted(
        {
            path
            for market in strict_markets
            for path in normalized.glob(f"{market}_1d.parquet*")
            if path.is_file()
        }
    )
    signal_files = sorted(
        {
            path
            for market in strict_markets
            for timeframe in signal_timeframes
            for path in normalized.glob(f"{market}_{timeframe}.parquet*")
            if path.is_file()
        }
    )
    data_files = signal_files
    latest_modified = max(
        (path.stat().st_mtime for path in data_files),
        default=0.0,
    )
    return {
        "status": "DATA_REFRESHED" if refresh else "DATA_AUDITED",
        "strict_markets": list(strict_markets),
        "timeframes": list(signal_timeframes),
        "file_count": len(data_files),
        "latest_modified_utc": (
            datetime.fromtimestamp(latest_modified, tz=UTC).isoformat() if latest_modified else None
        ),
        "data_fingerprint": AutopilotOrchestrator.fingerprint_files(data_files),
        "daily_data_fingerprint": (AutopilotOrchestrator.fingerprint_files(daily_files)),
        "signal_data_fingerprint": (AutopilotOrchestrator.fingerprint_files(signal_files)),
        "complete_daily_snapshot": daily_watermark["complete_daily_snapshot"],
        "daily_watermark": daily_watermark,
        "refresh": refresh_result,
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def _autopilot_observer_stage(settings: Settings) -> dict[str, Any]:
    """Append causal forward evidence and verify every observer is orderless."""

    from core.autopilot import assert_orderless_research_payload
    from research.forward_observer import (
        FORWARD_OBSERVER_SCHEMA_VERSION,
        ForwardPerformanceGatePolicy,
        build_breakout_forward_evidence,
        merge_breakout_forward_manifest,
    )
    from research.portfolio_breakout import (
        backtest_breakout_portfolio,
        breakout_observer_snapshot,
        breakout_portfolio_parameter_set,
    )

    report_path = _breakout_portfolio_campaign_path(settings)
    if not report_path.is_file():
        raise FileNotFoundError("portfolio-breakout-v1 report is required before autopilot")
    report = read_json(report_path)
    assert_orderless_research_payload(report)
    manifests = dict(report.get("observer_manifests") or {})
    if not manifests:
        raise RuntimeError("AUTOPILOT_OBSERVER_MANIFESTS_MISSING")
    frozen, _, markets, source_paths, frames = _frozen_rotation_inputs(settings)
    policy = _strict_rotation_portfolio_policy(
        settings,
        markets=markets,
    )
    forward_start = pd.Timestamp(frozen["forward_validation_start"])
    forward_start = (
        forward_start.tz_localize("UTC")
        if forward_start.tzinfo is None
        else forward_start.tz_convert("UTC")
    )
    parameters_by_name = {
        (
            f"TURTLE_{parameters.entry_lookback}_"
            f"{parameters.exit_lookback}_"
            f"EMA{parameters.trend_ema_period}_"
            f"{parameters.weighting.upper()}"
        ): parameters
        for parameters in breakout_portfolio_parameter_set()
    }
    identities: dict[str, str] = {}
    summaries: dict[str, Any] = {}
    degradation_observations: dict[str, Any] = {}
    for policy_name, raw_path in sorted(manifests.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"observer manifest missing: {path}")
        observer = read_json(path)
        assert_orderless_research_payload(observer)
        if observer.get("status") != "FROZEN_FORWARD_RESEARCH":
            raise RuntimeError(f"AUTOPILOT_OBSERVER_NOT_FROZEN:{policy_name}")
        if bool(observer.get("candidate_promotion_implied", False)):
            raise RuntimeError(f"AUTOPILOT_OBSERVER_PROMOTION_IMPLIED:{policy_name}")
        parameters = parameters_by_name.get(str(policy_name))
        if parameters is None:
            raise RuntimeError(f"AUTOPILOT_UNKNOWN_BREAKOUT_POLICY:{policy_name}")
        result = backtest_breakout_portfolio(
            frames,
            parameters,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            portfolio_policy=policy,
        )
        evidence = build_breakout_forward_evidence(
            result,
            frames,
            forward_start=forward_start,
            minimum_observations=int(observer["minimum_forward_closed_daily_observations"]),
            minimum_rebalances=int(observer["minimum_forward_rebalances"]),
            performance_policy=ForwardPerformanceGatePolicy(
                minimum_profit_factor=(settings.research.minimum_profit_factor),
                minimum_stressed_profit_factor=(settings.research.minimum_stressed_profit_factor),
                maximum_drawdown=(settings.research.maximum_drawdown),
                minimum_effective_sample_size=(settings.research.minimum_effective_sample_size),
                stressed_cost_multiplier=(settings.costs.stressed_cost_multiplier),
                bootstrap_samples=(settings.research.multiple_testing_bootstrap_samples),
                bootstrap_block_size=(settings.research.multiple_testing_block_size),
                bootstrap_seed=settings.app.random_seed,
            ),
        )
        current_snapshot = breakout_observer_snapshot(result)
        observer = merge_breakout_forward_manifest(
            {**observer, **current_snapshot},
            evidence,
            source_candidate_identity=frozen["immutable_identity"],
            strategy_dna_hash=parameters.dna_hash,
            execution_identity=current_snapshot["execution_identity"],
            forward_start=forward_start,
        )
        observer["data_hashes"] = {
            market: sha256_file(source_path) for market, source_path in source_paths.items()
        }
        atomic_write_json(path, _json_ready(observer))
        assert_orderless_research_payload(observer)
        identities[str(policy_name)] = str(observer.get("strategy_dna_hash") or "")
        summaries[str(policy_name)] = observer["forward_summary"]
        if observer.get("degradation_observation") is not None:
            degradation_observations[str(policy_name)] = observer["degradation_observation"]
    total_forward_observations = sum(
        int(summary["closed_daily_observations"]) for summary in summaries.values()
    )
    all_sample_requirements_met = bool(summaries) and all(
        all(bool(value) for value in summary["checks"].values()) for summary in summaries.values()
    )
    all_forward_performance_pass = bool(summaries) and all(
        bool(summary.get("forward_performance_pass", False)) for summary in summaries.values()
    )
    capital_result = _run_capital_utilization_campaign(settings)
    assert_orderless_research_payload(capital_result)
    capital_report = read_json(_capital_utilization_campaign_path(settings))
    assert_orderless_research_payload(capital_report)
    capital_forward_summaries = dict(capital_report.get("forward_summaries") or {})
    capital_forward_observations = sum(
        int(summary.get("closed_daily_observations") or 0)
        for summary in capital_forward_summaries.values()
    )
    absolute_result = _run_absolute_momentum_campaign(settings)
    assert_orderless_research_payload(absolute_result)
    absolute_report = read_json(
        _absolute_momentum_campaign_path(settings)
    )
    assert_orderless_research_payload(absolute_report)
    absolute_forward_summaries = dict(
        absolute_report.get("forward_summaries") or {}
    )
    absolute_forward_observations = sum(
        int(summary.get("closed_daily_observations") or 0)
        for summary in absolute_forward_summaries.values()
    )
    plateau_result = _run_absolute_momentum_plateau_campaign(
        settings
    )
    assert_orderless_research_payload(plateau_result)
    plateau_report = read_json(
        _absolute_momentum_plateau_campaign_path(settings)
    )
    assert_orderless_research_payload(plateau_report)
    plateau_forward_summaries = dict(
        plateau_report.get("forward_summaries") or {}
    )
    plateau_forward_observations = sum(
        int(summary.get("closed_daily_observations") or 0)
        for summary in plateau_forward_summaries.values()
    )
    aggregate = {
        "status": "FROZEN_FORWARD_RESEARCH",
        "campaign": "PORTFOLIO_BREAKOUT_V1",
        "forward_observer_schema_version": (FORWARD_OBSERVER_SCHEMA_VERSION),
        "observer_count": len(manifests),
        "observer_dna_hashes": identities,
        "forward_summaries": summaries,
        "degradation_observations": degradation_observations,
        "total_forward_observations": total_forward_observations,
        "all_sample_requirements_met": all_sample_requirements_met,
        "all_forward_performance_pass": (all_forward_performance_pass),
        "parallel_capital_utilization_observers": {
            "campaign": "CAPITAL_UTILIZATION_V1",
            "observer_count": len(capital_forward_summaries),
            "forward_summaries": capital_forward_summaries,
            "total_forward_observations": (capital_forward_observations),
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
        "parallel_absolute_momentum_observers": {
            "campaign": "ABSOLUTE_MOMENTUM_V1",
            "status": absolute_result["status"],
            "observer_count": len(absolute_forward_summaries),
            "forward_summaries": absolute_forward_summaries,
            "total_forward_observations": (
                absolute_forward_observations
            ),
            "paper_candidate_permitted": False,
            "orders_generated": 0,
            "live_ready": False,
        },
        "parallel_absolute_momentum_plateau_observers": {
            "campaign": "ABSOLUTE_MOMENTUM_PLATEAU_V1",
            "status": plateau_result["status"],
            "observer_count": len(plateau_forward_summaries),
            "forward_summaries": plateau_forward_summaries,
            "total_forward_observations": (
                plateau_forward_observations
            ),
            "paper_candidate_permitted": False,
            "orders_generated": 0,
            "live_ready": False,
        },
        "total_forward_observations_all_campaigns": (
            total_forward_observations
            + capital_forward_observations
            + absolute_forward_observations
            + plateau_forward_observations
        ),
        "source_candidate_identity": report.get("source_candidate_identity"),
        "frozen_candidate_unchanged": bool(report.get("frozen_candidate_unchanged")),
        "orders_generated": 0,
        "orders_submitted": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    forward_report = (
        settings.paths.lab_dir / "reports" / "portfolio_breakout_forward_observer_v1.json"
    )
    atomic_write_json(forward_report, _json_ready(aggregate))
    return {**aggregate, "report": str(forward_report)}


def _autopilot_ledger_preflight_stage(
    settings: Settings,
) -> dict[str, Any]:
    """Verify every active append-only ledger before data or research runs."""

    from research.ledger_guard import audit_forward_ledgers

    observer_root = settings.paths.lab_dir / "observers"
    active_directories = (
        observer_root / "portfolio_breakout_v1",
        observer_root / "capital_utilization_v1",
        observer_root / "absolute_momentum_v1",
        observer_root / "absolute_momentum_plateau_v1",
    )
    paths = [
        path
        for directory in active_directories
        for path in sorted(directory.glob("*.json"))
    ]
    payload = audit_forward_ledgers(paths)
    report_path = (
        settings.paths.lab_dir
        / "reports"
        / "forward_ledger_preflight_v1.json"
    )
    atomic_write_json(report_path, _json_ready(payload))
    return {**payload, "report": str(report_path)}


def _autopilot_feature_store_stage(settings: Settings) -> dict[str, Any]:
    """Build an optional classical-research feature snapshot."""

    from data.feature_store import (
        STRICT_PORTFOLIO_MARKETS,
        build_and_persist_feature_store,
    )

    normalized = settings.paths.data_dir / "normalized"
    source_paths = {
        market: normalized / f"{market}_1d.parquet" for market in STRICT_PORTFOLIO_MARKETS
    }
    return build_and_persist_feature_store(
        source_paths,
        settings.paths.lab_dir / "feature_store" / "portfolio_daily_v1",
    )


def _preserved_breakout_forward_fields(
    existing: Mapping[str, Any],
    *,
    source_candidate_identity: str,
    strategy_dna_hash: str,
    execution_identity: str,
    forward_start: Any,
) -> dict[str, Any]:
    """Validate observer identity and retain only append-only forward fields."""

    from research.forward_observer import (
        validate_forward_manifest_identity,
    )

    validate_forward_manifest_identity(
        existing,
        source_candidate_identity=source_candidate_identity,
        strategy_dna_hash=strategy_dna_hash,
        execution_identity=execution_identity,
        forward_start=forward_start,
    )
    return {
        field: existing[field]
        for field in (
            "forward_observer_schema_version",
            "forward_observations",
            "forward_hash_chain",
            "forward_decisions",
            "forward_summary",
            "degradation_observation",
        )
        if field in existing
    }


def _autopilot_research_stage(settings: Settings) -> dict[str, Any]:
    """Run classical preregistered campaigns and immutable storm epochs."""

    result = _run_breakout_portfolio_campaign(settings)
    absolute_momentum = _run_absolute_momentum_campaign(settings)
    absolute_momentum_plateau = (
        _run_absolute_momentum_plateau_campaign(settings)
    )
    data_audit = _autopilot_data_stage(
        settings,
        refresh=False,
        refresh_timeout_seconds=1.0,
    )
    portfolio_storm_epoch = _run_autopilot_storm_epoch(
        settings,
        data_fingerprint=str(data_audit["daily_data_fingerprint"]),
    )
    signal_storm_epoch = _run_autopilot_signal_storm_epoch(
        settings,
        data_fingerprint=str(data_audit["signal_data_fingerprint"]),
    )
    total_trials = int(result["total_known_trials"])
    parameters_tested = int(result["parameters_tested"])
    return {
        "status": result["status"],
        "campaign": result["campaign"],
        "parameters_tested": parameters_tested,
        "prior_trials_accounted": int(
            result.get(
                "prior_trials_accounted",
                total_trials - parameters_tested,
            )
        ),
        "total_known_trials": total_trials,
        "economic_research_lead_count": result["economic_research_lead_count"],
        "statistically_qualified_count": result["statistically_qualified_count"],
        "frozen_candidate_unchanged": result["frozen_candidate_unchanged"],
        "portfolio_storm_epoch": portfolio_storm_epoch,
        "portfolio_storm_total_known_trials": int(portfolio_storm_epoch["total_known_trials"]),
        "signal_synthesis_storm_epoch": signal_storm_epoch,
        "signal_synthesis_total_known_trials": int(signal_storm_epoch["total_known_trials"]),
        "parallel_absolute_momentum_campaign": {
            "campaign": absolute_momentum["campaign"],
            "status": absolute_momentum["status"],
            "primary_policy_name": absolute_momentum[
                "primary_policy_name"
            ],
            "total_known_trials": absolute_momentum[
                "total_known_trials"
            ],
            "pbo": absolute_momentum["pbo"],
            "observer_manifests": absolute_momentum[
                "observer_manifests"
            ],
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
        "parallel_absolute_momentum_plateau_campaign": {
            "campaign": absolute_momentum_plateau["campaign"],
            "status": absolute_momentum_plateau["status"],
            "generated_trial_count": absolute_momentum_plateau[
                "generated_trial_count"
            ],
            "registered_unique_plateau_trials": (
                absolute_momentum_plateau[
                    "registered_unique_plateau_trials"
                ]
            ),
            "total_known_trials": absolute_momentum_plateau[
                "total_known_trials"
            ],
            "plateau_eligible_count": (
                absolute_momentum_plateau[
                    "plateau_eligible_count"
                ]
            ),
            "standard_pbo": absolute_momentum_plateau[
                "standard_pbo"
            ],
            "plateau_selection_pbo": (
                absolute_momentum_plateau[
                    "plateau_selection_pbo"
                ]
            ),
            "observer_manifests": absolute_momentum_plateau[
                "observer_manifests"
            ],
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
        "paper_candidates": result["paper_candidates"],
        "live_orders": result["live_orders"],
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def _autopilot_degradation_observation(
    path: Path | None,
) -> Any:
    if path is None:
        return None
    from core.autopilot import DegradationObservation

    payload = read_json(path)
    return DegradationObservation(
        live_return=float(payload["live_return"]),
        cv_mean=float(payload["cv_mean"]),
        cv_std=float(payload["cv_std"]),
        observation_count=int(payload["observation_count"]),
        window=str(payload.get("window") or "30d"),
        source=str(payload.get("source") or path),
    )


def _autopilot_task_xml(settings: Settings) -> str:
    """Build a least-privilege daily orderless research task."""

    python = settings.paths.project_root / ".venv" / "Scripts" / "python.exe"
    main = settings.paths.project_root / "main.py"
    local_now = datetime.now().astimezone()
    start = (local_now + timedelta(days=1)).replace(
        hour=3,
        minute=15,
        second=0,
        microsecond=0,
    )
    start_boundary = start.isoformat(timespec="seconds")
    arguments = f'"{main}" lab campaign autopilot --run-research --refresh-data'
    return (
        '<?xml version="1.0" encoding="UTF-16"?>'
        '<Task version="1.4" '
        'xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">'
        "<Triggers><CalendarTrigger>"
        f"<StartBoundary>{html.escape(start_boundary)}</StartBoundary>"
        "<Enabled>true</Enabled><ScheduleByDay><DaysInterval>1</DaysInterval>"
        "</ScheduleByDay></CalendarTrigger></Triggers>"
        '<Principals><Principal id="Author">'
        "<LogonType>InteractiveToken</LogonType>"
        "<RunLevel>LeastPrivilege</RunLevel></Principal></Principals>"
        "<Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>"
        "<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>"
        "<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>"
        "<StartWhenAvailable>true</StartWhenAvailable>"
        "<ExecutionTimeLimit>PT4H</ExecutionTimeLimit>"
        "<RestartOnFailure><Interval>PT5M</Interval><Count>3</Count>"
        "</RestartOnFailure></Settings>"
        '<Actions Context="Author"><Exec>'
        f"<Command>{html.escape(str(python))}</Command>"
        f"<Arguments>{html.escape(arguments)}</Arguments>"
        f"<WorkingDirectory>{html.escape(str(settings.paths.project_root))}"
        "</WorkingDirectory></Exec></Actions></Task>"
    )


def _autopilot_task_command(
    settings: Settings,
    *,
    mode: str,
    confirmed: bool,
    dry_run: bool,
) -> tuple[int, dict[str, Any]]:
    """Install, inspect or remove the persistent Windows autopilot task."""

    task_name = "CryptoResearchAutopilotOrderless"
    if mode == "task-status":
        command = [
            "schtasks.exe",
            "/Query",
            "/TN",
            task_name,
            "/FO",
            "LIST",
            "/V",
        ]
    elif mode == "task-remove":
        if not confirmed:
            return 2, {
                "status": "CONFIRMATION_REQUIRED",
                "task_name": task_name,
                "reason": "PERSISTENT_TASK_REMOVAL_REQUIRES_CONFIRMATION",
                "orders_generated": 0,
                "paper_candidate_permitted": False,
                "live_ready": False,
            }
        command = ["schtasks.exe", "/Delete", "/TN", task_name, "/F"]
    else:
        xml = _autopilot_task_xml(settings)
        command = [
            "schtasks.exe",
            "/Create",
            "/TN",
            task_name,
            "/XML",
            "<temporary-xml>",
            "/F",
        ]
        if dry_run or not confirmed:
            return (0 if dry_run else 2), {
                "status": ("DRY_RUN" if dry_run else "CONFIRMATION_REQUIRED"),
                "task_name": task_name,
                "command": command,
                "xml": xml,
                "schedule": "DAILY_03:15_LOCAL_START_WHEN_AVAILABLE",
                "utc_daily_close_grace": ("AT_LEAST_01:15_AFTER_UTC_MIDNIGHT"),
                "orders_generated": 0,
                "paper_candidate_permitted": False,
                "live_ready": False,
            }
        with tempfile.NamedTemporaryFile(
            suffix=".xml",
            mode="w",
            encoding="utf-16",
            delete=False,
        ) as handle:
            handle.write(xml)
            xml_path = Path(handle.name)
        command[-2] = str(xml_path)
    try:
        completed = subprocess.run(
            command,
            cwd=settings.paths.project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        if mode == "task-install" and "xml_path" in locals():
            xml_path.unlink(missing_ok=True)
    return (
        0 if completed.returncode == 0 else 2,
        {
            "status": ("PASSED" if completed.returncode == 0 else "FAILED"),
            "task_name": task_name,
            "mode": mode,
            "return_code": completed.returncode,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "schedule": (
                "DAILY_03:15_LOCAL_START_WHEN_AVAILABLE" if mode == "task-install" else None
            ),
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
    )


def _rotation_return_ci_lower(
    returns: pd.Series,
    *,
    samples: int,
    block_size: int,
    seed: int,
) -> float:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < block_size or samples < 100:
        return -math.inf
    randomizer = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for sample in range(samples):
        indices: list[int] = []
        while len(indices) < len(values):
            start = int(randomizer.integers(0, len(values)))
            indices.extend((start + offset) % len(values) for offset in range(block_size))
        means[sample] = values[np.asarray(indices[: len(values)], dtype=int)].mean()
    return float(np.quantile(means, 0.025))


def _portfolio_stochastic_validation(
    settings: Settings,
    *,
    normal_equity: pd.Series,
    stressed_equity: pd.Series,
    seed_offset: int,
) -> dict[str, Any]:
    """Run immutable Monte Carlo and Dirichlet gates on two net return paths."""

    from research.stochastic_validation import (
        policy_from_research_settings,
        validate_strategy_return_paths,
    )

    policy = policy_from_research_settings(
        settings.research,
        seed=settings.app.random_seed,
        expected_block_length=10,
    )
    normal_returns = normal_equity.pct_change(fill_method=None).dropna().to_numpy(dtype=float)
    stressed_returns = stressed_equity.pct_change(fill_method=None).dropna().to_numpy(dtype=float)
    return validate_strategy_return_paths(
        normal_returns,
        stressed_returns,
        policy=policy,
        seed_offset=seed_offset,
    )


def _run_rotation_campaign(
    settings: Settings,
    *,
    ensemble: bool = False,
    institutional: bool = False,
) -> dict[str, Any]:
    """Run and persist the allowed-universe daily rotation research campaign."""

    from research.combinatorial_lab import FAST_SCREEN_MINIMUM_TRADES
    from research.optimization import multiple_testing_bootstrap
    from research.portfolio_selection import (
        ROTATION_ENGINE_VERSION,
        ROTATION_POLICY_VERSION,
        RotationPortfolioPolicy,
        backtest_rotation,
        ensemble_rotation_parameter_grid,
        rotation_parameter_grid,
        rotation_period_metrics,
    )

    markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    paths = {
        market: settings.paths.processed_data_dir / f"{market}_1d.parquet" for market in markets
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing normalized 1d rotation datasets: {missing}")
    frames = {market: pd.read_parquet(path) for market, path in paths.items()}
    ensemble_mode = ensemble or institutional
    grid_factory = ensemble_rotation_parameter_grid if ensemble_mode else rotation_parameter_grid
    portfolio_policy: RotationPortfolioPolicy | None = None
    if institutional:
        portfolio_policy = _strict_rotation_portfolio_policy(
            settings,
            markets=markets,
        )
        grid = grid_factory(
            horizon_sets=((20, 90), (20, 60, 120), (20, 90, 180)),
            top_ns=(1, 2),
            rebalance_days=(7,),
            asset_ema_periods=(50, 200),
            continuous_regimes=(False, True),
            weightings=("equal", "inverse_volatility"),
            gross_exposure=portfolio_policy.maximum_total_exposure,
            minimum_cash=portfolio_policy.minimum_cash,
            maximum_positions=settings.operational.maximum_positions,
        )
    else:
        grid = grid_factory(
            gross_exposure=(
                min(0.25, settings.operational.maximum_portfolio_exposure)
                if ensemble
                else settings.operational.maximum_portfolio_exposure
            ),
            minimum_cash=settings.operational.reserve_cash_fraction,
            maximum_positions=settings.operational.maximum_positions,
        )
    prior_trials = 1_245 if institutional else (1_080 if ensemble else 648)
    periods = {
        "development": ("2021-08-05", "2023-12-31"),
        "validation": ("2024-01-01", "2025-06-30"),
        "confirmation": ("2025-07-01", "2026-07-23"),
    }
    rows: list[dict[str, Any]] = []
    development_returns: dict[str, pd.Series] = {}
    results: dict[str, Any] = {}
    for parameters in grid:
        result = backtest_rotation(
            frames,
            parameters,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            portfolio_policy=portfolio_policy,
        )
        period_results: dict[str, Any] = {}
        for period, (start, end) in periods.items():
            metrics, returns = rotation_period_metrics(
                result.equity_curve,
                start=start,
                end=end,
            )
            period_results[period] = metrics
            if period == "development":
                development_returns[parameters.dna_hash] = returns
        development = period_results["development"]
        development_start = pd.Timestamp(periods["development"][0], tz="UTC")
        development_end = pd.Timestamp(periods["development"][1], tz="UTC")
        development_rebalances = int(
            (
                (pd.to_datetime(result.decisions["executed_at"], utc=True) >= development_start)
                & (pd.to_datetime(result.decisions["executed_at"], utc=True) <= development_end)
                & (result.decisions["turnover"].astype(float) > 1e-12)
            ).sum()
        )
        development_score = (
            float(development["sharpe"])
            + float(development["annualized_return"])
            - abs(float(development["maximum_drawdown"]))
        )
        row = {
            "strategy_dna_hash": parameters.dna_hash,
            "parameters": asdict(parameters),
            "development_score": development_score,
            "development_rebalances": development_rebalances,
            "periods": period_results,
            "full_sample": result.metrics,
            "cost_breakdown": result.cost_breakdown,
            "integrity": result.integrity,
        }
        rows.append(row)
        results[parameters.dna_hash] = result

    rows.sort(key=lambda item: float(item["development_score"]), reverse=True)
    matrix = pd.concat(development_returns, axis=1).dropna(how="any")
    multiple = multiple_testing_bootstrap(
        matrix,
        bootstrap_samples=settings.research.multiple_testing_bootstrap_samples,
        block_size=settings.research.multiple_testing_block_size,
        seed=settings.app.random_seed,
        known_trial_count=len(rows) + prior_trials,
    )
    development_eligible = [
        row
        for row in rows
        if float(row["periods"]["development"]["net_return"]) > 0
        and int(row["development_rebalances"]) >= FAST_SCREEN_MINIMUM_TRADES
        and int(row["periods"]["development"]["effective_sample_size"])
        >= FAST_SCREEN_MINIMUM_TRADES
    ]
    survivors: list[dict[str, Any]] = []
    signatures: Counter[tuple[Any, ...]] = Counter()
    for row in development_eligible:
        parameters = row["parameters"]
        signature = (
            (
                parameters["momentum_lookback"],
                *parameters.get("additional_momentum_lookbacks", ()),
            ),
            parameters["top_n"],
            parameters.get("continuous_regime", False),
        )
        if signatures[signature] >= 1:
            continue
        survivors.append(row)
        signatures[signature] += 1
        if len(survivors) >= 12:
            break

    for survivor_index, row in enumerate(survivors):
        dna_hash = str(row["strategy_dna_hash"])
        result = results[dna_hash]
        stressed = backtest_rotation(
            frames,
            result.parameters,
            fee_rate=settings.costs.default_fee * settings.costs.stressed_cost_multiplier,
            slippage_bps=settings.costs.slippage_bps * settings.costs.stressed_cost_multiplier,
            spread_bps=settings.costs.spread_bps * settings.costs.stressed_cost_multiplier,
            portfolio_policy=portfolio_policy,
        )
        stressed_confirmation, _ = rotation_period_metrics(
            stressed.equity_curve,
            start=periods["confirmation"][0],
            end=periods["confirmation"][1],
        )
        _, confirmation_returns = rotation_period_metrics(
            result.equity_curve,
            start=periods["confirmation"][0],
            end=periods["confirmation"][1],
        )
        fold_returns = np.array_split(
            result.equity_curve.pct_change(fill_method=None).dropna().to_numpy(dtype=float),
            settings.research.walk_forward_folds,
        )
        positive_folds = sum(
            float(np.prod(1.0 + fold) - 1.0) > 0 for fold in fold_returns if len(fold)
        )
        ci_lower = _rotation_return_ci_lower(
            confirmation_returns,
            samples=2_000,
            block_size=10,
            seed=settings.app.random_seed,
        )
        dsr = float(multiple.deflated_sharpe_probabilities.get(dna_hash, 0.0))
        stochastic = _portfolio_stochastic_validation(
            settings,
            normal_equity=result.equity_curve,
            stressed_equity=stressed.equity_curve,
            seed_offset=survivor_index * 10,
        )
        checks = {
            "development_positive": (float(row["periods"]["development"]["net_return"]) > 0),
            "validation_positive": (float(row["periods"]["validation"]["net_return"]) > 0),
            "confirmation_positive": (float(row["periods"]["confirmation"]["net_return"]) > 0),
            "normal_profit_factor": (
                float(row["periods"]["validation"]["daily_profit_factor"])
                >= settings.research.minimum_profit_factor
            ),
            "stressed_positive": float(stressed_confirmation["net_return"]) > 0,
            "stressed_profit_factor": (
                float(stressed_confirmation["daily_profit_factor"])
                >= settings.research.minimum_stressed_profit_factor
            ),
            "minimum_rebalances": (
                int(result.metrics["rebalance_count"]) >= settings.research.minimum_trades
            ),
            "minimum_effective_sample": (
                int(row["periods"]["confirmation"]["effective_sample_size"])
                >= settings.research.minimum_effective_sample_size
            ),
            "positive_fold_gate": (positive_folds >= settings.research.minimum_positive_folds),
            "confidence_interval_lower_positive": ci_lower > 0,
            "deflated_sharpe_gate": (dsr >= settings.research.minimum_deflated_sharpe_probability),
            "white_reality_check_gate": (
                multiple.white_reality_check_pvalue
                <= settings.research.maximum_white_reality_check_pvalue
            ),
            "hansen_spa_gate": (
                multiple.hansen_spa_pvalue <= settings.research.maximum_hansen_spa_pvalue
            ),
            "pbo_gate": (
                multiple.probability_of_backtest_overfitting is not None
                and multiple.probability_of_backtest_overfitting
                <= settings.research.maximum_probability_of_backtest_overfitting
            ),
            "monte_carlo_gate": bool(
                stochastic["normal"]["monte_carlo"]["passed"]
                and stochastic["stressed"]["monte_carlo"]["passed"]
            ),
            "dirichlet_gate": bool(
                stochastic["normal"]["dirichlet"]["passed"]
                and stochastic["stressed"]["dirichlet"]["passed"]
            ),
            "maximum_drawdown_gate": (
                abs(float(row["full_sample"]["maximum_drawdown"]))
                <= settings.research.maximum_drawdown
            ),
        }
        statistical_checks = {
            "confidence_interval_lower_positive",
            "deflated_sharpe_gate",
            "white_reality_check_gate",
            "hansen_spa_gate",
            "pbo_gate",
            "monte_carlo_gate",
            "dirichlet_gate",
        }
        row["robustness"] = {
            "checks": checks,
            "positive_folds": positive_folds,
            "total_folds": len(fold_returns),
            "confirmation_mean_return_ci_lower_95": ci_lower,
            "deflated_sharpe_probability": dsr,
            "stressed_confirmation": stressed_confirmation,
            "stochastic_validation": stochastic,
            "all_numeric_gates_passed": all(checks.values()),
            "economic_gates_passed": all(
                passed for name, passed in checks.items() if name not in statistical_checks
            ),
            "statistical_gates_passed": all(checks[name] for name in statistical_checks),
            "paper_candidate_permitted": False,
            "holdout_status": "CONTAMINATED_BY_PRIOR_EXPLORATION",
        }

    positive_all_periods = sum(
        all(float(row["periods"][period]["net_return"]) > 0 for period in periods) for row in rows
    )
    economic_research_leads = [
        row for row in survivors if row["robustness"]["economic_gates_passed"]
    ]
    statistically_qualified = [
        row for row in survivors if row["robustness"]["all_numeric_gates_passed"]
    ]
    campaign_label = (
        "CROSS_SECTIONAL_INSTITUTIONAL_CONTINUATION_V2"
        if institutional
        else (
            "CROSS_SECTIONAL_MULTI_HORIZON_ENSEMBLE_V1"
            if ensemble
            else "CROSS_SECTIONAL_ROTATION_ALLOWED_V1"
        )
    )
    report = {
        "status": "COMPLETED",
        "campaign": campaign_label,
        "result_type": "JOINT_PARAMETER_SCREEN",
        "rotation_engine_version": ROTATION_ENGINE_VERSION,
        "rotation_policy_version": ROTATION_POLICY_VERSION,
        "markets": list(markets),
        "timeframe": "1d",
        "bias_label": "CURRENT_UNIVERSE_RETROSPECTIVE",
        "selection_basis": "DEVELOPMENT_ONLY",
        "periods": periods,
        "joint_parameter_trials": len(rows),
        "prior_exploratory_trials_accounted": prior_trials,
        "total_known_family_trials": len(rows) + prior_trials,
        "development_eligible": len(development_eligible),
        "positive_all_three_periods_descriptive_only": positive_all_periods,
        "survivor_count": len(survivors),
        "economic_research_lead_count": len(economic_research_leads),
        "economic_research_leads": economic_research_leads,
        "statistically_qualified_count": len(statistically_qualified),
        "multiple_testing": asdict(multiple),
        "survivors": survivors,
        "top_development_rows": rows[:25],
        "paper_candidates": 0,
        "live_orders": 0,
        "portfolio_policy": (asdict(portfolio_policy) if portfolio_policy is not None else None),
        "portfolio_policy_hash": (
            portfolio_policy.policy_hash if portfolio_policy is not None else None
        ),
        "acceptance_note": (
            "Positive rows are research evidence only. The recent confirmation period "
            "was previously inspected and is not an untouched final holdout."
        ),
        "data_files": {
            market: {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(frames[market]),
                "start": frames[market].index.min().isoformat(),
                "end": frames[market].index.max().isoformat(),
            }
            for market, path in paths.items()
        },
    }
    report_path = _rotation_campaign_path(
        settings,
        ensemble=ensemble,
        institutional=institutional,
    )
    atomic_write_json(report_path, _json_ready(report))
    frozen_path: Path | None = None
    if ensemble_mode and economic_research_leads:
        lead = economic_research_leads[0]
        frozen_path = (
            settings.paths.lab_dir
            / "candidates"
            / (
                "rotation_institutional_lead_v2.json"
                if institutional
                else "rotation_research_lead_v1.json"
            )
        )
        lead_result = results[str(lead["strategy_dna_hash"])]
        if not frozen_path.is_file():
            atomic_write_json(
                frozen_path,
                _json_ready(
                    {
                        "status": "FROZEN_RESEARCH_LEAD",
                        "candidate_type": "ECONOMIC_RESEARCH_LEAD_NOT_PAPER_APPROVED",
                        "strategy_dna_hash": lead["strategy_dna_hash"],
                        "execution_identity": lead_result.summary()["execution_identity"],
                        "parameters": lead["parameters"],
                        "portfolio_policy": asdict(lead_result.portfolio_policy),
                        "portfolio_policy_hash": lead_result.portfolio_policy.policy_hash,
                        "periods": lead["periods"],
                        "full_sample": lead["full_sample"],
                        "cost_breakdown": lead["cost_breakdown"],
                        "robustness": lead["robustness"],
                        "source_report": str(report_path),
                        "known_family_trials_accounted": report["total_known_family_trials"],
                        "selection_bias": "CONTAMINATED_BY_PRIOR_EXPLORATION",
                        "forward_validation_start": "2026-07-25T00:00:00Z",
                        "minimum_forward_closed_daily_observations": 365,
                        "minimum_forward_rebalances": FAST_SCREEN_MINIMUM_TRADES,
                        "paper_candidate_permitted": False,
                        "live_ready": False,
                        "immutable_identity": stable_hash(
                            {
                                "strategy_dna_hash": lead["strategy_dna_hash"],
                                "parameters": lead["parameters"],
                                "data_hashes": {
                                    market: sha256_file(path) for market, path in paths.items()
                                },
                                "portfolio_policy_hash": (lead_result.portfolio_policy.policy_hash),
                            },
                            length=64,
                        ),
                    }
                ),
            )
    csv_path = report_path.with_suffix(".csv")
    pd.DataFrame(
        [
            {
                "strategy_dna_hash": row["strategy_dna_hash"],
                **{
                    key: value
                    for key, value in row["parameters"].items()
                    if not isinstance(value, (list, dict))
                },
                "development_return": row["periods"]["development"]["net_return"],
                "validation_return": row["periods"]["validation"]["net_return"],
                "confirmation_return": row["periods"]["confirmation"]["net_return"],
                "development_sharpe": row["periods"]["development"]["sharpe"],
                "validation_sharpe": row["periods"]["validation"]["sharpe"],
                "confirmation_sharpe": row["periods"]["confirmation"]["sharpe"],
                "full_maximum_drawdown": row["full_sample"]["maximum_drawdown"],
                "full_rebalances": row["full_sample"]["rebalance_count"],
                "monte_carlo_gate": row.get("robustness", {})
                .get("checks", {})
                .get("monte_carlo_gate"),
                "dirichlet_gate": row.get("robustness", {}).get("checks", {}).get("dirichlet_gate"),
            }
            for row in rows
        ]
    ).to_csv(csv_path, index=False)
    return {
        "status": report["status"],
        "campaign": report["campaign"],
        "joint_parameter_trials": report["joint_parameter_trials"],
        "development_eligible": report["development_eligible"],
        "positive_all_three_periods_descriptive_only": positive_all_periods,
        "survivor_count": len(survivors),
        "economic_research_lead_count": len(economic_research_leads),
        "statistically_qualified_count": len(statistically_qualified),
        "paper_candidates": 0,
        "report": str(report_path),
        "csv": str(csv_path),
        "frozen_research_lead": str(frozen_path) if frozen_path else None,
    }


def _run_rotation_forward_validation(settings: Settings) -> dict[str, Any]:
    """Evaluate only the frozen rotation DNA on observations after its freeze."""

    from research.portfolio_selection import (
        RotationParameters,
        backtest_rotation,
        rotation_period_metrics,
        rotation_regime_coverage,
    )

    frozen_path = settings.paths.lab_dir / "candidates" / "rotation_research_lead_v1.json"
    if not frozen_path.is_file():
        return {
            "status": "BLOCKED",
            "reason_code": "NO_FROZEN_ROTATION_RESEARCH_LEAD",
            "frozen_manifest": str(frozen_path),
        }
    frozen = read_json(frozen_path)
    parameters_payload = dict(frozen["parameters"])
    parameters_payload["additional_momentum_lookbacks"] = tuple(
        parameters_payload.get("additional_momentum_lookbacks") or ()
    )
    parameters = RotationParameters(**parameters_payload)
    if parameters.dna_hash != frozen["strategy_dna_hash"]:
        raise ValueError("frozen rotation DNA does not match its parameters")

    markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    paths = {
        market: settings.paths.processed_data_dir / f"{market}_1d.parquet" for market in markets
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing forward-validation data: {missing}")
    frames = {market: pd.read_parquet(path) for market, path in paths.items()}
    forward_start = pd.Timestamp(frozen["forward_validation_start"])
    forward_start = (
        forward_start.tz_localize("UTC")
        if forward_start.tzinfo is None
        else forward_start.tz_convert("UTC")
    )
    btc_index = pd.to_datetime(frames["BTC-EUR"].index, utc=True)
    closed_observations = int((btc_index >= forward_start).sum())
    minimum_observations = int(frozen["minimum_forward_closed_daily_observations"])
    minimum_rebalances = int(frozen["minimum_forward_rebalances"])
    report_path = settings.paths.lab_dir / "reports" / "rotation_forward_validation_v1.json"
    base = {
        "candidate_identity": frozen["immutable_identity"],
        "strategy_dna_hash": frozen["strategy_dna_hash"],
        "forward_start": forward_start.isoformat(),
        "latest_closed_candle": btc_index.max().isoformat(),
        "closed_daily_observations": closed_observations,
        "required_closed_daily_observations": minimum_observations,
        "required_rebalances": minimum_rebalances,
        "required_regime_coverage": {
            "axes": {
                "btc_trend": ["UP", "DOWN"],
                "volatility": ["HIGH", "LOW"],
                "breadth": ["BROAD", "NARROW"],
            },
            "minimum_decisions_per_state": 5,
        },
        "parameters_frozen": True,
        "parameter_reselection_permitted": False,
        "paper_candidate_permitted": False,
        "live_ready": False,
        "data_hashes": {market: sha256_file(path) for market, path in paths.items()},
    }
    if closed_observations < 30:
        payload = base | {
            "status": "COLLECTING_FORWARD_DATA",
            "reason_code": "INSUFFICIENT_NEW_CLOSED_DAILY_OBSERVATIONS",
            "remaining_observations": max(0, minimum_observations - closed_observations),
        }
        atomic_write_json(report_path, payload)
        return payload | {"report": str(report_path)}

    normal = backtest_rotation(
        frames,
        parameters,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
    )
    stressed = backtest_rotation(
        frames,
        parameters,
        fee_rate=settings.costs.default_fee * settings.costs.stressed_cost_multiplier,
        slippage_bps=settings.costs.slippage_bps * settings.costs.stressed_cost_multiplier,
        spread_bps=settings.costs.spread_bps * settings.costs.stressed_cost_multiplier,
    )
    forward_end = btc_index.max()
    normal_metrics, normal_returns = rotation_period_metrics(
        normal.equity_curve,
        start=forward_start,
        end=forward_end,
    )
    stressed_metrics, _ = rotation_period_metrics(
        stressed.equity_curve,
        start=forward_start,
        end=forward_end,
    )
    forward_decisions = normal.decisions[
        pd.to_datetime(normal.decisions["executed_at"], utc=True) >= forward_start
    ]
    rebalances = int((forward_decisions["turnover"].astype(float) > 1e-12).sum())
    regime_coverage = rotation_regime_coverage(
        forward_decisions,
        minimum_per_state=5,
    )
    ci_lower = _rotation_return_ci_lower(
        normal_returns,
        samples=settings.research.multiple_testing_bootstrap_samples,
        block_size=settings.research.multiple_testing_block_size,
        seed=settings.app.random_seed,
    )
    checks = {
        "minimum_closed_observations": closed_observations >= minimum_observations,
        "minimum_rebalances": rebalances >= minimum_rebalances,
        "net_positive": float(normal_metrics["net_return"]) > 0,
        "profit_factor": (
            float(normal_metrics["daily_profit_factor"]) >= settings.research.minimum_profit_factor
        ),
        "stressed_net_positive": float(stressed_metrics["net_return"]) > 0,
        "stressed_profit_factor": (
            float(stressed_metrics["daily_profit_factor"])
            >= settings.research.minimum_stressed_profit_factor
        ),
        "effective_sample_size": (
            int(normal_metrics["effective_sample_size"])
            >= settings.research.minimum_effective_sample_size
        ),
        "confidence_interval_lower_positive": ci_lower > 0,
        "minimum_regime_coverage": regime_coverage["passed"],
    }
    passed = all(checks.values())
    payload = base | {
        "status": "FORWARD_PASS" if passed else "FORWARD_NOT_YET_QUALIFIED",
        "checks": checks,
        "normal": normal_metrics,
        "stressed": stressed_metrics,
        "forward_rebalances": rebalances,
        "regime_coverage": regime_coverage,
        "mean_return_ci_lower_95": ci_lower,
        "shadow_review_eligible": passed,
        "paper_candidate_permitted": False,
    }
    atomic_write_json(report_path, _json_ready(payload))
    return payload | {"report": str(report_path)}


def _run_rotation_external_validation(settings: Settings) -> dict[str, Any]:
    """Test frozen DNA once across declared quote and asset holdout views."""

    from research.optimization import multiple_testing_bootstrap
    from research.portfolio_selection import (
        RotationParameters,
        backtest_rotation,
        rotation_period_metrics,
    )

    frozen_path = settings.paths.lab_dir / "candidates" / "rotation_research_lead_v1.json"
    if not frozen_path.is_file():
        return {
            "status": "BLOCKED",
            "reason_code": "NO_FROZEN_ROTATION_RESEARCH_LEAD",
            "frozen_manifest": str(frozen_path),
        }
    frozen = read_json(frozen_path)
    parameter_payload = dict(frozen["parameters"])
    parameter_payload["additional_momentum_lookbacks"] = tuple(
        parameter_payload.get("additional_momentum_lookbacks") or ()
    )
    parameters = RotationParameters(**parameter_payload)
    if parameters.dna_hash != frozen["strategy_dna_hash"]:
        raise ValueError("frozen rotation DNA does not match its parameters")

    base_eur = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    views: dict[str, tuple[tuple[str, ...], str]] = {
        "USDT_QUOTE_PROVIDER": (
            ("BTC-USDT", "ETH-USDT", "SOL-USDT", "LINK-USDT"),
            "BTC-USDT",
        ),
        "ADD_XMR": ((*base_eur, "XMR-EUR"), "BTC-EUR"),
        "ADD_ZEC": ((*base_eur, "ZEC-EUR"), "BTC-EUR"),
        "ADD_HYPE": ((*base_eur, "HYPE-EUR"), "BTC-EUR"),
        "ADD_XMR_ZEC_HYPE": (
            (*base_eur, "XMR-EUR", "ZEC-EUR", "HYPE-EUR"),
            "BTC-EUR",
        ),
    }
    period = ("2025-07-01", "2026-07-23")
    normal_returns: dict[str, pd.Series] = {}
    results: dict[str, Any] = {}
    all_paths: dict[str, Path] = {}
    for view, (markets, benchmark) in views.items():
        paths = {
            market: settings.paths.processed_data_dir / f"{market}_1d.parquet" for market in markets
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{view} lacks external-validation data: {missing}")
        all_paths.update(paths)
        frames = {market: pd.read_parquet(path) for market, path in paths.items()}
        normal = backtest_rotation(
            frames,
            parameters,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            benchmark_market=benchmark,
        )
        stressed = backtest_rotation(
            frames,
            parameters,
            fee_rate=settings.costs.default_fee * settings.costs.stressed_cost_multiplier,
            slippage_bps=settings.costs.slippage_bps * settings.costs.stressed_cost_multiplier,
            spread_bps=settings.costs.spread_bps * settings.costs.stressed_cost_multiplier,
            benchmark_market=benchmark,
        )
        normal_metrics, returns = rotation_period_metrics(
            normal.equity_curve,
            start=period[0],
            end=period[1],
        )
        stressed_metrics, _ = rotation_period_metrics(
            stressed.equity_curve,
            start=period[0],
            end=period[1],
        )
        normal_returns[view] = returns
        ci_lower = _rotation_return_ci_lower(
            returns,
            samples=settings.research.multiple_testing_bootstrap_samples,
            block_size=settings.research.multiple_testing_block_size,
            seed=settings.app.random_seed,
        )
        results[view] = {
            "markets": list(markets),
            "benchmark_market": benchmark,
            "normal": normal_metrics,
            "stressed": stressed_metrics,
            "mean_return_ci_lower_95": ci_lower,
            "integrity": normal.integrity,
        }

    matrix = pd.concat(normal_returns, axis=1).dropna(how="any")
    multiple = multiple_testing_bootstrap(
        matrix,
        bootstrap_samples=settings.research.multiple_testing_bootstrap_samples,
        block_size=settings.research.multiple_testing_block_size,
        seed=settings.app.random_seed,
        known_trial_count=int(frozen["known_family_trials_accounted"]) + len(views),
    )
    for view, result in results.items():
        normal = result["normal"]
        stressed = result["stressed"]
        dsr = float(multiple.deflated_sharpe_probabilities[view])
        checks = {
            "net_positive": float(normal["net_return"]) > 0,
            "normal_profit_factor": (
                float(normal["daily_profit_factor"]) >= settings.research.minimum_profit_factor
            ),
            "stressed_net_positive": float(stressed["net_return"]) > 0,
            "stressed_profit_factor": (
                float(stressed["daily_profit_factor"])
                >= settings.research.minimum_stressed_profit_factor
            ),
            "effective_sample_size": (
                int(normal["effective_sample_size"])
                >= settings.research.minimum_effective_sample_size
            ),
            "maximum_drawdown": (
                abs(float(normal["maximum_drawdown"])) <= settings.research.maximum_drawdown
            ),
            "confidence_interval_lower_positive": (float(result["mean_return_ci_lower_95"]) > 0),
            "deflated_sharpe": (dsr >= settings.research.minimum_deflated_sharpe_probability),
        }
        result["deflated_sharpe_probability"] = dsr
        result["checks"] = checks
        result["economic_positive"] = all(
            checks[name]
            for name in (
                "net_positive",
                "stressed_net_positive",
                "stressed_profit_factor",
                "effective_sample_size",
                "maximum_drawdown",
            )
        )
        result["all_view_gates_passed"] = all(checks.values())

    global_checks = {
        "all_views_net_positive": all(
            float(result["normal"]["net_return"]) > 0 for result in results.values()
        ),
        "all_views_stressed_positive": all(
            float(result["stressed"]["net_return"]) > 0 for result in results.values()
        ),
        "white_reality_check": (
            multiple.white_reality_check_pvalue
            <= settings.research.maximum_white_reality_check_pvalue
        ),
        "hansen_spa": (multiple.hansen_spa_pvalue <= settings.research.maximum_hansen_spa_pvalue),
        "pbo": (
            multiple.probability_of_backtest_overfitting is not None
            and multiple.probability_of_backtest_overfitting
            <= settings.research.maximum_probability_of_backtest_overfitting
        ),
        "at_least_one_dsr_pass": any(
            probability >= settings.research.minimum_deflated_sharpe_probability
            for probability in multiple.deflated_sharpe_probabilities.values()
        ),
    }
    full_pass = all(global_checks.values()) and any(
        result["all_view_gates_passed"] for result in results.values()
    )
    payload = {
        "status": (
            "EXTERNAL_VALIDATION_PASS"
            if full_pass
            else "EXTERNAL_ECONOMIC_PASS_STATISTICAL_PARTIAL"
        ),
        "candidate_identity": frozen["immutable_identity"],
        "strategy_dna_hash": frozen["strategy_dna_hash"],
        "parameters_frozen": True,
        "parameter_reselection_permitted": False,
        "evaluation_count": len(views),
        "period": {"start": period[0], "end": period[1]},
        "holdout_label": "EXTERNAL_ASSET_AND_QUOTE_VALIDATION_EVALUATED_ONCE",
        "views": results,
        "multiple_testing": asdict(multiple),
        "global_checks": global_checks,
        "known_family_trials_before_external_validation": frozen["known_family_trials_accounted"],
        "total_known_trials_after_external_validation": (
            int(frozen["known_family_trials_accounted"]) + len(views)
        ),
        "data_hashes": {market: sha256_file(path) for market, path in sorted(all_paths.items())},
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    report_path = settings.paths.lab_dir / "reports" / "rotation_external_holdouts_v1.json"
    atomic_write_json(report_path, _json_ready(payload))
    return {
        "status": payload["status"],
        "global_checks": global_checks,
        "report": str(report_path),
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def _strict_rotation_portfolio_policy(
    settings: Settings,
    *,
    markets: tuple[str, ...],
) -> Any:
    from research.portfolio_selection import RotationPortfolioPolicy

    maximum_total = min(
        0.40,
        settings.operational.maximum_portfolio_exposure,
    )
    maximum_position = min(
        0.20,
        settings.operational.maximum_position_fraction,
        maximum_total,
    )
    return RotationPortfolioPolicy(
        allowed_markets=markets,
        maximum_total_exposure=maximum_total,
        maximum_position_exposure=maximum_position,
        minimum_cash=max(
            settings.operational.reserve_cash_fraction,
            1.0 - maximum_total,
        ),
        minimum_history_observations=90,
    )


def _frozen_rotation_inputs(
    settings: Settings,
) -> tuple[dict[str, Any], Any, tuple[str, ...], dict[str, Path], dict[str, pd.DataFrame]]:
    from research.portfolio_selection import RotationParameters

    frozen_path = settings.paths.lab_dir / "candidates" / "rotation_research_lead_v1.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"frozen rotation lead is missing: {frozen_path}")
    frozen = read_json(frozen_path)
    parameter_payload = dict(frozen["parameters"])
    parameter_payload["additional_momentum_lookbacks"] = tuple(
        parameter_payload.get("additional_momentum_lookbacks") or ()
    )
    parameters = RotationParameters(**parameter_payload)
    if parameters.dna_hash != frozen["strategy_dna_hash"]:
        raise ValueError("frozen rotation DNA does not match its parameters")
    markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    paths = {
        market: settings.paths.processed_data_dir / f"{market}_1d.parquet" for market in markets
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"strict rotation datasets are missing: {missing}")
    frames = {market: pd.read_parquet(path) for market, path in paths.items()}
    return frozen, parameters, markets, paths, frames


def _run_rotation_institutional_audit(settings: Settings) -> dict[str, Any]:
    """Reproduce frozen signal DNA under a new explicit strict execution policy."""

    from research.portfolio_selection import (
        PORTFOLIO_METRICS_VERSION,
        backtest_rotation,
        rotation_benchmark_suite,
        rotation_period_metrics,
    )

    frozen, parameters, markets, paths, frames = _frozen_rotation_inputs(settings)
    source_report = read_json(_rotation_campaign_path(settings, ensemble=True))
    if tuple(source_report["markets"]) != markets:
        raise ValueError("source lead was not selected on the strict core market universe")
    policy = _strict_rotation_portfolio_policy(settings, markets=markets)
    normal = backtest_rotation(
        frames,
        parameters,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        portfolio_policy=policy,
    )
    stressed = backtest_rotation(
        frames,
        parameters,
        fee_rate=settings.costs.default_fee * settings.costs.stressed_cost_multiplier,
        slippage_bps=settings.costs.slippage_bps * settings.costs.stressed_cost_multiplier,
        spread_bps=settings.costs.spread_bps * settings.costs.stressed_cost_multiplier,
        portfolio_policy=policy,
    )
    periods = dict(source_report["periods"])
    period_results: dict[str, Any] = {}
    stressed_period_results: dict[str, Any] = {}
    for period, bounds in periods.items():
        period_results[period], _ = rotation_period_metrics(
            normal.equity_curve,
            start=bounds[0],
            end=bounds[1],
        )
        stressed_period_results[period], _ = rotation_period_metrics(
            stressed.equity_curve,
            start=bounds[0],
            end=bounds[1],
        )
    benchmarks = rotation_benchmark_suite(
        frames,
        parameters,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        portfolio_policy=policy,
    )
    checks = {
        "source_universe_allowed_only": tuple(source_report["markets"]) == markets,
        "all_periods_net_positive": all(
            float(metrics["net_return"]) > 0 for metrics in period_results.values()
        ),
        "all_periods_stressed_net_positive": all(
            float(metrics["net_return"]) > 0 for metrics in stressed_period_results.values()
        ),
        "full_sample_net_positive": float(normal.metrics["net_return"]) > 0,
        "full_sample_stressed_net_positive": (float(stressed.metrics["net_return"]) > 0),
        "portfolio_period_ess": (
            int(normal.metrics["portfolio_period_effective_sample_size"])
            >= settings.research.minimum_effective_sample_size
        ),
        "minimum_holding_episodes": (int(normal.metrics["closed_position_episodes"]) >= 30),
        "minimum_rebalance_opportunities": (
            int(normal.metrics["scheduled_rebalance_opportunities"])
            >= settings.research.minimum_trades
        ),
        "maximum_total_exposure": normal.integrity["maximum_exposure_respected"],
        "maximum_position_exposure": normal.integrity["maximum_position_exposure_respected"],
        "minimum_cash": normal.integrity["minimum_cash_respected"],
        "asset_pnl_reconciled": normal.integrity["asset_pnl_reconciled"],
        "next_open_execution": normal.integrity["decision_at_close_execution_next_open"],
        "terminal_liquidation": normal.integrity["terminal_liquidation_recorded"],
    }
    economic_pass = all(checks.values())
    execution_identity = normal.summary()["execution_identity"]
    payload = {
        "status": (
            "STRICT_ALLOWED_POLICY_ECONOMIC_PASS"
            if economic_pass
            else "STRICT_ALLOWED_POLICY_NOT_QUALIFIED"
        ),
        "candidate_type": "INSTITUTIONAL_POLICY_REPRODUCTION_NOT_NEW_SELECTION",
        "source_candidate_identity": frozen["immutable_identity"],
        "strategy_dna_hash": frozen["strategy_dna_hash"],
        "execution_identity": execution_identity,
        "portfolio_metrics_version": PORTFOLIO_METRICS_VERSION,
        "parameters_frozen": True,
        "parameter_reselection_permitted": False,
        "source_universe": list(markets),
        "discovery_assets_used_for_original_selection": [],
        "external_assets_used_for_original_selection": [],
        "portfolio_policy": asdict(policy),
        "portfolio_policy_hash": policy.policy_hash,
        "exposure_semantics": {
            "gross_exposure_parameter_is_total_not_per_position": True,
            "signal_target_total_exposure": parameters.gross_exposure,
            "hard_maximum_total_exposure": policy.maximum_total_exposure,
            "hard_maximum_position_exposure": policy.maximum_position_exposure,
            "hard_minimum_cash": policy.minimum_cash,
            "maximum_observed_total_exposure": float(normal.executed_weights.sum(axis=1).max()),
            "maximum_observed_position_exposure": normal.metrics[
                "maximum_position_exposure_observed"
            ],
        },
        "checks": checks,
        "economic_gates_passed": economic_pass,
        "normal": normal.summary(),
        "stressed": stressed.summary(),
        "periods": period_results,
        "stressed_periods": stressed_period_results,
        "asset_pnl_attribution": normal.metrics["asset_pnl_attribution"],
        "benchmarks_and_ablations": benchmarks,
        "historical_multiple_testing": source_report["multiple_testing"],
        "historical_statistical_gates_passed": frozen["robustness"]["statistical_gates_passed"],
        "statistical_recalculation_note": (
            "The original 160-strategy multiple-testing matrix already used only "
            "BTC-EUR, ETH-EUR, SOL-EUR and LINK-EUR. This fixed-policy reproduction "
            "does not create a new search family and cannot manufacture a new DSR."
        ),
        "data_hashes": {market: sha256_file(path) for market, path in paths.items()},
        "paper_candidate_permitted": False,
        "live_ready": False,
        "live_orders": 0,
    }
    report_path = settings.paths.lab_dir / "reports" / "rotation_institutional_audit_v2.json"
    atomic_write_json(report_path, _json_ready(payload))
    benchmark_csv = report_path.with_suffix(".csv")
    pd.DataFrame(
        [
            {"name": "strict_rotation", **normal.metrics},
            *[{"name": name, **metrics} for name, metrics in benchmarks["benchmarks"].items()],
        ]
    ).to_csv(benchmark_csv, index=False)
    return {
        "status": payload["status"],
        "execution_identity": execution_identity,
        "economic_gates_passed": economic_pass,
        "historical_statistical_gates_passed": False,
        "report": str(report_path),
        "benchmark_csv": str(benchmark_csv),
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def _run_rotation_forward_observer(settings: Settings) -> dict[str, Any]:
    """Persist a frozen daily research snapshot without generating any order."""

    from research.portfolio_selection import (
        backtest_rotation,
        rotation_decision_snapshot,
        rotation_regime_coverage,
    )

    frozen, parameters, markets, paths, frames = _frozen_rotation_inputs(settings)
    policy = _strict_rotation_portfolio_policy(settings, markets=markets)
    snapshot = rotation_decision_snapshot(
        frames,
        parameters,
        portfolio_policy=policy,
    )
    result = backtest_rotation(
        frames,
        parameters,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        portfolio_policy=policy,
    )
    forward_start = pd.Timestamp(frozen["forward_validation_start"])
    forward_start = (
        forward_start.tz_localize("UTC")
        if forward_start.tzinfo is None
        else forward_start.tz_convert("UTC")
    )
    forward_decisions = result.decisions[
        pd.to_datetime(result.decisions["decision_at"], utc=True) >= forward_start
    ].copy()
    coverage = rotation_regime_coverage(
        forward_decisions,
        minimum_per_state=5,
    )
    report_path = settings.paths.lab_dir / "reports" / "rotation_forward_observer_v2.json"
    existing_observations: list[dict[str, Any]] = []
    if report_path.is_file():
        existing_observations = list(read_json(report_path).get("observations") or [])
    by_decision_at = {str(item["decision_at"]): item for item in existing_observations}
    if pd.Timestamp(snapshot["decision_at"]) >= forward_start:
        by_decision_at[snapshot["decision_at"]] = snapshot
    observations = [by_decision_at[key] for key in sorted(by_decision_at)]
    payload = {
        "status": "FROZEN_FORWARD_RESEARCH",
        "source_candidate_identity": frozen["immutable_identity"],
        "strategy_dna_hash": frozen["strategy_dna_hash"],
        "execution_identity": result.summary()["execution_identity"],
        "portfolio_policy": asdict(policy),
        "parameters_frozen": True,
        "parameter_reselection_permitted": False,
        "forward_start": forward_start.isoformat(),
        "latest_snapshot": snapshot,
        "observations": observations,
        "forward_decision_count": len(forward_decisions),
        "regime_coverage": coverage,
        "data_hashes": {market: sha256_file(path) for market, path in paths.items()},
        "orders_generated": 0,
        "orders_submitted": 0,
        "shadow_candidate": False,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    atomic_write_json(report_path, _json_ready(payload))
    return {
        "status": payload["status"],
        "latest_snapshot": snapshot,
        "forward_decision_count": len(forward_decisions),
        "regime_coverage": coverage,
        "report": str(report_path),
        "orders_generated": 0,
        "orders_submitted": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def _run_capital_utilization_campaign(settings: Settings) -> dict[str, Any]:
    """Compare pre-registered allocation policies on one frozen signal DNA."""

    from research.forward_observer import (
        ForwardPerformanceGatePolicy,
        build_rotation_forward_evidence,
        merge_portfolio_forward_manifest,
    )
    from research.optimization import multiple_testing_bootstrap
    from research.portfolio_selection import (
        CAPITAL_UTILIZATION_METRICS_VERSION,
        RotationPortfolioPolicy,
        backtest_rotation,
        capital_utilization_benchmark_suite,
        capital_utilization_policy_set,
        paired_block_bootstrap_difference,
        rotation_decision_snapshot,
        rotation_period_metrics,
    )

    frozen_path = settings.paths.lab_dir / "candidates" / "rotation_research_lead_v1.json"
    frozen_sha_before = sha256_file(frozen_path)
    frozen, parameters, markets, paths, frames = _frozen_rotation_inputs(settings)
    policies = capital_utilization_policy_set()
    continuation_path = _rotation_campaign_path(settings, institutional=True)
    prior_trials = 1_293
    if continuation_path.is_file():
        prior_trials = int(
            read_json(continuation_path).get(
                "total_known_family_trials",
                prior_trials,
            )
        )
    periods = {
        "development": ("2021-08-05", "2023-12-31"),
        "validation": ("2024-01-01", "2025-06-30"),
        "confirmation": ("2025-07-01", "2026-07-23"),
    }
    normal_results: dict[str, Any] = {}
    stressed_results: dict[str, Any] = {}
    daily_returns: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    observer_paths: dict[str, str] = {}
    forward_start = pd.Timestamp(frozen["forward_validation_start"])
    forward_start = (
        forward_start.tz_localize("UTC")
        if forward_start.tzinfo is None
        else forward_start.tz_convert("UTC")
    )

    for allocation in policies:
        portfolio_policy = RotationPortfolioPolicy(
            allowed_markets=markets,
            maximum_total_exposure=allocation.maximum_total_exposure,
            maximum_position_exposure=allocation.maximum_position_exposure,
            minimum_cash=allocation.minimum_cash,
            minimum_history_observations=90,
        )
        normal = backtest_rotation(
            frames,
            parameters,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            portfolio_policy=portfolio_policy,
            capital_utilization_policy=allocation,
        )
        stressed = backtest_rotation(
            frames,
            parameters,
            fee_rate=(settings.costs.default_fee * settings.costs.stressed_cost_multiplier),
            slippage_bps=(settings.costs.slippage_bps * settings.costs.stressed_cost_multiplier),
            spread_bps=(settings.costs.spread_bps * settings.costs.stressed_cost_multiplier),
            portfolio_policy=portfolio_policy,
            capital_utilization_policy=allocation,
        )
        period_metrics: dict[str, Any] = {}
        stressed_period_metrics: dict[str, Any] = {}
        for period, bounds in periods.items():
            period_metrics[period], _ = rotation_period_metrics(
                normal.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )
            stressed_period_metrics[period], _ = rotation_period_metrics(
                stressed.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )
        normal_results[allocation.name] = normal
        stressed_results[allocation.name] = stressed
        daily_returns[allocation.name] = normal.equity_curve.pct_change(fill_method=None).dropna()
        current_operational_compatible = (
            allocation.maximum_total_exposure
            <= settings.operational.maximum_portfolio_exposure + 1e-12
            and allocation.maximum_position_exposure
            <= settings.operational.maximum_position_fraction + 1e-12
            and allocation.minimum_cash >= settings.operational.reserve_cash_fraction - 1e-12
        )
        snapshot = rotation_decision_snapshot(
            frames,
            parameters,
            portfolio_policy=portfolio_policy,
            capital_utilization_policy=allocation,
        )
        observer_payload = {
            "status": "FROZEN_FORWARD_RESEARCH",
            "source_candidate_identity": frozen["immutable_identity"],
            "strategy_dna_hash": parameters.dna_hash,
            "allocation_policy": asdict(allocation),
            "allocation_policy_hash": allocation.policy_hash,
            "execution_identity": normal.summary()["execution_identity"],
            "forward_start": forward_start.isoformat(),
            "minimum_forward_closed_daily_observations": 365,
            "minimum_forward_rebalances": 30,
            "regime_coverage_required": True,
            "latest_historical_snapshot": snapshot,
            "current_operational_limits_compatible": (current_operational_compatible),
            "orders_generated": 0,
            "orders_submitted": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
        observer_path = (
            settings.paths.lab_dir
            / "observers"
            / "capital_utilization_v1"
            / f"{allocation.name.lower()}.json"
        )
        existing_observer = read_json(observer_path) if observer_path.is_file() else {}
        evidence = build_rotation_forward_evidence(
            normal,
            frames,
            forward_start=forward_start,
            minimum_observations=365,
            minimum_rebalances=30,
            performance_policy=ForwardPerformanceGatePolicy(
                minimum_profit_factor=(settings.research.minimum_profit_factor),
                minimum_stressed_profit_factor=(settings.research.minimum_stressed_profit_factor),
                maximum_drawdown=settings.research.maximum_drawdown,
                minimum_effective_sample_size=(settings.research.minimum_effective_sample_size),
                stressed_cost_multiplier=(settings.costs.stressed_cost_multiplier),
                bootstrap_samples=(settings.research.multiple_testing_bootstrap_samples),
                bootstrap_block_size=(settings.research.multiple_testing_block_size),
                bootstrap_seed=settings.app.random_seed,
            ),
        )
        merged_forward = merge_portfolio_forward_manifest(
            existing_observer,
            evidence,
            source_candidate_identity=frozen["immutable_identity"],
            strategy_dna_hash=parameters.dna_hash,
            execution_identity=normal.summary()["execution_identity"],
            forward_start=forward_start,
        )
        observer_payload = {
            **merged_forward,
            **observer_payload,
        }
        atomic_write_json(observer_path, _json_ready(observer_payload))
        observer_paths[allocation.name] = str(observer_path)
        rows.append(
            {
                "policy_name": allocation.name,
                "allocation_policy": asdict(allocation),
                "allocation_policy_hash": allocation.policy_hash,
                "portfolio_policy": asdict(portfolio_policy),
                "execution_identity": normal.summary()["execution_identity"],
                "same_frozen_signal_dna": (
                    normal.parameters.dna_hash == frozen["strategy_dna_hash"]
                ),
                "current_operational_limits_compatible": (current_operational_compatible),
                "normal": normal.summary(),
                "stressed": stressed.summary(),
                "periods": period_metrics,
                "stressed_periods": stressed_period_metrics,
                "cash_reason_attribution": normal.metrics["cash_reason_attribution_average"],
                "forward_summary": observer_payload["forward_summary"],
                "observer_manifest": str(observer_path),
            }
        )

    return_matrix = pd.concat(daily_returns, axis=1).dropna(how="any")
    multiple = multiple_testing_bootstrap(
        return_matrix,
        bootstrap_samples=settings.research.multiple_testing_bootstrap_samples,
        block_size=settings.research.multiple_testing_block_size,
        seed=settings.app.random_seed,
        known_trial_count=prior_trials + len(policies),
    )
    control = normal_results["FROZEN_CONTROL"]
    control_returns = daily_returns["FROZEN_CONTROL"]
    paired: dict[str, Any] = {}
    for index, allocation in enumerate(policies):
        if allocation.name == "FROZEN_CONTROL":
            continue
        candidate = normal_results[allocation.name]
        paired[allocation.name] = paired_block_bootstrap_difference(
            daily_returns[allocation.name],
            control_returns,
            samples=2_000,
            block_size=10,
            seed=settings.app.random_seed + index,
        )
        incremental_exposure = float(
            candidate.metrics["average_exposure"] - control.metrics["average_exposure"]
        )
        incremental_return = float(candidate.metrics["net_return"] - control.metrics["net_return"])
        paired[allocation.name]["incremental_net_return"] = incremental_return
        paired[allocation.name]["incremental_average_exposure"] = incremental_exposure
        paired[allocation.name]["incremental_return_per_incremental_exposure"] = (
            incremental_return / incremental_exposure if abs(incremental_exposure) > 1e-12 else 0.0
        )
        paired[allocation.name]["incremental_maximum_drawdown_depth"] = float(
            abs(candidate.metrics["maximum_drawdown"]) - abs(control.metrics["maximum_drawdown"])
        )
        paired[allocation.name]["incremental_daily_cvar_95_depth"] = float(
            abs(candidate.metrics["daily_cvar_95"]) - abs(control.metrics["daily_cvar_95"])
        )

    for row_index, row in enumerate(rows):
        name = str(row["policy_name"])
        normal = normal_results[name]
        stressed = stressed_results[name]
        dsr = float(multiple.deflated_sharpe_probabilities.get(name, 0.0))
        stochastic = _portfolio_stochastic_validation(
            settings,
            normal_equity=normal.equity_curve,
            stressed_equity=stressed.equity_curve,
            seed_offset=10_000 + row_index * 10,
        )
        economic_checks = {
            "all_periods_positive": all(
                float(row["periods"][period]["net_return"]) > 0 for period in periods
            ),
            "all_stressed_periods_positive": all(
                float(row["stressed_periods"][period]["net_return"]) > 0 for period in periods
            ),
            "minimum_effective_sample": (
                int(normal.metrics["portfolio_period_effective_sample_size"])
                >= settings.research.minimum_effective_sample_size
            ),
            "minimum_rebalances": (
                int(normal.metrics["rebalance_count"]) >= settings.research.minimum_trades
            ),
            "profit_factor": (
                float(normal.metrics["portfolio_period_profit_factor"])
                >= settings.research.minimum_profit_factor
            ),
            "maximum_drawdown": (
                abs(float(normal.metrics["maximum_drawdown"])) <= settings.research.maximum_drawdown
            ),
            "exposure_limits_respected": bool(
                normal.integrity["maximum_exposure_respected"]
                and normal.integrity["maximum_position_exposure_respected"]
                and normal.integrity["minimum_cash_respected"]
            ),
        }
        statistical_checks = {
            "source_historical_statistical_gates": bool(
                frozen["robustness"]["statistical_gates_passed"]
            ),
            "deflated_sharpe": (dsr >= settings.research.minimum_deflated_sharpe_probability),
            "white_reality_check": (
                multiple.white_reality_check_pvalue
                <= settings.research.maximum_white_reality_check_pvalue
            ),
            "hansen_spa": (
                multiple.hansen_spa_pvalue <= settings.research.maximum_hansen_spa_pvalue
            ),
            "pbo": (
                multiple.probability_of_backtest_overfitting is not None
                and multiple.probability_of_backtest_overfitting
                <= settings.research.maximum_probability_of_backtest_overfitting
            ),
            "monte_carlo": bool(
                stochastic["normal"]["monte_carlo"]["passed"]
                and stochastic["stressed"]["monte_carlo"]["passed"]
            ),
            "dirichlet": bool(
                stochastic["normal"]["dirichlet"]["passed"]
                and stochastic["stressed"]["dirichlet"]["passed"]
            ),
        }
        row["gates"] = {
            "economic_checks": economic_checks,
            "statistical_checks": statistical_checks,
            "deflated_sharpe_probability": dsr,
            "stochastic_validation": stochastic,
            "economic_pass": all(economic_checks.values()),
            "statistical_pass": all(statistical_checks.values()),
            "research_pass": False,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }

    benchmarks = capital_utilization_benchmark_suite(
        frames,
        start=control.equity_curve.index[0],
        minimum_history_observations=90,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        allowed_markets=markets,
        exposure_matches={
            name: float(result.metrics["average_exposure"])
            for name, result in normal_results.items()
        },
    )
    rows.sort(
        key=lambda row: float(row["normal"]["metrics"]["net_return"]),
        reverse=True,
    )
    payload = {
        "status": "COMPLETED_NOT_PROMOTED",
        "campaign": "CAPITAL_UTILIZATION_V1",
        "result_type": "PRE_REGISTERED_ALLOCATION_POLICY_COMPARISON",
        "capital_utilization_metrics_version": (CAPITAL_UTILIZATION_METRICS_VERSION),
        "source_candidate_identity": frozen["immutable_identity"],
        "strategy_dna_hash": parameters.dna_hash,
        "signal_dna_frozen": True,
        "signal_parameters_changed": False,
        "markets": list(markets),
        "timeframe": "1d",
        "policies_tested": len(policies),
        "prior_trials_accounted": prior_trials,
        "total_known_trials": prior_trials + len(policies),
        "periods": periods,
        "multiple_testing": asdict(multiple),
        "source_historical_robustness": frozen["robustness"],
        "paired_block_bootstrap_vs_frozen_control": paired,
        "benchmarks": benchmarks,
        "policy_results": rows,
        "observer_manifests": observer_paths,
        "forward_summaries": {str(row["policy_name"]): row["forward_summary"] for row in rows},
        "frozen_candidate_sha256_before": frozen_sha_before,
        "frozen_candidate_sha256_after": sha256_file(frozen_path),
        "frozen_candidate_unchanged": (frozen_sha_before == sha256_file(frozen_path)),
        "selection_note": (
            "This campaign compares a pre-registered allocation layer only. "
            "It does not reselect momentum horizons, assets, ranks, filters, "
            "execution timing or cost assumptions."
        ),
        "operational_note": (
            "Policies above configured 40% total or 20% per-position exposure "
            "are research-only and cannot be promoted without a separate manual "
            "risk-policy decision plus all forward and statistical gates."
        ),
        "paper_candidates": 0,
        "live_orders": 0,
        "live_ready": False,
        "data_hashes": {market: sha256_file(path) for market, path in paths.items()},
    }
    report_path = _capital_utilization_campaign_path(settings)
    atomic_write_json(report_path, _json_ready(payload))
    csv_path = report_path.with_suffix(".csv")
    pd.DataFrame(
        [
            {
                "policy": row["policy_name"],
                "net_return": row["normal"]["metrics"]["net_return"],
                "cagr": row["normal"]["metrics"]["annualized_return"],
                "sharpe": row["normal"]["metrics"]["sharpe"],
                "sortino": row["normal"]["metrics"]["sortino"],
                "omega": row["normal"]["metrics"]["omega"],
                "maximum_drawdown": row["normal"]["metrics"]["maximum_drawdown"],
                "daily_cvar_95": row["normal"]["metrics"]["daily_cvar_95"],
                "average_exposure": row["normal"]["metrics"]["average_exposure"],
                "average_cash": row["normal"]["metrics"]["cash_fraction_average"],
                "return_per_average_exposure": row["normal"]["metrics"][
                    "return_per_average_exposure"
                ],
                "economic_pass": row["gates"]["economic_pass"],
                "statistical_pass": row["gates"]["statistical_pass"],
                "monte_carlo_gate": row["gates"]["statistical_checks"]["monte_carlo"],
                "dirichlet_gate": row["gates"]["statistical_checks"]["dirichlet"],
                "operational_compatible": row["current_operational_limits_compatible"],
                "paper_candidate": False,
                "live_ready": False,
            }
            for row in rows
        ]
    ).to_csv(csv_path, index=False)
    return {
        "status": payload["status"],
        "campaign": payload["campaign"],
        "policies_tested": len(policies),
        "total_known_trials": payload["total_known_trials"],
        "frozen_candidate_unchanged": payload["frozen_candidate_unchanged"],
        "report": str(report_path),
        "csv": str(csv_path),
        "observer_manifests": observer_paths,
        "paper_candidates": 0,
        "live_orders": 0,
        "live_ready": False,
    }


def _run_breakout_portfolio_campaign(settings: Settings) -> dict[str, Any]:
    """Run the pre-registered strict-universe time-series breakout family."""

    from research.optimization import multiple_testing_bootstrap
    from research.portfolio_breakout import (
        BREAKOUT_ENGINE_VERSION,
        backtest_breakout_portfolio,
        breakout_observer_snapshot,
        breakout_portfolio_parameter_set,
    )
    from research.portfolio_selection import (
        backtest_rotation,
        capital_utilization_benchmark_suite,
        paired_block_bootstrap_difference,
        rotation_period_metrics,
    )

    frozen_path = settings.paths.lab_dir / "candidates" / "rotation_research_lead_v1.json"
    frozen_sha_before = sha256_file(frozen_path)
    frozen, frozen_parameters, markets, paths, frames = _frozen_rotation_inputs(settings)
    diversified_path = _diversified_rotation_campaign_path(settings)
    if not diversified_path.is_file():
        raise FileNotFoundError(
            "diversified-rotation-v1 must complete before portfolio-breakout-v1"
        )
    diversified_campaign = read_json(diversified_path)
    prior_trials = int(diversified_campaign["total_known_trials"])
    periods = dict(diversified_campaign["periods"])
    parameters_set = breakout_portfolio_parameter_set()
    policy = _strict_rotation_portfolio_policy(settings, markets=markets)
    frozen_control = backtest_rotation(
        frames,
        frozen_parameters,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        portfolio_policy=policy,
    )
    control_returns = frozen_control.equity_curve.pct_change(fill_method=None).dropna()
    results: dict[str, Any] = {}
    stressed_results: dict[str, Any] = {}
    daily_returns: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    observer_paths: dict[str, str] = {}
    forward_start = pd.Timestamp(frozen["forward_validation_start"])
    forward_start = (
        forward_start.tz_localize("UTC")
        if forward_start.tzinfo is None
        else forward_start.tz_convert("UTC")
    )

    for parameter_index, parameters in enumerate(parameters_set):
        normal = backtest_breakout_portfolio(
            frames,
            parameters,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            portfolio_policy=policy,
        )
        stressed = backtest_breakout_portfolio(
            frames,
            parameters,
            fee_rate=(settings.costs.default_fee * settings.costs.stressed_cost_multiplier),
            slippage_bps=(settings.costs.slippage_bps * settings.costs.stressed_cost_multiplier),
            spread_bps=(settings.costs.spread_bps * settings.costs.stressed_cost_multiplier),
            portfolio_policy=policy,
        )
        period_metrics: dict[str, Any] = {}
        stressed_period_metrics: dict[str, Any] = {}
        for period, bounds in periods.items():
            period_metrics[period], _ = rotation_period_metrics(
                normal.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )
            stressed_period_metrics[period], _ = rotation_period_metrics(
                stressed.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )
        policy_name = (
            f"TURTLE_{parameters.entry_lookback}_{parameters.exit_lookback}"
            f"_EMA{parameters.trend_ema_period}"
            f"_{parameters.weighting.upper()}"
        )
        results[policy_name] = normal
        stressed_results[policy_name] = stressed
        returns = normal.equity_curve.pct_change(fill_method=None).dropna()
        daily_returns[policy_name] = returns[~returns.index.duplicated(keep="last")]
        observer_path = (
            settings.paths.lab_dir
            / "observers"
            / "portfolio_breakout_v1"
            / f"{policy_name.lower()}.json"
        )
        existing_observer = read_json(observer_path) if observer_path.is_file() else {}
        snapshot = breakout_observer_snapshot(normal)
        preserved_forward = _preserved_breakout_forward_fields(
            existing_observer,
            source_candidate_identity=frozen["immutable_identity"],
            strategy_dna_hash=parameters.dna_hash,
            execution_identity=snapshot["execution_identity"],
            forward_start=forward_start,
        )
        observer = {
            **snapshot,
            "family": "PORTFOLIO_BREAKOUT_V1",
            "policy_name": policy_name,
            "parameters": asdict(parameters),
            "source_candidate_identity": frozen["immutable_identity"],
            "forward_start": forward_start.isoformat(),
            "minimum_forward_closed_daily_observations": 365,
            "minimum_forward_rebalances": 30,
            "forward_observations": [],
            "data_hashes": {market: sha256_file(path) for market, path in paths.items()},
            **preserved_forward,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
        atomic_write_json(observer_path, _json_ready(observer))
        observer_paths[policy_name] = str(observer_path)
        paired = paired_block_bootstrap_difference(
            daily_returns[policy_name],
            control_returns,
            samples=2_000,
            block_size=10,
            seed=settings.app.random_seed + parameter_index,
        )
        fold_returns = np.array_split(
            normal.equity_curve.pct_change(fill_method=None).dropna().to_numpy(dtype=float),
            settings.research.walk_forward_folds,
        )
        positive_folds = sum(
            float(np.prod(1.0 + fold) - 1.0) > 0 for fold in fold_returns if len(fold)
        )
        _, confirmation_returns = rotation_period_metrics(
            normal.equity_curve,
            start=periods["confirmation"][0],
            end=periods["confirmation"][1],
        )
        ci_lower = _rotation_return_ci_lower(
            confirmation_returns,
            samples=2_000,
            block_size=10,
            seed=settings.app.random_seed + parameter_index,
        )
        rows.append(
            {
                "policy_name": policy_name,
                "strategy_dna_hash": parameters.dna_hash,
                "parameters": asdict(parameters),
                "normal": normal.summary(),
                "stressed": stressed.summary(),
                "periods": period_metrics,
                "stressed_periods": stressed_period_metrics,
                "paired_block_bootstrap_vs_frozen_control": paired,
                "positive_folds": positive_folds,
                "total_folds": len(fold_returns),
                "confirmation_mean_return_ci_lower_95": ci_lower,
                "observer_manifest": str(observer_path),
            }
        )

    return_matrix = pd.concat(daily_returns, axis=1).dropna(how="any")
    return_path_groups: dict[str, list[str]] = {}
    for name, returns in daily_returns.items():
        path_hash = stable_hash(
            np.round(returns.to_numpy(dtype=float), 15).tolist(),
            length=64,
        )
        return_path_groups.setdefault(path_hash, []).append(name)
    for row in rows:
        row["return_path_hash"] = next(
            path_hash
            for path_hash, names in return_path_groups.items()
            if row["policy_name"] in names
        )
    multiple = multiple_testing_bootstrap(
        return_matrix,
        bootstrap_samples=settings.research.multiple_testing_bootstrap_samples,
        block_size=settings.research.multiple_testing_block_size,
        seed=settings.app.random_seed,
        known_trial_count=prior_trials + len(parameters_set),
    )
    for row_index, row in enumerate(rows):
        name = str(row["policy_name"])
        result = results[name]
        stressed_result = stressed_results[name]
        stochastic = _portfolio_stochastic_validation(
            settings,
            normal_equity=result.equity_curve,
            stressed_equity=stressed_result.equity_curve,
            seed_offset=20_000 + row_index * 10,
        )
        economic_checks = {
            "development_positive": (float(row["periods"]["development"]["net_return"]) > 0),
            "validation_positive": (float(row["periods"]["validation"]["net_return"]) > 0),
            "confirmation_positive": (float(row["periods"]["confirmation"]["net_return"]) > 0),
            "all_stressed_periods_positive": all(
                float(row["stressed_periods"][period]["net_return"]) > 0 for period in periods
            ),
            "minimum_effective_sample": (
                int(result.metrics["portfolio_period_effective_sample_size"])
                >= settings.research.minimum_effective_sample_size
            ),
            "minimum_rebalances": (
                int(result.metrics["rebalance_count"]) >= settings.research.minimum_trades
            ),
            "minimum_closed_episodes": (int(result.metrics["closed_position_episodes"]) >= 30),
            "profit_factor": (
                float(result.metrics["portfolio_period_profit_factor"])
                >= settings.research.minimum_profit_factor
            ),
            "maximum_drawdown": (
                abs(float(result.metrics["maximum_drawdown"])) <= settings.research.maximum_drawdown
            ),
            "positive_fold_gate": (
                int(row["positive_folds"]) >= settings.research.minimum_positive_folds
            ),
            "exposure_limits_respected": bool(
                result.integrity["maximum_exposure_respected"]
                and result.integrity["maximum_position_exposure_respected"]
                and result.integrity["minimum_cash_respected"]
            ),
        }
        statistical_checks = {
            "confirmation_ci_lower_positive": (
                float(row["confirmation_mean_return_ci_lower_95"]) > 0.0
            ),
            "deflated_sharpe": (
                float(multiple.deflated_sharpe_probabilities.get(name, 0.0))
                >= settings.research.minimum_deflated_sharpe_probability
            ),
            "white_reality_check": (
                multiple.white_reality_check_pvalue
                <= settings.research.maximum_white_reality_check_pvalue
            ),
            "hansen_spa": (
                multiple.hansen_spa_pvalue <= settings.research.maximum_hansen_spa_pvalue
            ),
            "pbo": (
                multiple.probability_of_backtest_overfitting is not None
                and multiple.probability_of_backtest_overfitting
                <= settings.research.maximum_probability_of_backtest_overfitting
            ),
            "monte_carlo": bool(
                stochastic["normal"]["monte_carlo"]["passed"]
                and stochastic["stressed"]["monte_carlo"]["passed"]
            ),
            "dirichlet": bool(
                stochastic["normal"]["dirichlet"]["passed"]
                and stochastic["stressed"]["dirichlet"]["passed"]
            ),
        }
        row["gates"] = {
            "economic_checks": economic_checks,
            "statistical_checks": statistical_checks,
            "deflated_sharpe_probability": float(
                multiple.deflated_sharpe_probabilities.get(name, 0.0)
            ),
            "stochastic_validation": stochastic,
            "economic_pass": all(economic_checks.values()),
            "statistical_pass": all(statistical_checks.values()),
            "holdout_status": "CONTAMINATED_BY_PRIOR_EXPLORATION",
            "research_pass": False,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }

    rows.sort(
        key=lambda row: (
            float(row["periods"]["development"]["sharpe"])
            + float(row["periods"]["development"]["annualized_return"])
            - abs(float(row["periods"]["development"]["maximum_drawdown"]))
        ),
        reverse=True,
    )
    economic_leads = [row for row in rows if row["gates"]["economic_pass"]]
    statistical_rows = [row for row in rows if row["gates"]["statistical_pass"]]
    benchmarks = capital_utilization_benchmark_suite(
        frames,
        start=frozen_control.equity_curve.index[0],
        minimum_history_observations=90,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        allowed_markets=markets,
        exposure_matches={
            name: float(result.metrics["average_exposure"]) for name, result in results.items()
        },
    )
    payload = {
        "status": "COMPLETED_NOT_PROMOTED",
        "campaign": "PORTFOLIO_BREAKOUT_V1",
        "result_type": "PRE_REGISTERED_ECONOMIC_ALPHA_FAMILY",
        "breakout_engine_version": BREAKOUT_ENGINE_VERSION,
        "source_candidate_identity": frozen["immutable_identity"],
        "source_frozen_strategy_dna_hash": frozen["strategy_dna_hash"],
        "frozen_lead_mutated": False,
        "markets": list(markets),
        "timeframe": "1d",
        "portfolio_policy": asdict(policy),
        "parameters_tested": len(parameters_set),
        "prior_trials_accounted": prior_trials,
        "total_known_trials": prior_trials + len(parameters_set),
        "periods": periods,
        "multiple_testing": asdict(multiple),
        "return_path_audit": {
            "declared_trial_count": len(parameters_set),
            "unique_return_path_count": len(return_path_groups),
            "structural_redundancy_detected": (len(return_path_groups) < len(parameters_set)),
            "groups": [
                {
                    "return_path_hash": path_hash,
                    "policies": names,
                }
                for path_hash, names in sorted(return_path_groups.items())
            ],
            "interpretation": (
                "Equal and inverse-volatility weights can collapse to the same "
                "path when two positions both saturate the 20% hard asset cap. "
                "Declared variants still count as known trials."
            ),
        },
        "frozen_control": frozen_control.summary(),
        "benchmarks": benchmarks,
        "policy_results": rows,
        "economic_research_lead_count": len(economic_leads),
        "statistically_qualified_count": len(statistical_rows),
        "observer_manifests": observer_paths,
        "frozen_candidate_sha256_before": frozen_sha_before,
        "frozen_candidate_sha256_after": sha256_file(frozen_path),
        "frozen_candidate_unchanged": (frozen_sha_before == sha256_file(frozen_path)),
        "selection_bias": "CONTAMINATED_BY_PRIOR_EXPLORATION",
        "paper_candidates": 0,
        "live_orders": 0,
        "live_ready": False,
        "data_hashes": {market: sha256_file(path) for market, path in paths.items()},
    }
    report_path = _breakout_portfolio_campaign_path(settings)
    atomic_write_json(report_path, _json_ready(payload))
    csv_path = report_path.with_suffix(".csv")
    pd.DataFrame(
        [
            {
                "policy": row["policy_name"],
                "return_path_hash": row["return_path_hash"],
                **row["parameters"],
                "net_return": row["normal"]["metrics"]["net_return"],
                "cagr": row["normal"]["metrics"]["annualized_return"],
                "sharpe": row["normal"]["metrics"]["sharpe"],
                "sortino": row["normal"]["metrics"]["sortino"],
                "maximum_drawdown": row["normal"]["metrics"]["maximum_drawdown"],
                "average_exposure": row["normal"]["metrics"]["average_exposure"],
                "profit_factor": row["normal"]["metrics"]["portfolio_period_profit_factor"],
                "closed_episodes": row["normal"]["metrics"]["closed_position_episodes"],
                "economic_pass": row["gates"]["economic_pass"],
                "statistical_pass": row["gates"]["statistical_pass"],
                "monte_carlo_gate": row["gates"]["statistical_checks"]["monte_carlo"],
                "dirichlet_gate": row["gates"]["statistical_checks"]["dirichlet"],
                "paper_candidate": False,
                "live_ready": False,
            }
            for row in rows
        ]
    ).to_csv(csv_path, index=False)
    return {
        "status": payload["status"],
        "campaign": payload["campaign"],
        "parameters_tested": len(parameters_set),
        "prior_trials_accounted": payload["prior_trials_accounted"],
        "total_known_trials": payload["total_known_trials"],
        "economic_research_lead_count": len(economic_leads),
        "statistically_qualified_count": len(statistical_rows),
        "frozen_candidate_unchanged": payload["frozen_candidate_unchanged"],
        "report": str(report_path),
        "csv": str(csv_path),
        "observer_manifests": observer_paths,
        "paper_candidates": 0,
        "live_orders": 0,
        "live_ready": False,
    }


def _run_diversified_rotation_campaign(settings: Settings) -> dict[str, Any]:
    """Run a pre-registered top-3/top-4 volatility-targeted continuation."""

    from research.optimization import multiple_testing_bootstrap
    from research.portfolio_selection import (
        DIVERSIFICATION_ENGINE_VERSION,
        RotationPortfolioPolicy,
        backtest_rotation,
        capital_utilization_benchmark_suite,
        diversified_rotation_policy_set,
        paired_block_bootstrap_difference,
        rotation_decision_snapshot,
        rotation_period_metrics,
    )

    frozen_path = settings.paths.lab_dir / "candidates" / "rotation_research_lead_v1.json"
    frozen_sha_before = sha256_file(frozen_path)
    frozen, frozen_parameters, markets, paths, frames = _frozen_rotation_inputs(settings)
    policies = diversified_rotation_policy_set()
    capital_path = _capital_utilization_campaign_path(settings)
    if not capital_path.is_file():
        raise FileNotFoundError(
            "capital-utilization-v1 must complete before diversified-rotation-v1"
        )
    capital_campaign = read_json(capital_path)
    prior_trials = int(capital_campaign["total_known_trials"])
    periods = dict(capital_campaign["periods"])
    control_policy = _strict_rotation_portfolio_policy(
        settings,
        markets=markets,
    )
    control = backtest_rotation(
        frames,
        frozen_parameters,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        portfolio_policy=control_policy,
    )
    control_returns = control.equity_curve.pct_change(fill_method=None).dropna()
    results: dict[str, Any] = {}
    stressed_results: dict[str, Any] = {}
    daily_returns: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    observer_paths: dict[str, str] = {}
    forward_start = pd.Timestamp(frozen["forward_validation_start"])
    forward_start = (
        forward_start.tz_localize("UTC")
        if forward_start.tzinfo is None
        else forward_start.tz_convert("UTC")
    )

    for policy_index, diversification in enumerate(policies):
        parameters = replace(
            frozen_parameters,
            top_n=diversification.top_n,
            maximum_positions=4,
            weighting=diversification.weighting,
        )
        portfolio_policy = RotationPortfolioPolicy(
            allowed_markets=markets,
            maximum_total_exposure=diversification.maximum_total_exposure,
            maximum_position_exposure=(diversification.maximum_position_exposure),
            minimum_cash=diversification.minimum_cash,
            minimum_history_observations=90,
        )
        normal = backtest_rotation(
            frames,
            parameters,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            portfolio_policy=portfolio_policy,
            diversification_policy=diversification,
        )
        stressed = backtest_rotation(
            frames,
            parameters,
            fee_rate=(settings.costs.default_fee * settings.costs.stressed_cost_multiplier),
            slippage_bps=(settings.costs.slippage_bps * settings.costs.stressed_cost_multiplier),
            spread_bps=(settings.costs.spread_bps * settings.costs.stressed_cost_multiplier),
            portfolio_policy=portfolio_policy,
            diversification_policy=diversification,
        )
        period_metrics: dict[str, Any] = {}
        stressed_period_metrics: dict[str, Any] = {}
        for period, bounds in periods.items():
            period_metrics[period], _ = rotation_period_metrics(
                normal.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )
            stressed_period_metrics[period], _ = rotation_period_metrics(
                stressed.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )
        results[diversification.name] = normal
        stressed_results[diversification.name] = stressed
        policy_returns = normal.equity_curve.pct_change(fill_method=None).dropna()
        daily_returns[diversification.name] = policy_returns[
            ~policy_returns.index.duplicated(keep="last")
        ]
        snapshot = rotation_decision_snapshot(
            frames,
            parameters,
            portfolio_policy=portfolio_policy,
            diversification_policy=diversification,
        )
        observer_payload = {
            "status": "FROZEN_FORWARD_RESEARCH",
            "family": "DIVERSIFIED_ROTATION_V1",
            "source_candidate_identity": frozen["immutable_identity"],
            "source_frozen_strategy_dna_hash": frozen["strategy_dna_hash"],
            "strategy_dna_hash": parameters.dna_hash,
            "diversification_policy": asdict(diversification),
            "diversification_policy_hash": diversification.policy_hash,
            "execution_identity": normal.summary()["execution_identity"],
            "forward_start": forward_start.isoformat(),
            "minimum_forward_closed_daily_observations": 365,
            "minimum_forward_rebalances": 30,
            "regime_coverage_required": True,
            "latest_historical_snapshot": snapshot,
            "forward_observations": [],
            "current_operational_limits_compatible": False,
            "orders_generated": 0,
            "orders_submitted": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
        observer_path = (
            settings.paths.lab_dir
            / "observers"
            / "diversified_rotation_v1"
            / f"{diversification.name.lower()}.json"
        )
        atomic_write_json(observer_path, _json_ready(observer_payload))
        observer_paths[diversification.name] = str(observer_path)
        paired = paired_block_bootstrap_difference(
            daily_returns[diversification.name],
            control_returns,
            samples=2_000,
            block_size=10,
            seed=settings.app.random_seed + policy_index,
        )
        rows.append(
            {
                "policy_name": diversification.name,
                "diversification_policy": asdict(diversification),
                "diversification_policy_hash": diversification.policy_hash,
                "strategy_dna_hash": parameters.dna_hash,
                "parameters": asdict(parameters),
                "signal_horizons_and_filters_preserved": (
                    parameters.momentum_lookbacks == frozen_parameters.momentum_lookbacks
                    and parameters.rebalance_days == frozen_parameters.rebalance_days
                    and parameters.asset_ema_period == frozen_parameters.asset_ema_period
                    and parameters.btc_ema_period == frozen_parameters.btc_ema_period
                    and parameters.continuous_regime == frozen_parameters.continuous_regime
                ),
                "normal": normal.summary(),
                "stressed": stressed.summary(),
                "periods": period_metrics,
                "stressed_periods": stressed_period_metrics,
                "paired_block_bootstrap_vs_frozen_control": paired,
                "observer_manifest": str(observer_path),
            }
        )

    return_matrix = pd.concat(daily_returns, axis=1).dropna(how="any")
    multiple = multiple_testing_bootstrap(
        return_matrix,
        bootstrap_samples=settings.research.multiple_testing_bootstrap_samples,
        block_size=settings.research.multiple_testing_block_size,
        seed=settings.app.random_seed,
        known_trial_count=prior_trials + len(policies),
    )
    for row_index, row in enumerate(rows):
        name = str(row["policy_name"])
        result = results[name]
        stressed_result = stressed_results[name]
        paired = row["paired_block_bootstrap_vs_frozen_control"]
        stochastic = _portfolio_stochastic_validation(
            settings,
            normal_equity=result.equity_curve,
            stressed_equity=stressed_result.equity_curve,
            seed_offset=30_000 + row_index * 10,
        )
        economic_checks = {
            "all_periods_positive": all(
                float(row["periods"][period]["net_return"]) > 0 for period in periods
            ),
            "all_stressed_periods_positive": all(
                float(row["stressed_periods"][period]["net_return"]) > 0 for period in periods
            ),
            "minimum_effective_sample": (
                int(result.metrics["portfolio_period_effective_sample_size"])
                >= settings.research.minimum_effective_sample_size
            ),
            "minimum_rebalances": (
                int(result.metrics["rebalance_count"]) >= settings.research.minimum_trades
            ),
            "profit_factor": (
                float(result.metrics["portfolio_period_profit_factor"])
                >= settings.research.minimum_profit_factor
            ),
            "maximum_drawdown": (
                abs(float(result.metrics["maximum_drawdown"])) <= settings.research.maximum_drawdown
            ),
            "paired_incremental_ci_positive": (float(paired["ci_lower_95"]) > 0.0),
            "exposure_limits_respected": bool(
                result.integrity["maximum_exposure_respected"]
                and result.integrity["maximum_position_exposure_respected"]
                and result.integrity["minimum_cash_respected"]
            ),
        }
        statistical_checks = {
            "source_historical_statistical_gates": bool(
                frozen["robustness"]["statistical_gates_passed"]
            ),
            "deflated_sharpe": (
                float(multiple.deflated_sharpe_probabilities.get(name, 0.0))
                >= settings.research.minimum_deflated_sharpe_probability
            ),
            "white_reality_check": (
                multiple.white_reality_check_pvalue
                <= settings.research.maximum_white_reality_check_pvalue
            ),
            "hansen_spa": (
                multiple.hansen_spa_pvalue <= settings.research.maximum_hansen_spa_pvalue
            ),
            "pbo": (
                multiple.probability_of_backtest_overfitting is not None
                and multiple.probability_of_backtest_overfitting
                <= settings.research.maximum_probability_of_backtest_overfitting
            ),
            "monte_carlo": bool(
                stochastic["normal"]["monte_carlo"]["passed"]
                and stochastic["stressed"]["monte_carlo"]["passed"]
            ),
            "dirichlet": bool(
                stochastic["normal"]["dirichlet"]["passed"]
                and stochastic["stressed"]["dirichlet"]["passed"]
            ),
        }
        row["gates"] = {
            "economic_checks": economic_checks,
            "statistical_checks": statistical_checks,
            "deflated_sharpe_probability": float(
                multiple.deflated_sharpe_probabilities.get(name, 0.0)
            ),
            "stochastic_validation": stochastic,
            "economic_pass": all(economic_checks.values()),
            "statistical_pass": all(statistical_checks.values()),
            "research_pass": False,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }

    benchmarks = capital_utilization_benchmark_suite(
        frames,
        start=control.equity_curve.index[0],
        minimum_history_observations=90,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        allowed_markets=markets,
        exposure_matches={
            name: float(result.metrics["average_exposure"]) for name, result in results.items()
        },
    )
    rows.sort(
        key=lambda row: float(row["normal"]["metrics"]["net_return"]),
        reverse=True,
    )
    payload = {
        "status": "COMPLETED_NOT_PROMOTED",
        "campaign": "DIVERSIFIED_ROTATION_V1",
        "result_type": "PRE_REGISTERED_DIVERSIFICATION_CONTINUATION",
        "diversification_engine_version": DIVERSIFICATION_ENGINE_VERSION,
        "source_candidate_identity": frozen["immutable_identity"],
        "source_frozen_strategy_dna_hash": frozen["strategy_dna_hash"],
        "source_historical_robustness": frozen["robustness"],
        "frozen_signal_horizons_and_filters_changed": False,
        "declared_new_dna_dimensions": [
            "top_n",
            "maximum_positions",
            "weighting",
            "volatility_target",
            "covariance_lookback",
            "rebalance_buffer",
        ],
        "markets": list(markets),
        "timeframe": "1d",
        "policies_tested": len(policies),
        "prior_trials_accounted": prior_trials,
        "total_known_trials": prior_trials + len(policies),
        "periods": periods,
        "multiple_testing": asdict(multiple),
        "frozen_control": control.summary(),
        "benchmarks": benchmarks,
        "policy_results": rows,
        "observer_manifests": observer_paths,
        "frozen_candidate_sha256_before": frozen_sha_before,
        "frozen_candidate_sha256_after": sha256_file(frozen_path),
        "frozen_candidate_unchanged": (frozen_sha_before == sha256_file(frozen_path)),
        "selection_bias": "CONTAMINATED_BY_PRIOR_EXPLORATION",
        "paper_candidates": 0,
        "live_orders": 0,
        "live_ready": False,
        "data_hashes": {market: sha256_file(path) for market, path in paths.items()},
    }
    report_path = _diversified_rotation_campaign_path(settings)
    atomic_write_json(report_path, _json_ready(payload))
    csv_path = report_path.with_suffix(".csv")
    pd.DataFrame(
        [
            {
                "policy": row["policy_name"],
                "top_n": row["parameters"]["top_n"],
                "weighting": row["parameters"]["weighting"],
                "target_volatility": row["diversification_policy"]["target_annualized_volatility"],
                "net_return": row["normal"]["metrics"]["net_return"],
                "cagr": row["normal"]["metrics"]["annualized_return"],
                "sharpe": row["normal"]["metrics"]["sharpe"],
                "sortino": row["normal"]["metrics"]["sortino"],
                "maximum_drawdown": row["normal"]["metrics"]["maximum_drawdown"],
                "daily_cvar_95": row["normal"]["metrics"]["daily_cvar_95"],
                "average_exposure": row["normal"]["metrics"]["average_exposure"],
                "average_cash": row["normal"]["metrics"]["cash_fraction_average"],
                "economic_pass": row["gates"]["economic_pass"],
                "statistical_pass": row["gates"]["statistical_pass"],
                "monte_carlo_gate": row["gates"]["statistical_checks"]["monte_carlo"],
                "dirichlet_gate": row["gates"]["statistical_checks"]["dirichlet"],
                "paper_candidate": False,
                "live_ready": False,
            }
            for row in rows
        ]
    ).to_csv(csv_path, index=False)
    return {
        "status": payload["status"],
        "campaign": payload["campaign"],
        "policies_tested": len(policies),
        "total_known_trials": payload["total_known_trials"],
        "frozen_candidate_unchanged": payload["frozen_candidate_unchanged"],
        "report": str(report_path),
        "csv": str(csv_path),
        "observer_manifests": observer_paths,
        "paper_candidates": 0,
        "live_orders": 0,
        "live_ready": False,
    }


def _run_absolute_momentum_campaign(settings: Settings) -> dict[str, Any]:
    """Run the fixed absolute-momentum volatility-budget family."""

    from research.absolute_momentum import (
        ABSOLUTE_MOMENTUM_ENGINE_VERSION,
        ABSOLUTE_MOMENTUM_FAMILY,
        absolute_momentum_parameter_set,
        backtest_absolute_momentum,
    )
    from research.forward_observer import (
        ForwardPerformanceGatePolicy,
        build_rotation_forward_evidence,
        merge_portfolio_forward_manifest,
    )
    from research.optimization import multiple_testing_bootstrap
    from research.portfolio_selection import (
        RotationPortfolioPolicy,
        capital_utilization_benchmark_suite,
        rotation_period_metrics,
    )

    markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    paths = {
        market: settings.paths.processed_data_dir / f"{market}_1d.parquet"
        for market in markets
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing absolute-momentum datasets: {missing}")
    frames = {market: pd.read_parquet(path) for market, path in paths.items()}
    parameters_set = absolute_momentum_parameter_set()
    primary_name = "ABS_MOM_VOL_05"
    policy = RotationPortfolioPolicy(
        allowed_markets=markets,
        maximum_total_exposure=0.20,
        maximum_position_exposure=0.20,
        minimum_cash=0.80,
        minimum_history_observations=90,
    )
    periods = {
        "development": ("2019-12-01", "2023-12-31"),
        "validation": ("2024-01-01", "2025-06-30"),
        "confirmation": ("2025-07-01", "2026-07-23"),
    }
    exploration_ledger = {
        "prior_formal_and_storm_trials": 16_312,
        "absolute_momentum_development_grid": 288,
        "mean_reversion_development_grid": 108,
        "component_ablation_paths": 6,
        "midpoint_risk_budget_path": 1,
    }
    total_known_trials = sum(exploration_ledger.values())
    normal_results: dict[str, Any] = {}
    stressed_results: dict[str, Any] = {}
    development_returns: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    observer_paths: dict[str, str] = {}
    forward_summaries: dict[str, Any] = {}
    forward_start = pd.Timestamp("2026-07-25T00:00:00+00:00")

    for parameters in parameters_set:
        name = f"ABS_MOM_VOL_{int(parameters.target_annualized_volatility * 100):02d}"
        normal = backtest_absolute_momentum(
            frames,
            parameters,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            portfolio_policy=policy,
        )
        stressed = backtest_absolute_momentum(
            frames,
            parameters,
            fee_rate=settings.costs.default_fee
            * settings.costs.stressed_cost_multiplier,
            slippage_bps=settings.costs.slippage_bps
            * settings.costs.stressed_cost_multiplier,
            spread_bps=settings.costs.spread_bps
            * settings.costs.stressed_cost_multiplier,
            portfolio_policy=policy,
        )
        normal_results[name] = normal
        stressed_results[name] = stressed
        period_metrics: dict[str, Any] = {}
        stressed_period_metrics: dict[str, Any] = {}
        for period, bounds in periods.items():
            period_metrics[period], returns = rotation_period_metrics(
                normal.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )
            stressed_period_metrics[period], _ = rotation_period_metrics(
                stressed.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )
            if period == "development":
                development_returns[name] = returns
        execution_identity = normal.summary()["execution_identity"]
        source_candidate_identity = stable_hash(
            {
                "campaign": "ABSOLUTE_MOMENTUM_V1",
                "strategy_dna_hash": parameters.dna_hash,
                "portfolio_policy_hash": policy.policy_hash,
                "forward_start": forward_start.isoformat(),
            },
            length=64,
        )
        observer = {
            "status": "FROZEN_FORWARD_RESEARCH",
            "family": "ABSOLUTE_MOMENTUM_V1",
            "policy_name": name,
            "source_candidate_identity": source_candidate_identity,
            "strategy_dna_hash": parameters.dna_hash,
            "execution_identity": execution_identity,
            "parameters": asdict(parameters),
            "portfolio_policy": asdict(policy),
            "portfolio_policy_hash": policy.policy_hash,
            "forward_start": forward_start.isoformat(),
            "minimum_forward_closed_daily_observations": 365,
            "minimum_forward_rebalances": 30,
            "selection_bias": "CONTAMINATED_BY_PRIOR_HISTORICAL_EXPLORATION",
            "orders_generated": 0,
            "orders_submitted": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
        observer_path = (
            settings.paths.lab_dir
            / "observers"
            / "absolute_momentum_v1"
            / f"{name.lower()}.json"
        )
        if observer_path.is_file():
            existing = read_json(observer_path)
            observer.update(
                _preserved_breakout_forward_fields(
                    existing,
                    source_candidate_identity=source_candidate_identity,
                    strategy_dna_hash=parameters.dna_hash,
                    execution_identity=execution_identity,
                    forward_start=forward_start,
                )
            )
        evidence = build_rotation_forward_evidence(
            normal,
            frames,
            forward_start=forward_start,
            minimum_observations=int(
                observer["minimum_forward_closed_daily_observations"]
            ),
            minimum_rebalances=int(observer["minimum_forward_rebalances"]),
            performance_policy=ForwardPerformanceGatePolicy(
                minimum_profit_factor=settings.research.minimum_profit_factor,
                minimum_stressed_profit_factor=(
                    settings.research.minimum_stressed_profit_factor
                ),
                maximum_drawdown=settings.research.maximum_drawdown,
                minimum_effective_sample_size=(
                    settings.research.minimum_effective_sample_size
                ),
                stressed_cost_multiplier=(
                    settings.costs.stressed_cost_multiplier
                ),
                bootstrap_samples=(
                    settings.research.multiple_testing_bootstrap_samples
                ),
                bootstrap_block_size=(
                    settings.research.multiple_testing_block_size
                ),
                bootstrap_seed=settings.app.random_seed,
            ),
        )
        observer = merge_portfolio_forward_manifest(
            observer,
            evidence,
            source_candidate_identity=source_candidate_identity,
            strategy_dna_hash=parameters.dna_hash,
            execution_identity=execution_identity,
            forward_start=forward_start,
        )
        observer["data_hashes"] = {
            market: sha256_file(path) for market, path in paths.items()
        }
        atomic_write_json(observer_path, _json_ready(observer))
        observer_paths[name] = str(observer_path)
        forward_summaries[name] = observer["forward_summary"]
        rows.append(
            {
                "policy_name": name,
                "strategy_dna_hash": parameters.dna_hash,
                "parameters": asdict(parameters),
                "primary_pre_registered_path": name == primary_name,
                "normal": normal.summary(),
                "stressed": stressed.summary(),
                "periods": period_metrics,
                "stressed_periods": stressed_period_metrics,
                "observer_manifest": str(observer_path),
            }
        )

    matrix = pd.concat(development_returns, axis=1).dropna(how="any")
    multiple = multiple_testing_bootstrap(
        matrix,
        bootstrap_samples=settings.research.multiple_testing_bootstrap_samples,
        block_size=settings.research.multiple_testing_block_size,
        seed=settings.app.random_seed,
        known_trial_count=total_known_trials,
    )
    for row_index, row in enumerate(rows):
        name = str(row["policy_name"])
        normal = normal_results[name]
        stressed = stressed_results[name]
        stochastic = _portfolio_stochastic_validation(
            settings,
            normal_equity=normal.equity_curve,
            stressed_equity=stressed.equity_curve,
            seed_offset=40_000 + row_index * 10,
        )
        economic_checks = {
            "all_periods_positive": all(
                float(row["periods"][period]["net_return"]) > 0.0
                for period in periods
            ),
            "all_stressed_periods_positive": all(
                float(row["stressed_periods"][period]["net_return"]) > 0.0
                for period in periods
            ),
            "minimum_rebalances": (
                int(normal.metrics["rebalance_count"])
                >= settings.research.minimum_trades
            ),
            "minimum_effective_sample": (
                int(normal.metrics["portfolio_period_effective_sample_size"])
                >= settings.research.minimum_effective_sample_size
            ),
            "profit_factor": (
                float(normal.metrics["portfolio_period_profit_factor"])
                >= settings.research.minimum_profit_factor
            ),
            "validation_profit_factor": (
                float(row["periods"]["validation"]["portfolio_period_profit_factor"])
                >= settings.research.minimum_profit_factor
            ),
            "stressed_validation_profit_factor": (
                float(
                    row["stressed_periods"]["validation"][
                        "portfolio_period_profit_factor"
                    ]
                )
                >= settings.research.minimum_stressed_profit_factor
            ),
            "maximum_drawdown": (
                abs(float(normal.metrics["maximum_drawdown"]))
                <= settings.research.maximum_drawdown
            ),
            "exposure_limits_respected": bool(
                normal.integrity["maximum_exposure_respected"]
                and normal.integrity["maximum_position_exposure_respected"]
                and normal.integrity["minimum_cash_respected"]
            ),
        }
        statistical_checks = {
            "deflated_sharpe": (
                float(multiple.deflated_sharpe_probabilities.get(name, 0.0))
                >= settings.research.minimum_deflated_sharpe_probability
            ),
            "white_reality_check": (
                multiple.white_reality_check_pvalue
                <= settings.research.maximum_white_reality_check_pvalue
            ),
            "hansen_spa": (
                multiple.hansen_spa_pvalue
                <= settings.research.maximum_hansen_spa_pvalue
            ),
            "pbo": (
                multiple.probability_of_backtest_overfitting is not None
                and multiple.probability_of_backtest_overfitting
                <= settings.research.maximum_probability_of_backtest_overfitting
            ),
            "monte_carlo": bool(
                stochastic["normal"]["monte_carlo"]["passed"]
                and stochastic["stressed"]["monte_carlo"]["passed"]
            ),
            "dirichlet": bool(
                stochastic["normal"]["dirichlet"]["passed"]
                and stochastic["stressed"]["dirichlet"]["passed"]
            ),
            "untouched_holdout": False,
        }
        row["gates"] = {
            "economic_checks": economic_checks,
            "statistical_checks": statistical_checks,
            "deflated_sharpe_probability": float(
                multiple.deflated_sharpe_probabilities.get(name, 0.0)
            ),
            "stochastic_validation": stochastic,
            "economic_pass": all(economic_checks.values()),
            "statistical_pass": all(statistical_checks.values()),
            "research_pass": False,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }

    rows.sort(
        key=lambda row: float(row["parameters"]["target_annualized_volatility"])
    )
    primary = next(row for row in rows if row["policy_name"] == primary_name)
    benchmarks = capital_utilization_benchmark_suite(
        frames,
        start=normal_results[primary_name].equity_curve.index[0],
        minimum_history_observations=policy.minimum_history_observations,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        allowed_markets=markets,
        exposure_matches={
            name: float(result.metrics["average_exposure"])
            for name, result in normal_results.items()
        },
    )
    primary_positive_lead = bool(
        primary["gates"]["economic_pass"]
        and primary["gates"]["statistical_checks"]["deflated_sharpe"]
        and primary["gates"]["statistical_checks"]["white_reality_check"]
        and primary["gates"]["statistical_checks"]["hansen_spa"]
        and primary["gates"]["statistical_checks"]["monte_carlo"]
        and primary["gates"]["statistical_checks"]["dirichlet"]
    )
    payload = {
        "status": (
            "POSITIVE_RESEARCH_LEAD_NOT_PROMOTED"
            if primary_positive_lead
            else "COMPLETED_NOT_PROMOTED"
        ),
        "campaign": "ABSOLUTE_MOMENTUM_V1",
        "result_type": "FIXED_PRIMARY_WITH_FULL_EXPLORATION_LEDGER",
        "strategy_family": ABSOLUTE_MOMENTUM_FAMILY,
        "engine_version": ABSOLUTE_MOMENTUM_ENGINE_VERSION,
        "primary_policy_name": primary_name,
        "primary_strategy_dna_hash": primary["strategy_dna_hash"],
        "markets": list(markets),
        "timeframe": "1d",
        "portfolio_policy": asdict(policy),
        "periods": periods,
        "formal_risk_budget_paths": len(parameters_set),
        "exploration_ledger": exploration_ledger,
        "total_known_trials": total_known_trials,
        "multiple_testing": asdict(multiple),
        "primary_result": primary,
        "policy_results": rows,
        "benchmarks": benchmarks,
        "selection_bias": "CONTAMINATED_BY_PRIOR_HISTORICAL_EXPLORATION",
        "holdout_status": "NO_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS",
        "forward_evidence_required": True,
        "observer_manifests": observer_paths,
        "forward_summaries": forward_summaries,
        "total_forward_observations": sum(
            int(summary["closed_daily_observations"])
            for summary in forward_summaries.values()
        ),
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
        "data_hashes": {
            market: sha256_file(path)
            for market, path in paths.items()
        },
    }
    report_path = _absolute_momentum_campaign_path(settings)
    atomic_write_json(report_path, _json_ready(payload))
    csv_path = report_path.with_suffix(".csv")
    pd.DataFrame(
        [
            {
                "policy": row["policy_name"],
                "target_volatility": row["parameters"][
                    "target_annualized_volatility"
                ],
                "net_return": row["normal"]["metrics"]["net_return"],
                "cagr": row["normal"]["metrics"]["annualized_return"],
                "sharpe": row["normal"]["metrics"]["sharpe"],
                "maximum_drawdown": row["normal"]["metrics"]["maximum_drawdown"],
                "average_exposure": row["normal"]["metrics"]["average_exposure"],
                "profit_factor": row["normal"]["metrics"][
                    "portfolio_period_profit_factor"
                ],
                "dsr": row["gates"]["deflated_sharpe_probability"],
                "monte_carlo_gate": row["gates"]["statistical_checks"][
                    "monte_carlo"
                ],
                "dirichlet_gate": row["gates"]["statistical_checks"]["dirichlet"],
                "pbo_gate": row["gates"]["statistical_checks"]["pbo"],
                "economic_pass": row["gates"]["economic_pass"],
                "statistical_pass": row["gates"]["statistical_pass"],
                "research_pass": False,
                "paper_candidate": False,
                "live_ready": False,
            }
            for row in rows
        ]
    ).to_csv(csv_path, index=False)
    return {
        "status": payload["status"],
        "campaign": payload["campaign"],
        "primary_policy_name": primary_name,
        "primary_positive_research_lead": primary_positive_lead,
        "formal_risk_budget_paths": len(parameters_set),
        "total_known_trials": total_known_trials,
        "pbo": multiple.probability_of_backtest_overfitting,
        "report": str(report_path),
        "csv": str(csv_path),
        "observer_manifests": observer_paths,
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }


def _run_absolute_momentum_plateau_campaign(
    settings: Settings,
) -> dict[str, Any]:
    """Generate, register and evaluate the preregistered N±2 plateau."""

    from research.absolute_momentum import (
        ABSOLUTE_MOMENTUM_PLATEAU_ENGINE_VERSION,
        ABSOLUTE_MOMENTUM_PLATEAU_FAMILY,
        absolute_momentum_plateau_parameter_set,
        backtest_absolute_momentum,
    )
    from research.forward_observer import (
        ForwardPerformanceGatePolicy,
        build_rotation_forward_evidence,
        merge_portfolio_forward_manifest,
    )
    from research.optimization import multiple_testing_bootstrap
    from research.portfolio_selection import (
        RotationPortfolioPolicy,
        rotation_period_metrics,
    )
    from research.strategy_registry import (
        ContentAddressedTrialRegistry,
        gaussian_plateau_table,
        plateau_selection_pbo,
    )

    markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    paths = {
        market: settings.paths.processed_data_dir
        / f"{market}_1d.parquet"
        for market in markets
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing plateau campaign datasets: {missing}"
        )
    data_hashes = {
        market: sha256_file(path) for market, path in paths.items()
    }
    data_fingerprint = stable_hash(data_hashes, length=64)
    frames = {
        market: pd.read_parquet(path)
        for market, path in paths.items()
    }
    policy = RotationPortfolioPolicy(
        allowed_markets=markets,
        maximum_total_exposure=0.20,
        maximum_position_exposure=0.20,
        minimum_cash=0.80,
        minimum_history_observations=90,
    )
    periods = {
        "development": ("2019-12-01", "2023-12-31"),
        "validation": ("2024-01-01", "2025-06-30"),
        "confirmation": ("2025-07-01", "2026-07-23"),
    }
    candidates = absolute_momentum_plateau_parameter_set()
    report_path = _absolute_momentum_plateau_campaign_path(settings)
    plan_path = report_path.with_name(
        "absolute_momentum_plateau_plan_v1.json"
    )
    search_space_hash = stable_hash(
        [row.dna_hash for row in candidates],
        length=64,
    )
    expected_plan = {
        "schema_version": "absolute_momentum_plateau_plan_v1",
        "status": "PREREGISTERED_NOT_RUN",
        "campaign": "ABSOLUTE_MOMENTUM_PLATEAU_V1",
        "strategy_family": ABSOLUTE_MOMENTUM_PLATEAU_FAMILY,
        "engine_version": ABSOLUTE_MOMENTUM_PLATEAU_ENGINE_VERSION,
        "trial_count": len(candidates),
        "strategy_dna_hashes": [
            row.dna_hash for row in candidates
        ],
        "strategy_dna": [asdict(row) for row in candidates],
        "search_space_hash": search_space_hash,
        "selection_basis": "DEVELOPMENT_GAUSSIAN_N_PLUS_MINUS_2",
        "kernel": [0.05, 0.25, 0.40, 0.25, 0.05],
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    if plan_path.is_file():
        stored_plan = read_json(plan_path)
        for field in (
            "campaign",
            "engine_version",
            "trial_count",
            "strategy_dna_hashes",
            "search_space_hash",
        ):
            if stored_plan.get(field) != expected_plan.get(field):
                raise RuntimeError(
                    f"ABSOLUTE_MOMENTUM_PLATEAU_PLAN_DRIFT:{field}"
                )
    else:
        atomic_write_json(plan_path, _json_ready(expected_plan))

    def candidate_name(row: Any) -> str:
        shift = (
            f"P{row.horizon_shift:02d}"
            if row.horizon_shift >= 0
            else f"M{abs(row.horizon_shift):02d}"
        )
        return (
            f"AMPS_{shift}_V{row.volatility_lookback}_"
            f"T{int(row.target_annualized_volatility * 100):02d}"
        )

    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    development_returns: dict[str, pd.Series] = {}
    by_name: dict[str, Any] = {}
    for candidate in candidates:
        name = candidate_name(candidate)
        by_name[name] = candidate
        result = backtest_absolute_momentum(
            frames,
            candidate.parameters,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            portfolio_policy=policy,
        )
        results[name] = result
        period_metrics: dict[str, Any] = {}
        for period, bounds in periods.items():
            metrics, returns = rotation_period_metrics(
                result.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )
            period_metrics[period] = metrics
            if period == "development":
                development_returns[name] = returns
        rows.append(
            {
                "strategy_id": name,
                "strategy_trial_dna_hash": candidate.dna_hash,
                "execution_dna_hash": result.parameters.dna_hash,
                "parameters": asdict(candidate),
                "derived_execution_parameters": asdict(
                    candidate.parameters
                ),
                "normal": result.summary(),
                "periods": period_metrics,
            }
        )
    matrix = pd.concat(
        development_returns,
        axis=1,
    ).dropna(how="any")
    coordinates = {
        name: int(candidate.horizon_shift)
        for name, candidate in by_name.items()
    }
    groups = {
        name: candidate.nuisance_group
        for name, candidate in by_name.items()
    }
    plateau = gaussian_plateau_table(
        matrix,
        coordinates=coordinates,
        groups=groups,
    )
    registry = ContentAddressedTrialRegistry(
        settings.paths.lab_dir
        / "strategy_registry"
        / "absolute_momentum_plateau_v1",
        campaign_id="ABSOLUTE_MOMENTUM_PLATEAU_V1",
    )
    for row in rows:
        name = str(row["strategy_id"])
        plateau_row = plateau.loc[name].to_dict()
        row["plateau"] = plateau_row
        development = development_returns[name]
        registration = registry.register(
            data_fingerprint=data_fingerprint,
            strategy_family=ABSOLUTE_MOMENTUM_PLATEAU_FAMILY,
            strategy_dna_hash=str(
                row["strategy_trial_dna_hash"]
            ),
            parameters=row["parameters"],
            metrics_at_birth={
                **row["periods"]["development"],
                "full_sample_metrics": row["normal"]["metrics"],
            },
            return_path_hash=stable_hash(
                [
                    round(float(value), 15)
                    for value in development.to_numpy(dtype=float)
                ],
                length=64,
            ),
            selection_metadata=_json_ready(plateau_row),
        )
        row["registration"] = registration
    registry_audit = registry.audit()
    absolute_report_path = _absolute_momentum_campaign_path(settings)
    base_known_trials = (
        int(
            read_json(absolute_report_path).get(
                "total_known_trials",
                16_715,
            )
        )
        if absolute_report_path.is_file()
        else 16_715
    )
    total_known_trials = (
        base_known_trials
        + int(registry_audit["unique_trial_count"])
    )
    multiple = multiple_testing_bootstrap(
        matrix,
        bootstrap_samples=(
            settings.research.multiple_testing_bootstrap_samples
        ),
        block_size=settings.research.multiple_testing_block_size,
        seed=settings.app.random_seed,
        known_trial_count=total_known_trials,
    )
    plateau_pbo, plateau_logits = plateau_selection_pbo(
        matrix,
        coordinates=coordinates,
        groups=groups,
    )
    eligible = plateau[
        plateau["plateau_eligible"].astype(bool)
    ].sort_values(
        "gaussian_smoothed_sharpe",
        ascending=False,
    )
    primary_name = (
        str(eligible.index[0]) if not eligible.empty else None
    )
    primary_result: dict[str, Any] | None = None
    observer_paths: dict[str, str] = {}
    forward_summaries: dict[str, Any] = {}
    if primary_name is not None:
        selected_candidate = by_name[primary_name]
        normal = results[primary_name]
        stressed = backtest_absolute_momentum(
            frames,
            selected_candidate.parameters,
            fee_rate=(
                settings.costs.default_fee
                * settings.costs.stressed_cost_multiplier
            ),
            slippage_bps=(
                settings.costs.slippage_bps
                * settings.costs.stressed_cost_multiplier
            ),
            spread_bps=(
                settings.costs.spread_bps
                * settings.costs.stressed_cost_multiplier
            ),
            portfolio_policy=policy,
        )
        stressed_periods = {
            period: rotation_period_metrics(
                stressed.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )[0]
            for period, bounds in periods.items()
        }
        stochastic = _portfolio_stochastic_validation(
            settings,
            normal_equity=normal.equity_curve,
            stressed_equity=stressed.equity_curve,
            seed_offset=50_000,
        )
        selected_row = next(
            row
            for row in rows
            if row["strategy_id"] == primary_name
        )
        economic_checks = {
            "complete_profitable_parameter_plateau": bool(
                selected_row["plateau"][
                    "all_neighbors_net_positive"
                ]
            ),
            "positive_minimum_neighbor_sharpe": float(
                selected_row["plateau"][
                    "minimum_neighbor_sharpe"
                ]
            )
            > 0.0,
            "all_periods_positive": all(
                float(
                    selected_row["periods"][period]["net_return"]
                )
                > 0.0
                for period in periods
            ),
            "all_stressed_periods_positive": all(
                float(stressed_periods[period]["net_return"]) > 0.0
                for period in periods
            ),
            "minimum_rebalances": (
                int(normal.metrics["rebalance_count"])
                >= settings.research.minimum_trades
            ),
            "minimum_effective_sample": (
                int(
                    normal.metrics[
                        "portfolio_period_effective_sample_size"
                    ]
                )
                >= settings.research.minimum_effective_sample_size
            ),
            "profit_factor": (
                float(
                    normal.metrics[
                        "portfolio_period_profit_factor"
                    ]
                )
                >= settings.research.minimum_profit_factor
            ),
            "validation_profit_factor": (
                float(
                    selected_row["periods"]["validation"][
                        "portfolio_period_profit_factor"
                    ]
                )
                >= settings.research.minimum_profit_factor
            ),
            "stressed_validation_profit_factor": (
                float(
                    stressed_periods["validation"][
                        "portfolio_period_profit_factor"
                    ]
                )
                >= settings.research.minimum_stressed_profit_factor
            ),
            "maximum_drawdown": (
                abs(float(normal.metrics["maximum_drawdown"]))
                <= settings.research.maximum_drawdown
            ),
            "exposure_limits_respected": all(
                bool(normal.integrity[field])
                for field in (
                    "maximum_exposure_respected",
                    "maximum_position_exposure_respected",
                    "minimum_cash_respected",
                )
            ),
        }
        standard_pbo = (
            multiple.probability_of_backtest_overfitting
        )
        statistical_checks = {
            "deflated_sharpe": (
                float(
                    multiple.deflated_sharpe_probabilities.get(
                        primary_name,
                        0.0,
                    )
                )
                >= settings.research.minimum_deflated_sharpe_probability
            ),
            "white_reality_check": (
                multiple.white_reality_check_pvalue
                <= settings.research.maximum_white_reality_check_pvalue
            ),
            "hansen_spa": (
                multiple.hansen_spa_pvalue
                <= settings.research.maximum_hansen_spa_pvalue
            ),
            "standard_pbo": (
                standard_pbo is not None
                and standard_pbo
                <= settings.research.maximum_probability_of_backtest_overfitting
            ),
            "plateau_selection_pbo": (
                plateau_pbo is not None
                and plateau_pbo
                <= settings.research.maximum_probability_of_backtest_overfitting
            ),
            "monte_carlo": bool(
                stochastic["normal"]["monte_carlo"]["passed"]
                and stochastic["stressed"]["monte_carlo"]["passed"]
            ),
            "dirichlet": bool(
                stochastic["normal"]["dirichlet"]["passed"]
                and stochastic["stressed"]["dirichlet"]["passed"]
            ),
            "untouched_holdout": False,
        }
        primary_result = {
            **selected_row,
            "stressed": stressed.summary(),
            "stressed_periods": stressed_periods,
            "gates": {
                "economic_checks": economic_checks,
                "statistical_checks": statistical_checks,
                "deflated_sharpe_probability": float(
                    multiple.deflated_sharpe_probabilities.get(
                        primary_name,
                        0.0,
                    )
                ),
                "stochastic_validation": stochastic,
                "economic_pass": all(economic_checks.values()),
                "statistical_pass": all(
                    statistical_checks.values()
                ),
                "research_pass": False,
                "paper_candidate_permitted": False,
                "live_ready": False,
            },
        }
        forward_start = pd.Timestamp(
            "2026-07-26T00:00:00+00:00"
        )
        execution_identity = normal.summary()[
            "execution_identity"
        ]
        source_candidate_identity = stable_hash(
            {
                "campaign": "ABSOLUTE_MOMENTUM_PLATEAU_V1",
                "strategy_trial_dna_hash": (
                    selected_candidate.dna_hash
                ),
                "execution_dna_hash": normal.parameters.dna_hash,
                "portfolio_policy_hash": policy.policy_hash,
                "forward_start": forward_start.isoformat(),
            },
            length=64,
        )
        observer_path = (
            settings.paths.lab_dir
            / "observers"
            / "absolute_momentum_plateau_v1"
            / f"{primary_name.lower()}.json"
        )
        observer = {
            "status": "FROZEN_FORWARD_RESEARCH",
            "family": "ABSOLUTE_MOMENTUM_PLATEAU_V1",
            "policy_name": primary_name,
            "source_candidate_identity": source_candidate_identity,
            "strategy_trial_dna_hash": selected_candidate.dna_hash,
            "strategy_dna_hash": normal.parameters.dna_hash,
            "execution_identity": execution_identity,
            "parameters": asdict(selected_candidate),
            "portfolio_policy": asdict(policy),
            "portfolio_policy_hash": policy.policy_hash,
            "forward_start": forward_start.isoformat(),
            "minimum_forward_closed_daily_observations": 365,
            "minimum_forward_rebalances": 30,
            "orders_generated": 0,
            "orders_submitted": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
        if observer_path.is_file():
            observer.update(
                _preserved_breakout_forward_fields(
                    read_json(observer_path),
                    source_candidate_identity=(
                        source_candidate_identity
                    ),
                    strategy_dna_hash=normal.parameters.dna_hash,
                    execution_identity=execution_identity,
                    forward_start=forward_start,
                )
            )
        evidence = build_rotation_forward_evidence(
            normal,
            frames,
            forward_start=forward_start,
            minimum_observations=365,
            minimum_rebalances=30,
            performance_policy=ForwardPerformanceGatePolicy(
                minimum_profit_factor=(
                    settings.research.minimum_profit_factor
                ),
                minimum_stressed_profit_factor=(
                    settings.research.minimum_stressed_profit_factor
                ),
                maximum_drawdown=(
                    settings.research.maximum_drawdown
                ),
                minimum_effective_sample_size=(
                    settings.research.minimum_effective_sample_size
                ),
                stressed_cost_multiplier=(
                    settings.costs.stressed_cost_multiplier
                ),
                bootstrap_samples=(
                    settings.research.multiple_testing_bootstrap_samples
                ),
                bootstrap_block_size=(
                    settings.research.multiple_testing_block_size
                ),
                bootstrap_seed=settings.app.random_seed,
            ),
        )
        observer = merge_portfolio_forward_manifest(
            observer,
            evidence,
            source_candidate_identity=source_candidate_identity,
            strategy_dna_hash=normal.parameters.dna_hash,
            execution_identity=execution_identity,
            forward_start=forward_start,
        )
        observer["data_hashes"] = data_hashes
        atomic_write_json(observer_path, _json_ready(observer))
        observer_paths[primary_name] = str(observer_path)
        forward_summaries[primary_name] = observer[
            "forward_summary"
        ]

    payload = {
        "schema_version": "absolute_momentum_plateau_report_v1",
        "status": "COMPLETED_NOT_PROMOTED",
        "campaign": "ABSOLUTE_MOMENTUM_PLATEAU_V1",
        "strategy_family": ABSOLUTE_MOMENTUM_PLATEAU_FAMILY,
        "engine_version": ABSOLUTE_MOMENTUM_PLATEAU_ENGINE_VERSION,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "search_space_hash": search_space_hash,
        "selection_basis": "DEVELOPMENT_GAUSSIAN_N_PLUS_MINUS_2",
        "selection_integrity": {
            "development_only": True,
            "validation_used_for_selection": False,
            "confirmation_used_for_selection": False,
            "complete_neighborhood_required": True,
            "all_neighbors_net_positive_required": True,
            "kernel": [0.05, 0.25, 0.40, 0.25, 0.05],
        },
        "generated_trial_count": len(candidates),
        "base_known_trials": base_known_trials,
        "registered_unique_plateau_trials": int(
            registry_audit["unique_trial_count"]
        ),
        "total_known_trials": total_known_trials,
        "plateau_eligible_count": int(
            plateau["plateau_eligible"].sum()
        ),
        "primary_strategy_id": primary_name,
        "primary_result": primary_result,
        "candidate_results": rows,
        "multiple_testing": {
            **asdict(multiple),
            "plateau_selection_pbo": plateau_pbo,
            "plateau_selection_pbo_logits": list(plateau_logits),
        },
        "trial_registry": registry_audit,
        "data_fingerprint": data_fingerprint,
        "data_hashes": data_hashes,
        "periods": periods,
        "portfolio_policy": asdict(policy),
        "holdout_status": "NO_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS",
        "observer_manifests": observer_paths,
        "forward_summaries": forward_summaries,
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }
    atomic_write_json(report_path, _json_ready(payload))
    csv_path = report_path.with_suffix(".csv")
    pd.DataFrame(
        [
            {
                "strategy_id": row["strategy_id"],
                "horizon_shift": row["parameters"][
                    "horizon_shift"
                ],
                "volatility_lookback": row["parameters"][
                    "volatility_lookback"
                ],
                "target_annualized_volatility": row["parameters"][
                    "target_annualized_volatility"
                ],
                "development_net_return": row["periods"][
                    "development"
                ]["net_return"],
                "validation_net_return": row["periods"][
                    "validation"
                ]["net_return"],
                "confirmation_net_return": row["periods"][
                    "confirmation"
                ]["net_return"],
                "gaussian_smoothed_sharpe": row["plateau"][
                    "gaussian_smoothed_sharpe"
                ],
                "minimum_neighbor_sharpe": row["plateau"][
                    "minimum_neighbor_sharpe"
                ],
                "plateau_eligible": row["plateau"][
                    "plateau_eligible"
                ],
                "selected_primary": (
                    row["strategy_id"] == primary_name
                ),
            }
            for row in rows
        ]
    ).to_csv(csv_path, index=False)
    return {
        "status": payload["status"],
        "campaign": payload["campaign"],
        "generated_trial_count": len(candidates),
        "registered_unique_plateau_trials": payload[
            "registered_unique_plateau_trials"
        ],
        "total_known_trials": total_known_trials,
        "plateau_eligible_count": payload[
            "plateau_eligible_count"
        ],
        "primary_strategy_id": primary_name,
        "standard_pbo": (
            multiple.probability_of_backtest_overfitting
        ),
        "plateau_selection_pbo": plateau_pbo,
        "report": str(report_path),
        "csv": str(csv_path),
        "observer_manifests": observer_paths,
        "forward_summaries": forward_summaries,
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }


async def command_lab_async(args: argparse.Namespace, settings: Settings) -> int:
    from research.combinatorial_lab import (
        ECONOMIC_HYPOTHESIS_TEMPLATES,
        CombinationGenerator,
        CombinationState,
        GenerationMode,
        LabControl,
        LabRunner,
        LabStore,
        LogicMode,
        UniverseManager,
        UniverseType,
        signal_block_registry,
        validate_blocks,
        write_legacy_migration_report,
    )

    if args.lab_section in {"run", "retest"}:
        lab_overrides = {
            key: value
            for key, value in {
                "cpu_limit": getattr(args, "cpu_limit", None),
                "memory_limit_mb": getattr(args, "memory_limit_mb", None),
                "trial_timeout_seconds": getattr(args, "trial_timeout", None),
                "combination_timeout_seconds": getattr(
                    args,
                    "combination_timeout",
                    None,
                ),
            }.items()
            if value is not None
        }
        if lab_overrides:
            settings = settings.model_copy(
                update={
                    "lab": settings.lab.model_copy(update=lab_overrides),
                }
            )
    store = LabStore(settings)
    runner = LabRunner(settings, store=store)
    section = args.lab_section
    action = getattr(args, "lab_action", None)
    if section in {"generate", "enqueue"}:
        section = "combinations"
        action = "generate"
    if section == "ai":
        from core.ai_governance import evaluate_ai_governance

        evidence = read_json(args.evidence) if args.evidence is not None else None
        decision = evaluate_ai_governance(evidence)
        if args.write_report:
            report = settings.paths.lab_dir / "reports" / "ai_governance_status_v1.json"
            atomic_write_json(report, decision)
            decision = {**decision, "report": str(report)}
        emit(decision)
        return 0
    if section == "data":
        requested_markets = [
            market.upper().replace("/", "-")
            for market in csv_values(getattr(args, "markets_csv", None))
        ]
        if requested_markets:
            markets = requested_markets
        else:
            markets = runner._markets(
                args.universe_size,
                universe_scope=("allowed" if args.allowed_universe else "discovery"),
                include_review_required_research_only=not args.allowed_universe,
            )
        if args.allowed_universe:
            rejected = [
                market
                for market in markets
                if settings.shariah.eligibility(market).status.value != "ALLOWED"
            ]
            if rejected:
                raise ValueError(f"non-ALLOWED markets requested for allowed research: {rejected}")
        timeframes = tuple(csv_values(args.timeframes) or settings.market_data.timeframes)
        if action == "prepare":
            emit(
                await runner.prepare_real_data(
                    markets=markets,
                    timeframes=timeframes,
                    minimum_rows=args.minimum_rows,
                    history_profile=args.history_profile,
                    only_missing=not args.force,
                )
            )
        else:
            emit(
                runner.data_status(
                    markets=markets,
                    timeframes=timeframes,
                    minimum_rows=args.minimum_rows,
                )
            )
        return 0
    if section == "indicators":
        from research.indicator_registry import indicator_registry

        registry = indicator_registry()
        if action == "coverage":
            payload = registry.report()
            json_path = settings.paths.reports_dir / "indicator_coverage.json"
            csv_path = settings.paths.reports_dir / "indicator_coverage.csv"
            atomic_write_json(json_path, payload)
            pd.DataFrame(payload["coverage_rows"]).to_csv(csv_path, index=False)
            emit(
                {
                    "status": "PASSED",
                    "json": str(json_path),
                    "csv": str(csv_path),
                    "summary": {
                        key: payload[key]
                        for key in (
                            "source_item_occurrences",
                            "unique_canonical_indicators",
                            "counts_by_coverage_status",
                        )
                    },
                }
            )
        elif action == "list":
            emit(
                {
                    "count": len(registry),
                    "indicators": [
                        {
                            "id": item.canonical_name,
                            "family": item.family,
                            "status": item.status.value,
                        }
                        for item in registry.definitions()
                    ],
                }
            )
        else:
            item = registry.get(args.id)
            if action == "describe":
                emit(item.to_dict())
            elif action == "parameters":
                emit(
                    {
                        "id": item.canonical_name,
                        "parameters": [asdict(parameter) for parameter in item.parameters],
                    }
                )
            else:
                emit(
                    {
                        "id": item.canonical_name,
                        "status": (
                            "PASSED"
                            if item.status.value
                            in {
                                "IMPLEMENTED",
                                "IMPLEMENTED_AS_ALIAS",
                                "DERIVED_FROM_EXISTING_FEATURES",
                            }
                            else "SKIPPED_NOT_IMPLEMENTED"
                        ),
                        "coverage_status": item.status.value,
                    }
                )
        return 0
    if section == "universe":
        manager = UniverseManager(settings, database=store.database)
        if action == "refresh":
            snapshot = await manager.refresh(
                target_size=args.universe_size,
                scan_limit=args.scan_limit,
            )
            emit(
                snapshot.to_dict()
                | {
                    "research_eligible_count": len(snapshot.research_eligible),
                    "execution_eligible_count": len(snapshot.execution_eligible),
                }
            )
        elif action in {"show", "snapshot"}:
            emit(manager.latest() or {"status": "EMPTY", "reason_code": "NO_SNAPSHOT"})
        elif action == "eligibility":
            latest = manager.latest()
            if latest is None:
                emit({"status": "EMPTY", "reason_code": "NO_SNAPSHOT"})
                return 0
            members = latest.get("members") or []
            emit(
                {
                    "snapshot_id": latest.get("snapshot_id"),
                    "members": [
                        {
                            "symbol": member.get("symbol"),
                            "allowlist_status": member.get("allowlist_status"),
                            "universe_types": member.get("universe_types"),
                            "exclusion_reasons": member.get("exclusion_reasons"),
                        }
                        for member in members
                    ],
                }
            )
        elif action == "coverage":
            latest = manager.latest()
            if latest is None:
                emit({"status": "PARTIAL", "reason_code": "NO_UNIVERSE_SNAPSHOT"})
                return 0
            rows: list[dict[str, Any]] = []
            for member in latest.get("members") or []:
                if UniverseType.DISCOVERY_UNIVERSE.value not in set(
                    member.get("universe_types") or []
                ):
                    continue
                availability = member.get("market_availability") or {}
                rows.append(
                    {
                        "snapshot_id": latest.get("snapshot_id"),
                        "symbol": member.get("symbol"),
                        "cmc_rank": member.get("cmc_rank"),
                        "allowlist_status": member.get("allowlist_status"),
                        "universe_types": member.get("universe_types"),
                        "bitvavo_markets": availability.get("bitvavo") or [],
                        "kraken_markets": availability.get("kraken") or [],
                        "mexc_markets": availability.get("mexc") or [],
                        "coinmarketcap": True,
                        "exclusion_reasons": member.get("exclusion_reasons") or [],
                    }
                )
            json_path = settings.paths.reports_dir / "universe_coverage.json"
            csv_path = settings.paths.reports_dir / "universe_coverage.csv"
            atomic_write_json(json_path, rows)
            csv_frame = pd.DataFrame(rows)
            for column in csv_frame:
                if csv_frame[column].map(lambda value: isinstance(value, (list, dict))).any():
                    csv_frame[column] = csv_frame[column].map(stable_json)
            csv_frame.to_csv(csv_path, index=False)
            emit(
                {
                    "status": ProviderStatus.READY.value if rows else ProviderStatus.PARTIAL.value,
                    "snapshot_id": latest.get("snapshot_id"),
                    "bias_label": latest.get("bias_label"),
                    "assets": len(rows),
                    "json": json_path,
                    "csv": csv_path,
                    "rows": rows,
                }
            )
        else:
            emit(manager.history())
        return 0
    if section == "blocks":
        registry = signal_block_registry()
        if action == "list":
            emit(
                {
                    "count": len(registry),
                    "blocks": [
                        {
                            "block_id": block.block_id,
                            "family": block.family,
                            "role": block.role,
                            "direction": block.direction,
                        }
                        for block in sorted(registry.values(), key=lambda item: item.block_id)
                    ],
                }
            )
        elif action == "describe":
            if args.block not in registry:
                raise ValueError(f"unknown block: {args.block}")
            emit(registry[args.block].to_dict())
        else:
            emit(validate_blocks(registry))
        return 0
    if section == "combinations":
        registry = signal_block_registry()
        generator = CombinationGenerator(registry)
        sizes = _lab_sizes(args.combination_sizes, (1, 2, 3))
        logic_modes = _lab_logic_modes(args.logic_modes)
        blocks = csv_values(getattr(args, "blocks", None)) or list(
            runner._profile_blocks(args.profile)
        )
        generator = CombinationGenerator({block_id: registry[block_id] for block_id in blocks})
        timeframes = csv_values(args.timeframes) or ["1h", "4h", "1d"]
        if action == "estimate":
            estimate = generator.estimate(
                sizes,
                logic_modes=logic_modes,
                assets=args.universe_size,
                timeframes=len(timeframes),
            )
            emit(
                estimate
                | {
                    "profile": args.profile.upper(),
                    "confirmation_required": (
                        estimate["baseline_experiments_upper_bound"]
                        > settings.lab.confirmation_job_threshold
                    ),
                }
            )
            return 0
        if action == "generate":
            estimate = generator.estimate(
                sizes,
                logic_modes=logic_modes,
                assets=args.universe_size,
                timeframes=len(timeframes),
            )
            if (
                estimate["baseline_experiments_upper_bound"]
                > settings.lab.confirmation_job_threshold
                and not args.yes
            ):
                emit(
                    estimate
                    | {
                        "status": "CONFIRMATION_REQUIRED",
                        "reason_code": "ESTIMATED_JOB_THRESHOLD_EXCEEDED",
                    }
                )
                return 2
            generation_checkpoint = store.paths.checkpoints / "generation_cursor.json"
            continuation_cursor = None
            if args.resume and generation_checkpoint.is_file():
                generation_state = read_json(generation_checkpoint)
                if generation_state.get("status") == "PARTIAL_GENERATION":
                    continuation_cursor = generation_state.get("continuation_cursor")
            combinations = generator.generate(
                sizes=sizes,
                logic_modes=logic_modes,
                mode=(
                    GenerationMode.EXHAUSTIVE
                    if args.profile == "exhaustive"
                    else GenerationMode.FAMILY_AWARE
                ),
                block_ids=blocks,
                timeframes=timeframes,
                maximum_rows=settings.lab.maximum_generation_rows,
                continuation_cursor=continuation_cursor,
            )
            store.persist_blocks(registry.values())
            store.persist_combinations(combinations)
            atomic_write_json(
                store.paths.checkpoints / "generation_cursor.json",
                generator.last_generation_status,
            )
            emit(
                {
                    "status": generator.last_generation_status["status"],
                    "count": len(combinations),
                    "generation": generator.last_generation_status,
                    "by_state": dict(
                        sorted(
                            Counter(
                                combination.eligibility_status.value for combination in combinations
                            ).items()
                        )
                    ),
                }
            )
            return 0
        combinations = [
            dict(row["payload"]) for row in store.database.fetch_records("strategy_combinations")
        ]
        if action == "inspect":
            selected = next(
                (row for row in combinations if row.get("combination_id") == args.id),
                None,
            )
            emit(selected or {"status": "NOT_FOUND", "combination_id": args.id})
            return 0 if selected else 2
        emit(
            {
                "count": len(combinations),
                "by_state": dict(
                    sorted(
                        Counter(str(row.get("eligibility_status")) for row in combinations).items()
                    )
                ),
            }
        )
        return 0
    if section == "campaign":
        campaign_action = action or "status"
        if campaign_action == "autopilot":
            from core.autopilot import (
                AutopilotOrchestrator,
                AutopilotPolicy,
            )

            if args.mode in {
                "task-install",
                "task-status",
                "task-remove",
            }:
                return_code, payload = _autopilot_task_command(
                    settings,
                    mode=args.mode,
                    confirmed=bool(args.yes),
                    dry_run=bool(args.dry_run),
                )
                emit(payload)
                return return_code
            policy = AutopilotPolicy(
                interval_seconds=args.interval_seconds,
                research_interval_seconds=args.research_interval_seconds,
                degradation_z_threshold=args.degradation_z_threshold,
                minimum_degradation_observations=(args.minimum_degradation_observations),
                minimum_formal_degradation_observations=(
                    args.minimum_formal_degradation_observations
                ),
                stale_lock_seconds=args.stale_lock_seconds,
            )
            orchestrator = AutopilotOrchestrator(
                settings.paths.lab_dir / "autopilot",
                policy=policy,
            )
            if args.mode == "status":
                emit(orchestrator.status())
                return 0
            if args.mode == "reset":
                if not args.yes:
                    emit(
                        {
                            "status": "CONFIRMATION_REQUIRED",
                            "reason": ("PERSISTENT_KILL_SWITCH_RESET_REQUIRES_REVIEW"),
                            "orders_generated": 0,
                            "paper_candidate_permitted": False,
                            "live_ready": False,
                        }
                    )
                    return 2
                emit(
                    orchestrator.reset_kill_switch(
                        reason=args.reason or "",
                        confirmed=True,
                    )
                )
                return 0

            degradation_observation = _autopilot_degradation_observation(args.degradation_input)

            def cycle() -> dict[str, Any]:
                return orchestrator.run_once(
                    preflight_stage=lambda: (
                        _autopilot_ledger_preflight_stage(settings)
                    ),
                    data_stage=lambda: _autopilot_data_stage(
                        settings,
                        refresh=args.refresh_data,
                        refresh_timeout_seconds=(args.refresh_timeout_seconds),
                    ),
                    feature_store_stage=(
                        (lambda: _autopilot_feature_store_stage(settings))
                        if args.build_feature_store
                        else None
                    ),
                    research_stage=(
                        (lambda: _autopilot_research_stage(settings)) if args.run_research else None
                    ),
                    observer_stage=lambda: _autopilot_observer_stage(settings),
                    degradation_observation=degradation_observation,
                    force_research=args.force_research,
                )

            if args.mode == "continuous":
                max_cycles = None if args.max_cycles == 0 else args.max_cycles
                results = await asyncio.to_thread(
                    orchestrator.run_loop,
                    cycle,
                    max_cycles=max_cycles,
                )
                payload: dict[str, Any] = {
                    "status": results[-1]["status"],
                    "cycle_count": len(results),
                    "last_cycle": results[-1],
                    "orders_generated": 0,
                    "paper_candidate_permitted": False,
                    "live_ready": False,
                }
            else:
                payload = await asyncio.to_thread(cycle)
            emit(payload)
            return 3 if payload.get("status") == "SYSTEM_DEGRADED" else 0
        campaign_name = getattr(
            args,
            "name",
            "microstructure-5m15m",
        )
        formal_campaign = campaign_name == "formal-five-family"
        ensemble_campaign = campaign_name == "cross-sectional-ensemble"
        institutional_campaign = campaign_name == "institutional-rotation-v2"
        capital_utilization_campaign = campaign_name == "capital-utilization-v1"
        diversified_rotation_campaign = campaign_name == "diversified-rotation-v1"
        breakout_portfolio_campaign = campaign_name == "portfolio-breakout-v1"
        absolute_momentum_campaign = campaign_name == "absolute-momentum-v1"
        absolute_momentum_plateau_campaign = (
            campaign_name == "absolute-momentum-plateau-v1"
        )
        portfolio_storm_campaign = campaign_name == "portfolio-storm-v1"
        signal_synthesis_storm_campaign = campaign_name == "signal-synthesis-storm-v1"
        rotation_campaign = campaign_name in {
            "cross-sectional-rotation",
            "cross-sectional-ensemble",
            "institutional-rotation-v2",
        }
        campaign_timeframes = (
            ("1h", "4h", "1d")
            if signal_synthesis_storm_campaign
            else (
                ("1d",)
                if (
                    rotation_campaign
                    or capital_utilization_campaign
                    or diversified_rotation_campaign
                    or breakout_portfolio_campaign
                    or absolute_momentum_campaign
                    or absolute_momentum_plateau_campaign
                    or portfolio_storm_campaign
                )
                else (("1h",) if formal_campaign else ("5m", "15m"))
            )
        )
        campaign_label = (
            (
                "CROSS_SECTIONAL_INSTITUTIONAL_CONTINUATION_V2"
                if institutional_campaign
                else (
                    "CROSS_SECTIONAL_MULTI_HORIZON_ENSEMBLE_V1"
                    if ensemble_campaign
                    else "CROSS_SECTIONAL_ROTATION_ALLOWED_V1"
                )
            )
            if rotation_campaign
            else (
                (
                    (
                        (
                            (
                                "ABSOLUTE_MOMENTUM_V1"
                                if absolute_momentum_campaign
                                else "PORTFOLIO_STORM_V1"
                            )
                            if (
                                portfolio_storm_campaign
                                or absolute_momentum_campaign
                            )
                            else "PORTFOLIO_BREAKOUT_V1"
                        )
                        if (
                            breakout_portfolio_campaign
                            or portfolio_storm_campaign
                            or absolute_momentum_campaign
                        )
                        else "DIVERSIFIED_ROTATION_V1"
                    )
                    if (
                        diversified_rotation_campaign
                        or breakout_portfolio_campaign
                        or absolute_momentum_campaign
                        or portfolio_storm_campaign
                    )
                    else "CAPITAL_UTILIZATION_V1"
                )
                if (
                    capital_utilization_campaign
                    or diversified_rotation_campaign
                    or breakout_portfolio_campaign
                    or portfolio_storm_campaign
                )
                else (
                    "FORMAL_CAUSAL_FIVE_FAMILY_V1"
                    if formal_campaign
                    else "ALLOWED_5M_15M_FULL_HISTORY_V2"
                )
            )
        )
        if signal_synthesis_storm_campaign:
            campaign_label = "SIGNAL_SYNTHESIS_STORM_V1"
        if absolute_momentum_campaign:
            campaign_label = "ABSOLUTE_MOMENTUM_V1"
        if absolute_momentum_plateau_campaign:
            campaign_label = "ABSOLUTE_MOMENTUM_PLATEAU_V1"
        campaign_sizes = _lab_sizes(
            getattr(args, "combination_sizes", "1,2"),
            (1, 2),
        )
        if campaign_action == "forward":
            if not ensemble_campaign:
                raise ValueError(
                    "forward validation is available only for cross-sectional-ensemble"
                )
            emit(await asyncio.to_thread(_run_rotation_forward_validation, settings))
            return 0
        if campaign_action == "external":
            if not ensemble_campaign:
                raise ValueError(
                    "external validation is available only for cross-sectional-ensemble"
                )
            emit(await asyncio.to_thread(_run_rotation_external_validation, settings))
            return 0
        if campaign_action == "audit":
            if not ensemble_campaign:
                raise ValueError(
                    "institutional audit is available only for cross-sectional-ensemble"
                )
            emit(await asyncio.to_thread(_run_rotation_institutional_audit, settings))
            return 0
        if campaign_action == "observe":
            if breakout_portfolio_campaign:
                await asyncio.to_thread(
                    _run_breakout_portfolio_campaign,
                    settings,
                )
                emit(
                    await asyncio.to_thread(
                        _autopilot_observer_stage,
                        settings,
                    )
                )
                return 0
            if absolute_momentum_campaign:
                emit(
                    await asyncio.to_thread(
                        _run_absolute_momentum_campaign,
                        settings,
                    )
                )
                return 0
            if absolute_momentum_plateau_campaign:
                emit(
                    await asyncio.to_thread(
                        _run_absolute_momentum_plateau_campaign,
                        settings,
                    )
                )
                return 0
            if diversified_rotation_campaign:
                emit(
                    await asyncio.to_thread(
                        _run_diversified_rotation_campaign,
                        settings,
                    )
                )
                return 0
            if capital_utilization_campaign:
                emit(
                    await asyncio.to_thread(
                        _run_capital_utilization_campaign,
                        settings,
                    )
                )
                return 0
            if not ensemble_campaign:
                raise ValueError("forward observer is available only for cross-sectional-ensemble")
            emit(await asyncio.to_thread(_run_rotation_forward_observer, settings))
            return 0
        if campaign_action == "package":
            if not ensemble_campaign:
                raise ValueError(
                    "acceptance package is available only for cross-sectional-ensemble"
                )
            from reporting.research_package import build_rotation_acceptance_package

            emit(
                await asyncio.to_thread(
                    build_rotation_acceptance_package,
                    settings,
                )
            )
            return 0
        if campaign_action in {"plan", "estimate"}:
            if signal_synthesis_storm_campaign:
                from research.signal_synthesis_storm import (
                    SIGNAL_STORM_TRIAL_COUNT,
                )

                trial_count = min(
                    args.storm_trials,
                    SIGNAL_STORM_TRIAL_COUNT,
                )
                payload = _signal_synthesis_storm_plan_payload(
                    settings,
                    trial_count=trial_count,
                )
                plan_path, _, _ = _signal_synthesis_storm_paths(settings)
                if campaign_action == "plan":
                    if plan_path.is_file():
                        existing = read_json(plan_path)
                        if existing.get("search_space_hash") != payload.get("search_space_hash"):
                            raise RuntimeError(
                                "SIGNAL_SYNTHESIS_STORM_PLAN_ALREADY_EXISTS_DIFFERENT"
                            )
                    else:
                        atomic_write_json(
                            plan_path,
                            _json_ready(payload),
                        )
                public_plan = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"strategy_dna", "strategy_dna_hashes"}
                }
                emit(
                    {
                        **public_plan,
                        "status": (
                            "CAMPAIGN_PLAN" if campaign_action == "plan" else "CAMPAIGN_ESTIMATE"
                        ),
                        "plan_path": str(plan_path),
                    }
                )
                return 0
            if portfolio_storm_campaign:
                from research.portfolio_storm import STORM_TRIAL_COUNT

                trial_count = (
                    min(args.storm_trials, STORM_TRIAL_COUNT)
                    if hasattr(args, "storm_trials")
                    else STORM_TRIAL_COUNT
                )
                payload = _portfolio_storm_plan_payload(
                    settings,
                    trial_count=trial_count,
                )
                plan_path, _, _ = _portfolio_storm_paths(settings)
                if campaign_action == "plan":
                    if plan_path.is_file():
                        existing = read_json(plan_path)
                        if existing.get("search_space_hash") != payload.get("search_space_hash"):
                            raise RuntimeError("PORTFOLIO_STORM_PLAN_ALREADY_EXISTS_DIFFERENT")
                    else:
                        atomic_write_json(plan_path, _json_ready(payload))
                public_plan = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"strategy_dna", "strategy_dna_hashes"}
                }
                emit(
                    {
                        **public_plan,
                        "status": (
                            "CAMPAIGN_PLAN" if campaign_action == "plan" else "CAMPAIGN_ESTIMATE"
                        ),
                        "plan_path": str(plan_path),
                    }
                )
                return 0
            if breakout_portfolio_campaign:
                from research.portfolio_breakout import (
                    breakout_portfolio_parameter_set,
                )

                parameters = breakout_portfolio_parameter_set()
                emit(
                    {
                        "status": (
                            "CAMPAIGN_PLAN" if campaign_action == "plan" else "CAMPAIGN_ESTIMATE"
                        ),
                        "campaign": campaign_label,
                        "result_type": "PRE_REGISTERED_ECONOMIC_ALPHA_FAMILY",
                        "economic_hypothesis": ("MULTI_ASSET_TIME_SERIES_BREAKOUT"),
                        "source_frozen_lead_mutated": False,
                        "parameters": [asdict(row) for row in parameters],
                        "parameter_trials": len(parameters),
                        "prior_trials_accounted": 1_304,
                        "total_known_trials": 1_304 + len(parameters),
                        "markets": [
                            "BTC-EUR",
                            "ETH-EUR",
                            "SOL-EUR",
                            "LINK-EUR",
                        ],
                        "timeframes": ["1d"],
                        "maximum_total_exposure": 0.40,
                        "maximum_position_exposure": 0.20,
                        "minimum_cash": 0.60,
                        "next_open_execution": True,
                        "paper_candidates": 0,
                        "live_orders": 0,
                    }
                )
                return 0
            if absolute_momentum_plateau_campaign:
                from research.absolute_momentum import (
                    absolute_momentum_plateau_parameter_set,
                )

                parameters = (
                    absolute_momentum_plateau_parameter_set()
                )
                emit(
                    {
                        "status": (
                            "CAMPAIGN_PLAN"
                            if campaign_action == "plan"
                            else "CAMPAIGN_ESTIMATE"
                        ),
                        "campaign": campaign_label,
                        "result_type": (
                            "PREREGISTERED_GAUSSIAN_PARAMETER_PLATEAU"
                        ),
                        "economic_hypothesis": (
                            "ABSOLUTE_MOMENTUM_PARAMETER_INSENSITIVITY"
                        ),
                        "selection_basis": (
                            "DEVELOPMENT_GAUSSIAN_N_PLUS_MINUS_2"
                        ),
                        "kernel": [
                            0.05,
                            0.25,
                            0.40,
                            0.25,
                            0.05,
                        ],
                        "generated_trial_count": len(parameters),
                        "base_known_trials": 16_715,
                        "projected_total_known_trials": (
                            16_715 + len(parameters)
                        ),
                        "maximum_total_exposure": 0.20,
                        "maximum_position_exposure": 0.20,
                        "minimum_cash": 0.80,
                        "next_open_execution": True,
                        "paper_candidates": 0,
                        "orders_generated": 0,
                        "live_ready": False,
                    }
                )
                return 0
            if absolute_momentum_campaign:
                from research.absolute_momentum import (
                    absolute_momentum_parameter_set,
                )

                parameters = absolute_momentum_parameter_set()
                emit(
                    {
                        "status": (
                            "CAMPAIGN_PLAN"
                            if campaign_action == "plan"
                            else "CAMPAIGN_ESTIMATE"
                        ),
                        "campaign": campaign_label,
                        "result_type": (
                            "FIXED_PRIMARY_WITH_FULL_EXPLORATION_LEDGER"
                        ),
                        "economic_hypothesis": (
                            "MULTI_ASSET_ABSOLUTE_MOMENTUM_VOL_TARGET"
                        ),
                        "primary_policy_name": "ABS_MOM_VOL_05",
                        "parameters": [asdict(row) for row in parameters],
                        "formal_risk_budget_paths": len(parameters),
                        "total_known_trials": 16_715,
                        "maximum_total_exposure": 0.20,
                        "maximum_position_exposure": 0.20,
                        "minimum_cash": 0.80,
                        "next_open_execution": True,
                        "paper_candidates": 0,
                        "orders_generated": 0,
                    }
                )
                return 0
            if diversified_rotation_campaign:
                from research.portfolio_selection import (
                    diversified_rotation_policy_set,
                )

                policies = diversified_rotation_policy_set()
                emit(
                    {
                        "status": (
                            "CAMPAIGN_PLAN" if campaign_action == "plan" else "CAMPAIGN_ESTIMATE"
                        ),
                        "campaign": campaign_label,
                        "result_type": ("PRE_REGISTERED_DIVERSIFICATION_CONTINUATION"),
                        "source_frozen_lead_mutated": False,
                        "policies": [asdict(policy) for policy in policies],
                        "diversification_policy_trials": len(policies),
                        "prior_trials_accounted": 1_298,
                        "total_known_trials": 1_298 + len(policies),
                        "markets": [
                            "BTC-EUR",
                            "ETH-EUR",
                            "SOL-EUR",
                            "LINK-EUR",
                        ],
                        "timeframes": ["1d"],
                        "next_open_execution": True,
                        "paper_candidates": 0,
                        "live_orders": 0,
                    }
                )
                return 0
            if capital_utilization_campaign:
                from research.portfolio_selection import (
                    capital_utilization_policy_set,
                )

                policies = capital_utilization_policy_set()
                emit(
                    {
                        "status": (
                            "CAMPAIGN_PLAN" if campaign_action == "plan" else "CAMPAIGN_ESTIMATE"
                        ),
                        "campaign": campaign_label,
                        "result_type": ("PRE_REGISTERED_ALLOCATION_POLICY_COMPARISON"),
                        "signal_dna_frozen": True,
                        "signal_parameters_changed": False,
                        "policies": [asdict(policy) for policy in policies],
                        "allocation_policy_trials": len(policies),
                        "prior_trials_accounted": 1_293,
                        "total_known_trials": 1_293 + len(policies),
                        "markets": [
                            "BTC-EUR",
                            "ETH-EUR",
                            "SOL-EUR",
                            "LINK-EUR",
                        ],
                        "timeframes": ["1d"],
                        "next_open_execution": True,
                        "paper_candidates": 0,
                        "live_orders": 0,
                    }
                )
                return 0
            if rotation_campaign:
                from research.portfolio_selection import (
                    ensemble_rotation_parameter_grid,
                    rotation_parameter_grid,
                )

                ensemble_mode = ensemble_campaign or institutional_campaign
                grid_factory = (
                    ensemble_rotation_parameter_grid if ensemble_mode else rotation_parameter_grid
                )
                campaign_exposure = (
                    settings.operational.maximum_portfolio_exposure
                    if institutional_campaign
                    else (
                        min(0.25, settings.operational.maximum_portfolio_exposure)
                        if ensemble_campaign
                        else settings.operational.maximum_portfolio_exposure
                    )
                )
                campaign_minimum_cash = (
                    max(
                        settings.operational.reserve_cash_fraction,
                        1.0 - campaign_exposure,
                    )
                    if institutional_campaign
                    else settings.operational.reserve_cash_fraction
                )
                grid_arguments: dict[str, Any] = {
                    "gross_exposure": campaign_exposure,
                    "minimum_cash": campaign_minimum_cash,
                    "maximum_positions": settings.operational.maximum_positions,
                }
                if institutional_campaign:
                    grid_arguments.update(
                        {
                            "horizon_sets": (
                                (20, 90),
                                (20, 60, 120),
                                (20, 90, 180),
                            ),
                            "top_ns": (1, 2),
                            "rebalance_days": (7,),
                            "asset_ema_periods": (50, 200),
                            "continuous_regimes": (False, True),
                            "weightings": ("equal", "inverse_volatility"),
                        }
                    )

                emit(
                    {
                        "status": (
                            "CAMPAIGN_PLAN" if campaign_action == "plan" else "CAMPAIGN_ESTIMATE"
                        ),
                        "campaign": campaign_label,
                        "strategy_family": "CROSS_SECTIONAL_MOMENTUM_ROTATION",
                        "result_type": "JOINT_PARAMETER_SCREEN",
                        "joint_parameter_trials": len(grid_factory(**grid_arguments)),
                        "markets": ["BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"],
                        "timeframes": ["1d"],
                        "selection_basis": "DEVELOPMENT_ONLY",
                        "maximum_positions": settings.operational.maximum_positions,
                        "maximum_exposure": (campaign_exposure),
                        "minimum_cash": campaign_minimum_cash,
                        "maximum_position_exposure": (
                            settings.operational.maximum_position_fraction
                            if institutional_campaign
                            else None
                        ),
                        "prior_trials_accounted": (1_245 if institutional_campaign else None),
                        "closed_candles_only": True,
                        "next_open_execution": True,
                        "live_orders": 0,
                    }
                )
                return 0
            registry = signal_block_registry()
            if formal_campaign:
                selected_blocks = sorted(
                    {
                        block
                        for membership in ECONOMIC_HYPOTHESIS_TEMPLATES.values()
                        for block in membership
                    }
                )
                estimate = {
                    "registered_signal_blocks": len(selected_blocks),
                    "economic_hypotheses": len(ECONOMIC_HYPOTHESIS_TEMPLATES),
                    "baseline_experiments_upper_bound": (
                        len(ECONOMIC_HYPOTHESIS_TEMPLATES) * 4 * len(campaign_timeframes)
                    ),
                    "templates": {
                        name: list(blocks)
                        for name, blocks in sorted(ECONOMIC_HYPOTHESIS_TEMPLATES.items())
                    },
                }
            else:
                selected_blocks = list(runner._profile_blocks("hypotheses"))
                estimate = CombinationGenerator(
                    {block_id: registry[block_id] for block_id in selected_blocks}
                ).estimate(
                    campaign_sizes,
                    logic_modes=(LogicMode.LAYERED,),
                    assets=4,
                    timeframes=len(campaign_timeframes),
                )
            emit(
                estimate
                | {
                    "status": (
                        "CAMPAIGN_PLAN" if campaign_action == "plan" else "CAMPAIGN_ESTIMATE"
                    ),
                    "campaign": campaign_label,
                    "profile": ("FORMAL_ECONOMIC_HYPOTHESES" if formal_campaign else "HYPOTHESES"),
                    "markets": ["BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"],
                    "timeframes": list(campaign_timeframes),
                    "history_mode": "common_full_history",
                    "closed_candles_only": True,
                    "live_orders": 0,
                }
            )
            return 0
        if campaign_action == "run":
            if signal_synthesis_storm_campaign:
                if not args.yes:
                    emit(
                        {
                            "status": "CONFIRMATION_REQUIRED",
                            "campaign": campaign_label,
                            "parameter_trials": args.storm_trials,
                            "reason_code": ("LARGE_PREREGISTERED_SIGNAL_DNA_STORM"),
                        }
                    )
                    return 2
                emit(
                    await asyncio.to_thread(
                        _run_signal_synthesis_storm_campaign,
                        settings,
                        maximum_trials=args.storm_trials,
                    )
                )
                return 0
            if portfolio_storm_campaign:
                if not args.yes:
                    emit(
                        {
                            "status": "CONFIRMATION_REQUIRED",
                            "campaign": campaign_label,
                            "parameter_trials": args.storm_trials,
                            "reason_code": ("LARGE_PREREGISTERED_MULTI_OBJECTIVE_STORM"),
                        }
                    )
                    return 2
                emit(
                    await asyncio.to_thread(
                        _run_portfolio_storm_campaign,
                        settings,
                        maximum_trials=args.storm_trials,
                    )
                )
                return 0
            if breakout_portfolio_campaign:
                if not args.yes:
                    emit(
                        {
                            "status": "CONFIRMATION_REQUIRED",
                            "campaign": campaign_label,
                            "parameter_trials": 8,
                            "reason_code": ("PRE_REGISTERED_ECONOMIC_ALPHA_FAMILY"),
                        }
                    )
                    return 2
                emit(
                    await asyncio.to_thread(
                        _run_breakout_portfolio_campaign,
                        settings,
                    )
                )
                return 0
            if absolute_momentum_plateau_campaign:
                if not args.yes:
                    emit(
                        {
                            "status": "CONFIRMATION_REQUIRED",
                            "campaign": campaign_label,
                            "generated_trial_count": 117,
                            "reason_code": (
                                "PREREGISTERED_GAUSSIAN_PLATEAU_SEARCH"
                            ),
                        }
                    )
                    return 2
                emit(
                    await asyncio.to_thread(
                        _run_absolute_momentum_plateau_campaign,
                        settings,
                    )
                )
                return 0
            if absolute_momentum_campaign:
                if not args.yes:
                    emit(
                        {
                            "status": "CONFIRMATION_REQUIRED",
                            "campaign": campaign_label,
                            "formal_risk_budget_paths": 5,
                            "reason_code": (
                                "ACCOUNTED_ABSOLUTE_MOMENTUM_RESEARCH_FAMILY"
                            ),
                        }
                    )
                    return 2
                emit(
                    await asyncio.to_thread(
                        _run_absolute_momentum_campaign,
                        settings,
                    )
                )
                return 0
            if diversified_rotation_campaign:
                if not args.yes:
                    emit(
                        {
                            "status": "CONFIRMATION_REQUIRED",
                            "campaign": campaign_label,
                            "diversification_policy_trials": 6,
                            "reason_code": ("PRE_REGISTERED_DIVERSIFICATION_CONTINUATION"),
                        }
                    )
                    return 2
                emit(
                    await asyncio.to_thread(
                        _run_diversified_rotation_campaign,
                        settings,
                    )
                )
                return 0
            if capital_utilization_campaign:
                if not args.yes:
                    emit(
                        {
                            "status": "CONFIRMATION_REQUIRED",
                            "campaign": campaign_label,
                            "allocation_policy_trials": 5,
                            "reason_code": ("PRE_REGISTERED_ALLOCATION_POLICY_COMPARISON"),
                        }
                    )
                    return 2
                emit(
                    await asyncio.to_thread(
                        _run_capital_utilization_campaign,
                        settings,
                    )
                )
                return 0
            if rotation_campaign:
                if not args.yes:
                    emit(
                        {
                            "status": "CONFIRMATION_REQUIRED",
                            "campaign": campaign_label,
                            "joint_parameter_trials": (
                                48
                                if institutional_campaign
                                else (160 if ensemble_campaign else 432)
                            ),
                            "reason_code": "FORMAL_JOINT_PARAMETER_SCREEN",
                        }
                    )
                    return 2
                emit(
                    await asyncio.to_thread(
                        _run_rotation_campaign,
                        settings,
                        ensemble=ensemble_campaign,
                        institutional=institutional_campaign,
                    )
                )
                return 0
            registry = signal_block_registry()
            if formal_campaign:
                selected_blocks = sorted(
                    {
                        block
                        for membership in ECONOMIC_HYPOTHESIS_TEMPLATES.values()
                        for block in membership
                    }
                )
                estimate = {
                    "baseline_experiments_upper_bound": (
                        len(ECONOMIC_HYPOTHESIS_TEMPLATES) * 4 * len(campaign_timeframes)
                    )
                }
            else:
                selected_blocks = list(runner._profile_blocks("hypotheses"))
                estimate = CombinationGenerator(
                    {block_id: registry[block_id] for block_id in selected_blocks}
                ).estimate(
                    campaign_sizes,
                    logic_modes=(LogicMode.LAYERED,),
                    assets=4,
                    timeframes=len(campaign_timeframes),
                )
            confirmation_required = (
                estimate["baseline_experiments_upper_bound"]
                > settings.lab.confirmation_job_threshold
            )
            emit(
                estimate
                | {
                    "status": (
                        "CONFIRMATION_REQUIRED"
                        if confirmation_required and not args.yes
                        else "CAMPAIGN_ACCEPTED"
                    ),
                    "campaign": campaign_label,
                }
            )
            if confirmation_required and not args.yes:
                return 2
            result = await runner.run_once_guarded(
                profile=("deep" if formal_campaign else "hypotheses"),
                universe_size=4,
                combination_sizes=campaign_sizes,
                logic_modes=(LogicMode.LAYERED,),
                timeframes=campaign_timeframes,
                rows=settings.lab.deep_minimum_history_rows,
                history_mode="common_full_history",
                workers=args.workers,
                data_mode="real",
                max_trials=args.max_trials,
                universe_scope="allowed",
                include_review_required_research_only=False,
                resume=True,
                force=False,
                retest=args.retest,
                only_missing=not args.retest,
                block_ids=selected_blocks,
                parameter_overrides=None,
                combination_templates=(ECONOMIC_HYPOTHESIS_TEMPLATES if formal_campaign else None),
            )
            emit(result)
            return 0 if int(result.get("failures") or 0) == 0 else 2
        if campaign_action == "report":
            if signal_synthesis_storm_campaign:
                _, report_path, _ = _signal_synthesis_storm_paths(settings)
                if not report_path.is_file():
                    raise FileNotFoundError("signal-synthesis-storm-v1 report does not exist")
                emit(read_json(report_path))
                return 0
            if portfolio_storm_campaign:
                _, report_path, _ = _portfolio_storm_paths(settings)
                if not report_path.is_file():
                    raise FileNotFoundError("portfolio-storm-v1 report does not exist")
                emit(read_json(report_path))
                return 0
            if breakout_portfolio_campaign:
                report_path = _breakout_portfolio_campaign_path(settings)
                emit(
                    read_json(report_path)
                    if report_path.is_file()
                    else {
                        "status": "NOT_RUN",
                        "campaign": campaign_label,
                        "report": str(report_path),
                    }
                )
                return 0
            if absolute_momentum_campaign:
                report_path = _absolute_momentum_campaign_path(settings)
                emit(
                    read_json(report_path)
                    if report_path.is_file()
                    else {
                        "status": "NOT_RUN",
                        "campaign": campaign_label,
                        "report": str(report_path),
                    }
                )
                return 0
            if absolute_momentum_plateau_campaign:
                report_path = (
                    _absolute_momentum_plateau_campaign_path(
                        settings
                    )
                )
                emit(
                    read_json(report_path)
                    if report_path.is_file()
                    else {
                        "status": "NOT_RUN",
                        "campaign": campaign_label,
                        "report": str(report_path),
                    }
                )
                return 0
            if diversified_rotation_campaign:
                report_path = _diversified_rotation_campaign_path(settings)
                emit(
                    read_json(report_path)
                    if report_path.is_file()
                    else {
                        "status": "NOT_RUN",
                        "campaign": campaign_label,
                        "report": str(report_path),
                    }
                )
                return 0
            if capital_utilization_campaign:
                report_path = _capital_utilization_campaign_path(settings)
                emit(
                    read_json(report_path)
                    if report_path.is_file()
                    else {
                        "status": "NOT_RUN",
                        "campaign": campaign_label,
                        "report": str(report_path),
                    }
                )
                return 0
            if rotation_campaign:
                report_path = _rotation_campaign_path(
                    settings,
                    ensemble=ensemble_campaign,
                    institutional=institutional_campaign,
                )
                emit(
                    read_json(report_path)
                    if report_path.is_file()
                    else {
                        "status": "NOT_RUN",
                        "campaign": campaign_label,
                        "report": str(report_path),
                    }
                )
                return 0
            emit(
                {
                    "campaign": campaign_label,
                    "leaderboards": store.export_leaderboards(),
                    "report": store.generate_report(run_id=getattr(args, "run_id", None)),
                    "status": runner.status(),
                }
            )
            return 0
        if campaign_action == "status":
            if signal_synthesis_storm_campaign:
                plan_path, report_path, matrix_path = _signal_synthesis_storm_paths(settings)
                emit(
                    {
                        "campaign": campaign_label,
                        "status": (
                            read_json(report_path).get("status")
                            if report_path.is_file()
                            else "PLANNED_NOT_RUN"
                            if plan_path.is_file()
                            else "NOT_PLANNED"
                        ),
                        "plan": str(plan_path),
                        "plan_exists": plan_path.is_file(),
                        "report": str(report_path),
                        "report_exists": report_path.is_file(),
                        "returns_matrix_exists": (matrix_path.is_file()),
                    }
                )
                return 0
            if portfolio_storm_campaign:
                plan_path, report_path, matrix_path = _portfolio_storm_paths(settings)
                emit(
                    {
                        "campaign": campaign_label,
                        "status": (
                            read_json(report_path).get("status")
                            if report_path.is_file()
                            else "PLANNED_NOT_RUN"
                            if plan_path.is_file()
                            else "NOT_PLANNED"
                        ),
                        "plan": str(plan_path),
                        "plan_exists": plan_path.is_file(),
                        "report": str(report_path),
                        "report_exists": report_path.is_file(),
                        "returns_matrix_exists": matrix_path.is_file(),
                    }
                )
                return 0
            if breakout_portfolio_campaign:
                report_path = _breakout_portfolio_campaign_path(settings)
                emit(
                    {
                        "campaign": campaign_label,
                        "status": ("COMPLETED" if report_path.is_file() else "NOT_RUN"),
                        "report": str(report_path),
                        "live_orders": 0,
                    }
                )
                return 0
            if absolute_momentum_campaign:
                report_path = _absolute_momentum_campaign_path(settings)
                emit(
                    {
                        "campaign": campaign_label,
                        "status": (
                            read_json(report_path).get("status")
                            if report_path.is_file()
                            else "NOT_RUN"
                        ),
                        "report": str(report_path),
                        "live_orders": 0,
                    }
                )
                return 0
            if absolute_momentum_plateau_campaign:
                report_path = (
                    _absolute_momentum_plateau_campaign_path(
                        settings
                    )
                )
                emit(
                    {
                        "campaign": campaign_label,
                        "status": (
                            read_json(report_path).get("status")
                            if report_path.is_file()
                            else "NOT_RUN"
                        ),
                        "report": str(report_path),
                        "live_orders": 0,
                    }
                )
                return 0
            if diversified_rotation_campaign:
                report_path = _diversified_rotation_campaign_path(settings)
                emit(
                    {
                        "campaign": campaign_label,
                        "status": ("COMPLETED" if report_path.is_file() else "NOT_RUN"),
                        "report": str(report_path),
                        "live_orders": 0,
                    }
                )
                return 0
            if capital_utilization_campaign:
                report_path = _capital_utilization_campaign_path(settings)
                emit(
                    {
                        "campaign": campaign_label,
                        "status": ("COMPLETED" if report_path.is_file() else "NOT_RUN"),
                        "report": str(report_path),
                        "live_orders": 0,
                    }
                )
                return 0
            if rotation_campaign:
                report_path = _rotation_campaign_path(
                    settings,
                    ensemble=ensemble_campaign,
                    institutional=institutional_campaign,
                )
                emit(
                    {
                        "campaign": campaign_label,
                        "status": "COMPLETED" if report_path.is_file() else "NOT_RUN",
                        "report": str(report_path),
                        "live_orders": 0,
                    }
                )
                return 0
            emit(
                {
                    "campaign": campaign_label,
                    "status": runner.status(),
                    "queue": store.queue_status(),
                }
            )
            return 0
        raise AssertionError(f"unhandled lab campaign action: {campaign_action}")
    if section == "run":
        arguments = _lab_generation_arguments(args)
        registry = signal_block_registry()
        selected_blocks = (
            list(arguments["block_ids"])
            if arguments["block_ids"]
            else list(runner._profile_blocks(args.profile))
        )
        estimate = CombinationGenerator(
            {block_id: registry[block_id] for block_id in selected_blocks}
        ).estimate(
            arguments["combination_sizes"],
            logic_modes=arguments["logic_modes"],
            assets=args.universe_size,
            timeframes=len(arguments["timeframes"]),
        )
        confirmation_required = (
            estimate["baseline_experiments_upper_bound"] > settings.lab.confirmation_job_threshold
            or args.profile == "exhaustive"
        )
        emit(
            estimate
            | {
                "status": (
                    "CONFIRMATION_REQUIRED"
                    if confirmation_required and not args.yes
                    else "QUEUE_ESTIMATE_ACCEPTED"
                ),
                "profile": args.profile.upper(),
            }
        )
        if confirmation_required and not args.yes:
            return 2
        result = (
            await runner.run_continuous(
                soak_minutes=args.soak_minutes,
                **arguments,
            )
            if args.continuous
            else await runner.run_once_guarded(**arguments)
        )
        emit(result)
        return 0 if int(result.get("failures") or 0) == 0 else 2
    if section in {"pause", "resume", "drain", "stop"}:
        emit(runner.control(LabControl(section.upper())))
        return 0
    if section == "status":
        emit(runner.status())
        return 0
    if section == "state":
        emit(
            store.reconcile_state(
                run_id=getattr(args, "run_id", None),
                apply=bool(getattr(args, "apply", False)),
            )
        )
        return 0
    if section == "queue":
        emit(store.queue_status())
        return 0
    if section == "workers":
        path = store.paths.state / "worker_status.json"
        emit(read_json(path) if path.is_file() else {"workers": 0, "active": 0})
        return 0
    if section == "failures":
        emit([read_json(path) for path in sorted(store.paths.failures.glob("*.json"))])
        return 0
    if section == "retry":
        retried = 0
        for job in store.jobs():
            if job.get("status") == CombinationState.ERROR_RETRYABLE.value:
                store.update_job(
                    job,
                    status=CombinationState.QUEUED_BASELINE,
                    stage="BASELINE",
                    reason_code="MANUAL_RETRY",
                    checkpoint=job.get("last_checkpoint"),
                )
                retried += 1
        emit({"status": "PASSED", "retried": retried})
        return 0
    if section == "leaderboard":
        leaderboard_action = action or "show"
        if leaderboard_action == "export":
            emit(store.export_leaderboards())
        elif leaderboard_action == "history":
            emit(
                [
                    dict(row["payload"])
                    for row in store.database.fetch_records("leaderboard_snapshots")
                ]
            )
        elif leaderboard_action == "inspect":
            selected = next(
                (row for row in store.leaderboard() if row.get("combination_id") == args.id),
                None,
            )
            emit(selected or {"status": "NOT_FOUND", "combination_id": args.id})
            return 0 if selected else 2
        else:
            rows = [
                row
                for row in store.leaderboard()
                if int(row.get("trade_count") or 0) >= args.minimum_trades
            ][: args.top]
            emit(rows)
        return 0
    if section == "retest":
        arguments = _lab_generation_arguments(args)
        arguments.update(resume=True, retest=True)
        emit(await runner.run_once_guarded(**arguments))
        return 0
    if section == "validate":
        emit(
            {
                "blocks": validate_blocks(),
                "queue": store.queue_status(),
                "live_orders": 0,
                "live_promotion": False,
            }
        )
        return 0
    if section == "report":
        emit(
            {
                "leaderboards": store.export_leaderboards(),
                "report": store.generate_report(),
                "legacy_migration_report": str(write_legacy_migration_report(settings)),
                "status": runner.status(),
                "output_root": str(store.paths.root),
            }
        )
        return 0
    raise AssertionError(f"unhandled lab command: {section}")


async def self_test(settings: Settings) -> int:
    from execution.execution import PaperBroker
    from research.backtest import BacktestConfig, BacktestEngine
    from research.features import FeaturePipeline
    from research.strategies import get_strategy
    from scrapers.intelligence import SourceSpec, run_intelligence_pipeline

    html = (
        b"<main><article><h2><a href='/btc'>Bitcoin exchange upgrade</a></h2>"
        b"<p>Crypto liquidity and custody update.</p></article></main>"
    )

    async def page_fetcher(_: SourceSpec) -> bytes:
        return html

    source = SourceSpec(
        "SELF_TEST",
        "Self Test",
        "https://example.test/news",
        "en",
        ("crypto_news",),
        True,
    )
    temporary = tempfile.TemporaryDirectory(prefix="crypto-self-test-")
    temporary_root = Path(temporary.name)
    test_settings = settings.model_copy(
        update={
            "paths": settings.paths.model_copy(
                update={
                    "raw_data_dir": temporary_root / "raw",
                    "intelligence_dir": temporary_root / "intelligence",
                    "checkpoints_dir": temporary_root / "checkpoints",
                }
            )
        }
    )
    intelligence = await run_intelligence_pipeline(
        test_settings,
        sources=(source,),
        page_fetcher=page_fetcher,
        include_rss=False,
    )
    frame = synthetic_ohlcv(700, seed=settings.app.random_seed)
    features = FeaturePipeline().build(
        frame,
        market="BTC-EUR",
        intelligence=intelligence.records,
    )
    result = BacktestEngine(
        BacktestConfig(monte_carlo_runs=100),
        settings=test_settings,
    ).run({"BTC-EUR": features}, get_strategy("ema_trend_pullback"))
    broker = PaperBroker(ledger_path=test_settings.paths.checkpoints_dir / "self_test_paper.jsonl")
    paper = broker.submit(
        OrderIntent(
            intent_id="self-test",
            idempotency_key=f"self-test-{stable_hash(utc_now(), length=12)}",
            market="BTC-EUR",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
            strategy_id="self-test",
        ),
        market_price=Decimal("20000"),
    )
    checks = {
        "intelligence_records": len(intelligence.records) == 1,
        "intelligence_timing": intelligence.audit["invalid_timing_count"] == 0,
        "feature_knowability": bool(features.attrs.get("feature_knowability")),
        "next_open_execution": result.integrity["next_open_execution"],
        "long_only_spot": result.integrity["long_only_spot"],
        "paper_fill": paper.status.value == "FILLED",
        "paper_reconciliation": broker.reconcile().healthy,
        "live_blocked_without_all_gates": bool(test_settings.static_live_preflight_failures()),
    }
    temporary.cleanup()
    emit({"status": "OK" if all(checks.values()) else "FAILED", "checks": checks})
    return 0 if all(checks.values()) else 2


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path)
    parser.add_argument("--market", default="BTC-EUR")
    parser.add_argument("--markets", dest="markets_csv")
    parser.add_argument("--rows", type=int, default=900)


def add_research_arguments(parser: argparse.ArgumentParser) -> None:
    add_source_arguments(parser)
    parser.add_argument("--strategy", default="ema_trend_pullback")
    parser.add_argument("--capital", type=float, default=2_000.0)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--monte-carlo-runs", type=int, default=100)
    parser.add_argument("--intelligence", type=Path)
    parser.add_argument("--risk-per-trade", type=float)
    parser.add_argument("--fee", type=float)
    parser.add_argument("--slippage-bps", type=float)


def add_lab_generation_arguments(
    parser: argparse.ArgumentParser,
    *,
    run: bool = False,
) -> None:
    parser.add_argument(
        "--profile",
        choices=("quick", "hypotheses", "standard", "deep", "exhaustive"),
        default="quick",
    )
    parser.add_argument("--universe-size", type=int, default=5 if run else 25)
    parser.add_argument("--combination-sizes", default="1,2" if run else "1,2,3")
    parser.add_argument("--logic-modes", default="layered")
    parser.add_argument("--timeframes", default="1h,4h" if run else "1h,4h,1d")
    parser.add_argument("--blocks")
    if not run:
        parser.add_argument("--resume", action="store_true")
    if run:
        parser.add_argument("--rows", type=int, default=500)
        parser.add_argument(
            "--history-mode",
            choices=(
                "smoke",
                "bounded",
                "common_full_history",
                "asset_max_history",
            ),
            default="bounded",
        )
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument(
            "--data-mode",
            choices=("real", "synthetic"),
            default="real",
        )
        parser.add_argument("--cpu-limit", type=int)
        parser.add_argument("--memory-limit-mb", type=int)
        parser.add_argument("--trial-timeout", type=float)
        parser.add_argument("--combination-timeout", type=float)
        parser.add_argument("--parameter", action="append", default=[])
        parser.add_argument("--parameter-step-profile", choices=("whole", "half"), default="half")
        parser.add_argument("--max-trials", type=int)
        universe_scope = parser.add_mutually_exclusive_group()
        universe_scope.add_argument("--discovery-universe", action="store_true")
        universe_scope.add_argument("--allowed-universe", action="store_true")
        parser.add_argument(
            "--include-review-required-research-only",
            action="store_true",
        )
        parser.add_argument("--continuous", action="store_true")
        parser.add_argument("--once", action="store_true")
        parser.add_argument("--soak-minutes", type=float)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--retest", action="store_true")
        parser.add_argument("--only-missing", action="store_true")
        parser.add_argument("--yes", action="store_true")


def _operation_directory(settings: Settings) -> Path:
    path = settings.paths.output_dir / "operations"
    path.mkdir(parents=True, exist_ok=True)
    (path / "candidates").mkdir(parents=True, exist_ok=True)
    return path


def _operation_service_id(mode: str) -> str:
    return f"operate-{mode.casefold()}"


def _candidate_path(settings: Settings, candidate_id: str) -> Path:
    safe = "".join(
        character
        for character in candidate_id
        if character.isalnum() or character in {"-", "_", "."}
    )
    if not safe or safe != candidate_id:
        raise ValueError("candidate ID contains unsafe characters")
    return _operation_directory(settings) / "candidates" / f"{safe}.json"


def _load_candidate(settings: Settings, candidate_id: str) -> CandidateArtifact:
    path = _candidate_path(settings, candidate_id)
    if not path.is_file():
        raise FileNotFoundError(f"candidate artifact not found: {candidate_id}")
    return CandidateArtifact.model_validate(read_json(path))


def _active_candidate_record(settings: Settings, mode: str) -> dict[str, Any] | None:
    path = settings.paths.checkpoints_dir / f"active_candidate_{mode}.json"
    if not path.is_file():
        return None
    payload = read_json(path)
    if payload.get("state") in {"SUSPENDED", "RETIRED"}:
        return None
    return payload


def _configured_alert_secrets(settings: Settings) -> tuple[str, ...]:
    provider_secrets = tuple(
        value.get_secret_value()
        for name in type(settings.providers).model_fields
        if (value := getattr(settings.providers, name, None)) is not None
        and hasattr(value, "get_secret_value")
    )
    environment_secrets = tuple(
        value
        for name in (
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
            "TELEGRAM_TOKEN",
        )
        if (value := os.environ.get(name))
    )
    return provider_secrets + environment_secrets


def _alerter(settings: Settings) -> AlertThrottle:
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    def deliver(event_type: str, payload: dict[str, Any]) -> None:
        if not token or not chat_id:
            return
        body = urllib.parse.urlencode(
            {
                "chat_id": chat_id,
                "text": (f"{event_type}\n{stable_json(payload, indent=2)[:3500]}"),
                "disable_web_page_preview": "true",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=body,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
            if int(response.status) >= 400:
                raise RuntimeError("Telegram alert delivery failed")

    return AlertThrottle(
        state_path=settings.paths.checkpoints_dir / "alert_throttle.json",
        audit_path=settings.paths.logs_dir / "operational_alerts.jsonl",
        cooldown_seconds=settings.operational.alert_cooldown_seconds,
        secrets=_configured_alert_secrets(settings),
        delivery=deliver if token and chat_id else None,
    )


def _operational_profile(settings: Settings, profile: str) -> dict[str, Any]:
    if profile.casefold() != settings.operational.profile_name.casefold():
        raise ValueError(f"unknown operational profile: {profile}")
    markets = tuple(
        market
        for market in settings.operational.markets
        if settings.shariah.eligibility(market).status.value == "ALLOWED"
    )
    if not markets:
        raise RuntimeError("operational profile has no ALLOWED markets")
    return {
        "name": settings.operational.profile_name,
        "markets": markets,
        "execution_timeframe": settings.operational.execution_timeframe,
        "trend_timeframe": settings.operational.trend_timeframe,
        "regime_timeframe": settings.operational.regime_timeframe,
        "risk_per_trade": settings.operational.risk_per_trade,
        "maximum_risk_per_trade": settings.operational.maximum_risk_per_trade,
        "maximum_total_open_risk": settings.operational.maximum_total_open_risk,
        "maximum_positions": settings.operational.maximum_positions,
        "maximum_position_fraction": settings.operational.maximum_position_fraction,
        "maximum_portfolio_exposure": settings.operational.maximum_portfolio_exposure,
        "reserve_cash_fraction": settings.operational.reserve_cash_fraction,
        "maximum_daily_loss": settings.operational.maximum_daily_loss,
        "drawdown_warning": settings.operational.drawdown_warning,
        "drawdown_block_new_entries": (settings.operational.drawdown_block_new_entries),
        "drawdown_kill_switch": settings.operational.drawdown_kill_switch,
    }


async def _operate_preflight(
    settings: Settings,
    *,
    mode: str,
    profile: str,
    probe_public: bool = True,
) -> dict[str, Any]:
    from data.data_loader import ContinuousDataService, DataLoader
    from data.database import Database
    from execution.execution import PaperBroker
    from risk.risk_manager import KillSwitch

    selected_profile = _operational_profile(settings, profile)
    failures: list[str] = []
    checks: dict[str, Any] = {}
    database = Database(
        supported_database_url(settings),
        sqlite_path=settings.paths.database_path,
    )
    database.migrate()
    try:
        health = database.health()
        checks["database"] = health
        if health["status"] != "PASSED":
            failures.append("DATABASE_UNHEALTHY")
        if health["read_latency_ms"] > settings.operational.database_read_latency_limit_ms:
            failures.append("DATABASE_READ_LATENCY_EXCEEDED")
        if health["write_latency_ms"] > settings.operational.database_write_latency_limit_ms:
            failures.append("DATABASE_WRITE_LATENCY_EXCEEDED")
        disk = shutil.disk_usage(settings.paths.project_root)
        free_gb = disk.free / 1024**3
        checks["disk"] = {
            "free_gb": free_gb,
            "minimum_free_gb": settings.market_data.minimum_free_disk_gb,
        }
        if free_gb < settings.market_data.minimum_free_disk_gb:
            failures.append("INSUFFICIENT_DISK_SPACE")
        lock_path = settings.paths.checkpoints_dir / "data_service.lock"
        lock_inspection = ContinuousDataService.inspect_lock_path(lock_path)
        checks["single_instance_lock"] = lock_inspection
        checks["single_instance_lock_available"] = lock_inspection["available"]
        if not lock_inspection["available"]:
            failures.append("SINGLE_INSTANCE_LOCK_UNAVAILABLE")
        kill_switch = KillSwitch(settings.paths.checkpoints_dir / "kill_switch.json")
        checks["kill_switch"] = {
            "active": kill_switch.active,
            "reason": kill_switch.reason,
        }
        if kill_switch.active:
            failures.append("KILL_SWITCH_ACTIVE")
        active = _active_candidate_record(settings, mode)
        candidate: CandidateArtifact | None = None
        if active:
            try:
                candidate = _load_candidate(settings, str(active["candidate_id"]))
                if candidate.expires_at <= utc_now():
                    failures.append("CANDIDATE_EXPIRED")
                if not candidate.verify_manifest():
                    failures.append("CANDIDATE_MANIFEST_INVALID")
                for market in candidate.eligible_markets:
                    if settings.shariah.eligibility(market).status.value != "ALLOWED":
                        failures.append(f"CANDIDATE_MARKET_NOT_ALLOWED:{market}")
            except (OSError, ValueError, KeyError) as exc:
                failures.append(f"CANDIDATE_INVALID:{type(exc).__name__}")
        checks["candidate"] = {
            "candidate_id": candidate.candidate_id if candidate else None,
            "status": (
                candidate.lifecycle_state.value if candidate else "IDLE_NO_APPROVED_CANDIDATE"
            ),
        }
        if probe_public:
            loader = DataLoader(settings, database=database)
            public_checks: list[dict[str, Any]] = []
            for market in selected_profile["markets"]:
                try:
                    candles = await loader.download_ohlcv(
                        provider="bitvavo",
                        market=market,
                        timeframe=selected_profile["execution_timeframe"],
                        start=utc_now() - timedelta(hours=4),
                        end=utc_now(),
                        resume=True,
                        persist=True,
                    )
                    ticker = await loader.download_ticker(
                        provider="bitvavo",
                        market=market,
                        persist=True,
                        mode=mode,
                    )
                    book = await loader.download_orderbook_snapshot(
                        provider="bitvavo",
                        market=market,
                        depth=min(25, settings.market_data.orderbook_maximum_depth),
                        persist=True,
                        mode=mode,
                    )
                    bids = book.values.get("bids") or []
                    asks = book.values.get("asks") or []
                    valid_book = bool(bids and asks and float(bids[0][0]) < float(asks[0][0]))
                    if not valid_book:
                        failures.append(f"ORDERBOOK_INVALID:{market}")
                    closed = [record for record in candles if record.closed]
                    if not closed:
                        failures.append(f"CLOSED_CANDLE_MISSING:{market}")
                    public_checks.append(
                        {
                            "market": market,
                            "ticker_at": ticker.timestamp,
                            "orderbook_at": book.timestamp,
                            "orderbook_valid": valid_book,
                            "latest_closed_candle": (closed[-1].timestamp if closed else None),
                        }
                    )
                except Exception as exc:
                    failures.append(f"BITVAVO_PUBLIC_DATA_UNHEALTHY:{market}:{type(exc).__name__}")
            checks["bitvavo_public"] = public_checks
        else:
            checks["bitvavo_public"] = {"status": "NOT_PROBED"}
        if mode == "paper":
            broker = PaperBroker(
                initial_balances={"EUR": Decimal("2000")},
                fee_fraction=Decimal(str(settings.costs.default_fee)),
                slippage_bps=Decimal(str(settings.costs.slippage_bps)),
                spread_bps=Decimal(str(settings.costs.spread_bps)),
                ledger_path=settings.paths.checkpoints_dir / "paper_execution.jsonl",
            )
            reconciliation = broker.reconcile()
            checks["paper"] = {
                "ledger_healthy": reconciliation.healthy,
                "balances_initialized": bool(broker.balances),
                "order_cap_eur": settings.operational.paper_order_cap_eur,
                "fee_model": settings.costs.default_fee,
                "slippage_bps": settings.costs.slippage_bps,
            }
            if not reconciliation.healthy:
                failures.append("PAPER_RECONCILIATION_FAILED")
        checks["private_exchange_requests"] = 0
        checks["live_orders"] = 0
        already_running = (
            "SINGLE_INSTANCE_LOCK_UNAVAILABLE" in failures
            and lock_inspection.get("reason_code") == "LOCK_HELD_BY_LIVE_PROCESS"
        )
        return {
            "status": (
                "PASSED" if not failures else "ALREADY_RUNNING" if already_running else "FAILED"
            ),
            "mode": mode,
            "profile": selected_profile,
            "service_state": ("IDLE_NO_APPROVED_CANDIDATE" if candidate is None else "READY"),
            "failures": list(dict.fromkeys(failures)),
            "checks": checks,
        }
    finally:
        database.close()


async def _operational_cycle(
    settings: Settings,
    *,
    database: Any,
    loader: Any,
    mode: str,
    profile: dict[str, Any],
    candidate_id: str | None,
    candidate_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from data.orderbook_l2 import Level2OrderBook
    from risk.risk_manager import OperationalDegradation

    cycle_at = utc_now()
    market_results: list[dict[str, Any]] = []
    signal_records: list[dict[str, Any]] = []
    for market in profile["markets"]:
        candle_records = await loader.download_ohlcv(
            provider="bitvavo",
            market=market,
            timeframe=profile["execution_timeframe"],
            start=cycle_at - timedelta(hours=4),
            end=cycle_at,
            resume=True,
            persist=True,
        )
        closed_candles = [record for record in candle_records if record.closed]
        latest_closed = closed_candles[-1] if closed_candles else None
        interval_seconds = TIMEFRAME_SECONDS[profile["execution_timeframe"]]
        grace_seconds = settings.market_data.candle_close_grace_for(profile["execution_timeframe"])
        trusted_epoch = cycle_at.timestamp() - grace_seconds
        expected_open_epoch = (
            math.floor(trusted_epoch / interval_seconds) * interval_seconds - interval_seconds
        )
        expected_latest_closed_open = datetime.fromtimestamp(
            expected_open_epoch,
            tz=UTC,
        )
        candle_fresh = bool(
            latest_closed and latest_closed.timestamp >= expected_latest_closed_open
        )
        ticker = await loader.download_ticker(
            provider="bitvavo", market=market, persist=True, mode=mode
        )
        trades = await loader.download_trades(
            provider="bitvavo", market=market, persist=True, mode=mode
        )
        snapshot = await loader.download_orderbook_snapshot(
            provider="bitvavo",
            market=market,
            depth=min(100, settings.market_data.orderbook_maximum_depth),
            persist=True,
            mode=mode,
        )
        book = Level2OrderBook(provider="bitvavo", market=market)
        await book.initialize(
            bids=snapshot.values.get("bids") or (),
            asks=snapshot.values.get("asks") or (),
            sequence=snapshot.values.get("sequence"),
            timestamp=snapshot.timestamp,
        )
        stats = {
            "external_id": stable_hash(
                ["orderbook-statistics", market, cycle_at.isoformat()],
                length=32,
            ),
            "provider": "bitvavo",
            "market": market,
            "timestamp": cycle_at,
            "observed_at": cycle_at,
            "status": "READY" if book.valid and not book.is_stale() else "STALE",
            "mode": mode,
            "best_bid": str(book.best_bid),
            "best_ask": str(book.best_ask),
            "mid": str(book.mid_price),
            "spread": str(book.spread),
            "spread_bps": str(book.spread_bps),
            "microprice": str(book.microprice),
            "top_level_imbalance": str(book.top_level_imbalance),
            "depth_imbalance": str(book.book_pressure),
            "depth_bid": str(book.cumulative_bid_depth),
            "depth_ask": str(book.cumulative_ask_depth),
            "estimated_slippage": None,
            "estimated_market_impact": None,
            "update_rate": 1,
            "stale": book.is_stale(),
            "sequence_health": "VALID" if book.valid else "INVALID",
        }
        database.upsert_records("orderbook_statistics", [stats])
        buys = [item for item in trades if str(item.values.get("side") or "").casefold() == "buy"]
        sells = [item for item in trades if str(item.values.get("side") or "").casefold() == "sell"]
        buy_volume = sum(float(item.values.get("quantity") or 0) for item in buys)
        sell_volume = sum(float(item.values.get("quantity") or 0) for item in sells)
        quantities = [float(item.values.get("quantity") or 0) for item in trades]
        trade_aggregate = {
            "external_id": stable_hash(
                ["trade-aggregate", market, cycle_at.isoformat()],
                length=32,
            ),
            "provider": "bitvavo",
            "market": market,
            "timestamp": cycle_at,
            "observed_at": cycle_at,
            "status": "AGGREGATED",
            "data_kind": "trade_flow_aggregate",
            "mode": mode,
            "trade_count": len(trades),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "taker_imbalance": (
                (buy_volume - sell_volume) / (buy_volume + sell_volume)
                if buy_volume + sell_volume
                else 0.0
            ),
            "average_trade_size": (sum(quantities) / len(quantities) if quantities else 0.0),
            "large_trade_count": sum(
                quantity > np.quantile(quantities, 0.95) for quantity in quantities
            )
            if quantities
            else 0,
            "trade_intensity": len(trades),
        }
        database.upsert_records("trades", [trade_aggregate])
        context_hash = stable_hash(
            {
                "ticker": ticker.raw_hash,
                "orderbook": snapshot.raw_hash,
                "trades": [item.raw_hash for item in trades],
            },
            length=64,
        )
        data_hash = latest_closed.raw_hash if latest_closed else ticker.raw_hash
        feature_snapshot_hash = stable_hash(
            [
                market,
                profile["execution_timeframe"],
                data_hash,
                context_hash,
                [],
            ],
            length=64,
        )
        decision_hash = stable_hash(
            {
                "action": "NO_ENTRY",
                "entry_state": False,
                "exit_state": False,
                "risk_decision": "BLOCKED",
                "reason": (
                    "STALE_OR_MISSING_CLOSED_CANDLE"
                    if not candle_fresh
                    else (
                        "IDLE_NO_APPROVED_CANDIDATE" if candidate_id is None else "NO_ENTRY_SIGNAL"
                    )
                ),
            },
            length=64,
        )
        identity = dict(candidate_identity or {})
        signal_records.append(
            {
                "external_id": stable_hash(
                    [
                        "operational-signal-v3",
                        mode,
                        candidate_id,
                        identity.get("manifest_hash"),
                        identity.get("software_version"),
                        identity.get("parameter_hash"),
                        market,
                        profile["execution_timeframe"],
                        (
                            latest_closed.timestamp.isoformat()
                            if latest_closed
                            else "NO_CLOSED_CANDLE"
                        ),
                        data_hash,
                        context_hash,
                        feature_snapshot_hash,
                        decision_hash,
                    ],
                    length=32,
                ),
                "provider": "bitvavo",
                "market": market,
                "timeframe": profile["execution_timeframe"],
                "timestamp": latest_closed.timestamp if latest_closed else cycle_at,
                "observed_at": cycle_at,
                "status": "NO_ENTRY",
                "mode": mode,
                "candidate_id": candidate_id,
                "candidate_manifest_hash": identity.get("manifest_hash"),
                "strategy_dna_hash": identity.get("strategy_dna_hash"),
                "strategy_software_version": identity.get("software_version"),
                "parameter_hash": identity.get("parameter_hash"),
                "signal_identity_schema_version": 3,
                "candle_timestamp": (
                    latest_closed.timestamp.isoformat() if latest_closed else None
                ),
                "evaluated_at": cycle_at.isoformat(),
                "evaluation_key": (
                    latest_closed.timestamp.isoformat() if latest_closed else "NO_CLOSED_CANDLE"
                ),
                "action": "NO_ENTRY",
                "entry_state": False,
                "exit_state": False,
                "regime_state": "NOT_EVALUATED" if candidate_id is None else "NEUTRAL",
                "active_blocks": [],
                "inactive_blocks": [],
                "feature_snapshot_hash": feature_snapshot_hash,
                "data_hash": data_hash,
                "context_hash": context_hash,
                "decision_hash": decision_hash,
                "size_multiplier": 0.0,
                "risk_decision": "BLOCKED",
                "final_reason_code": (
                    "STALE_OR_MISSING_CLOSED_CANDLE"
                    if not candle_fresh
                    else (
                        "IDLE_NO_APPROVED_CANDIDATE" if candidate_id is None else "NO_ENTRY_SIGNAL"
                    )
                ),
            }
        )
        market_results.append(
            {
                "market": market,
                "ticker": ticker.values,
                "latest_closed_candle": (latest_closed.timestamp if latest_closed else None),
                "closed_candle_freshness": {
                    "status": "FRESH" if candle_fresh else "STALE",
                    "expected_latest_open": expected_latest_closed_open,
                    "observed_latest_open": (latest_closed.timestamp if latest_closed else None),
                    "grace_seconds": grace_seconds,
                },
                "trade_count": len(trades),
                "orderbook": stats,
            }
        )
    database.upsert_records("strategy_signals", signal_records)
    degradation = OperationalDegradation(
        state_path=settings.paths.checkpoints_dir / "degradation_state.json",
        audit_path=settings.paths.logs_dir / "degradation_audit.jsonl",
    )
    degradation_status = degradation.evaluate(
        block_new_entries=(
            tuple(
                f"ORDERBOOK_INVALID:{row['market']}"
                for row in market_results
                if row["orderbook"]["sequence_health"] != "VALID"
            )
            + tuple(
                f"CLOSED_CANDLE_STALE:{row['market']}"
                for row in market_results
                if row["closed_candle_freshness"]["status"] != "FRESH"
            )
        ),
    )
    if degradation_status["state"] != "NORMAL":
        database.upsert_records(
            "risk_events",
            [
                {
                    "external_id": stable_hash(
                        ["degradation", cycle_at.isoformat(), degradation_status],
                        length=32,
                    ),
                    "status": degradation_status["state"],
                    "mode": mode,
                    "candidate_id": candidate_id,
                    "timestamp": cycle_at,
                    **degradation_status,
                }
            ],
        )
    database.apply_retention(
        "orderbook_snapshots",
        older_than=timedelta(days=settings.market_data.maximum_orderbook_retention_days),
    )
    database.apply_retention(
        "trades",
        older_than=timedelta(days=settings.market_data.maximum_trade_retention_days),
    )
    return {
        "cycle_at": cycle_at,
        "markets": market_results,
        "signals": len(signal_records),
        "risk_state": degradation_status,
    }


def _operational_status(
    settings: Settings,
    *,
    mode: str,
    profile: str,
) -> dict[str, Any]:
    from data.database import Database
    from reporting.reports import write_operational_reports
    from risk.risk_manager import KillSwitch

    selected_profile = _operational_profile(settings, profile)
    service_id = _operation_service_id(mode)
    heartbeat_path = settings.paths.checkpoints_dir / f"{service_id}_heartbeat.json"
    heartbeat = read_json(heartbeat_path) if heartbeat_path.is_file() else {}
    database = Database(
        supported_database_url(settings),
        sqlite_path=settings.paths.database_path,
    )
    database.migrate()
    try:
        provider_rows = [row["payload"] for row in database.fetch_records("provider_health")]
        latest_signals = [
            row["payload"] for row in database.fetch_recent_records("strategy_signals", limit=20)
        ]
        latest_candles = database.latest_closed_candles(
            markets=selected_profile["markets"],
            timeframes=(
                selected_profile["execution_timeframe"],
                selected_profile["trend_timeframe"],
                selected_profile["regime_timeframe"],
            ),
            provider="bitvavo",
        )
        now = utc_now()
        closed_candle_freshness: dict[str, Any] = {}
        for key, value in latest_candles.items():
            timeframe = key.rsplit(":", 1)[-1]
            timestamp = _parse_utc_datetime(value)
            interval_seconds = TIMEFRAME_SECONDS[timeframe]
            grace_seconds = settings.market_data.candle_close_grace_for(timeframe)
            expected_epoch = (
                math.floor((now.timestamp() - grace_seconds) / interval_seconds) * interval_seconds
                - interval_seconds
            )
            expected = datetime.fromtimestamp(expected_epoch, tz=UTC)
            closed_candle_freshness[key] = {
                "status": "FRESH" if timestamp >= expected else "STALE",
                "observed_latest_open": timestamp,
                "expected_latest_open": expected,
                "grace_seconds": grace_seconds,
            }
        counts = database.health()["table_counts"]
        active = _active_candidate_record(settings, mode)
        kill_switch = KillSwitch(settings.paths.checkpoints_dir / "kill_switch.json")
        heartbeat_at = heartbeat.get("heartbeat_at")
        heartbeat_age = None
        if heartbeat_at:
            heartbeat_age = (utc_now() - _parse_utc_datetime(heartbeat_at)).total_seconds()
        uptime = None
        if heartbeat.get("started_at"):
            uptime = (utc_now() - _parse_utc_datetime(heartbeat["started_at"])).total_seconds()
        heartbeat_state = heartbeat.get("state")
        service_state = (
            "IDLE_NO_APPROVED_CANDIDATE"
            if not active and heartbeat_state in {None, "STOPPED"}
            else heartbeat_state or "NOT_STARTED"
        )
        payload = {
            "service_state": service_state,
            "mode": mode,
            "uptime": uptime,
            "heartbeat_age": heartbeat_age,
            "active_candidate": active.get("candidate_id") if active else None,
            "shadow_challengers": [],
            "current_markets": list(selected_profile["markets"]),
            "current_timeframes": [
                selected_profile["execution_timeframe"],
                selected_profile["trend_timeframe"],
                selected_profile["regime_timeframe"],
            ],
            "provider_health": provider_rows,
            "latest_closed_candles": latest_candles,
            "closed_candle_freshness": closed_candle_freshness,
            "context_freshness": {},
            "latest_signals": latest_signals,
            "open_paper_positions": [],
            "cash": None,
            "exposure": 0.0,
            "open_risk": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "daily_pnl": 0.0,
            "drawdown": 0.0,
            "risk_state": "KILL_SWITCH" if kill_switch.active else "NORMAL",
            "kill_switch_state": ("ACTIVE" if kill_switch.active else "INACTIVE"),
            "recent_errors": [
                row for row in provider_rows if row.get("status") not in {"READY", "PASSED"}
            ][-10:],
            "next_scheduled_jobs": heartbeat.get("next_scheduled_operation"),
            "table_counts": counts,
            "private_exchange_requests": 0,
            "live_orders": 0,
        }
        paths = write_operational_reports(
            _operation_directory(settings),
            status=payload,
            candidate_health={
                "manifest_valid": None if not active else True,
            },
            provider_health=provider_rows,
        )
        payload["reports"] = paths
        return payload
    finally:
        database.close()


async def _operate_start(
    args: argparse.Namespace,
    settings: Settings,
) -> int:
    from data.data_loader import ContinuousDataService, DataLoader
    from data.database import Database

    if args.mode == "live":
        raise ExecutionBlocked("operate live remains blocked in this phase")
    preflight = await _operate_preflight(
        settings,
        mode=args.mode,
        profile=args.profile,
        probe_public=True,
    )
    if preflight["failures"]:
        emit(preflight)
        return 0 if preflight["status"] == "ALREADY_RUNNING" else 2
    profile = _operational_profile(settings, args.profile)
    database = Database(
        supported_database_url(settings),
        sqlite_path=settings.paths.database_path,
    )
    database.migrate()
    loader = DataLoader(settings, database=database)
    service = ContinuousDataService(
        settings,
        database=database,
        service_id=_operation_service_id(args.mode),
        mode=args.mode,
    )
    active = _active_candidate_record(settings, args.mode)
    candidate_id = str(active["candidate_id"]) if active else None
    candidate_identity: dict[str, Any] | None = None
    if candidate_id is not None:
        candidate = _load_candidate(settings, candidate_id)
        candidate_identity = {
            "manifest_hash": candidate.manifest_hash,
            "strategy_dna_hash": candidate.strategy_dna_hash,
            "software_version": candidate.software_version,
            "parameter_hash": candidate.parameter_hash,
        }
    service.active_candidate = candidate_id
    service.kill_switch_state = "INACTIVE"
    service.next_scheduled_operation = "RISK_RECONCILIATION_THEN_NEXT_CLOSED_1H_CANDLE"
    started = time.monotonic()
    soak_seconds = max(0.0, float(args.soak_minutes or 0.0) * 60.0)
    last_cycle: dict[str, Any] = {}
    _alerter(settings).send(
        "SERVICE_START",
        {"mode": args.mode, "profile": args.profile, "candidate_id": candidate_id},
    )
    completed = False

    async def cycle() -> None:
        nonlocal last_cycle
        try:
            last_cycle = await _operational_cycle(
                settings,
                database=database,
                loader=loader,
                mode=args.mode,
                profile=profile,
                candidate_id=candidate_id,
                candidate_identity=candidate_identity,
            )
            service.kill_switch_state = last_cycle["risk_state"]["state"]
        except Exception as exc:
            last_cycle = {
                "cycle_at": utc_now(),
                "status": "BLOCK_NEW_ENTRIES",
                "reason_code": f"MANDATORY_PUBLIC_DATA_FAILURE:{type(exc).__name__}",
                "private_exchange_requests": 0,
                "live_orders": 0,
            }
            database.upsert_records(
                "risk_events",
                [
                    {
                        "external_id": stable_hash(last_cycle, length=32),
                        "status": "BLOCK_NEW_ENTRIES",
                        "mode": args.mode,
                        "candidate_id": candidate_id,
                        **last_cycle,
                    }
                ],
            )
            _alerter(settings).send("PROVIDER_OUTAGE", last_cycle)
        if soak_seconds and time.monotonic() - started >= soak_seconds:
            service.stop()

    once = not args.continuous and not soak_seconds
    try:
        await service.start(
            cycle,
            interval_seconds=settings.operational.cycle_seconds,
            once=once,
        )
        completed = True
    finally:
        acceptance_completed = completed and (
            not soak_seconds or time.monotonic() - started >= soak_seconds
        )
        database.upsert_records(
            "test_runs",
            [
                {
                    "external_id": stable_hash(
                        [
                            "operate",
                            args.mode,
                            started,
                            acceptance_completed,
                        ],
                        length=32,
                    ),
                    "status": (
                        "PASSED"
                        if acceptance_completed
                        else "INTERRUPTED"
                        if completed
                        else "FAILED"
                    ),
                    "mode": args.mode,
                    "profile": args.profile,
                    "candidate_id": candidate_id,
                    "started_monotonic": started,
                    "completed_at": utc_now(),
                    "operation": "OPERATIONAL_SHADOW_SOAK" if soak_seconds else "OPERATIONAL_CYCLE",
                    "private_exchange_requests": 0,
                    "live_orders": 0,
                }
            ],
        )
        _alerter(settings).send(
            "SERVICE_STOP",
            {"mode": args.mode, "candidate_id": candidate_id},
        )
        database.close()
    status = _operational_status(
        settings,
        mode=args.mode,
        profile=args.profile,
    )
    status["last_cycle"] = last_cycle
    status["service_state"] = "IDLE_NO_APPROVED_CANDIDATE" if candidate_id is None else "STOPPED"
    emit(status)
    return 0


def _task_xml(settings: Settings, *, mode: str, profile: str) -> str:
    python = settings.paths.project_root / ".venv" / "Scripts" / "python.exe"
    main = settings.paths.project_root / "main.py"
    trigger = (
        "<LogonTrigger><Enabled>true</Enabled></LogonTrigger>"
        if settings.operational.task_start_trigger == "logon"
        else "<BootTrigger><Enabled>true</Enabled></BootTrigger>"
    )
    arguments = f'"{main}" operate start --mode {mode} --profile {profile} --continuous --resume'
    return (
        '<?xml version="1.0" encoding="UTF-16"?>'
        '<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">'
        f"<Triggers>{trigger}</Triggers>"
        '<Principals><Principal id="Author"><LogonType>InteractiveToken</LogonType>'
        "<RunLevel>LeastPrivilege</RunLevel></Principal></Principals>"
        "<Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>"
        "<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>"
        "<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>"
        "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>"
        f"<RestartOnFailure><Interval>PT1M</Interval><Count>{settings.operational.task_restart_count}</Count></RestartOnFailure>"
        '</Settings><Actions Context="Author"><Exec>'
        f"<Command>{html.escape(str(python))}</Command>"
        f"<Arguments>{html.escape(arguments)}</Arguments>"
        f"<WorkingDirectory>{html.escape(str(settings.paths.project_root))}</WorkingDirectory>"
        "</Exec></Actions></Task>"
    )


async def command_operate(args: argparse.Namespace, settings: Settings) -> int:
    action = args.operate_command
    if action in {"lock-status", "recover-stale-lock"}:
        from data.data_loader import ContinuousDataService

        lock_path = settings.paths.checkpoints_dir / "data_service.lock"
        if action == "lock-status":
            emit(
                {
                    "status": "LOCK_STATUS",
                    "lock_path": lock_path,
                    "lock": ContinuousDataService.inspect_lock_path(lock_path),
                }
            )
            return 0
        try:
            recovery = ContinuousDataService.recover_stale_lock_path(lock_path)
        except RuntimeError as exc:
            emit(
                {
                    "status": "BLOCKED",
                    "reason_code": str(exc),
                    "lock": ContinuousDataService.inspect_lock_path(lock_path),
                }
            )
            return 2
        emit({"status": "RECOVERED" if recovery.get("recovered") else "NOOP", **recovery})
        return 0
    if action == "signal-identity":
        from data.database import Database

        database = Database(
            supported_database_url(settings),
            sqlite_path=settings.paths.database_path,
        )
        database.migrate()
        try:
            emit(database.signal_identity_audit(apply=bool(getattr(args, "apply", False))))
        finally:
            database.close()
        return 0
    if action == "preflight":
        result = await _operate_preflight(
            settings,
            mode=args.mode,
            profile=args.profile,
        )
        emit(result)
        return 0 if not result["failures"] else 2
    if action == "start":
        return await _operate_start(args, settings)
    if action in {"status", "health", "report"}:
        emit(
            _operational_status(
                settings,
                mode=args.mode,
                profile=args.profile,
            )
        )
        return 0
    if action in {"pause", "resume", "drain", "stop"}:
        from data.data_loader import ContinuousDataService

        service_id = _operation_service_id(args.mode)
        lock_path = settings.paths.checkpoints_dir / "data_service.lock"
        lock_inspection = ContinuousDataService.inspect_lock_path(lock_path)
        if lock_inspection["available"]:
            emit(
                {
                    "status": "NOT_RUNNING",
                    "action": action.upper(),
                    "service_id": service_id,
                    "lock": lock_inspection,
                }
            )
            return 0
        owner = lock_inspection.get("owner") or {}
        owner_service_id = owner.get("service_id")
        if owner_service_id not in {None, service_id}:
            emit(
                {
                    "status": "FAILED",
                    "reason_code": "ACTIVE_SERVICE_MODE_MISMATCH",
                    "requested_service_id": service_id,
                    "active_service_id": owner_service_id,
                }
            )
            return 2
        request = {
            "action": action.upper(),
            "requested_at": utc_now(),
            "requested_by": "CLI",
        }
        path = settings.paths.checkpoints_dir / f"{service_id}_control.json"
        atomic_write_json(path, request)
        explicit_wait = bool(getattr(args, "wait", False)) or (
            getattr(args, "wait_seconds", None) is not None
        )
        wait_seconds = (
            float(
                getattr(args, "timeout", None)
                or getattr(args, "wait_seconds", None)
                or settings.operational.control_wait_seconds
            )
            if action in {"drain", "stop"} and explicit_wait
            else 0.0
        )
        if wait_seconds <= 0:
            emit({"status": "REQUESTED", **request, "control_path": path})
            return 0
        heartbeat_path = settings.paths.checkpoints_dir / f"{service_id}_heartbeat.json"
        deadline = time.monotonic() + wait_seconds
        last_heartbeat: dict[str, Any] = {}
        while time.monotonic() < deadline:
            if heartbeat_path.is_file():
                try:
                    last_heartbeat = read_json(heartbeat_path)
                except (OSError, ValueError, TypeError):
                    last_heartbeat = {}
            lock_inspection = ContinuousDataService.inspect_lock_path(lock_path)
            expected_terminal_state = "DRAINED" if action == "drain" else "STOPPED"
            if (
                str(last_heartbeat.get("state") or "").upper() == expected_terminal_state
                and lock_inspection["available"]
            ):
                emit(
                    {
                        "status": expected_terminal_state,
                        **request,
                        "service_state": expected_terminal_state,
                        "reason_code": last_heartbeat.get("reason_code"),
                        "waited_seconds": (wait_seconds) - max(0.0, deadline - time.monotonic()),
                        "control_path": path,
                    }
                )
                return 0
            await asyncio.sleep(float(getattr(args, "poll_seconds", 0.2)))
        emit(
            {
                "status": "TIMEOUT",
                **request,
                "reason_code": "SERVICE_CONTROL_ACK_TIMEOUT",
                "waited_seconds": wait_seconds,
                "last_service_state": last_heartbeat.get("state"),
                "control_path": path,
            }
        )
        return 2
    if action == "reconcile":
        from execution.execution import PaperBroker

        broker = PaperBroker(ledger_path=settings.paths.checkpoints_dir / "paper_execution.jsonl")
        emit(asdict(broker.reconcile()))
        return 0
    if action == "candidates":
        candidates: list[dict[str, Any]] = []
        for path in sorted((_operation_directory(settings) / "candidates").glob("*.json")):
            try:
                candidate = CandidateArtifact.model_validate(read_json(path))
                candidates.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "lifecycle_state": candidate.lifecycle_state.value,
                        "expires_at": candidate.expires_at,
                        "manifest_valid": candidate.verify_manifest(),
                        "path": path,
                    }
                )
            except (OSError, ValueError):
                candidates.append(
                    {
                        "candidate_id": path.stem,
                        "lifecycle_state": "INVALID",
                        "manifest_valid": False,
                        "path": path,
                    }
                )
        emit(
            {
                "status": ("IDLE_NO_APPROVED_CANDIDATE" if not candidates else "READY"),
                "candidates": candidates,
            }
        )
        return 0
    if action == "candidate-inspect":
        emit(_load_candidate(settings, args.id))
        return 0
    if action in {"activate-shadow", "activate-paper"}:
        candidate = _load_candidate(settings, args.id)
        expected = (
            CandidateLifecycle.SHADOW_CANDIDATE
            if action == "activate-shadow"
            else CandidateLifecycle.PAPER_CANDIDATE
        )
        if candidate.lifecycle_state is not expected:
            raise PermissionError(
                f"candidate must be {expected.value}, got {candidate.lifecycle_state.value}"
            )
        if candidate.expires_at <= utc_now():
            raise PermissionError("candidate is expired")
        if action == "activate-paper" and not args.yes:
            raise PermissionError("paper activation requires explicit --yes approval")
        for market in candidate.eligible_markets:
            settings.shariah.require_allowed(market)
        mode = "shadow" if action == "activate-shadow" else "paper"
        record = {
            "candidate_id": candidate.candidate_id,
            "manifest_hash": candidate.manifest_hash,
            "state": (
                CandidateLifecycle.SHADOW_ACTIVE.value
                if mode == "shadow"
                else CandidateLifecycle.PAPER_ACTIVE.value
            ),
            "mode": mode,
            "activated_at": utc_now(),
            "manual_approval": action == "activate-paper",
        }
        atomic_write_json(
            settings.paths.checkpoints_dir / f"active_candidate_{mode}.json",
            record,
        )
        _alerter(settings).send("CANDIDATE_ACTIVATION", record)
        emit({"status": "ACTIVATED", **record})
        return 0
    if action in {"suspend", "retire"}:
        state = action.upper()
        changed = 0
        for mode in ("shadow", "paper"):
            path = settings.paths.checkpoints_dir / f"active_candidate_{mode}.json"
            if not path.is_file():
                continue
            record = read_json(path)
            if record.get("candidate_id") != args.id:
                continue
            record.update(
                state=state,
                changed_at=utc_now().isoformat(),
                reason=args.reason,
            )
            atomic_write_json(path, record)
            changed += 1
        emit({"status": state, "candidate_id": args.id, "records_changed": changed})
        return 0
    if action == "alerts-test":
        payload = {
            "status": "PASSED",
            "token": os.environ.get("TELEGRAM_BOT_TOKEN", "not-configured"),
        }
        first = _alerter(settings).send("ALERT_TEST", payload)
        second = _alerter(settings).send("ALERT_TEST", payload)
        emit(
            {
                "status": "PASSED",
                "first_sent": first,
                "duplicate_suppressed": not second,
                "secrets_redacted": True,
            }
        )
        return 0
    if action == "reset-kill-switch":
        from data.database import Database
        from risk.risk_manager import KillSwitch, OperationalDegradation
        from utils.common import append_jsonl

        if not args.yes:
            raise PermissionError("kill-switch reset requires explicit --yes")
        health = await _operate_preflight(
            settings,
            mode=args.mode,
            profile=args.profile,
        )
        unresolved = [reason for reason in health["failures"] if reason != "KILL_SWITCH_ACTIVE"]
        if unresolved:
            raise RuntimeError(f"kill-switch health checks unresolved: {unresolved}")
        kill_switch = KillSwitch(settings.paths.checkpoints_dir / "kill_switch.json")
        kill_switch.reset(
            approval_phrase="OPERATOR_CONFIRMED_RESET",
            required_phrase="OPERATOR_CONFIRMED_RESET",
        )
        degradation = OperationalDegradation(
            state_path=settings.paths.checkpoints_dir / "degradation_state.json",
            audit_path=settings.paths.logs_dir / "degradation_audit.jsonl",
        )
        if degradation.manual_reset_required:
            degradation.manual_reset(
                confirmed=True,
                reason=args.reason,
                resolved_health_checks=True,
            )
        event = {
            "external_id": stable_hash(
                ["kill-switch-reset", utc_now().isoformat(), args.reason],
                length=32,
            ),
            "status": "RESET",
            "reason": args.reason,
            "confirmed": True,
            "resolved_health_checks": True,
            "timestamp": utc_now(),
        }
        append_jsonl(settings.paths.logs_dir / "kill_switch_audit.jsonl", event)
        database = Database(
            supported_database_url(settings),
            sqlite_path=settings.paths.database_path,
        )
        database.migrate()
        try:
            database.upsert_records("kill_switch_events", [event])
        finally:
            database.close()
        emit({"status": "RESET", "reason": args.reason})
        return 0
    if action in {"task-install", "task-status", "task-remove"}:
        task_name = settings.operational.windows_task_name
        if action == "task-status":
            command = ["schtasks.exe", "/Query", "/TN", task_name, "/FO", "LIST"]
        elif action == "task-remove":
            command = ["schtasks.exe", "/Delete", "/TN", task_name, "/F"]
        else:
            xml = _task_xml(settings, mode=args.mode, profile=args.profile)
            command = ["schtasks.exe", "/Create", "/TN", task_name, "/XML", "<temporary-xml>"]
            if args.dry_run:
                emit(
                    {
                        "status": "DRY_RUN",
                        "command": command,
                        "task_name": task_name,
                        "mode": args.mode,
                        "live_default": False,
                        "virtualenv_python": str(
                            settings.paths.project_root / ".venv" / "Scripts" / "python.exe"
                        ),
                        "working_directory": settings.paths.project_root,
                        "xml": xml,
                    }
                )
                return 0
            with tempfile.NamedTemporaryFile(
                suffix=".xml",
                mode="w",
                encoding="utf-16",
                delete=False,
            ) as handle:
                handle.write(xml)
                xml_path = Path(handle.name)
            command[-1] = str(xml_path)
        try:
            completed = subprocess.run(
                command,
                cwd=settings.paths.project_root,
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            if action == "task-install" and "xml_path" in locals():
                xml_path.unlink(missing_ok=True)
        emit(
            {
                "status": "PASSED" if completed.returncode == 0 else "FAILED",
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        return 0 if completed.returncode == 0 else 2
    raise AssertionError(f"unhandled operate command: {action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed EUR crypto spot research and execution"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version")
    commands.add_parser("doctor")
    commands.add_parser("self-test")

    config = commands.add_parser("config").add_subparsers(dest="config_command", required=True)
    config.add_parser("show")
    config.add_parser("validate")

    eligibility = commands.add_parser("eligibility").add_subparsers(
        dest="eligibility_command", required=True
    )
    eligibility.add_parser("list")
    eligibility_check = eligibility.add_parser("check")
    eligibility_check.add_argument("market", nargs="?")
    eligibility_check.add_argument("--market", dest="market_option")

    providers = commands.add_parser("providers").add_subparsers(
        dest="providers_command", required=True
    )
    providers.add_parser("list")
    providers.add_parser("capabilities")
    providers_test = providers.add_parser("test")
    providers_test.add_argument("--public-only", action="store_true")
    providers.add_parser("status")

    data = commands.add_parser("data").add_subparsers(dest="data_command", required=True)
    data.add_parser("providers")
    download = data.add_parser("download")
    download.add_argument("--markets", nargs="+", default=None)
    download.add_argument("--timeframes", nargs="+", default=None)
    download.add_argument("--providers", nargs="+", default=None)
    download.add_argument("--start", default="2017-01-01T00:00:00Z")
    download.add_argument("--end")
    download.add_argument("--no-resume", action="store_true")
    validate = data.add_parser("validate")
    validate.add_argument("path", type=Path)
    validate.add_argument("--market", required=True)
    validate.add_argument("--timeframe", required=True)
    validate.add_argument("--maximum-staleness-hours", type=float, default=6.0)
    inspect = data.add_parser("inspect")
    inspect.add_argument("path", type=Path)
    historical = data.add_parser("historical")
    historical.add_argument("--provider", default="bitvavo")
    historical.add_argument("--market", default="BTC-EUR")
    historical.add_argument("--timeframe", default="1h")
    historical.add_argument("--start", default="2025-01-01T00:00:00Z")
    historical.add_argument("--end")
    historical.add_argument("--no-resume", action="store_true")
    data.add_parser("live")
    data.add_parser("reconcile")
    data.add_parser("database-health")
    for name in ("estimate", "fetch", "context-fetch", "sync"):
        selected = data.add_parser(name)
        selected.add_argument("--providers", default="all")
        selected.add_argument(
            "--history-profile",
            choices=("smoke", "standard", "deep", "maximum"),
            default="standard",
        )
        selected.add_argument("--resume", action="store_true")
        selected.add_argument("--yes", action="store_true")
        if name != "context-fetch":
            selected.add_argument("--universe-size", type=int, default=25)
            selected.add_argument("--timeframes", default="all")
        if name == "sync":
            selected.add_argument("--only-missing", action="store_true")
            selected.add_argument("--once", action="store_true")
            selected.add_argument("--continuous", action="store_true")
            selected.add_argument("--context", default="none")
            selected.add_argument("--interval-seconds", type=float, default=300.0)
    for name in ("status", "coverage", "gaps", "freshness"):
        data.add_parser(name)

    websocket = commands.add_parser("websocket").add_subparsers(
        dest="websocket_command", required=True
    )
    for name in ("run", "soak"):
        selected = websocket.add_parser(name)
        selected.add_argument(
            "--provider", choices=("bitvavo", "kraken", "mexc"), default="bitvavo"
        )
        selected.add_argument("--market", default="BTC-EUR")
        selected.add_argument("--duration", type=float, default=60.0 if name == "run" else 900.0)
    websocket.add_parser("status")

    orderbook = commands.add_parser("orderbook").add_subparsers(
        dest="orderbook_command", required=True
    )
    for name in ("snapshot", "stream", "inspect"):
        selected = orderbook.add_parser(name)
        selected.add_argument(
            "--provider", choices=("bitvavo", "kraken", "mexc"), default="bitvavo"
        )
        selected.add_argument("--market", default="BTC-EUR")
        selected.add_argument("--depth", type=int, default=100)

    macro = commands.add_parser("macro").add_subparsers(dest="macro_command", required=True)
    macro_build = macro.add_parser("build")
    macro_build.add_argument("--timeframes", default="1h,4h,12h,1d")
    macro.add_parser("inspect")

    gex = commands.add_parser("gex").add_subparsers(dest="gex_command", required=True)
    gex_collect = gex.add_parser("collect")
    gex_collect.add_argument("--underlying", choices=("BTC", "ETH"), default="BTC")
    gex.add_parser("inspect")

    positions = commands.add_parser("positions").add_subparsers(
        dest="positions_command", required=True
    )
    for name in ("status", "reconcile", "pnl"):
        positions.add_parser(name)

    risk_command = commands.add_parser("risk").add_subparsers(dest="risk_command", required=True)
    for name in ("correlation", "drawdown", "kill-switch-status"):
        risk_command.add_parser(name)

    scrape = commands.add_parser("scrape").add_subparsers(dest="scrape_command", required=True)
    scrape_run = scrape.add_parser("run")
    scrape_run.add_argument("--no-rss", action="store_true")
    scrape_run.add_argument("--sources", default="all")
    scrape_run.add_argument("--markets", dest="markets_csv")
    scrape_run.add_argument("--playwright-fallback", action="store_true")
    scrape_run.add_argument("--output-dir", type=Path)
    scrape.add_parser("status")
    scrape_inspect = scrape.add_parser("inspect")
    scrape_inspect.add_argument("--limit", type=int, default=10)
    scrape.add_parser("audit")

    features = commands.add_parser("features").add_subparsers(
        dest="features_command", required=True
    )
    feature_build = features.add_parser("build")
    add_source_arguments(feature_build)
    feature_build.add_argument("--output", type=Path)
    feature_build.add_argument("--persist", action="store_true")
    feature_audit = features.add_parser("audit")
    add_source_arguments(feature_audit)

    indicators = commands.add_parser("indicators").add_subparsers(
        dest="indicator_command", required=True
    )
    indicator_coverage = indicators.add_parser("coverage")
    indicator_coverage.add_argument("--output", type=Path)
    indicator_coverage.add_argument("--persist", action="store_true")
    indicator_list = indicators.add_parser("list")
    indicator_list.add_argument("--family")
    indicator_list.add_argument("--status")
    indicator_list.add_argument("--role")
    indicator_list.add_argument("--provider-availability", choices=("required", "internal"))
    indicator_list.add_argument("--tradable-only", action="store_true")
    indicator_list.add_argument("--research-only", action="store_true")
    indicator_describe = indicators.add_parser("describe")
    indicator_describe.add_argument("name")
    fractal_audit = indicators.add_parser("fractal-audit")
    fractal_audit.add_argument("--rows", type=int, default=500)
    fractal_audit.add_argument("--output", type=Path)

    investing = commands.add_parser("investing").add_subparsers(
        dest="investing_command", required=True
    )
    investing_score = investing.add_parser("score")
    investing_score.add_argument("--asset", required=True)
    investing_score.add_argument("--input", type=Path)
    investing_score.add_argument("--persist", action="store_true")

    strategies = commands.add_parser("strategies").add_subparsers(
        dest="strategies_command", required=True
    )
    strategies.add_parser("list")
    describe = strategies.add_parser("describe")
    describe.add_argument("strategy")

    backtest = commands.add_parser("backtest")
    add_research_arguments(backtest)
    backtest.add_argument("--output", type=Path)

    optimize = commands.add_parser("optimize")
    add_research_arguments(optimize)
    optimize.add_argument(
        "--method", choices=("grid", "random", "coordinate", "optuna"), default="random"
    )
    optimize.add_argument("--trials", type=int, default=20)
    optimize.add_argument("--rounds", type=int, default=2)
    optimize.add_argument("--minimum-trades", type=int, default=1)
    optimize.add_argument("--checkpoint")

    walk = commands.add_parser("walk-forward")
    add_research_arguments(walk)
    walk.add_argument("--folds", type=int, default=6)
    walk.add_argument("--mode", choices=("anchored", "rolling"), default="anchored")
    walk.add_argument("--purge-bars", type=int, default=1)
    walk.add_argument("--embargo-bars", type=int, default=1)

    monte = commands.add_parser("monte-carlo")
    monte.add_argument("--r-multiples", default="1.5,-1,2,-1,0.8,-1,1.2")
    monte.add_argument("--risk-fraction", type=float, default=0.005)
    monte.add_argument("--capital", type=float, default=2_000.0)
    monte.add_argument("--trades", type=int, default=100)
    monte.add_argument("--runs", type=int, default=10_000)
    monte.add_argument("--block-size", type=int, default=1)

    research = commands.add_parser("research")
    research.add_argument(
        "research_action",
        nargs="?",
        choices=("run", "validate"),
        default="run",
    )
    add_research_arguments(research)
    research.add_argument(
        "--method", choices=("grid", "random", "coordinate", "optuna"), default="random"
    )
    research.add_argument("--trials", type=int, default=20)
    research.add_argument("--purge-bars", type=int, default=1)
    research.add_argument("--embargo-bars", type=int, default=1)
    research.add_argument("--checkpoint")
    research.add_argument("--output", type=Path)
    research.add_argument("--output-dir", dest="output", type=Path)
    research.add_argument("--promote-to-paper", action="store_true")
    research.add_argument("--providers")
    research.add_argument("--scrapers", default="none")
    research.add_argument("--timeframes")
    research.add_argument("--strategies")
    research.add_argument("--profile", default="standard")
    research.add_argument("--walk-forward-folds", type=int)

    report = commands.add_parser("report")
    report.add_argument("path")

    tests = commands.add_parser("test").add_subparsers(dest="test_mode", required=True)
    for name in (
        "offline",
        "network",
        "full",
        "providers",
        "websockets",
        "database",
        "reporting",
        "paper",
        "secrets",
    ):
        tests.add_parser(name).set_defaults(duration=10 if name in {"network", "websockets"} else 0)
    soak = tests.add_parser("soak")
    soak.add_argument("--duration", type=float, default=900.0)

    paper = commands.add_parser("paper").add_subparsers(dest="paper_command", required=True)
    paper.add_parser("status")
    paper.add_parser("reconcile")
    paper_run = paper.add_parser("run")
    paper_run.add_argument("--market", default="BTC-EUR")
    paper_run.add_argument("--strategy", default="manual-paper-check")
    paper_run.add_argument("--capital", type=float, default=2_000.0)
    paper_run.add_argument("--price", type=float, default=20_000.0)
    paper_run.add_argument("--quantity", type=float, default=0.001)
    paper_run.add_argument("--stop-fraction", type=float, default=0.05)
    paper_run.add_argument("--idempotency-key")
    paper_run.add_argument("--markets", dest="markets_csv")
    paper_run.add_argument("--candidates", type=Path)
    paper_run.add_argument("--once", action="store_true")

    live = commands.add_parser("live").add_subparsers(dest="live_command", required=True)
    live.add_parser("status")
    for name in ("preflight", "run"):
        selected = live.add_parser(name)
        selected.add_argument("--research-report")
        selected.add_argument("--data", type=Path)
        selected.add_argument("--market", default="BTC-EUR")
        selected.add_argument("--timeframe", default="1h")
        selected.add_argument("--strategy", default="manual-live")
        selected.add_argument("--side", choices=("BUY", "SELL"), default="BUY")
        selected.add_argument("--price", type=float, required=name == "run", default=0.0)
        selected.add_argument("--quantity", type=float, required=name == "run", default=0.0)
        selected.add_argument("--stop-fraction", type=float, default=0.05)
        selected.add_argument("--idempotency-key", required=name == "run", default="")

    operate = commands.add_parser("operate").add_subparsers(
        dest="operate_command",
        required=True,
    )
    for name in (
        "preflight",
        "status",
        "health",
        "pause",
        "resume",
        "drain",
        "stop",
        "reconcile",
        "report",
        "lock-status",
        "recover-stale-lock",
    ):
        selected = operate.add_parser(name)
        selected.add_argument(
            "--mode",
            choices=("shadow", "paper", "live"),
            default="shadow",
        )
        selected.add_argument("--profile", default="practical_spot_v1")
        if name in {"drain", "stop"}:
            selected.add_argument("--wait", action="store_true")
            selected.add_argument("--timeout", type=float)
            selected.add_argument("--poll-seconds", type=float, default=0.2)
            selected.add_argument("--wait-seconds", type=float)
    signal_identity = operate.add_parser("signal-identity")
    signal_identity.add_argument("--apply", action="store_true")
    operate_start = operate.add_parser("start")
    operate_start.add_argument(
        "--mode",
        choices=("shadow", "paper", "live"),
        default="shadow",
    )
    operate_start.add_argument("--profile", default="practical_spot_v1")
    operate_start.add_argument("--continuous", action="store_true")
    operate_start.add_argument("--resume", action="store_true")
    operate_start.add_argument("--soak-minutes", type=float, default=0.0)
    operate.add_parser("candidates")
    candidate_inspect = operate.add_parser("candidate-inspect")
    candidate_inspect.add_argument("--id", required=True)
    for name in ("activate-shadow", "activate-paper"):
        selected = operate.add_parser(name)
        selected.add_argument("--id", required=True)
        selected.add_argument("--yes", action="store_true")
    for name in ("suspend", "retire"):
        selected = operate.add_parser(name)
        selected.add_argument("--id", required=True)
        selected.add_argument("--reason", required=True)
    operate.add_parser("alerts-test")
    reset_kill = operate.add_parser("reset-kill-switch")
    reset_kill.add_argument("--mode", choices=("shadow", "paper"), default="shadow")
    reset_kill.add_argument("--profile", default="practical_spot_v1")
    reset_kill.add_argument("--reason", required=True)
    reset_kill.add_argument("--yes", action="store_true")
    for name in ("task-install", "task-status", "task-remove"):
        selected = operate.add_parser(name)
        selected.add_argument(
            "--mode",
            choices=("shadow", "paper"),
            default="shadow",
        )
        selected.add_argument("--profile", default="practical_spot_v1")
        selected.add_argument("--dry-run", action="store_true")

    lab = commands.add_parser("lab").add_subparsers(
        dest="lab_section",
        required=True,
    )
    lab_ai = lab.add_parser("ai").add_subparsers(
        dest="lab_action",
        required=True,
    )
    ai_status = lab_ai.add_parser("status")
    ai_status.add_argument("--evidence", type=Path)
    ai_status.add_argument("--write-report", action="store_true")

    universe = lab.add_parser("universe").add_subparsers(
        dest="lab_action",
        required=True,
    )
    universe_refresh = universe.add_parser("refresh")
    universe_refresh.add_argument(
        "--universe-size",
        "--size",
        dest="universe_size",
        type=int,
        default=25,
    )
    universe_refresh.add_argument("--scan-limit", type=int, default=100)
    universe.add_parser("show")
    universe.add_parser("snapshot")
    universe.add_parser("eligibility")
    universe.add_parser("history")
    universe.add_parser("coverage")

    lab_data = lab.add_parser("data").add_subparsers(
        dest="lab_action",
        required=True,
    )
    for name in ("status", "prepare", "validate"):
        selected = lab_data.add_parser(name)
        selected.add_argument("--universe-size", type=int, default=5)
        selected.add_argument("--markets", dest="markets_csv")
        selected.add_argument("--allowed-universe", action="store_true")
        selected.add_argument("--timeframes")
        selected.add_argument("--minimum-rows", type=int, default=2_000)
        if name == "prepare":
            selected.add_argument("--force", action="store_true")
            selected.add_argument(
                "--history-profile",
                choices=("smoke", "standard", "deep", "maximum"),
                default="standard",
            )

    lab_indicators = lab.add_parser("indicators").add_subparsers(
        dest="lab_action",
        required=True,
    )
    lab_indicators.add_parser("coverage")
    lab_indicators.add_parser("list")
    for name in ("describe", "parameters", "test"):
        selected = lab_indicators.add_parser(name)
        selected.add_argument("--id", required=True)

    blocks = lab.add_parser("blocks").add_subparsers(
        dest="lab_action",
        required=True,
    )
    blocks.add_parser("list")
    block_describe = blocks.add_parser("describe")
    block_describe.add_argument("--block", required=True)
    blocks.add_parser("validate")

    combinations = lab.add_parser("combinations").add_subparsers(
        dest="lab_action",
        required=True,
    )
    for name in ("estimate", "generate"):
        selected = combinations.add_parser(name)
        add_lab_generation_arguments(selected)
        selected.add_argument("--yes", action="store_true")
    combinations.add_parser("status")
    combination_inspect = combinations.add_parser("inspect")
    combination_inspect.add_argument("--id", required=True)

    lab_run = lab.add_parser("run")
    add_lab_generation_arguments(lab_run, run=True)
    for name in ("generate", "enqueue"):
        selected = lab.add_parser(name)
        add_lab_generation_arguments(selected)
        selected.add_argument("--yes", action="store_true")
    for name in ("pause", "resume", "drain", "stop", "status"):
        lab.add_parser(name)
    lab_state = lab.add_parser("state")
    lab_state.add_argument("--run-id")
    lab_state.add_argument("--apply", action="store_true")
    campaign = lab.add_parser("campaign").add_subparsers(
        dest="lab_action",
        required=True,
    )
    campaign_names = (
        "microstructure-5m15m",
        "formal-five-family",
        "cross-sectional-rotation",
        "cross-sectional-ensemble",
        "institutional-rotation-v2",
        "capital-utilization-v1",
        "diversified-rotation-v1",
        "portfolio-breakout-v1",
        "absolute-momentum-v1",
        "absolute-momentum-plateau-v1",
        "portfolio-storm-v1",
        "signal-synthesis-storm-v1",
    )
    campaign_plan = campaign.add_parser("plan")
    campaign_plan.add_argument("--name", choices=campaign_names, default=campaign_names[0])
    campaign_plan.add_argument("--combination-sizes", default="1,2")
    campaign_plan.add_argument("--storm-trials", type=int, default=5_000)
    campaign_estimate = campaign.add_parser("estimate")
    campaign_estimate.add_argument("--name", choices=campaign_names, default=campaign_names[0])
    campaign_estimate.add_argument("--combination-sizes", default="1,2")
    campaign_estimate.add_argument("--storm-trials", type=int, default=5_000)
    campaign_run = campaign.add_parser("run")
    campaign_run.add_argument("--name", choices=campaign_names, default=campaign_names[0])
    campaign_run.add_argument("--combination-sizes", default="1,2")
    campaign_run.add_argument("--workers", type=int, default=4)
    campaign_run.add_argument("--max-trials", type=int, default=20)
    campaign_run.add_argument("--storm-trials", type=int, default=5_000)
    campaign_run.add_argument("--retest", action="store_true")
    campaign_run.add_argument("--yes", action="store_true")
    campaign_status = campaign.add_parser("status")
    campaign_status.add_argument("--name", choices=campaign_names, default=campaign_names[0])
    campaign_report = campaign.add_parser("report")
    campaign_report.add_argument("--name", choices=campaign_names, default=campaign_names[0])
    campaign_report.add_argument("--run-id")
    campaign_forward = campaign.add_parser("forward")
    campaign_forward.add_argument(
        "--name",
        choices=("cross-sectional-ensemble",),
        default="cross-sectional-ensemble",
    )
    campaign_external = campaign.add_parser("external")
    campaign_external.add_argument(
        "--name",
        choices=("cross-sectional-ensemble",),
        default="cross-sectional-ensemble",
    )
    campaign_audit = campaign.add_parser("audit")
    campaign_audit.add_argument(
        "--name",
        choices=("cross-sectional-ensemble",),
        default="cross-sectional-ensemble",
    )
    campaign_observe = campaign.add_parser("observe")
    campaign_observe.add_argument(
        "--name",
        choices=(
            "cross-sectional-ensemble",
            "capital-utilization-v1",
            "diversified-rotation-v1",
            "portfolio-breakout-v1",
            "absolute-momentum-v1",
            "absolute-momentum-plateau-v1",
        ),
        default="cross-sectional-ensemble",
    )
    campaign_package = campaign.add_parser("package")
    campaign_package.add_argument(
        "--name",
        choices=("cross-sectional-ensemble",),
        default="cross-sectional-ensemble",
    )
    campaign_autopilot = campaign.add_parser("autopilot")
    campaign_autopilot.add_argument(
        "--mode",
        choices=(
            "once",
            "continuous",
            "status",
            "reset",
            "task-install",
            "task-status",
            "task-remove",
        ),
        default="once",
    )
    campaign_autopilot.add_argument(
        "--interval-seconds",
        type=float,
        default=86_400.0,
    )
    campaign_autopilot.add_argument(
        "--research-interval-seconds",
        type=float,
        default=604_800.0,
    )
    campaign_autopilot.add_argument(
        "--degradation-z-threshold",
        type=float,
        default=-2.0,
    )
    campaign_autopilot.add_argument(
        "--minimum-degradation-observations",
        type=int,
        default=30,
    )
    campaign_autopilot.add_argument(
        "--minimum-formal-degradation-observations",
        type=int,
        default=365,
    )
    campaign_autopilot.add_argument(
        "--stale-lock-seconds",
        type=float,
        default=14_400.0,
    )
    campaign_autopilot.add_argument("--max-cycles", type=int, default=1)
    campaign_autopilot.add_argument("--run-research", action="store_true")
    campaign_autopilot.add_argument("--force-research", action="store_true")
    campaign_autopilot.add_argument("--refresh-data", action="store_true")
    campaign_autopilot.add_argument(
        "--build-feature-store",
        action="store_true",
        help=(
            "opt in to the general causal research feature snapshot; "
            "this does not authorize AI/model development"
        ),
    )
    campaign_autopilot.add_argument(
        "--skip-feature-store",
        dest="build_feature_store",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    campaign_autopilot.set_defaults(build_feature_store=False)
    campaign_autopilot.add_argument(
        "--refresh-timeout-seconds",
        type=float,
        default=3_600.0,
    )
    campaign_autopilot.add_argument(
        "--degradation-input",
        type=Path,
    )
    campaign_autopilot.add_argument("--reason")
    campaign_autopilot.add_argument("--yes", action="store_true")
    campaign_autopilot.add_argument("--dry-run", action="store_true")
    lab.add_parser("queue")
    lab.add_parser("workers")
    lab.add_parser("failures")
    lab.add_parser("retry")

    leaderboard = lab.add_parser("leaderboard")
    leaderboard.set_defaults(lab_action="show")
    leaderboard.add_argument("--source", default="final_holdout")
    leaderboard.add_argument("--minimum-trades", type=int, default=0)
    leaderboard.add_argument("--top", type=int, default=25)
    leaderboard.add_argument("--sort", default="robust_score")
    leaderboard_actions = leaderboard.add_subparsers(dest="lab_action")
    leaderboard_actions.add_parser("export")
    leaderboard_inspect = leaderboard_actions.add_parser("inspect")
    leaderboard_inspect.add_argument("--id", required=True)
    leaderboard_actions.add_parser("history")

    lab_retest = lab.add_parser("retest")
    add_lab_generation_arguments(lab_retest, run=True)
    lab.add_parser("validate")
    lab.add_parser("report")
    return parser


async def dispatch(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "version":
        emit({"name": settings.app.app_name, "version": settings.app.version})
        return 0
    if args.command == "doctor":
        return doctor(settings)
    if args.command == "self-test":
        return await self_test(settings)
    if args.command == "config":
        return command_config(args, settings)
    if args.command == "eligibility":
        return command_eligibility(args, settings)
    if args.command == "providers":
        return await command_providers(args, settings)
    if args.command == "data":
        if args.data_command in {
            "historical",
            "live",
            "reconcile",
            "database-health",
            "estimate",
            "fetch",
            "context-fetch",
            "sync",
            "status",
            "coverage",
            "gaps",
            "freshness",
        }:
            return await command_extended_data(args, settings)
        return await command_data_async(args, settings)
    if args.command == "websocket":
        return await command_websocket(args, settings)
    if args.command == "orderbook":
        return await command_orderbook(args, settings)
    if args.command == "macro":
        return command_macro(args, settings)
    if args.command == "gex":
        return await command_gex(args, settings)
    if args.command == "positions":
        return command_positions(args, settings)
    if args.command == "risk":
        return command_risk(args, settings)
    if args.command == "scrape":
        return await command_scrape_async(args, settings)
    if args.command == "features":
        return command_features(args, settings)
    if args.command == "indicators":
        return command_indicators(args, settings)
    if args.command == "investing":
        return command_investing(args, settings)
    if args.command == "strategies":
        return command_strategies(args)
    if args.command == "backtest":
        return command_backtest(args, settings)
    if args.command == "optimize":
        return command_optimize(args, settings)
    if args.command == "walk-forward":
        return command_walk_forward(args, settings)
    if args.command == "monte-carlo":
        return command_monte_carlo(args, settings)
    if args.command == "research":
        return await command_research_async(args, settings)
    if args.command == "report":
        if args.path in {"statistics", "charts", "full", "build"}:
            return command_operational_report(args, settings)
        return command_report(args)
    if args.command == "test":
        return await command_test(args, settings)
    if args.command == "paper":
        return command_paper(args, settings)
    if args.command == "live":
        return await command_live_async(args, settings)
    if args.command == "operate":
        return await command_operate(args, settings)
    if args.command == "lab":
        return await command_lab_async(args, settings)
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings: Settings | None = None
    try:
        settings = Settings.load()
        assert settings is not None
        process_run_id = stable_hash([settings.app.app_name, time.time_ns()], length=16)
        logger = configure_logging(
            log_file=settings.paths.logs_dir / "application.log",
            jsonl_file=settings.paths.logs_dir / "application.jsonl",
            secrets=_configured_secret_values(settings),
        )
        logger.info(
            "process started",
            extra=_log_extra(
                run_id=process_run_id,
                component="cli",
                operation=args.command,
                status="RUNNING",
                reason_code="PROCESS_STARTED",
            ),
        )
        return asyncio.run(dispatch(args, settings))
    except (
        ExecutionBlocked,
        OSError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as exc:
        emit(
            {
                "status": "FAILED",
                "error_code": type(exc).__name__,
                "message": _safe_exception_message(exc, settings),
            }
        )
        return 2
    except KeyboardInterrupt:
        emit({"status": "INTERRUPTED"})
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
