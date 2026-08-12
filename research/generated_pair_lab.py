"""Continuous generated-strategy evaluation on causal synthetic crypto pairs.

The canonical simple lab remains responsible for native EUR markets.  This
companion consumes the same immutable DNA queue and evaluates pair-compatible
close-derived strategies on synchronized EUR-leg ratios.  It never invents a
native BTC market, cross volume, order book, short leg, or live authority.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from config.settings import TIMEFRAME_SECONDS, Settings, normalize_timeframe
from research.combinatorial_lab import (
    BlockRole,
    CombinatorialStrategy,
    GenerationMode,
    LogicMode,
)
from research.features import FeaturePipeline
from research.relative_pair_15m import (
    RelativeStrategySpec,
    _load_ohlcv,
    _robustness,
    _simulate,
    build_synthetic_cross,
)
from research.simple_strategy_lab import SimpleStrategyResearchFactory
from utils.common import atomic_write_json, atomic_write_text, stable_hash, utc_iso

PAIR_LAB_VERSION = "1.0.0"
DEFAULT_TIMEFRAMES = ("15m", "1h", "4h", "1d", "1W")


@dataclass(frozen=True, slots=True)
class GeneratedPair:
    symbol: str
    base_market: str
    benchmark_market: str


GENERATED_PAIRS = (
    GeneratedPair("ETH/BTC", "ETH-EUR", "BTC-EUR"),
    GeneratedPair("SOL/BTC", "SOL-EUR", "BTC-EUR"),
    GeneratedPair("LINK/BTC", "LINK-EUR", "BTC-EUR"),
    GeneratedPair("ADA/BTC", "ADA-EUR", "BTC-EUR"),
    GeneratedPair("TAO/BTC", "TAO-EUR", "BTC-EUR"),
    GeneratedPair("NPC/BTC", "NPC-EUR", "BTC-EUR"),
    GeneratedPair("BTC/ETH", "BTC-EUR", "ETH-EUR"),
    GeneratedPair("SOL/ETH", "SOL-EUR", "ETH-EUR"),
    GeneratedPair("LINK/ETH", "LINK-EUR", "ETH-EUR"),
    GeneratedPair("ADA/ETH", "ADA-EUR", "ETH-EUR"),
    GeneratedPair("TAO/ETH", "TAO-EUR", "ETH-EUR"),
    GeneratedPair("NPC/ETH", "NPC-EUR", "ETH-EUR"),
)
DEFAULT_PAIR_SYMBOLS = tuple(pair.symbol for pair in GENERATED_PAIRS)

PAIR_SAFE_FAMILIES = frozenset(
    {
        "PRICE_RETURNS",
        "TREND",
        "MOMENTUM",
        "VOLATILITY",
        "STATISTICAL_REGIME",
        "STATISTICS_CYCLES",
    }
)

PAIR_UNSAFE_FEATURE_TOKENS = (
    "open",
    "high",
    "low",
    "volume",
    "vwap",
    "money_flow",
    "mfi",
    "cci",
    "williams",
    "atr",
    "true_range",
    "adx",
    "plus_di",
    "minus_di",
    "dmi",
    "aroon",
    "supertrend",
    "donchian",
    "choppiness",
    "keltner",
    "parkinson",
    "garman",
    "rogers",
    "yang_zhang",
    "fractal",
    "candle",
    "engulf",
    "hammer",
    "doji",
    "marubozu",
    "swing",
    "bos",
    "choch",
    "fvg",
    "liquidity",
    "order",
    "spread",
    "depth",
    "funding",
    "interest",
    "basis",
    "gamma",
    "gex",
    "sentiment",
    "fear",
    "breadth",
    "dominance",
    "intelligence",
    "ichimoku",
    "vortex",
    "median_price",
    "typical_price",
    "close_location",
    "btc_relative",
    "rolling_beta",
    "rolling_correlation",
)


def _pair_lab_dir(settings: Settings) -> Path:
    return settings.paths.output_dir / "research" / "generated_pair_lab"


def _finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _database_path(settings: Settings) -> Path:
    return _pair_lab_dir(settings) / "generated_pair_lab.sqlite3"


def _connect(settings: Settings) -> sqlite3.Connection:
    path = _database_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS dna_batches (
            scope_hash TEXT NOT NULL,
            strategy_dna_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            result_count INTEGER NOT NULL,
            positive_count INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (scope_hash, strategy_dna_hash)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS route_results (
            scope_hash TEXT NOT NULL,
            strategy_dna_hash TEXT NOT NULL,
            pair_symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (scope_hash, strategy_dna_hash, pair_symbol, timeframe)
        )
        """
    )
    connection.commit()
    return connection


def _normal_lab_snapshot(settings: Settings) -> dict[str, Any]:
    state_path = (
        settings.paths.output_dir
        / "research"
        / "simple_strategy_lab"
        / "current_registry_state.json"
    )
    if not state_path.is_file():
        return {"status": "NOT_INITIALIZED"}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    queue_path = Path(str(state["queue_path"]))
    if not queue_path.is_file():
        return {"status": "QUEUE_MISSING", "queue_path": str(queue_path)}
    with sqlite3.connect(queue_path) as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) FROM strategy_queue GROUP BY status"
        ).fetchall()
    status_counts = {str(status): int(count) for status, count in rows}
    service = queue_path.parent / "service_status.json"
    service_payload = (
        json.loads(service.read_text(encoding="utf-8")) if service.is_file() else {}
    )
    return _finite_json({
        "status": str(service_payload.get("status") or "READY"),
        "registry_hash": state.get("registry_hash"),
        "registered_signal_blocks": state.get("registered_signal_blocks"),
        "queue_path": str(queue_path),
        "status_counts": status_counts,
        "total_dna": sum(status_counts.values()),
        "normal_currency_scope": "EUR_SPOT_MARKETS",
        "continuous_service_pid": service_payload.get("pid"),
        "orders_generated": 0,
        "orders_submitted": 0,
    })


def pair_compatibility(
    block_ids: Iterable[str],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    blocks = []
    for block_id in block_ids:
        block = registry.get(str(block_id))
        if block is None:
            reasons.append(f"UNKNOWN_BLOCK:{block_id}")
            continue
        blocks.append(block)
        if str(block.family) not in PAIR_SAFE_FAMILIES:
            reasons.append(f"UNSAFE_FAMILY:{block.family}")
        referenced = {
            str(block.feature),
            str(block.compare_feature or ""),
            *(str(value) for value in block.required_features),
        }
        for feature in referenced:
            lowered = feature.casefold()
            token = next(
                (candidate for candidate in PAIR_UNSAFE_FEATURE_TOKENS if candidate in lowered),
                None,
            )
            if token:
                reasons.append(f"UNSAFE_PAIR_FEATURE:{feature}:{token}")
        if block.role is BlockRole.RISK_OVERLAY:
            reasons.append(f"PAIR_RISK_OVERLAY_UNSUPPORTED:{block.block_id}")
    if blocks and not any(block.role is BlockRole.ENTRY_TRIGGER for block in blocks):
        reasons.append("NO_ENTRY_CAPABLE_BLOCK")
    unique = tuple(dict.fromkeys(reasons))
    return {
        "compatible": not unique,
        "reasons": list(unique),
        "contract": (
            "CLOSE_DERIVED_SYNTHETIC_RATIO"
            if not unique
            else "NORMAL_MARKETS_ONLY"
        ),
    }


def _scope_hash(
    *,
    registry_hash: str,
    pairs: Iterable[GeneratedPair],
    timeframes: Iterable[str],
    maximum_rows: int,
) -> str:
    return stable_hash(
        {
            "engine_version": PAIR_LAB_VERSION,
            "registry_hash": registry_hash,
            "pairs": [pair.symbol for pair in pairs],
            "timeframes": list(timeframes),
            "maximum_rows": maximum_rows,
            "cost_contract": "BOTH_EUR_LEGS_NORMAL_AND_STRESSED",
        },
        length=64,
    )


def _select_pairs(values: Iterable[str] | None) -> tuple[GeneratedPair, ...]:
    requested = {str(value).strip().upper().replace("-", "/") for value in (values or ())}
    selected = tuple(pair for pair in GENERATED_PAIRS if not requested or pair.symbol in requested)
    if not selected:
        raise ValueError(f"no registered generated pairs selected: {sorted(requested)}")
    return selected


def _source_queue_path(factory: SimpleStrategyResearchFactory) -> Path:
    if not factory.queue_path.is_file():
        raise FileNotFoundError(factory.queue_path)
    return factory.queue_path


def _unseen_dna(
    settings: Settings,
    factory: SimpleStrategyResearchFactory,
    *,
    scope_hash: str,
    limit: int,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("pair-lab batch size must be positive")
    with _connect(settings) as pair_db:
        seen = {
            str(row[0])
            for row in pair_db.execute(
                """
                SELECT strategy_dna_hash FROM dna_batches
                WHERE scope_hash = ? AND status IN ('COMPLETE', 'PAIR_INCOMPATIBLE')
                """,
                (scope_hash,),
            ).fetchall()
        }
    selected: list[dict[str, Any]] = []
    last_complexity = 0
    last_hash = ""
    with sqlite3.connect(_source_queue_path(factory)) as source:
        source.row_factory = sqlite3.Row
        while len(selected) < limit:
            rows = source.execute(
                """
                SELECT strategy_dna_hash, payload_json
                FROM strategy_queue
                WHERE status != 'EXCLUDED'
                  AND (complexity > ? OR (complexity = ? AND strategy_dna_hash > ?))
                ORDER BY complexity, strategy_dna_hash
                LIMIT 500
                """,
                (last_complexity, last_complexity, last_hash),
            ).fetchall()
            if not rows:
                break
            last_payload = json.loads(str(rows[-1]["payload_json"]))
            last_complexity = int(last_payload.get("combination_size") or 0)
            last_hash = str(rows[-1]["strategy_dna_hash"])
            for row in rows:
                dna = str(row["strategy_dna_hash"])
                if dna in seen:
                    continue
                payload = json.loads(str(row["payload_json"]))
                payload["strategy_dna_hash"] = dna
                selected.append(payload)
                if len(selected) >= limit:
                    break
    return selected


def _build_pair_features(
    settings: Settings,
    pair: GeneratedPair,
    timeframe: str,
    *,
    maximum_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized = settings.paths.processed_data_dir
    base = _load_ohlcv(normalized / f"{pair.base_market}_{timeframe}.parquet")
    benchmark = _load_ohlcv(
        normalized / f"{pair.benchmark_market}_{timeframe}.parquet"
    )
    cross = build_synthetic_cross(base, benchmark, symbol=pair.symbol)
    inverse = build_synthetic_cross(
        benchmark,
        base,
        symbol="/".join(reversed(pair.symbol.split("/"))),
    )
    if maximum_rows > 0:
        common = cross.index.intersection(inverse.index)[-maximum_rows:]
        cross = cross.reindex(common)
        inverse = inverse.reindex(common)
        base = base.reindex(common)
        benchmark = benchmark.reindex(common)
    pair_market = pair.symbol.replace("/", "-")
    inverse_market = "-".join(reversed(pair.symbol.split("/")))
    for frame, market_id, synthetic_symbol in (
        (cross, pair_market, pair.symbol),
        (inverse, inverse_market, f"INVERSE:{pair.symbol}"),
    ):
        frame.attrs.update(
            market=market_id,
            synthetic_symbol=synthetic_symbol,
            timeframe=timeframe,
            synthetic_pair=True,
            native_market=False,
            volume_eligible=False,
            intrabar_structure_eligible=False,
            data_provenance={
                "base_market": pair.base_market,
                "benchmark_market": pair.benchmark_market,
                "no_forward_fill": True,
            },
        )
    pipeline = FeaturePipeline()
    base_features = pipeline.build(cross, market=pair_market)
    inverse_features = pipeline.build(
        inverse,
        market=inverse_market,
    )
    return base, benchmark, cross, base_features, inverse_features


def _rotation_targets(base_output: Any, inverse_output: Any) -> pd.Series:
    index = base_output.entry.index.intersection(inverse_output.entry.index)
    base_entry = base_output.entry.reindex(index).fillna(False).astype(bool)
    inverse_entry = inverse_output.entry.reindex(index).fillna(False).astype(bool)
    base_fresh = base_entry & ~base_entry.shift(1, fill_value=False)
    inverse_fresh = inverse_entry & ~inverse_entry.shift(1, fill_value=False)
    base_exit = base_output.exit.reindex(index).fillna(False).astype(bool)
    inverse_exit = inverse_output.exit.reindex(index).fillna(False).astype(bool)
    state = "CASH"
    held = 0
    cooldown = 0
    rows: list[str] = []
    for timestamp in index:
        cooldown = max(0, cooldown - 1)
        if state == "CASH":
            held = 0
            if cooldown == 0 and bool(base_fresh.loc[timestamp]) and not bool(
                inverse_fresh.loc[timestamp]
            ):
                state = "BASE"
            elif cooldown == 0 and bool(inverse_fresh.loc[timestamp]) and not bool(
                base_fresh.loc[timestamp]
            ):
                state = "BENCHMARK"
        else:
            held += 1
            exit_now = (
                state == "BASE"
                and (bool(base_exit.loc[timestamp]) or bool(inverse_fresh.loc[timestamp]))
            ) or (
                state == "BENCHMARK"
                and (bool(inverse_exit.loc[timestamp]) or bool(base_fresh.loc[timestamp]))
            )
            if held >= 4 and exit_now:
                state = "CASH"
                held = 0
                cooldown = 4
        rows.append(state)
    return pd.Series(rows, index=index, dtype="string")


def _evaluate_route(
    settings: Settings,
    strategy: CombinatorialStrategy,
    pair: GeneratedPair,
    timeframe: str,
    *,
    maximum_rows: int,
    simulations: int,
) -> dict[str, Any]:
    base, benchmark, cross, features, inverse_features = _build_pair_features(
        settings,
        pair,
        timeframe,
        maximum_rows=maximum_rows,
    )
    base_output = strategy.generate(features)
    inverse_output = strategy.generate(inverse_features)
    targets = _rotation_targets(base_output, inverse_output)
    spec = RelativeStrategySpec(
        strategy_id=strategy.strategy_id,
        mechanism="generated_pair_rotation",
        stop_atr=2.0,
        target_atr=3.0,
        maximum_holding_bars=120,
    )
    periods_per_year = 365.25 * 86_400 / TIMEFRAME_SECONDS[timeframe]
    common = {
        "fee_fraction": settings.costs.default_fee,
        "spread_bps": settings.costs.spread_bps,
        "slippage_bps": settings.costs.slippage_bps,
        "periods_per_year": periods_per_year,
    }
    normal = _simulate(
        base,
        benchmark,
        targets,
        spec,
        cost_multiplier=1.0,
        **common,
    )
    stressed = _simulate(
        base,
        benchmark,
        targets,
        spec,
        cost_multiplier=settings.costs.stressed_cost_multiplier,
        **common,
    )
    episodes = normal.pop("episode_pnl")
    normal.pop("equity_curve")
    stressed.pop("episode_pnl")
    stressed.pop("equity_curve")
    for metrics in (normal, stressed):
        profit_factor = float(metrics["closed_position_profit_factor"])
        metrics["profit_factor_infinite"] = not math.isfinite(profit_factor)
    positive = (
        float(normal["net_total_return"]) > 0
        and float(normal["closed_position_profit_factor"]) > 1.0
        and float(normal["net_expectancy_eur"]) > 0
    )
    robustness = (
        _robustness(
            episodes,
            simulations=simulations,
            seed=settings.app.random_seed,
        )
        if positive
        else {"status": "DEFERRED_UNTIL_ECONOMIC_POSITIVE", "sample": len(episodes)}
    )
    return _finite_json({
        "pair": pair.symbol,
        "base_market": pair.base_market,
        "benchmark_market": pair.benchmark_market,
        "timeframe": timeframe,
        "rows": len(cross),
        "start": cross.index.min().isoformat(),
        "end": cross.index.max().isoformat(),
        "normal": normal,
        "stressed": stressed,
        "robustness": robustness,
        "backtest_positive": positive,
        "stressed_positive": (
            float(stressed["net_total_return"]) > 0
            and float(stressed["closed_position_profit_factor"]) > 1.0
            and float(stressed["net_expectancy_eur"]) > 0
        ),
        "sample_warning": (
            "SMALL_SAMPLE"
            if int(normal["closed_holding_episodes"]) < 8
            else None
        ),
        "recommended_phase": "PAPER_CANDIDATE" if positive else "RESEARCH_ONLY",
        "native_pair": False,
        "live_authority": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    })


def _persist_batch(
    settings: Settings,
    *,
    scope_hash: str,
    dna: str,
    status: str,
    reason: str | None,
    payload: Mapping[str, Any],
    routes: Iterable[Mapping[str, Any]],
) -> None:
    route_rows = list(routes)
    now = utc_iso()
    with _connect(settings) as connection:
        for route in route_rows:
            connection.execute(
                """
                INSERT OR REPLACE INTO route_results (
                    scope_hash, strategy_dna_hash, pair_symbol, timeframe,
                    payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    scope_hash,
                    dna,
                    str(route["pair"]),
                    str(route["timeframe"]),
                    json.dumps(_finite_json(route), sort_keys=True, allow_nan=False),
                    now,
                ),
            )
        connection.execute(
            """
            INSERT OR REPLACE INTO dna_batches (
                scope_hash, strategy_dna_hash, status, reason, result_count,
                positive_count, payload_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scope_hash,
                dna,
                status,
                reason,
                len(route_rows),
                sum(bool(row.get("backtest_positive")) for row in route_rows),
                json.dumps(_finite_json(dict(payload)), sort_keys=True, allow_nan=False),
                now,
            ),
        )
        connection.commit()


def _write_reports(settings: Settings, *, scope_hash: str) -> dict[str, str]:
    output = _pair_lab_dir(settings)
    output.mkdir(parents=True, exist_ok=True)
    with _connect(settings) as connection:
        batch_rows = connection.execute(
            """
            SELECT status, reason, COUNT(*) AS count
            FROM dna_batches WHERE scope_hash = ?
            GROUP BY status, reason ORDER BY status, reason
            """,
            (scope_hash,),
        ).fetchall()
        route_rows = connection.execute(
            "SELECT payload_json FROM route_results WHERE scope_hash = ?",
            (scope_hash,),
        ).fetchall()
    routes = [
        _finite_json(json.loads(str(row["payload_json"]))) for row in route_rows
    ]
    routes.sort(
        key=lambda row: (
            bool(row["backtest_positive"]),
            _number(row["normal"]["sharpe"]),
            _number(row["normal"]["net_total_return"]),
        ),
        reverse=True,
    )
    status_counts = Counter()
    reason_counts = Counter()
    for row in batch_rows:
        status_counts[str(row["status"])] += int(row["count"])
        if row["reason"]:
            reason_counts[str(row["reason"])] += int(row["count"])
    payload = {
        "schema_version": "generated_pair_lab_status_v1",
        "generated_at": utc_iso(),
        "scope_hash": scope_hash,
        "pair_lab_version": PAIR_LAB_VERSION,
        "normal_currency_lab": _normal_lab_snapshot(settings),
        "pair_dna_status_counts": dict(sorted(status_counts.items())),
        "pair_incompatibility_reason_counts": dict(sorted(reason_counts.items())),
        "pair_route_result_count": len(routes),
        "pair_positive_route_count": sum(bool(row["backtest_positive"]) for row in routes),
        "top_pair_results": routes[:100],
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    status_path = output / "status.json"
    leaderboard_json = output / "leaderboard.json"
    leaderboard_csv = output / "leaderboard.csv"
    leaderboard_md = output / "leaderboard.md"
    atomic_write_json(status_path, payload)
    atomic_write_json(leaderboard_json, {"rows": routes})
    columns = (
        "rank",
        "strategy_dna_hash",
        "pair",
        "timeframe",
        "backtest_positive",
        "net_total_return",
        "profit_factor",
        "sharpe",
        "maximum_drawdown",
        "episodes",
        "stressed_profit_factor",
        "recommended_phase",
    )
    with leaderboard_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rank, row in enumerate(routes, start=1):
            normal, stressed = row["normal"], row["stressed"]
            writer.writerow(
                {
                    "rank": rank,
                    "strategy_dna_hash": row["strategy_dna_hash"],
                    "pair": row["pair"],
                    "timeframe": row["timeframe"],
                    "backtest_positive": row["backtest_positive"],
                    "net_total_return": normal["net_total_return"],
                    "profit_factor": normal["closed_position_profit_factor"],
                    "sharpe": normal["sharpe"],
                    "maximum_drawdown": normal["maximum_drawdown"],
                    "episodes": normal["closed_holding_episodes"],
                    "stressed_profit_factor": stressed["closed_position_profit_factor"],
                    "recommended_phase": row["recommended_phase"],
                }
            )
    lines = [
        "# Generated crypto-pair lab",
        "",
        "Synthetic ratios are evaluated through EUR spot legs; no native pair, short, or order is implied.",
        "",
        "| # | DNA | Pair | TF | Return | PF | Sharpe | DD | N | Stress PF | Phase |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(routes[:100], start=1):
        normal, stressed = row["normal"], row["stressed"]
        lines.append(
            f"| {rank} | `{row['strategy_dna_hash'][:12]}` | {row['pair']} | "
            f"{row['timeframe']} | {normal['net_total_return']:.2%} | "
            f"{_number(normal['closed_position_profit_factor']):.3f} | "
            f"{_number(normal['sharpe']):.2f} | "
            f"{_number(normal['maximum_drawdown']):.2%} | "
            f"{normal['closed_holding_episodes']} | "
            f"{_number(stressed['closed_position_profit_factor']):.3f} | "
            f"{row['recommended_phase']} |"
        )
    atomic_write_text(leaderboard_md, "\n".join(lines) + "\n")
    return {
        "status": str(status_path.resolve()),
        "leaderboard_json": str(leaderboard_json.resolve()),
        "leaderboard_csv": str(leaderboard_csv.resolve()),
        "leaderboard_markdown": str(leaderboard_md.resolve()),
        "database": str(_database_path(settings).resolve()),
    }


def run_generated_pair_batch(
    settings: Settings,
    *,
    pairs: Iterable[str] | None = None,
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
    batch_size: int = 2,
    maximum_rows: int = 2_000,
    simulations: int = 500,
) -> dict[str, Any]:
    selected_pairs = _select_pairs(pairs)
    selected_timeframes = tuple(
        dict.fromkeys(normalize_timeframe(value) for value in timeframes)
    )
    if not selected_timeframes:
        raise ValueError("pair-lab requires at least one timeframe")
    if simulations < 100:
        raise ValueError("pair-lab simulations must be at least 100")
    factory = SimpleStrategyResearchFactory(settings)
    scope_hash = _scope_hash(
        registry_hash=factory.registry_hash,
        pairs=selected_pairs,
        timeframes=selected_timeframes,
        maximum_rows=maximum_rows,
    )
    scan_limit = max(250, batch_size * 100)
    jobs = _unseen_dna(
        settings,
        factory,
        scope_hash=scope_hash,
        limit=scan_limit,
    )
    processed: list[dict[str, Any]] = []
    compatible_evaluated = 0
    for payload in jobs:
        dna = str(payload["strategy_dna_hash"])
        block_ids = tuple(str(value) for value in payload.get("block_ids") or ())
        compatibility = pair_compatibility(block_ids, factory.registry)
        summary = {
            "strategy_dna_hash": dna,
            "block_ids": list(block_ids),
            "families": list(payload.get("families") or ()),
            "compatibility": compatibility,
        }
        if not compatibility["compatible"]:
            reason = "|".join(compatibility["reasons"][:10])
            _persist_batch(
                settings,
                scope_hash=scope_hash,
                dna=dna,
                status="PAIR_INCOMPATIBLE",
                reason=reason,
                payload=summary,
                routes=(),
            )
            processed.append({**summary, "status": "PAIR_INCOMPATIBLE"})
            continue
        if compatible_evaluated >= batch_size:
            continue
        compatible_evaluated += 1
        logic_mode = LogicMode(str(payload.get("logic_mode") or "LAYERED"))
        combination = factory.generator.materialize_membership(
            block_ids,
            logic_mode=logic_mode,
            mode=GenerationMode.EXHAUSTIVE,
            timeframes=selected_timeframes,
        )
        if combination.strategy_dna_hash != dna:
            raise ValueError(f"strategy DNA mismatch for pair lab: {dna}")
        strategy = CombinatorialStrategy(combination, factory.registry)
        routes: list[dict[str, Any]] = []
        errors: list[str] = []
        for pair in selected_pairs:
            for timeframe in selected_timeframes:
                if timeframe not in combination.common_supported_timeframes:
                    continue
                paths = (
                    settings.paths.processed_data_dir / f"{pair.base_market}_{timeframe}.parquet",
                    settings.paths.processed_data_dir
                    / f"{pair.benchmark_market}_{timeframe}.parquet",
                )
                if any(not path.is_file() for path in paths):
                    errors.append(f"DATA_PENDING:{pair.symbol}:{timeframe}")
                    continue
                try:
                    route = _evaluate_route(
                        settings,
                        strategy,
                        pair,
                        timeframe,
                        maximum_rows=maximum_rows,
                        simulations=simulations,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(
                        f"{pair.symbol}:{timeframe}:{type(exc).__name__}:{str(exc)[:120]}"
                    )
                    continue
                route.update(
                    strategy_dna_hash=dna,
                    block_ids=list(block_ids),
                    logic_mode=logic_mode.value,
                )
                routes.append(route)
        status = "COMPLETE" if routes else "ERROR_RETRYABLE"
        reason = "|".join(errors[:10]) if errors else None
        _persist_batch(
            settings,
            scope_hash=scope_hash,
            dna=dna,
            status=status,
            reason=reason,
            payload={**summary, "errors": errors},
            routes=routes,
        )
        processed.append(
            {
                **summary,
                "status": status,
                "route_count": len(routes),
                "positive_count": sum(bool(row["backtest_positive"]) for row in routes),
                "errors": errors,
            }
        )
    artifacts = _write_reports(settings, scope_hash=scope_hash)
    return {
        "schema_version": "generated_pair_lab_batch_v1",
        "generated_at": utc_iso(),
        "status": "BATCH_COMPLETE" if jobs else "QUEUE_CAUGHT_UP",
        "scope_hash": scope_hash,
        "source_registry_hash": factory.registry_hash,
        "source_queue": str(factory.queue_path.resolve()),
        "pairs": [pair.symbol for pair in selected_pairs],
        "timeframes": list(selected_timeframes),
        "requested_batch_size": batch_size,
        "scan_limit": scan_limit,
        "scanned_count": len(jobs),
        "compatible_evaluated_count": compatible_evaluated,
        "processed_count": len(processed),
        "processed": processed,
        "normal_currency_lab": _normal_lab_snapshot(settings),
        "artifacts": artifacts,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def generated_pair_lab_status(settings: Settings) -> dict[str, Any]:
    status_path = _pair_lab_dir(settings) / "status.json"
    payload = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {
            "status": "NOT_RUN",
            "pair_dna_status_counts": {},
            "pair_route_result_count": 0,
            "pair_positive_route_count": 0,
        }
    )
    service_path = _pair_lab_dir(settings) / "service_status.json"
    service = (
        json.loads(service_path.read_text(encoding="utf-8"))
        if service_path.is_file()
        else {"status": "NOT_STARTED"}
    )
    return {
        **payload,
        "normal_currency_lab": _normal_lab_snapshot(settings),
        "service": service,
        "database": str(_database_path(settings).resolve()),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


__all__ = [
    "DEFAULT_TIMEFRAMES",
    "GENERATED_PAIRS",
    "PAIR_LAB_VERSION",
    "generated_pair_lab_status",
    "pair_compatibility",
    "run_generated_pair_batch",
]
