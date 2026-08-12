"""Causal cross-asset lead-lag falsification for EUR crypto spot.

The campaign tests the negative lead-lag ("seesaw") hypothesis reported in
the cited primary research.  It is deliberately research-only: simultaneous
target returns are clustered into one equal-weight portfolio episode, all
decisions use a fully closed leader candle, and execution occurs at the next
hourly open.  No function in this module has a broker or live-authority path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.common import atomic_write_json, sha256_file, stable_hash, utc_iso

CAMPAIGN_VERSION = "cross_asset_seesaw_campaign_v1"
PRIMARY_RESEARCH = (
    {
        "title": "A Seesaw Effect in the Cryptocurrency Market",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3465924",
        "hypothesis": "large-coin intraday returns negatively predict other coins",
    },
    {
        "title": "Measuring Quantile Dependence and Directional Predictability",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3758495",
        "hypothesis": "Bitcoin-altcoin predictability is state and tail dependent",
    },
)
DEFAULT_LEADERS = (
    "BTC-EUR",
    "ETH-EUR",
    "XRP-EUR",
    "LTC-EUR",
    "BCH-EUR",
)
DEFAULT_TARGETS = (
    "SOL-EUR",
    "LINK-EUR",
    "ADA-EUR",
    "DOGE-EUR",
    "TRX-EUR",
    "HYPE-EUR",
    "AVAX-EUR",
    "BNB-EUR",
)


@dataclass(frozen=True)
class SeesawParameters:
    direction: str
    leader_threshold: float
    holding_bars: int
    leader_markets: tuple[str, ...] = DEFAULT_LEADERS
    target_markets: tuple[str, ...] = DEFAULT_TARGETS
    minimum_targets: int = 4
    timeframe: str = "1h"

    def __post_init__(self) -> None:
        if self.direction not in {"NEGATIVE_SEESAW", "POSITIVE_CONTINUATION"}:
            raise ValueError("unsupported cross-asset direction")
        if not 0.0 < self.leader_threshold < 0.20:
            raise ValueError("leader threshold must be in (0, 0.20)")
        if self.holding_bars < 1:
            raise ValueError("holding bars must be positive")
        if self.minimum_targets < 2:
            raise ValueError("minimum targets must be at least two")
        if self.timeframe != "1h":
            raise ValueError("seesaw campaign is frozen to causal 1h candles")
        if set(self.leader_markets) & set(self.target_markets):
            raise ValueError("leader and target universes must not overlap")

    @property
    def strategy_dna_hash(self) -> str:
        return stable_hash(
            {
                "strategy": "CROSS_ASSET_SEESAW",
                "campaign_version": CAMPAIGN_VERSION,
                "parameters": asdict(self),
            }
        )


def _load_hourly_frame(path: Path) -> tuple[pd.DataFrame, str]:
    before = sha256_file(path)
    raw = pd.read_parquet(path)
    after = sha256_file(path)
    if before != after:
        raise RuntimeError(f"DATA_CHANGED_DURING_READ:{path}")
    required = {"timestamp", "closed", "values"}
    if not required.issubset(raw.columns):
        raise ValueError(f"NORMALIZED_CANDLE_SCHEMA_MISSING:{path}")
    raw = raw.loc[raw["closed"].eq(True)].copy()  # noqa: E712
    values = pd.DataFrame(raw["values"].tolist())
    if not {"open", "close"}.issubset(values.columns):
        raise ValueError(f"NORMALIZED_CANDLE_VALUES_MISSING:{path}")
    frame = pd.DataFrame(
        {
            "open": pd.to_numeric(values["open"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(values["close"], errors="coerce").to_numpy(),
        },
        index=pd.to_datetime(raw["timestamp"], utc=True, errors="coerce"),
    )
    frame = frame.loc[~frame.index.isna()].dropna().sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    if frame.empty or (frame[["open", "close"]] <= 0.0).any().any():
        raise ValueError(f"NORMALIZED_CANDLE_VALUES_INVALID:{path}")
    return frame, before


def load_campaign_frames(
    settings: Settings,
    parameters: SeesawParameters,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    root = settings.paths.processed_data_dir / "bitvavo"
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for market in sorted(set(parameters.leader_markets + parameters.target_markets)):
        path = root / market / f"{parameters.timeframe}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"NORMALIZED_CANDLE_FILE_MISSING:{path}")
        frames[market], hashes[market] = _load_hourly_frame(path)
    return frames, hashes


def leader_index_returns(
    frames: Mapping[str, pd.DataFrame],
    leader_markets: Sequence[str],
) -> pd.Series:
    returns = pd.concat(
        {market: frames[market]["close"].pct_change() for market in leader_markets},
        axis=1,
    )
    return returns.mean(axis=1, skipna=False).dropna()


def simulate_portfolio_episodes(
    frames: Mapping[str, pd.DataFrame],
    parameters: SeesawParameters,
    *,
    round_trip_cost: float,
) -> list[dict[str, Any]]:
    """Return non-overlapping, event-clustered next-open portfolio episodes."""

    leader = leader_index_returns(frames, parameters.leader_markets)
    if parameters.direction == "NEGATIVE_SEESAW":
        signals = leader <= -parameters.leader_threshold
    else:
        signals = leader >= parameters.leader_threshold
    active_until: pd.Timestamp | None = None
    episodes: list[dict[str, Any]] = []
    for signal_at in leader.index[signals]:
        entry_at = signal_at + pd.offsets.Hour(1)
        exit_at = entry_at + pd.offsets.Hour(parameters.holding_bars)
        if active_until is not None and entry_at <= active_until:
            continue
        target_returns: dict[str, float] = {}
        for market in parameters.target_markets:
            frame = frames[market]
            if entry_at not in frame.index or exit_at not in frame.index:
                continue
            target_returns[market] = float(
                frame.at[exit_at, "open"] / frame.at[entry_at, "open"]
                - 1.0
                - round_trip_cost
            )
        if len(target_returns) < parameters.minimum_targets:
            continue
        episodes.append(
            {
                "signal_at": signal_at.isoformat(),
                "entry_at": entry_at.isoformat(),
                "exit_at": exit_at.isoformat(),
                "leader_return": float(leader.loc[signal_at]),
                "target_count": len(target_returns),
                "portfolio_return": float(np.mean(list(target_returns.values()))),
                "target_returns": dict(sorted(target_returns.items())),
            }
        )
        active_until = exit_at
    return episodes


def episode_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [float(row["portfolio_return"]) for row in episodes],
        dtype=float,
    )
    if not len(values):
        return {
            "episode_count": 0,
            "mean_return": None,
            "net_return_sum": 0.0,
            "profit_factor": None,
            "win_rate": None,
        }
    profit = float(values[values > 0.0].sum())
    loss = float(-values[values < 0.0].sum())
    return {
        "episode_count": int(len(values)),
        "mean_return": float(values.mean()),
        "net_return_sum": float(values.sum()),
        "profit_factor": profit / loss if loss > 0.0 else None,
        "win_rate": float((values > 0.0).mean()),
    }


def chronological_metrics(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    ordered = sorted(episodes, key=lambda row: str(row["entry_at"]))
    first = int(len(ordered) * 0.60)
    second = int(len(ordered) * 0.80)
    return {
        "development": episode_metrics(ordered[:first]),
        "validation": episode_metrics(ordered[first:second]),
        "holdout": episode_metrics(ordered[second:]),
    }


def bootstrap_mean_return(
    episodes: Sequence[Mapping[str, Any]],
    *,
    simulations: int = 5_000,
    seed: int = 20260811,
) -> dict[str, Any]:
    values = np.asarray(
        [float(row["portfolio_return"]) for row in episodes],
        dtype=float,
    )
    if not len(values):
        return {
            "simulations": simulations,
            "seed": seed,
            "mean_return_ci_95": [None, None],
            "probability_mean_not_positive": None,
        }
    rng = np.random.default_rng(seed)
    means = np.empty(simulations, dtype=float)
    for start in range(0, simulations, 250):
        size = min(250, simulations - start)
        sampled = rng.choice(values, size=(size, len(values)), replace=True)
        means[start : start + size] = sampled.mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "simulations": simulations,
        "seed": seed,
        "mean_return_ci_95": [float(lower), float(upper)],
        "probability_mean_not_positive": float((means <= 0.0).mean()),
    }


def _passes_candidate_gates(result: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    normal = dict(result["normal"])
    stressed = dict(result["stressed"])
    periods = dict(result["periods"])
    bootstrap = dict(result["bootstrap"])
    if int(normal.get("episode_count") or 0) < 100:
        reasons.append("MINIMUM_EVENT_SAMPLE_FAILED")
    if float(normal.get("profit_factor") or 0.0) <= 1.10:
        reasons.append("NORMAL_PROFIT_FACTOR_FAILED")
    if float(stressed.get("profit_factor") or 0.0) <= 1.0:
        reasons.append("DOUBLE_COST_PROFIT_FACTOR_FAILED")
    for name in ("development", "validation", "holdout"):
        if float((periods.get(name) or {}).get("net_return_sum") or 0.0) <= 0.0:
            reasons.append(f"{name.upper()}_NET_RETURN_FAILED")
    if float((bootstrap.get("mean_return_ci_95") or [0.0])[0] or 0.0) <= 0.0:
        reasons.append("BOOTSTRAP_LOWER_BOUND_FAILED")
    return not reasons, reasons


def run_cross_asset_seesaw_campaign(settings: Settings) -> dict[str, Any]:
    """Run the fixed 40-trial research grid and write an orderless artifact."""

    one_way_cost = (
        float(settings.costs.default_fee)
        + float(settings.costs.slippage_bps) / 10_000.0
        + float(settings.costs.spread_bps) / 20_000.0
    )
    normal_cost = 2.0 * one_way_cost
    stressed_cost = normal_cost * float(settings.costs.stressed_cost_multiplier)
    thresholds = (0.005, 0.010, 0.015, 0.020)
    holding_bars = (1, 3, 6, 12, 24)
    directions = ("NEGATIVE_SEESAW", "POSITIVE_CONTINUATION")
    base = SeesawParameters(
        direction="NEGATIVE_SEESAW",
        leader_threshold=0.020,
        holding_bars=24,
    )
    frames, data_hashes = load_campaign_frames(settings, base)
    results: list[dict[str, Any]] = []
    paper_candidates: list[dict[str, Any]] = []
    for direction in directions:
        for threshold in thresholds:
            for holding in holding_bars:
                parameters = SeesawParameters(
                    direction=direction,
                    leader_threshold=threshold,
                    holding_bars=holding,
                )
                normal_episodes = simulate_portfolio_episodes(
                    frames,
                    parameters,
                    round_trip_cost=normal_cost,
                )
                stressed_episodes = simulate_portfolio_episodes(
                    frames,
                    parameters,
                    round_trip_cost=stressed_cost,
                )
                result: dict[str, Any] = {
                    "strategy_dna_hash": parameters.strategy_dna_hash,
                    "parameters": asdict(parameters),
                    "normal": episode_metrics(normal_episodes),
                    "stressed": episode_metrics(stressed_episodes),
                    "periods": chronological_metrics(normal_episodes),
                    "bootstrap": bootstrap_mean_return(normal_episodes),
                    "integrity": {
                        "closed_candles_only": True,
                        "decision_at_close_execution_next_open": True,
                        "one_non_overlapping_portfolio_episode": True,
                        "simultaneous_targets_clustered": True,
                        "long_only_spot": True,
                        "costs_included": True,
                        "paper_only": True,
                    },
                }
                passed, reasons = _passes_candidate_gates(result)
                result["gates"] = {
                    "paper_candidate_permitted": passed,
                    "reason_codes": reasons,
                }
                results.append(result)
                if passed:
                    paper_candidates.append(
                        {
                            "strategy_dna_hash": parameters.strategy_dna_hash,
                            "parameters": asdict(parameters),
                            "paper_only": True,
                            "live_authority": False,
                        }
                    )
    payload: dict[str, Any] = {
        "schema_version": CAMPAIGN_VERSION,
        "generated_at": utc_iso(),
        "status": "COMPLETED_NOT_PROMOTED" if not paper_candidates else "REVIEW",
        "hypothesis_family": "CROSS_ASSET_SEESAW_LEAD_LAG",
        "primary_research": list(PRIMARY_RESEARCH),
        "trial_count": len(results),
        "multiple_testing_accounted": True,
        "data_hashes": data_hashes,
        "normal_round_trip_cost": normal_cost,
        "stressed_round_trip_cost": stressed_cost,
        "results": results,
        "paper_candidates": paper_candidates,
        "paper_candidate_count": len(paper_candidates),
        "live_ready": False,
        "live_authority": False,
        "orders_generated": 0,
        "orders_submitted": 0,
        "private_exchange_requests": 0,
    }
    payload["artifact_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "generated_at"}
    )
    output = (
        settings.paths.lab_dir
        / "reports"
        / "cross_asset_seesaw_campaign_v1.json"
    )
    atomic_write_json(output, payload)
    return {**payload, "artifact_path": str(output.resolve())}


__all__ = [
    "SeesawParameters",
    "bootstrap_mean_return",
    "chronological_metrics",
    "episode_metrics",
    "leader_index_returns",
    "load_campaign_frames",
    "run_cross_asset_seesaw_campaign",
    "simulate_portfolio_episodes",
]
