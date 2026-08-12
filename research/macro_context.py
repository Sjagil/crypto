"""Causal, cadence-aware crypto macro context integrated with FeaturePipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from config.settings import TIMEFRAME_SECONDS, normalize_timeframe
from data.derivatives_context import CryptoGEXAnalyzer, OptionsContract, annualize_funding
from utils.common import atomic_write_json

GROUPS = (
    "sentiment",
    "dominance",
    "relative_strength",
    "breadth",
    "derivatives",
    "flows",
    "onchain",
    "global_macro",
    "events",
    "gex",
)


@dataclass(frozen=True)
class MacroSourceSpec:
    provider: str
    source_frequency: str
    expected_cadence: timedelta
    maximum_age: timedelta
    window_interpretation: Literal["bars", "hours", "days"] = "days"
    units: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("macro provider is required")
        if self.expected_cadence <= timedelta(0) or self.maximum_age <= timedelta(0):
            raise ValueError("cadence and maximum age must be positive")


@dataclass(frozen=True)
class MacroContextConfig:
    group_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "sentiment": 0.08,
            "dominance": 0.12,
            "relative_strength": 0.12,
            "breadth": 0.12,
            "derivatives": 0.14,
            "flows": 0.08,
            "onchain": 0.08,
            "global_macro": 0.12,
            "events": 0.06,
            "gex": 0.08,
        }
    )
    high_impact_pre_hours: float = 2.0
    high_impact_post_hours: float = 1.0
    token_unlock_pre_hours: float = 72.0
    material_unlock_fraction: float = 0.01
    funding_z_overheated: float = 2.0
    funding_z_crowded_short: float = -2.0
    gex_extreme_concentration: float = 0.40

    def __post_init__(self) -> None:
        if set(self.group_weights) != set(GROUPS):
            raise ValueError("group weights must cover every macro feature group")
        if any(value < 0 for value in self.group_weights.values()):
            raise ValueError("group weights cannot be negative")
        if sum(self.group_weights.values()) <= 0:
            raise ValueError("at least one group weight must be positive")


def _utc_index(value: pd.DataFrame | pd.Index) -> pd.DatetimeIndex:
    index = value.index if isinstance(value, pd.DataFrame) else value
    result = pd.DatetimeIndex(pd.to_datetime(index, utc=True)).sort_values()
    if result.has_duplicates:
        raise ValueError("base timestamps contain duplicates")
    return result


def _source(
    frame: pd.DataFrame,
    *,
    available_at_col: str | None,
    spec: MacroSourceSpec,
) -> pd.DataFrame:
    selected = frame.copy()
    if available_at_col:
        if available_at_col not in selected:
            raise ValueError(f"missing availability column: {available_at_col}")
        selected.index = pd.to_datetime(selected.pop(available_at_col), utc=True)
    elif not isinstance(selected.index, pd.DatetimeIndex):
        raise TypeError("macro source requires DatetimeIndex or available_at column")
    selected.index = pd.to_datetime(selected.index, utc=True)
    selected = selected.sort_index()
    selected = selected[~selected.index.duplicated(keep="last")]
    for column, unit in spec.units.items():
        if column not in selected:
            continue
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        if unit == "percent_points":
            selected[column] /= 100.0
        elif unit in {"fraction", "index", "currency", "count", "price"}:
            pass
        else:
            raise ValueError(f"unsupported declared unit {unit!r} for {column}")
    return selected


def causal_align(
    base_index: pd.DatetimeIndex,
    source: pd.DataFrame,
    *,
    group: str,
    spec: MacroSourceSpec,
) -> pd.DataFrame:
    left = pd.DataFrame({"timestamp": base_index})
    right = source.reset_index(names="source_available_at")
    aligned = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on="source_available_at",
        direction="backward",
        tolerance=pd.Timedelta(spec.maximum_age),
    ).set_index("timestamp")
    aligned.index = base_index
    age = (
        aligned.index.to_series(index=aligned.index) - aligned["source_available_at"]
    ).dt.total_seconds() / 3_600
    aligned[f"{group}_source_time"] = aligned.pop("source_available_at")
    aligned[f"{group}_age_hours"] = age
    aligned[f"{group}_provider"] = spec.provider
    aligned[f"{group}_fresh"] = age.le(spec.maximum_age.total_seconds() / 3_600)
    return aligned


def cadence_change(
    series: pd.Series,
    *,
    window: int | float,
    unit: Literal["bars", "hours", "days"],
    expected_cadence: timedelta,
) -> pd.Series:
    """Backward point-in-time change with explicit calendar or bar semantics."""
    selected = pd.to_numeric(series, errors="coerce").sort_index()
    if unit == "bars":
        periods = int(window)
        if periods < 1:
            raise ValueError("bar window must be positive")
        return selected.pct_change(periods=periods, fill_method=None)
    delta = timedelta(hours=float(window)) if unit == "hours" else timedelta(days=float(window))
    left = pd.DataFrame(
        {"timestamp": selected.index, "current": selected.to_numpy()}
    )
    targets = pd.DataFrame(
        {"target": selected.index - pd.Timedelta(delta), "row": np.arange(len(selected))}
    ).sort_values("target")
    history = pd.DataFrame(
        {"source_time": selected.index, "prior": selected.to_numpy()}
    )
    matched = pd.merge_asof(
        targets,
        history,
        left_on="target",
        right_on="source_time",
        direction="backward",
        tolerance=pd.Timedelta(expected_cadence) * 1.5,
    ).sort_values("row")
    prior = matched["prior"].to_numpy(dtype=float)
    current = left["current"].to_numpy(dtype=float)
    result = np.divide(
        current,
        prior,
        out=np.full(len(current), np.nan),
        where=np.isfinite(prior) & (prior != 0),
    ) - 1
    return pd.Series(result, index=selected.index)


def time_zscore(series: pd.Series, window: str = "90D") -> pd.Series:
    selected = pd.to_numeric(series, errors="coerce")
    rolling = selected.rolling(window, min_periods=5)
    return (selected - rolling.mean()) / rolling.std(ddof=0).replace(0, np.nan)


def _first(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _sentiment(frame: pd.DataFrame, spec: MacroSourceSpec) -> pd.DataFrame:
    value = _first(frame, ("fear_greed", "value", "index"))
    out = pd.DataFrame(index=frame.index)
    out["sentiment_fear_greed"] = value
    out["sentiment_change_7d"] = cadence_change(
        value, window=7, unit="days", expected_cadence=spec.expected_cadence
    )
    out["sentiment_zscore_90d"] = time_zscore(value)
    out["sentiment_regime"] = pd.cut(
        value,
        [-np.inf, 24, 44, 55, 74, np.inf],
        labels=["extreme_fear", "fear", "neutral", "greed", "extreme_greed"],
    ).astype("string")
    return out


def _dominance(frame: pd.DataFrame, spec: MacroSourceSpec) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    aliases = {
        "btc_dominance": ("btc_dominance", "btc_dominance_fraction"),
        "stablecoin_dominance": (
            "stablecoin_dominance",
            "stablecoin_dominance_fraction",
        ),
        "total_market_cap": ("total_market_cap", "total"),
        "total2_market_cap": ("total2_market_cap", "total2"),
        "total3_market_cap": ("total3_market_cap", "total3"),
    }
    for label, names in aliases.items():
        value = _first(frame, names)
        out[f"dominance_{label}"] = value
        out[f"dominance_{label}_change_7d"] = cadence_change(
            value, window=7, unit="days", expected_cadence=spec.expected_cadence
        )
    return out


def _relative(frame: pd.DataFrame, spec: MacroSourceSpec) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    btc = _first(frame, ("btc", "BTC", "btc_price", "BTC-EUR"))
    out["relative_btc_return_7"] = cadence_change(
        btc, window=7, unit="days", expected_cadence=spec.expected_cadence
    )
    out["relative_btc_return_30"] = cadence_change(
        btc, window=30, unit="days", expected_cadence=spec.expected_cadence
    )
    for asset in ("eth", "sol"):
        price = _first(frame, (asset, asset.upper(), f"{asset}_price", f"{asset.upper()}-EUR"))
        ratio = price / btc.replace(0, np.nan)
        out[f"relative_{asset}_btc"] = ratio
        out[f"relative_{asset}_btc_change_7d"] = cadence_change(
            ratio, window=7, unit="days", expected_cadence=spec.expected_cadence
        )
    return out


def _breadth(frame: pd.DataFrame, spec: MacroSourceSpec) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(index=frame.index)
    if numeric.empty:
        return out
    for days in (20, 50, 200):
        bars = max(
            2,
            round(
                timedelta(days=days).total_seconds()
                / spec.expected_cadence.total_seconds()
            ),
        )
        ema = numeric.ewm(span=bars, min_periods=2, adjust=False).mean()
        out[f"breadth_fraction_above_ema{days}"] = numeric.gt(ema).mean(axis=1)
        out[f"breadth_fraction_above_mean_{days}d"] = out[
            f"breadth_fraction_above_ema{days}"
        ]
    returns_1d = numeric.apply(
        lambda value: cadence_change(
            value, window=1, unit="days", expected_cadence=spec.expected_cadence
        )
    )
    returns_7d = numeric.apply(
        lambda value: cadence_change(
            value, window=7, unit="days", expected_cadence=spec.expected_cadence
        )
    )
    returns_30d = numeric.apply(
        lambda value: cadence_change(
            value, window=30, unit="days", expected_cadence=spec.expected_cadence
        )
    )
    out["breadth_positive_return_1d"] = returns_1d.gt(0).mean(axis=1)
    out["breadth_positive_return_7d"] = returns_7d.gt(0).mean(axis=1)
    out["breadth_positive_return_30d"] = returns_30d.gt(0).mean(axis=1)
    out["breadth_advance_decline"] = (
        returns_1d.gt(0).sum(axis=1) - returns_1d.lt(0).sum(axis=1)
    )
    out["breadth_equal_weighted_return_1d"] = returns_1d.mean(axis=1)
    out["breadth_median_return_1d"] = returns_1d.median(axis=1)
    out["breadth_cross_sectional_dispersion_1d"] = returns_1d.std(
        axis=1,
        ddof=0,
    )
    btc_column = next(
        (name for name in numeric if str(name).casefold() in {"btc", "btc-eur", "btc-usd", "btc-usdt"}),
        None,
    )
    out["breadth_btc_relative_fraction_1d"] = (
        returns_1d.gt(returns_1d[btc_column], axis=0).mean(axis=1)
        if btc_column is not None
        else np.nan
    )
    out["breadth_new_high_fraction"] = numeric.eq(
        numeric.rolling("30D", min_periods=2).max()
    ).mean(axis=1)
    out["breadth_new_low_fraction"] = numeric.eq(
        numeric.rolling("30D", min_periods=2).min()
    ).mean(axis=1)
    out["breadth_breadth_risk_on"] = out["breadth_fraction_above_mean_50d"] >= 0.55
    out["breadth_breadth_risk_off"] = out["breadth_fraction_above_mean_50d"] <= 0.35
    return out


def _derivatives(frame: pd.DataFrame, spec: MacroSourceSpec) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    funding = _first(frame, ("funding_rate",))
    interval = _first(frame, ("funding_interval_seconds",))
    out["derivatives_funding_rate"] = funding
    out["derivatives_funding_interval_seconds"] = interval
    out["derivatives_annualized_funding"] = pd.Series(
        [
            annualize_funding(rate, seconds)
            if np.isfinite(rate) and np.isfinite(seconds) and seconds > 0
            else np.nan
            for rate, seconds in zip(funding, interval, strict=False)
        ],
        index=frame.index,
    )
    out["derivatives_funding_zscore"] = time_zscore(funding, "30D")
    oi = _first(frame, ("open_interest",))
    out["derivatives_open_interest"] = oi
    out["derivatives_open_interest_change_7d"] = cadence_change(
        oi, window=7, unit="days", expected_cadence=spec.expected_cadence
    )
    for name in (
        "basis",
        "perpetual_premium",
        "long_liquidations",
        "short_liquidations",
        "liquidation_imbalance",
    ):
        out[f"derivatives_{name}"] = _first(frame, (name,))
    out["derivatives_leverage_overheated"] = (
        out["derivatives_funding_zscore"] >= 2
    ) | (out["derivatives_open_interest_change_7d"] >= 0.10)
    out["derivatives_deleveraging_event"] = (
        out["derivatives_open_interest_change_7d"] <= -0.10
    )
    return out


def _flows(frame: pd.DataFrame, spec: MacroSourceSpec) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for asset in ("btc", "eth"):
        value = _first(frame, (f"{asset}_etf_flow", f"{asset}_flow"))
        out[f"flows_{asset}_etf_flow"] = value
        out[f"flows_{asset}_etf_flow_7d"] = value.rolling("7D", min_periods=1).sum()
    return out


def _onchain(frame: pd.DataFrame, spec: MacroSourceSpec) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in (
        "exchange_netflow",
        "mvrv",
        "sopr",
        "nupl",
        "active_addresses",
        "transaction_count",
    ):
        value = _first(frame, (name,))
        out[f"onchain_{name}"] = value
        out[f"onchain_{name}_change_7d"] = cadence_change(
            value, window=7, unit="days", expected_cadence=spec.expected_cadence
        )
    return out


def _global(frame: pd.DataFrame, spec: MacroSourceSpec) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in (
        "dxy",
        "nasdaq",
        "sp500",
        "vix",
        "policy_rate",
        "ten_year_yield",
        "credit_spread",
        "liquidity",
    ):
        value = _first(frame, (name,))
        out[f"global_{name}"] = value
        out[f"global_{name}_change_7d"] = cadence_change(
            value, window=7, unit="days", expected_cadence=spec.expected_cadence
        )
    out["global_global_risk_on"] = (
        out["global_nasdaq_change_7d"].gt(0)
        & out["global_dxy_change_7d"].lt(0)
        & out["global_vix"].lt(25)
    )
    out["global_global_risk_off"] = (
        out["global_nasdaq_change_7d"].lt(0)
        | out["global_dxy_change_7d"].gt(0)
        | out["global_vix"].gt(25)
    )
    return out


def _events(
    base: pd.DatetimeIndex,
    frame: pd.DataFrame,
    config: MacroContextConfig,
) -> pd.DataFrame:
    """Vectorized per-event interval assignment; never scans all events per candle."""
    out = pd.DataFrame(
        {
            "events_high_impact_event_risk": False,
            "events_token_unlock_risk": False,
            "events_event_count": 0,
        },
        index=base,
    )
    if frame.empty:
        return out
    event_time_col = "event_at" if "event_at" in frame else "event_time"
    if event_time_col not in frame:
        raise ValueError("events require event_at or event_time")
    events = frame.copy()
    events[event_time_col] = pd.to_datetime(events[event_time_col], utc=True)
    availability = pd.Series(events.index, index=events.index)
    for index, event in events.iterrows():
        event_time = pd.Timestamp(event[event_time_col])
        known_at = pd.Timestamp(availability.loc[index])
        impact = str(event.get("impact", "")).casefold()
        event_type = str(event.get("event_type", event.get("type", ""))).casefold()
        if impact == "high":
            start = max(
                known_at,
                event_time - timedelta(hours=config.high_impact_pre_hours),
            )
            end = event_time + timedelta(hours=config.high_impact_post_hours)
            mask = (base >= start) & (base <= end)
            out.loc[mask, "events_high_impact_event_risk"] = True
            out.loc[mask, "events_event_count"] += 1
        unlock_fraction = pd.to_numeric(event.get("unlock_fraction"), errors="coerce")
        if (
            event_type == "token_unlock"
            and pd.notna(unlock_fraction)
            and float(unlock_fraction) >= config.material_unlock_fraction
        ):
            start = max(
                known_at,
                event_time - timedelta(hours=config.token_unlock_pre_hours),
            )
            mask = (base >= start) & (base <= event_time)
            out.loc[mask, "events_token_unlock_risk"] = True
            out.loc[mask, "events_event_count"] += 1
    return out


def _gex(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in (
        "call_gex_proxy",
        "put_gex_proxy",
        "gross_gex_proxy",
        "net_gex_proxy",
        "nearest_expiry_concentration",
        "dominant_gamma_strike",
        "gamma_concentration",
        "gamma_flip_proxy",
        "spot_distance_from_dominant_gamma",
    ):
        out[f"gex_{name}"] = _first(frame, (name,))
    out["gex_stale"] = frame.get("stale", False)
    return out


def _regimes(features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)
    def column(name: str, default: Any = np.nan) -> pd.Series:
        return features[name] if name in features else pd.Series(default, index=features.index)

    btc7 = column("relative_btc_return_7")
    btc30 = column("relative_btc_return_30")
    btc_dom = column("dominance_btc_dominance_change_7d")
    stable = column("dominance_stablecoin_dominance_change_7d")
    breadth_on = column("breadth_breadth_risk_on", False).fillna(False).astype(bool)
    breadth_off = column("breadth_breadth_risk_off", False).fillna(False).astype(bool)
    global_on = column("global_global_risk_on", False).fillna(False).astype(bool)
    global_off = column("global_global_risk_off", False).fillna(False).astype(bool)
    leverage = column("derivatives_leverage_overheated", False).fillna(False).astype(bool)
    deleveraging = column("derivatives_deleveraging_event", False).fillna(False).astype(bool)
    event = (
        column("events_high_impact_event_risk", False).fillna(False).astype(bool)
        | column("events_token_unlock_risk", False).fillna(False).astype(bool)
    )
    gex_extreme = column("gex_gamma_concentration", 0).fillna(0).ge(0.4)
    out["btc_led_market"] = (btc7.gt(0) & btc_dom.gt(0)).fillna(False)
    out["broad_altcoin_market"] = (btc7.gt(0) & btc_dom.lt(0)).fillna(False)
    out["altcoin_capitulation"] = (btc7.lt(0) & btc_dom.gt(0)).fillna(False)
    out["stablecoin_rotation"] = (stable.gt(0) & btc7.lt(0)).fillna(False)
    components = pd.DataFrame(index=features.index)
    components["btc"] = np.select([btc30.gt(0), btc30.lt(0)], [1, -1], default=0)
    components["breadth"] = np.select([breadth_on, breadth_off], [1, -1], default=0)
    components["global"] = np.select([global_on, global_off], [1, -1], default=0)
    components["leverage"] = np.select([deleveraging, leverage], [-0.5, -1], default=0)
    components["events"] = np.where(event, -1, 0)
    components["gex"] = np.where(gex_extreme, -0.5, 0)
    out["crypto_risk_score"] = components.sum(axis=1)
    out["crypto_risk_on"] = out["crypto_risk_score"].ge(2)
    out["crypto_risk_off"] = out["crypto_risk_score"].le(-2)
    out["primary_crypto_regime"] = np.select(
        [
            deleveraging,
            leverage,
            out["altcoin_capitulation"],
            out["broad_altcoin_market"],
            out["btc_led_market"],
            out["crypto_risk_on"],
            out["crypto_risk_off"],
        ],
        [
            "deleveraging_event",
            "leverage_overheated",
            "altcoin_capitulation",
            "broad_altcoin_risk_on",
            "btc_led_risk_on",
            "crypto_risk_on",
            "crypto_risk_off",
        ],
        default="neutral_or_mixed",
    )
    advisory = pd.Series(0.70, index=features.index)
    advisory.loc[out["crypto_risk_on"]] = 1.0
    advisory.loc[out["crypto_risk_off"]] = 0.35
    advisory.loc[event | gex_extreme] = np.minimum(advisory.loc[event | gex_extreme], 0.5)
    out["research_exposure_multiplier_advisory"] = advisory
    out["risk_manager_approval_required"] = True
    return out


class MacroContextEngine:
    def __init__(self, config: MacroContextConfig | None = None) -> None:
        self.config = config or MacroContextConfig()

    def build(
        self,
        base: pd.DataFrame | pd.Index,
        *,
        fear_greed: pd.DataFrame | None = None,
        dominance: pd.DataFrame | None = None,
        relative_prices: pd.DataFrame | None = None,
        breadth_prices: pd.DataFrame | None = None,
        derivatives: pd.DataFrame | None = None,
        etf_flows: pd.DataFrame | None = None,
        onchain: pd.DataFrame | None = None,
        global_macro: pd.DataFrame | None = None,
        events: pd.DataFrame | None = None,
        gex: pd.DataFrame | None = None,
        source_specs: Mapping[str, MacroSourceSpec] | None = None,
        availability_columns: Mapping[str, str | None] | None = None,
    ) -> pd.DataFrame:
        base_index = _utc_index(base)
        datasets = {
            "sentiment": fear_greed,
            "dominance": dominance,
            "relative_strength": relative_prices,
            "breadth": breadth_prices,
            "derivatives": derivatives,
            "flows": etf_flows,
            "onchain": onchain,
            "global_macro": global_macro,
            "events": events,
            "gex": gex,
        }
        specs = dict(source_specs or {})
        availability = dict(availability_columns or {})
        blocks: list[pd.DataFrame] = []
        group_feature_columns: dict[str, list[str]] = {}
        for group, raw in datasets.items():
            if raw is None:
                continue
            if group not in specs:
                raise ValueError(f"source_specs must declare cadence and units for {group}")
            spec = specs[group]
            prepared = _source(
                raw,
                available_at_col=availability.get(group),
                spec=spec,
            )
            if group == "events":
                features = _events(base_index, prepared, self.config)
                aligned = causal_align(
                    base_index,
                    prepared.drop(columns=["event_at", "event_time"], errors="ignore"),
                    group=group,
                    spec=spec,
                )
            else:
                aligned = causal_align(base_index, prepared, group=group, spec=spec)
                aligned_values = aligned.drop(
                    columns=[
                        f"{group}_source_time",
                        f"{group}_age_hours",
                        f"{group}_provider",
                        f"{group}_fresh",
                    ]
                )
                calculators = {
                    "sentiment": lambda value: _sentiment(value, spec),
                    "dominance": lambda value: _dominance(value, spec),
                    "relative_strength": lambda value: _relative(value, spec),
                    "breadth": lambda value: _breadth(value, spec),
                    "derivatives": lambda value: _derivatives(value, spec),
                    "flows": lambda value: _flows(value, spec),
                    "onchain": lambda value: _onchain(value, spec),
                    "global_macro": lambda value: _global(value, spec),
                    "gex": _gex,
                }
                features = calculators[group](aligned_values)
            metadata = aligned[
                [
                    f"{group}_source_time",
                    f"{group}_age_hours",
                    f"{group}_provider",
                    f"{group}_fresh",
                ]
            ]
            blocks.extend([features, metadata])
            group_feature_columns[group] = list(features.columns)
        result = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=base_index)
        for group in GROUPS:
            columns = group_feature_columns.get(group, [])
            result[f"{group}_completeness"] = (
                result[columns].notna().mean(axis=1) if columns else 0.0
            )
        total_weight = sum(self.config.group_weights.values())
        result["weighted_total_completeness"] = sum(
            result[f"{group}_completeness"] * self.config.group_weights[group]
            for group in GROUPS
        ) / total_weight
        result["macro_context_usable"] = result["weighted_total_completeness"].ge(0.40)
        result = pd.concat([result, _regimes(result)], axis=1)
        result.attrs["group_weights"] = dict(self.config.group_weights)
        result.attrs["source_specs"] = {
            name: {
                "provider": spec.provider,
                "source_frequency": spec.source_frequency,
                "expected_cadence_seconds": spec.expected_cadence.total_seconds(),
                "window_interpretation": spec.window_interpretation,
                "maximum_age_seconds": spec.maximum_age.total_seconds(),
                "units": dict(spec.units),
            }
            for name, spec in specs.items()
        }
        result.attrs["gex_assumptions"] = CryptoGEXAnalyzer.assumption_metadata
        result.attrs["canonical_macro_context"] = True
        result.attrs["point_in_time_aligned"] = True
        result.attrs["provenance_engine"] = "MacroContextEngine"
        return result

    @staticmethod
    def latest_snapshot(features: pd.DataFrame) -> dict[str, Any]:
        if features.empty:
            raise ValueError("features is empty")
        row = features.iloc[-1]
        keys = (
            "primary_crypto_regime",
            "crypto_risk_score",
            "crypto_risk_on",
            "crypto_risk_off",
            "research_exposure_multiplier_advisory",
            "weighted_total_completeness",
            "macro_context_usable",
        )
        return {
            "timestamp": str(features.index[-1]),
            **{
                key: (
                    row.get(key).item()
                    if isinstance(row.get(key), np.generic)
                    else row.get(key)
                )
                for key in keys
            },
        }


def summarize_gex_contracts(
    contracts: list[OptionsContract | Mapping[str, Any]],
) -> pd.DataFrame:
    summary = CryptoGEXAnalyzer().calculate(contracts)
    available = max(
        (
            item.available_at if isinstance(item, OptionsContract) else item["available_at"]
            for item in contracts
        ),
        default=pd.Timestamp.now(tz="UTC"),
    )
    scalar = {
        key: value
        for key, value in summary.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    return pd.DataFrame([scalar], index=pd.DatetimeIndex([available]))


def build_macro_features(base: pd.DataFrame | pd.Index, **datasets: Any) -> pd.DataFrame:
    return MacroContextEngine().build(base, **datasets)


def _flatten_persisted_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "values" in frame:
        expanded = pd.json_normalize(
            frame["values"].map(lambda value: value if isinstance(value, dict) else {})
        )
        expanded.index = frame.index
        frame = pd.concat([frame.drop(columns=["values"]), expanded], axis=1)
    timestamp = next(
        (
            name
            for name in ("available_at", "observed_at", "observation_time", "timestamp")
            if name in frame
        ),
        None,
    )
    if timestamp is None:
        raise ValueError(f"{path.name} has no point-in-time timestamp")
    frame.index = pd.to_datetime(frame[timestamp], utc=True, errors="coerce")
    frame = frame.loc[frame.index.notna()].sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def _combine_context_frames(paths: list[Path]) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            selected = _flatten_persisted_frame(path)
        except (OSError, ValueError, TypeError):
            continue
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return None
    return pd.concat(frames, axis=0, sort=False).sort_index()


def _stored_market_prices(processed_dir: Path, timeframe: str) -> pd.DataFrame | None:
    selected_timeframe = normalize_timeframe(timeframe)
    candidates = set(processed_dir.glob(f"*_{selected_timeframe}.parquet"))
    candidates.update(processed_dir.glob(f"*/*/{selected_timeframe}.parquet"))
    series: dict[str, pd.Series] = {}
    for path in sorted(candidates):
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        if "timestamp" in frame:
            frame.index = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        elif not isinstance(frame.index, pd.DatetimeIndex):
            continue
        else:
            frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        if "close" in frame:
            close = pd.to_numeric(frame["close"], errors="coerce")
        elif "values" in frame:
            close = pd.to_numeric(
                frame["values"].map(
                    lambda value: value.get("close") if isinstance(value, dict) else np.nan
                ),
                errors="coerce",
            )
        else:
            continue
        if "canonical_market" in frame and frame["canonical_market"].notna().any():
            market = str(frame["canonical_market"].dropna().iloc[-1])
        elif path.parent != processed_dir:
            market = path.parent.name
        else:
            market = path.stem.removesuffix(f"_{selected_timeframe}")
        asset = market.split("-")[0].lower()
        candidate = pd.Series(close.to_numpy(), index=frame.index, name=asset).dropna()
        if candidate.empty:
            continue
        existing = series.get(asset)
        if existing is None or len(candidate) > len(existing):
            series[asset] = candidate
    if not series:
        return None
    return pd.concat(series.values(), axis=1).sort_index()


def build_persisted_macro_context(
    *,
    context_dir: Path,
    processed_dir: Path,
    timeframes: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Build the canonical engine from only persisted, point-in-time source data."""

    context_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(context_dir.glob("*.parquet"))
    grouped_paths = {
        "sentiment": [path for path in files if "alternative_me" in path.name],
        "dominance": [path for path in files if "coinmarketcap_global" in path.name],
        "global_macro": [
            path
            for path in files
            if path.name.startswith(("fred_", "eodhd_"))
            and "economic_event" not in path.name
        ],
        "events": [path for path in files if "economic_event" in path.name],
        "onchain": [
            path
            for path in files
            if "defillama_" in path.name and "stablecoin" in path.name
        ],
        "derivatives": [path for path in files if "derivatives_" in path.name],
        "gex": [path for path in files if path.name.startswith("gex_")],
    }
    raw_groups = {
        name: _combine_context_frames(paths)
        for name, paths in grouped_paths.items()
    }
    for name, frame in raw_groups.items():
        if frame is not None and name != "global_macro":
            raw_groups[name] = frame[
                ~frame.index.duplicated(keep="last")
            ]
    fred_names = {
        "DFF": "policy_rate",
        "DGS10": "ten_year_yield",
        "BAMLH0A0HYM2": "credit_spread",
        "M2SL": "liquidity",
        "WALCL": "fed_balance_sheet",
        "NFCI": "financial_conditions",
        "VIX.INDX": "vix",
        "GSPC.INDX": "sp500",
        "NDX.INDX": "nasdaq",
    }
    global_raw = raw_groups["global_macro"]
    if global_raw is not None and "source_symbol" in global_raw:
        value_columns = [
            name
            for name in ("value", "close", "adjusted_close", "Value")
            if name in global_raw
        ]
        value_column = "_macro_value" if value_columns else None
        if value_column:
            global_raw[value_column] = (
                global_raw[value_columns]
                .apply(pd.to_numeric, errors="coerce")
                .bfill(axis=1)
                .iloc[:, 0]
            )
            global_raw = (
                global_raw.assign(
                    macro_name=global_raw["source_symbol"].map(fred_names),
                    _pit_index=global_raw.index,
                )
                .dropna(subset=["macro_name"])
                .pivot_table(
                    index="_pit_index",
                    columns="macro_name",
                    values=value_column,
                    aggfunc="last",
                )
            )
            global_raw.index = pd.to_datetime(global_raw.index, utc=True)
            raw_groups["global_macro"] = global_raw
    outputs: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for requested_timeframe in timeframes:
        timeframe = normalize_timeframe(requested_timeframe)
        prices = _stored_market_prices(processed_dir, timeframe)
        datasets = {
            "fear_greed": raw_groups["sentiment"],
            "dominance": raw_groups["dominance"],
            "relative_prices": prices,
            "breadth_prices": prices,
            "derivatives": raw_groups["derivatives"],
            "onchain": raw_groups["onchain"],
            "global_macro": raw_groups["global_macro"],
            "events": raw_groups["events"],
            "gex": raw_groups["gex"],
        }
        populated = {
            name: frame
            for name, frame in datasets.items()
            if frame is not None and not frame.empty
        }
        if not populated:
            outputs.append(
                {
                    "timeframe": timeframe,
                    "status": "PARTIAL",
                    "reason_code": "NO_PERSISTED_CONTEXT_DATA",
                }
            )
            continue
        indexes = [pd.DatetimeIndex(frame.index) for frame in populated.values()]
        end = min(
            max(index.max() for index in indexes),
            pd.Timestamp.now(tz="UTC"),
        )
        start = min(index.min() for index in indexes)
        start = max(start, end - pd.to_timedelta(3650, unit="D"))
        if start >= end:
            start = end - pd.Timedelta(days=30)
        rule = {
            "1W": "W-MON",
            "1mo": "MS",
        }.get(
            timeframe,
            pd.to_timedelta(TIMEFRAME_SECONDS[timeframe], unit="s"),
        )
        base = pd.date_range(start=start.floor("h"), end=end, freq=rule, tz="UTC")
        specs: dict[str, MacroSourceSpec] = {}
        engine_arguments: dict[str, pd.DataFrame] = {}
        price_cadence = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
        mapping = {
            "fear_greed": ("sentiment", timedelta(days=1), timedelta(days=2)),
            "dominance": ("dominance", timedelta(hours=1), timedelta(days=2)),
            "relative_prices": (
                "relative_strength",
                price_cadence,
                max(timedelta(days=2), price_cadence * 2),
            ),
            "breadth_prices": (
                "breadth",
                price_cadence,
                max(timedelta(days=2), price_cadence * 2),
            ),
            "derivatives": ("derivatives", timedelta(minutes=15), timedelta(hours=2)),
            "onchain": ("onchain", timedelta(days=1), timedelta(days=3)),
            "global_macro": ("global_macro", timedelta(days=1), timedelta(days=35)),
            "events": ("events", timedelta(hours=1), timedelta(days=14)),
            "gex": ("gex", timedelta(minutes=15), timedelta(hours=2)),
        }
        for argument, frame in populated.items():
            group, cadence, maximum_age = mapping[argument]
            engine_arguments[argument] = frame
            provider_values = (
                sorted(set(frame["provider"].dropna().astype(str)))
                if "provider" in frame
                else ["persisted_market_data"]
            )
            specs[group] = MacroSourceSpec(
                provider=",".join(provider_values),
                source_frequency=str(cadence),
                expected_cadence=cadence,
                maximum_age=maximum_age,
                units={},
            )
        result = MacroContextEngine().build(
            base,
            **engine_arguments,
            source_specs=specs,
        )
        target = context_dir / f"macro_context_{timeframe}.parquet"
        temporary = target.with_name(f".{target.name}.tmp")
        result.to_parquet(temporary)
        temporary.replace(target)
        for argument, frame in populated.items():
            group = mapping[argument][0]
            completeness = float(result[f"{group}_completeness"].mean())
            fresh_column = f"{group}_fresh"
            coverage_rows.append(
                {
                    "timeframe": timeframe,
                    "feature_group": group,
                    "provider": specs[group].provider,
                    "coverage_start": frame.index.min(),
                    "coverage_end": frame.index.max(),
                    "cadence": specs[group].source_frequency,
                    "maximum_age_seconds": specs[group].maximum_age.total_seconds(),
                    "completeness": completeness,
                    "stale_fraction": (
                        float(1 - result[fresh_column].fillna(False).mean())
                        if fresh_column in result
                        else 1.0
                    ),
                    "point_in_time_status": (
                        "CURRENT_UNIVERSE_RETROSPECTIVE"
                        if group == "breadth"
                        else "AVAILABLE_AT_ALIGNED"
                    ),
                    "missing_reason": (
                        "CAP_WEIGHTED_AND_SECTOR_BREADTH_REQUIRE_HISTORICAL_POINT_IN_TIME_WEIGHTS"
                        if group == "breadth"
                        else None
                    ),
                }
            )
        missing = sorted(set(GROUPS) - set(specs))
        for group in missing:
            coverage_rows.append(
                {
                    "timeframe": timeframe,
                    "feature_group": group,
                    "provider": None,
                    "coverage_start": None,
                    "coverage_end": None,
                    "cadence": None,
                    "maximum_age_seconds": None,
                    "completeness": 0.0,
                    "stale_fraction": 1.0,
                    "point_in_time_status": "UNAVAILABLE",
                    "missing_reason": "NO_PERSISTED_PROVIDER_DATA",
                }
            )
        outputs.append(
            {
                "timeframe": timeframe,
                "status": "READY",
                "rows": len(result),
                "columns": len(result.columns),
                "output": target,
                "snapshot": MacroContextEngine.latest_snapshot(result),
                "missing_groups": missing,
            }
        )
    report = {
        "generated_at": datetime.now(tz=UTC),
        "status": "READY" if any(row["status"] == "READY" for row in outputs) else "PARTIAL",
        "outputs": outputs,
        "coverage": coverage_rows,
        "breadth_limitations": [
            "Current discovery membership is labeled CURRENT_UNIVERSE_RETROSPECTIVE.",
            "Cap-weighted and sector breadth remain unavailable without historical point-in-time weights and sector membership.",
            "Native EUR, USD and USDT quote histories remain unconverted; no execution-ready EUR series is fabricated.",
        ],
    }
    atomic_write_json(context_dir / "macro_context_coverage.json", report)
    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(
            context_dir / "macro_context_coverage.csv",
            index=False,
        )
    return report


__all__ = [
    "GROUPS",
    "MacroContextConfig",
    "MacroContextEngine",
    "MacroSourceSpec",
    "build_persisted_macro_context",
    "build_macro_features",
    "cadence_change",
    "causal_align",
    "summarize_gex_contracts",
]
