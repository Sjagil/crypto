"""Formal, deterministic coverage registry for the supplied indicator masterlist.

The registry is deliberately broader than the executable feature frame.  An item
is never silently treated as implemented: every source item receives exactly one
coverage status and complete operational metadata.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from utils.common import stable_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTERLIST_GLOB = "indicator_masterlist_*.json"
TECHNICAL_TIMEFRAMES = (
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "1W",
    "1M",
)


class CoverageStatus(StrEnum):
    IMPLEMENTED = "IMPLEMENTED"
    IMPLEMENTED_AS_ALIAS = "IMPLEMENTED_AS_ALIAS"
    DERIVED_FROM_EXISTING_FEATURES = "DERIVED_FROM_EXISTING_FEATURES"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    INVESTING_ONLY = "INVESTING_ONLY"
    DATA_PROVIDER_REQUIRED = "DATA_PROVIDER_REQUIRED"
    DATA_CURRENTLY_UNAVAILABLE = "DATA_CURRENTLY_UNAVAILABLE"
    NOT_CAUSAL_AND_REJECTED = "NOT_CAUSAL_AND_REJECTED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    NOT_APPLICABLE_TO_CRYPTO_SPOT = "NOT_APPLICABLE_TO_CRYPTO_SPOT"


class IndicatorRole(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    FILTER = "FILTER"
    SIZING = "SIZING"
    REGIME = "REGIME"
    REPORTING = "REPORTING"
    LABEL = "LABEL"


@dataclass(frozen=True)
class IndicatorParameter:
    name: str
    parameter_type: str
    minimum: int | float | str | None
    maximum: int | float | str | None
    default: int | float | str | bool | None
    step: int | float | str | None = None
    integer_only: bool = False


@dataclass(frozen=True)
class IndicatorDefinition:
    canonical_name: str
    display_name: str
    family: str
    subfamily: str
    description: str
    input_columns: tuple[str, ...]
    minimum_history: int
    supported_timeframes: tuple[str, ...]
    output_columns: tuple[str, ...]
    parameters: tuple[IndicatorParameter, ...]
    causality_lag_bars: int
    warmup_bars: int
    repaints: bool
    tradable: bool
    research_only: bool
    investing_only: bool
    external_data_required: bool
    provider_requirements: tuple[str, ...]
    missing_data_policy: str
    normalization: str
    redundancy_group: str
    compatible_roles: tuple[IndicatorRole, ...]
    primary_role: IndicatorRole
    applicable_assets: tuple[str, ...]
    unit: str
    expected_cadence: str
    version: str
    status: CoverageStatus
    alias_of: str | None = None
    combinable: bool = False
    approximation: bool = False
    source_lines: tuple[int, ...] = ()
    configuration_hash: str = ""

    def with_hash(self) -> "IndicatorDefinition":
        payload = asdict(replace(self, configuration_hash=""))
        return replace(self, configuration_hash=stable_hash(payload, length=64))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.casefold().replace("+", " plus ").replace("%", " percent ")
    return re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_")


_EXTERNAL_FAMILIES = frozenset(
    {
        "DERIVATIVES",
        "OPTIONS",
        "ON_CHAIN",
        "STABLECOIN",
        "CRYPTO_MACRO",
        "SENTIMENT",
        "INSTITUTIONAL_FLOW",
        "DEFI_CHAIN_FUNDAMENTALS",
        "INVESTING_FUNDAMENTALS",
        "HOLDER_DISTRIBUTION",
        "GLOBAL_MACRO",
        "EVENTS",
    }
)
_INVESTING_FAMILIES = frozenset(
    {
        "DEFI_CHAIN_FUNDAMENTALS",
        "INVESTING_FUNDAMENTALS",
        "HOLDER_DISTRIBUTION",
        "INVESTING_SCORE",
    }
)
_CONTEXT_FAMILIES = frozenset(
    {
        "DERIVATIVES",
        "OPTIONS",
        "ON_CHAIN",
        "STABLECOIN",
        "CRYPTO_MACRO",
        "SENTIMENT",
        "INSTITUTIONAL_FLOW",
        "GLOBAL_MACRO",
        "EVENTS",
        "STATISTICS_CYCLES",
        "FRACTAL",
    }
)
_PROVIDERS: dict[str, tuple[str, ...]] = {
    "DERIVATIVES": ("mexc", "coinglass_or_equivalent"),
    "OPTIONS": ("deribit",),
    "ON_CHAIN": ("glassnode_cryptoquant_or_equivalent",),
    "STABLECOIN": ("defillama_or_equivalent",),
    "CRYPTO_MACRO": ("coinmarketcap",),
    "SENTIMENT": ("alternative_me_rss_forward_only",),
    "INSTITUTIONAL_FLOW": ("point_in_time_flow_provider",),
    "DEFI_CHAIN_FUNDAMENTALS": ("defillama_or_equivalent",),
    "INVESTING_FUNDAMENTALS": ("coinmarketcap_and_manual_research",),
    "HOLDER_DISTRIBUTION": ("chain_specific_provider",),
    "GLOBAL_MACRO": ("fred_eodhd_or_equivalent",),
    "EVENTS": ("point_in_time_calendar_provider",),
}

# Names with a concrete, causal implementation in the active feature, data,
# backtest, risk, or reporting stack.
_IMPLEMENTED = frozenset(
    canonical_slug(value)
    for value in (
        "Open|High|Low|Close|Typical price: (High + Low + Close) / 3|"
        "Median price: (High + Low) / 2|Weighted close: (High + Low + 2 × Close) / 4|"
        "OHLC average: (Open + High + Low + Close) / 4|Absolute return|Percentage return|"
        "Log return|Cumulative return|Rolling return|Relative return tegenover BTC|"
        "Relative strength tegenover een benchmark|SMA: Simple Moving Average|"
        "EMA: Exponential Moving Average|RMA: Wilder’s Moving Average|"
        "MA slope|MA separation|Distance from moving average|Moving-average compression|"
        "ADX: Average Directional Index|DMI|+DI|-DI|Aroon Up|Aroon Down|Aroon Oscillator|"
        "Supertrend|Donchian Channels|Choppiness Index|RSI|Stochastic RSI|MACD|"
        "MACD Histogram|PPO: Percentage Price Oscillator|Rate of Change|ROC percentage|"
        "CCI: Commodity Channel Index|Williams %R|Connors RSI|True Range|ATR|"
        "NATR: Normalized ATR|Bollinger Bands|Bollinger Bandwidth|Bollinger %B|"
        "Keltner Channels|Standard deviation|Variance|Historical volatility|"
        "Realized volatility|EWMA volatility|Parkinson volatility|Garman-Klass volatility|"
        "Rogers-Satchell volatility|Yang-Zhang volatility|Bollinger Band squeeze|"
        "Bollinger-Keltner squeeze|Spot volume|Quote volume|Base-asset volume|"
        "Dollar volume|Volume moving average|Relative volume|Volume z-score|OBV: On-Balance Volume|"
        "Chaikin Money Flow|Money Flow Index|VWAP|Rolling VWAP|Anchored VWAP|"
        "Bid-ask spread|Percentage spread|Effective spread|Market depth|Bid liquidity|Ask liquidity|"
        "Order-book imbalance|Weighted order-book imbalance|Microprice|Midprice|Book pressure|"
        "Order-book slope|Liquidity gaps|Slippage estimate|Market impact estimate|"
        "Swing high|Swing low|Higher high|Higher low|Lower high|Lower low|Break of Structure|"
        "Change of Character|Confirmed fractals|Williams fractals|Pivot highs|Pivot lows|"
        "Equal highs|Equal lows|Liquidity sweep|Failed breakout|Fair Value Gap|"
        "Imbalance zones|Supply zones|Demand zones|Hammer|Inverted Hammer|Hanging Man|"
        "Shooting Star|Bullish Engulfing|Bearish Engulfing|Morning Star|Evening Star|"
        "Three White Soldiers|Three Black Crows|Harami|Doji|Spinning Top|High-Wave Candle|"
        "Rising Three Methods|Falling Three Methods|Open Interest|Open Interest change|"
        "Funding rate|Perpetual premium|Futures basis|Annualized basis|Contango|Backwardation|"
        "Long liquidations|Short liquidations|Liquidation imbalance|Implied Volatility|"
        "Realized Volatility|Gamma exposure|Expected move|Mean|Median|Z-score|Rolling z-score|"
        "Quantiles|Percentiles|Skewness|Kurtosis|Correlation|Covariance|Beta|Alpha|"
        "Linear regression|R-squared|Regression residual|Autocorrelation|Cross-sectional rank|"
        "Total return|Annualized return|CAGR|Maximum Drawdown|Drawdown duration|"
        "Time under water|Standard deviation|Downside deviation|Probability of ruin|"
        "Sharpe Ratio|Sortino Ratio|Calmar Ratio|Profit Factor|Gross Profit|Gross Loss|"
        "Win Rate|Loss Rate|Average Win|Average Loss|Payoff Ratio|Expectancy|Median Trade|"
        "Average Trade|Maximum Consecutive Losses|Maximum Consecutive Wins|Holding Time|"
        "MAE: Maximum Adverse Excursion|MFE: Maximum Favorable Excursion|Exposure|Turnover|"
        "Transaction costs|Slippage|In-sample performance|Out-of-sample performance|"
        "Walk-forward performance|Fold consistency|Monte Carlo probability of loss|"
        "Monte Carlo drawdown|Parameter stability|Sensitivity analysis|Effective number of trades|"
        "Regime robustness|Asset robustness|Fee sensitivity|Slippage sensitivity|"
        "3-candle confirmed fractal|5-candle confirmed fractal|7-candle confirmed fractal|"
        "fractal pivot timestamp|fractal confirmation timestamp|bars since fractal high|"
        "bars since fractal low|distance to fractal high price|distance to fractal high percent|"
        "distance to fractal high ATR|distance to fractal low price|distance to fractal low percent|"
        "distance to fractal low ATR|fractal high breakout|fractal low breakdown|"
        "bullish fractal sweep|bearish fractal sweep|higher fractal high|higher fractal low|"
        "lower fractal high|lower fractal low|bullish fractal BOS|bearish fractal BOS|"
        "bullish fractal CHoCH|bearish fractal CHoCH|fractal range position|fractal density|"
        "fractal amplitude ATR|fractal trend score|multi-timeframe fractal alignment|"
        "fractal breakout volume confirmation|fractal significance score|"
        "fractal structure trailing stop|fractal retest entry|Higuchi Fractal Dimension|"
        "Katz Fractal Dimension|Petrosian Fractal Dimension|Fractal Dimension Index|"
        "rolling Hurst exponent|FRAMA"
    ).split("|")
)
_DERIVED = frozenset(
    canonical_slug(value)
    for value in (
        "Excess return|Risk-adjusted return|Rolling alpha|Rolling beta|"
        "moving-average expansion|9/21 EMA crossover|20/50 EMA crossover|"
        "50/200 SMA crossover|Golden cross|Death cross|Price above/below 200-day SMA|"
        "Momentum|RSI slope|Momentum acceleration|Momentum deceleration|"
        "Volatility percentile|Volatility z-score|Volatility contraction|"
        "Volatility expansion|ATR stop|ATR trailing stop|Position sizing op basis van ATR|"
        "Volatility targeting|Volume percentile|Total return|Monthly return|"
        "Rolling return|Excess return|Benchmark-relative return|Average Drawdown|"
        "Worst day/week/month|Exit efficiency|Capture ratio|Break-even fee|"
        "Break-even slippage"
    ).split("|")
)
_ALIASES = {
    canonical_slug("SMMA: Smoothed Moving Average"): canonical_slug(
        "RMA: Wilder’s Moving Average"
    ),
    canonical_slug("Price Volume Trend"): canonical_slug("Volume Price Trend"),
    canonical_slug("ROC percentage"): canonical_slug("Rate of Change"),
    canonical_slug("Annualized issuance"): canonical_slug("Issuance"),
}
_MANUAL_REVIEW_TERMS = (
    "order block",
    "breaker block",
    "mitigation block",
    "smart-money",
    "whale positioning",
    "liquidity pools",
    "stop clusters",
    "liquidity walls",
    "iceberg detection",
)
_NON_CAUSAL_LABELS = {
    canonical_slug("fractal efficiency"),
    canonical_slug("post-fractal MFE"),
    canonical_slug("post-fractal MAE"),
}
_UNAVAILABLE = {
    canonical_slug("Jurik Moving Average"),
    canonical_slug("Dealer positioning"),
    canonical_slug("Probability distribution uit options pricing"),
}
_NOT_SPOT = {
    canonical_slug("Earnings-anchored VWAP"),
    canonical_slug("Margin borrowing ratio"),
}


def _source_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted((PROJECT_ROOT / "config").glob(MASTERLIST_GLOB)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items.extend(payload["items"])
    if not items:
        raise FileNotFoundError("indicator masterlist catalog is missing")
    return items


def _default_inputs(family: str) -> tuple[str, ...]:
    if family in {"PRICE_RETURNS", "TREND", "MOMENTUM", "VOLATILITY", "FRACTAL"}:
        return ("open", "high", "low", "close", "volume")
    if family in {"VOLUME_FLOW", "MARKET_STRUCTURE", "CANDLESTICK"}:
        return ("open", "high", "low", "close", "volume")
    if family == "ORDERFLOW_MICROSTRUCTURE":
        return ("trades_or_l2_orderbook",)
    if family == "PERFORMANCE_RISK":
        return ("trades", "equity_curve")
    return ("provider_value", "observed_at", "usable_at")


def _provider_requirements(family: str, slug: str) -> tuple[str, ...]:
    if family == "ORDERFLOW_MICROSTRUCTURE":
        return ("l2_or_trade_stream",)
    if "open_interest" in slug and family == "FRACTAL":
        return ("derivatives_context",)
    return _PROVIDERS.get(family, ())


def _status(family: str, display_name: str) -> CoverageStatus:
    slug = canonical_slug(display_name)
    lower = display_name.casefold()
    if slug in _ALIASES:
        return CoverageStatus.IMPLEMENTED_AS_ALIAS
    if slug in _NON_CAUSAL_LABELS:
        return CoverageStatus.NOT_CAUSAL_AND_REJECTED
    if slug in _UNAVAILABLE:
        return CoverageStatus.DATA_CURRENTLY_UNAVAILABLE
    if slug in _NOT_SPOT:
        return CoverageStatus.NOT_APPLICABLE_TO_CRYPTO_SPOT
    if any(term in lower for term in _MANUAL_REVIEW_TERMS):
        return CoverageStatus.MANUAL_REVIEW_REQUIRED
    if family in _INVESTING_FAMILIES:
        return CoverageStatus.INVESTING_ONLY
    if slug in _IMPLEMENTED:
        return CoverageStatus.IMPLEMENTED
    if slug in _DERIVED or family == "PERFORMANCE_RISK":
        return CoverageStatus.DERIVED_FROM_EXISTING_FEATURES
    if family in _EXTERNAL_FAMILIES:
        return CoverageStatus.DATA_PROVIDER_REQUIRED
    return CoverageStatus.RESEARCH_ONLY


def _primary_role(family: str, status: CoverageStatus) -> IndicatorRole:
    if status is CoverageStatus.NOT_CAUSAL_AND_REJECTED:
        return IndicatorRole.LABEL
    if family in {"PERFORMANCE_RISK", "INVESTING_SCORE"}:
        return IndicatorRole.REPORTING
    if family in {"VOLATILITY", "ORDERFLOW_MICROSTRUCTURE"}:
        return IndicatorRole.SIZING
    if family in _CONTEXT_FAMILIES:
        return IndicatorRole.REGIME
    return IndicatorRole.FILTER


def _parameter_for(family: str, display_name: str) -> tuple[IndicatorParameter, ...]:
    if family in {
        "PRICE_RETURNS",
        "TREND",
        "MOMENTUM",
        "VOLATILITY",
        "VOLUME_FLOW",
        "MARKET_STRUCTURE",
        "STATISTICS_CYCLES",
    }:
        return (
            IndicatorParameter(
                name="period",
                parameter_type="INTEGER",
                minimum=2,
                maximum=500,
                default=20,
                step=1,
                integer_only=True,
            ),
        )
    if family == "FRACTAL":
        window = 5
        if display_name.startswith("3-candle"):
            window = 3
        elif display_name.startswith("7-candle"):
            window = 7
        return (
            IndicatorParameter(
                name="window",
                parameter_type="ODD_INTEGER",
                minimum=3,
                maximum=7,
                default=window,
                step=2,
                integer_only=True,
            ),
        )
    return ()


class IndicatorRegistry:
    """Validated item-level registry plus programmatic coverage reports."""

    def __init__(self, definitions: Iterable[IndicatorDefinition]) -> None:
        materialized = tuple(definitions)
        self._definitions = {item.canonical_name: item for item in materialized}
        if len(self._definitions) != len(materialized):
            raise ValueError("duplicate canonical indicator names")
        self.validate()

    @classmethod
    def build(cls) -> "IndicatorRegistry":
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for item in _source_items():
            family = str(item["family"])
            display = str(item["display_name"]).strip()
            slug = canonical_slug(display)
            key = (family, slug)
            if key not in grouped:
                grouped[key] = {"display": display, "lines": []}
            if item.get("source_line") is not None:
                grouped[key]["lines"].append(int(item["source_line"]))

        definitions: list[IndicatorDefinition] = []
        for (family, slug), source in sorted(grouped.items()):
            status = _status(family, source["display"])
            external = family in _EXTERNAL_FAMILIES or family == "ORDERFLOW_MICROSTRUCTURE"
            roles = (IndicatorRole.REGIME, IndicatorRole.FILTER)
            if family not in _CONTEXT_FAMILIES and family not in _INVESTING_FAMILIES:
                roles = (
                    IndicatorRole.ENTRY,
                    IndicatorRole.EXIT,
                    IndicatorRole.FILTER,
                    IndicatorRole.REGIME,
                    IndicatorRole.SIZING,
                )
            primary = _primary_role(family, status)
            tradable = status in {
                CoverageStatus.IMPLEMENTED,
                CoverageStatus.DERIVED_FROM_EXISTING_FEATURES,
            } and family not in {
                "PERFORMANCE_RISK",
                "INVESTING_SCORE",
                "INVESTING_FUNDAMENTALS",
                "DEFI_CHAIN_FUNDAMENTALS",
                "HOLDER_DISTRIBUTION",
            }
            if family in {"DERIVATIVES", "OPTIONS", "EVENTS", "SENTIMENT"}:
                tradable = False
            lag = 0
            if family == "FRACTAL" and status is CoverageStatus.IMPLEMENTED:
                lag = 1 if source["display"].startswith("3-candle") else (
                    3 if source["display"].startswith("7-candle") else 2
                )
            alias_target_slug = _ALIASES.get(slug)
            alias_of = (
                f"{family.casefold()}.{alias_target_slug}"
                if alias_target_slug is not None
                else None
            )
            parameters = _parameter_for(family, source["display"])
            definition = IndicatorDefinition(
                canonical_name=f"{family.casefold()}.{slug}",
                display_name=source["display"],
                family=family,
                subfamily=slug.split("_", 1)[0] or family.casefold(),
                description=(
                    f"Coverage specification item: {source['display']}. "
                    f"Operational status is {status.value}."
                ),
                input_columns=_default_inputs(family),
                minimum_history=max(
                    [int(parameter.default) for parameter in parameters if parameter.default]
                    or [1]
                ),
                supported_timeframes=TECHNICAL_TIMEFRAMES,
                output_columns=(slug,),
                parameters=parameters,
                causality_lag_bars=lag,
                warmup_bars=max(
                    [int(parameter.default) for parameter in parameters if parameter.default]
                    or [0]
                ),
                repaints=False,
                tradable=tradable,
                research_only=status is CoverageStatus.RESEARCH_ONLY,
                investing_only=status is CoverageStatus.INVESTING_ONLY,
                external_data_required=external,
                provider_requirements=_provider_requirements(family, slug),
                missing_data_policy="NAN_WITH_AVAILABILITY" if external else "WARMUP_NAN",
                normalization="SOURCE_UNIT" if external else "NONE_OR_ROLLING",
                redundancy_group=f"{family.casefold()}:{slug.split('_', 1)[0]}",
                compatible_roles=roles,
                primary_role=primary,
                applicable_assets=("CRYPTO_SPOT",),
                unit="provider_declared" if external else "dimensionless_or_price",
                expected_cadence="provider_declared" if external else "candle_close",
                version="1.0.0",
                status=status,
                alias_of=alias_of,
                combinable=tradable and status is not CoverageStatus.IMPLEMENTED_AS_ALIAS,
                approximation="proxy" in slug or "profile" in slug,
                source_lines=tuple(sorted(set(source["lines"]))),
            ).with_hash()
            definitions.append(definition)
        return cls(definitions)

    def validate(self) -> None:
        for name, item in self._definitions.items():
            if name != item.canonical_name:
                raise ValueError("registry key/canonical name mismatch")
            if item.configuration_hash != item.with_hash().configuration_hash:
                raise ValueError(f"non-deterministic configuration hash: {name}")
            if item.alias_of and item.alias_of not in self._definitions:
                raise ValueError(f"alias target does not exist: {name} -> {item.alias_of}")
            if item.repaints and item.tradable:
                raise ValueError(f"repainting item cannot be tradable: {name}")
            if item.research_only and item.tradable:
                raise ValueError(f"research-only item cannot be tradable: {name}")
            if item.investing_only and item.tradable:
                raise ValueError(f"investing-only item cannot be tradable: {name}")

    def __len__(self) -> int:
        return len(self._definitions)

    def definitions(self) -> tuple[IndicatorDefinition, ...]:
        return tuple(self._definitions[name] for name in sorted(self._definitions))

    def get(self, canonical_name: str) -> IndicatorDefinition:
        return self._definitions[canonical_name]

    def resolve(self, canonical_name: str) -> IndicatorDefinition:
        item = self.get(canonical_name)
        return self.get(item.alias_of) if item.alias_of else item

    def report(self) -> dict[str, Any]:
        definitions = self.definitions()
        by_family = Counter(item.family for item in definitions)
        by_status = Counter(item.status.value for item in definitions)
        redundancy: dict[str, list[str]] = defaultdict(list)
        for item in definitions:
            redundancy[item.redundancy_group].append(item.canonical_name)
        operational = {
            CoverageStatus.IMPLEMENTED,
            CoverageStatus.IMPLEMENTED_AS_ALIAS,
            CoverageStatus.DERIVED_FROM_EXISTING_FEATURES,
        }
        coverage_map = {
            CoverageStatus.IMPLEMENTED: "IMPLEMENTED_AND_REGISTERED",
            CoverageStatus.IMPLEMENTED_AS_ALIAS: "IMPLEMENTED_AND_REGISTERED",
            CoverageStatus.DERIVED_FROM_EXISTING_FEATURES: "IMPLEMENTED_AND_REGISTERED",
            CoverageStatus.DATA_PROVIDER_REQUIRED: "DATA_PROVIDER_MISSING",
            CoverageStatus.DATA_CURRENTLY_UNAVAILABLE: "DATA_PROVIDER_MISSING",
            CoverageStatus.MANUAL_REVIEW_REQUIRED: "FORMULA_NOT_OBJECTIVE",
            CoverageStatus.NOT_APPLICABLE_TO_CRYPTO_SPOT: "OUT_OF_SCOPE",
            CoverageStatus.NOT_CAUSAL_AND_REJECTED: "UNSUPPORTED_WITH_REASON",
            CoverageStatus.RESEARCH_ONLY: "UNSUPPORTED_WITH_REASON",
            CoverageStatus.INVESTING_ONLY: "OUT_OF_SCOPE",
        }
        coverage_rows = [
            {
                "indicator_id": item.canonical_name,
                "display_name": item.display_name,
                "family": item.family,
                "coverage_status": coverage_map[item.status],
                "registry_status": item.status.value,
                "implemented": item.status in operational,
                "registered": True,
                "tradable": item.tradable,
                "combinable": item.combinable,
                "provider_requirements": list(item.provider_requirements),
                "reason": item.description,
                "configuration_hash": item.configuration_hash,
            }
            for item in definitions
        ]
        coverage_counts = Counter(row["coverage_status"] for row in coverage_rows)
        return {
            "registry_version": "1.0.0",
            "registry_hash": stable_hash(
                [item.configuration_hash for item in definitions], length=64
            ),
            "source_item_occurrences": len(_source_items()),
            "unique_canonical_indicators": len(definitions),
            "counts_by_family": dict(sorted(by_family.items())),
            "counts_by_status": dict(sorted(by_status.items())),
            "counts_by_coverage_status": dict(sorted(coverage_counts.items())),
            "coverage_rows": coverage_rows,
            "operational": [
                item.canonical_name for item in definitions if item.status in operational
            ],
            "missing_external_data": [
                item.canonical_name
                for item in definitions
                if item.external_data_required
                and item.status is CoverageStatus.DATA_PROVIDER_REQUIRED
            ],
            "aliases": {
                item.canonical_name: item.alias_of
                for item in definitions
                if item.alias_of is not None
            },
            "non_combinable": [
                item.canonical_name for item in definitions if not item.combinable
            ],
            "causality_or_repainting_blocked": [
                item.canonical_name
                for item in definitions
                if item.repaints
                or item.status is CoverageStatus.NOT_CAUSAL_AND_REJECTED
            ],
            "redundancy_groups": {
                group: names
                for group, names in sorted(redundancy.items())
                if len(names) > 1
            },
        }


def indicator_registry() -> IndicatorRegistry:
    return IndicatorRegistry.build()


def indicator_coverage_report() -> dict[str, Any]:
    return indicator_registry().report()


__all__ = [
    "CoverageStatus",
    "IndicatorDefinition",
    "IndicatorParameter",
    "IndicatorRegistry",
    "IndicatorRole",
    "canonical_slug",
    "indicator_coverage_report",
    "indicator_registry",
]
