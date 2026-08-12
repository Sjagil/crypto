"""Data-only derivatives context and transparent crypto GEX proxies."""

from __future__ import annotations

import math
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.contracts import NormalizedDataRecord, require_utc
from utils.common import sha256_text, stable_json, utc_now

JsonRequester = Callable[
    [str, str, Mapping[str, Any] | None, Mapping[str, str] | None],
    Awaitable[Any],
]
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
MEXC_CONTRACT_REST = "https://contract.mexc.com/api/v1/contract"
DERIBIT_REST = "https://www.deribit.com/api/v2/public"


def annualize_funding(rate: float, interval_seconds: float) -> float:
    if interval_seconds <= 0:
        raise ValueError("funding interval must be positive")
    return float(rate) * SECONDS_PER_YEAR / float(interval_seconds)


class OptionsContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    underlying: str
    expiry: datetime
    strike: float = Field(gt=0)
    option_type: Literal["call", "put"]
    spot_or_index_price: float = Field(gt=0)
    open_interest: float = Field(ge=0)
    gamma: float = Field(ge=0)
    contract_multiplier: float = Field(gt=0)
    gamma_source: str = "DERIVED_FROM_DERIBIT_MARK_IV"
    flow_direction_score: float | None = Field(default=None, ge=-1, le=1)
    flow_confidence: float | None = Field(default=None, ge=0, le=1)
    observed_at: datetime
    available_at: datetime
    stale: bool = False

    _expiry = field_validator("expiry")(require_utc)
    _observed = field_validator("observed_at")(require_utc)
    _available = field_validator("available_at")(require_utc)


class FundingRateCollector:
    """Public context collector. No order or authenticated endpoint exists here."""

    def __init__(self, *, requester: JsonRequester) -> None:
        self.request = requester

    async def collect(
        self,
        *,
        provider: str = "mexc",
        market: str = "BTC-USDT",
        run_id: str | None = None,
    ) -> list[NormalizedDataRecord]:
        if provider.casefold() != "mexc":
            raise ValueError("public derivatives context currently supports MEXC")
        run = run_id or str(uuid.uuid4())
        observed = utc_now()
        source_symbol = market.replace("-", "_").upper()
        funding, open_interest, ticker = await self._download(source_symbol)
        interval_hours = float(
            funding.get("collectCycle")
            or funding.get("fundingInterval")
            or ticker.get("fundingRateInterval")
            or 8
        )
        interval_seconds = interval_hours * 3_600
        rate = float(funding.get("fundingRate") or ticker.get("fundingRate") or 0)
        mark = float(ticker.get("fairPrice") or ticker.get("markPrice") or 0)
        index = float(ticker.get("indexPrice") or 0)
        timestamp_value = (
            funding.get("timestamp") or ticker.get("timestamp") or observed.timestamp() * 1_000
        )
        timestamp = datetime.fromtimestamp(float(timestamp_value) / 1_000, tz=UTC)
        values = {
            "event_time": timestamp.isoformat(),
            "arrival_time": observed.isoformat(),
            "source_available_at": observed.isoformat(),
            "source": "mexc_public_contract_context",
            "funding_rate": rate,
            "funding_interval_seconds": interval_seconds,
            "funding_periods_per_year": SECONDS_PER_YEAR / interval_seconds,
            "annualized_funding": annualize_funding(rate, interval_seconds),
            "open_interest": float(
                open_interest.get("holdVol")
                or open_interest.get("openInterest")
                or ticker.get("holdVol")
                or 0
            ),
            "perpetual_base_volume_24h": float(
                ticker.get("volume24") or 0
            ),
            "perpetual_quote_volume_24h": float(
                ticker.get("amount24") or 0
            ),
            "mark_price": mark,
            "index_price": index,
            "perpetual_premium": (mark / index - 1) if mark and index else None,
            "basis": (mark - index) if mark and index else None,
            "long_liquidations": None,
            "short_liquidations": None,
            "liquidation_imbalance": None,
            "liquidation_status": "UNAVAILABLE_PUBLIC_ENDPOINT",
            "execution_permitted": False,
        }
        raw = {"funding": funding, "open_interest": open_interest, "ticker": ticker}
        return [
            NormalizedDataRecord(
                provider="mexc",
                source_symbol=source_symbol,
                canonical_market=market,
                timestamp=timestamp,
                observed_at=observed,
                available_at=observed,
                data_kind="derivatives_context",
                retrieval_run_id=run,
                raw_hash=sha256_text(stable_json(raw)),
                raw_payload=raw,
                values=values,
            )
        ]

    async def _download(self, symbol: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        funding_raw = await self.request(
            "GET", f"{MEXC_CONTRACT_REST}/funding_rate/{symbol}", None, None
        )
        ticker_raw = await self.request(
            "GET", f"{MEXC_CONTRACT_REST}/ticker", {"symbol": symbol}, None
        )
        # MEXC exposes aggregate open interest as ``holdVol`` on the documented
        # public ticker response; there is no public ``open_interest`` route.
        ticker = dict(ticker_raw.get("data") or {})
        return (
            dict(funding_raw.get("data") or {}),
            {"holdVol": ticker.get("holdVol")},
            ticker,
        )


class CryptoGEXAnalyzer:
    """Calculates unsigned gross GEX and an explicitly heuristic signed proxy.

    Gross convention: gamma * OI * multiplier * spot^2 * 1%. The signed
    heuristic assigns calls positive and puts negative. It is not a claim about
    dealer positioning.
    """

    assumption_metadata = {
        "gross_convention": "gamma * open_interest * contract_multiplier * spot^2 * 0.01",
        "signed_heuristic": "calls positive; puts negative",
        "dealer_positioning_known": False,
        "flow_adjusted_convention": (
            "gross_gex * observed_flow_direction_score * flow_confidence; "
            "only emitted when both inputs are observed"
        ),
        "warning": (
            "Open interest does not reveal dealer direction; signed GEX is a "
            "research heuristic, not observed dealer gamma."
        ),
        "execution_permitted": False,
    }

    def calculate(
        self,
        contracts: Iterable[OptionsContract | Mapping[str, Any]],
        *,
        stale_after: timedelta = timedelta(hours=2),
        now: datetime | None = None,
    ) -> dict[str, Any]:
        rows = [
            item.model_dump() if isinstance(item, OptionsContract) else dict(item)
            for item in contracts
        ]
        if not rows:
            return {
                "status": "SKIPPED",
                "reason_code": "NO_OPTIONS_CHAIN",
                "assumptions": self.assumption_metadata,
            }
        frame = pd.DataFrame(rows)
        observed_now = now or utc_now()
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
        frame["stale"] = frame["stale"].astype(bool) | (
            observed_now - frame["available_at"] > stale_after
        )
        frame["gross_gex"] = (
            frame["gamma"].astype(float)
            * frame["open_interest"].astype(float)
            * frame["contract_multiplier"].astype(float)
            * frame["spot_or_index_price"].astype(float).pow(2)
            * 0.01
        )
        frame["signed_gex"] = np.where(
            frame["option_type"].str.casefold().eq("call"),
            frame["gross_gex"],
            -frame["gross_gex"],
        )
        flow_score = pd.to_numeric(
            frame.get("flow_direction_score"), errors="coerce"
        )
        flow_confidence = pd.to_numeric(
            frame.get("flow_confidence"), errors="coerce"
        )
        frame["flow_adjusted_gex"] = (
            frame["gross_gex"] * flow_score * flow_confidence
        )
        frame["expiry"] = pd.to_datetime(frame["expiry"], utc=True)
        frame["hours_to_expiry"] = (
            frame["expiry"] - pd.Timestamp(observed_now)
        ).dt.total_seconds() / 3_600.0
        by_strike = (
            frame.groupby("strike", as_index=False)[
                ["gross_gex", "signed_gex", "flow_adjusted_gex"]
            ]
            .sum()
            .sort_values("strike")
        )
        by_expiry = (
            frame.groupby("expiry", as_index=False)[
                ["gross_gex", "signed_gex", "flow_adjusted_gex"]
            ]
            .sum()
            .sort_values("expiry")
        )
        calls = float(frame.loc[frame["option_type"].eq("call"), "gross_gex"].sum())
        puts = float(frame.loc[frame["option_type"].eq("put"), "gross_gex"].sum())
        gross = calls + puts
        net = calls - puts
        flow_observed = bool(
            frame["flow_adjusted_gex"].notna().all()
            and len(frame)
        )
        flow_adjusted = (
            float(frame["flow_adjusted_gex"].sum())
            if flow_observed
            else None
        )
        dominant = by_strike.loc[by_strike["gross_gex"].idxmax()]
        nearest_expiry = pd.to_datetime(frame["expiry"], utc=True).min()
        nearest = float(
            frame.loc[
                pd.to_datetime(frame["expiry"], utc=True).eq(nearest_expiry),
                "gross_gex",
            ].sum()
        )
        spot = float(frame["spot_or_index_price"].iloc[-1])
        calls_above = frame.loc[
            frame["option_type"].eq("call") & frame["strike"].ge(spot)
        ]
        puts_below = frame.loc[
            frame["option_type"].eq("put") & frame["strike"].le(spot)
        ]

        def wall(selected: pd.DataFrame) -> float | None:
            if selected.empty:
                return None
            grouped = selected.groupby("strike")["gross_gex"].sum()
            return float(grouped.idxmax())

        near_spot = frame.loc[
            frame["strike"].between(spot * 0.98, spot * 1.02)
        ]

        def horizon_gex(minimum_hours: float, maximum_hours: float) -> dict[str, float]:
            selected = frame.loc[
                frame["hours_to_expiry"].gt(minimum_hours)
                & frame["hours_to_expiry"].le(maximum_hours)
            ]
            return {
                "absolute_gex": float(selected["gross_gex"].sum()),
                "convention_signed_gex": float(selected["signed_gex"].sum()),
                "contract_count": int(len(selected)),
            }
        signs = np.sign(by_strike["signed_gex"].to_numpy())
        flip_indices = np.where(np.diff(signs) != 0)[0]
        flip = None
        if flip_indices.size:
            index = int(flip_indices[np.argmin(np.abs(
                by_strike.iloc[flip_indices]["strike"].to_numpy() - spot
            ))])
            left, right = by_strike.iloc[index], by_strike.iloc[index + 1]
            denominator = abs(left["signed_gex"]) + abs(right["signed_gex"])
            if denominator:
                flip = float(
                    left["strike"]
                    + (right["strike"] - left["strike"])
                    * abs(left["signed_gex"])
                    / denominator
                )
        return {
            "status": "PARTIAL" if bool(frame["stale"].any()) else "PASSED",
            "absolute_gex": gross,
            "convention_signed_gex": net,
            "call_gex_proxy": calls,
            "put_gex_proxy": puts,
            "gross_gex_proxy": gross,
            "net_gex_proxy": net,
            "flow_adjusted_gex": flow_adjusted,
            "flow_adjusted_status": (
                "AVAILABLE" if flow_observed else "UNAVAILABLE_MISSING_OPTION_FLOW"
            ),
            "gex_by_strike": by_strike.to_dict(orient="records"),
            "gex_by_expiry": by_expiry.assign(
                expiry=lambda value: value["expiry"].astype(str)
            ).to_dict(orient="records"),
            "nearest_expiry_concentration": nearest / gross if gross else 0.0,
            "dominant_gamma_strike": float(dominant["strike"]),
            "max_gamma_strike": float(dominant["strike"]),
            "call_wall": wall(calls_above),
            "put_wall": wall(puts_below),
            "gamma_concentration": float(dominant["gross_gex"]) / gross if gross else 0.0,
            "gex_concentration_within_2pct": (
                float(near_spot["gross_gex"].sum()) / gross if gross else 0.0
            ),
            "zero_day_gex": horizon_gex(0.0, 24.0),
            "weekly_gex": horizon_gex(0.0, 24.0 * 7.0),
            "monthly_gex": horizon_gex(24.0 * 7.0, 24.0 * 45.0),
            "gamma_flip_proxy": flip,
            "spot_distance_from_dominant_gamma": (
                spot / float(dominant["strike"]) - 1
            ),
            "stale": bool(frame["stale"].any()),
            "contract_count": len(frame),
            "assumptions": self.assumption_metadata,
        }


class DeribitOptionsCollector:
    """Collects public Deribit crypto options; no private method is exposed."""

    def __init__(self, *, requester: JsonRequester) -> None:
        self.request = requester

    async def collect(self, underlying: str = "BTC") -> list[OptionsContract]:
        observed = utc_now()
        currency = underlying.upper()
        instruments_raw = await self.request(
            "GET",
            f"{DERIBIT_REST}/get_instruments",
            {"currency": currency, "kind": "option", "expired": "false"},
            None,
        )
        summary_raw = await self.request(
            "GET",
            f"{DERIBIT_REST}/get_book_summary_by_currency",
            {"currency": currency, "kind": "option"},
            None,
        )
        summaries = {
            item["instrument_name"]: item
            for item in summary_raw.get("result", [])
        }
        result: list[OptionsContract] = []
        for instrument in instruments_raw.get("result", []):
            summary = summaries.get(instrument["instrument_name"], {})
            index_price = summary.get("underlying_price")
            mark_iv = summary.get("mark_iv")
            if mark_iv is None or index_price is None:
                continue
            expiry = datetime.fromtimestamp(
                float(instrument["expiration_timestamp"]) / 1_000, tz=UTC
            )
            years = max((expiry - observed).total_seconds() / SECONDS_PER_YEAR, 1e-9)
            volatility = float(mark_iv) / 100.0
            spot = float(index_price)
            strike = float(instrument["strike"])
            if volatility <= 0:
                continue
            d1 = (
                math.log(spot / strike) + 0.5 * volatility**2 * years
            ) / (volatility * math.sqrt(years))
            gamma = (
                math.exp(-0.5 * d1**2)
                / math.sqrt(2 * math.pi)
                / (spot * volatility * math.sqrt(years))
            )
            result.append(
                OptionsContract(
                    provider="deribit",
                    underlying=currency,
                    expiry=expiry,
                    strike=strike,
                    option_type=(
                        "call" if instrument["option_type"].casefold() == "call" else "put"
                    ),
                    spot_or_index_price=spot,
                    open_interest=float(summary.get("open_interest", 0)),
                    gamma=float(gamma),
                    contract_multiplier=float(instrument.get("contract_size", 1)),
                    observed_at=observed,
                    available_at=observed,
                    stale=False,
                )
            )
        return result


__all__ = [
    "CryptoGEXAnalyzer",
    "DeribitOptionsCollector",
    "FundingRateCollector",
    "OptionsContract",
    "annualize_funding",
]
