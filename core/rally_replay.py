"""Causal forensic replay for broad active-swing market moves."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from core.active_trading import _records_to_ohlcv_frame
from data.data_loader import DataLoader
from data.market_data import drop_open_candles
from utils.common import atomic_write_json, atomic_write_text, utc_now

DEFAULT_REPLAY_MARKETS = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "ADA-EUR",
    "BNB-EUR",
    "BCH-EUR",
    "LTC-EUR",
    "SUI-EUR",
    "TAO-EUR",
    "NPC-EUR",
)


def _load_local(
    settings: Settings,
    market: str,
    timeframe: str,
    *,
    now: datetime,
) -> pd.DataFrame:
    path = settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
    if not isinstance(frame.index, pd.DatetimeIndex):
        return pd.DataFrame()
    frame.index = (
        frame.index.tz_localize("UTC")
        if frame.index.tz is None
        else frame.index.tz_convert("UTC")
    )
    frame = frame.loc[:, ["open", "high", "low", "close", "volume"]]
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna().sort_index()
    return drop_open_candles(frame, timeframe=timeframe, now=now)


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.copy()
    close = selected["close"]
    previous = close.shift(1)
    true_range = pd.concat(
        [
            selected["high"] - selected["low"],
            (selected["high"] - previous).abs(),
            (selected["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    selected["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    selected["ema_20"] = close.ewm(span=20, adjust=False).mean()
    selected["return_15m"] = close.pct_change()
    selected["return_1h"] = close.pct_change(4)
    selected["atr_fraction"] = selected["atr_14"] / close
    selected["normalized_return_1h"] = (
        selected["return_1h"] / selected["atr_fraction"].replace(0, np.nan)
    )
    selected["relative_volume_20"] = selected["volume"] / selected[
        "volume"
    ].shift(1).rolling(20, min_periods=10).median()
    return selected


def _find_replay_candidate(
    market: str,
    feature: pd.DataFrame,
    five_minute: pd.DataFrame,
    breadth: pd.Series,
    btc_return: pd.Series,
    *,
    start: datetime,
    end: datetime,
    roundtrip_cost_fraction: float,
) -> dict[str, Any]:
    intraday = feature.loc[(feature.index >= start) & (feature.index <= end)]
    prior_impulse: tuple[pd.Timestamp, float] | None = None
    impulse_setup: dict[str, Any] | None = None
    setup: dict[str, Any] | None = None
    for timestamp, row in intraday.iterrows():
        market_breadth = float(breadth.get(timestamp, np.nan))
        if not np.isfinite(market_breadth):
            continue
        normalized = float(row.get("normalized_return_1h") or 0.0)
        rvol = float(row.get("relative_volume_20") or 0.0)
        asset_return = float(row.get("return_1h") or 0.0)
        benchmark_return = float(btc_return.get(timestamp, 0.0) or 0.0)
        relative_strength = asset_return - benchmark_return
        if prior_impulse is not None:
            impulse_time, impulse_high = prior_impulse
            bars_since = int((timestamp - impulse_time).total_seconds() // 900)
            atr = float(row.get("atr_14") or 0.0)
            pullback_atr = (
                (impulse_high - float(row["close"])) / atr if atr > 0 else 99.0
            )
            if (
                1 <= bars_since <= 4
                and 0.0 <= pullback_atr <= 0.90
                and float(row.get("return_15m") or 0.0) <= 0.002
                and float(row["close"]) > float(row["ema_20"])
                and market_breadth >= 0.55
            ):
                setup = {
                    "family": "FIRST_PULLBACK_AFTER_IMPULSE_V1",
                    "setup_time": timestamp,
                    "impulse_high": impulse_high,
                    "breadth": market_breadth,
                    "relative_strength_1h": relative_strength,
                }
                break
        impulse = bool(
            normalized >= 0.80
            and rvol >= 1.0
            and market_breadth >= 0.55
            and float(row["close"]) > float(row["ema_20"])
        )
        if impulse:
            prior_impulse = (timestamp, float(row["high"]))
            if relative_strength > 0.0 or market == "BTC-EUR":
                # Keep the first causal impulse as a fallback, but continue
                # observing the next four closed 15m bars for the explicitly
                # requested first controlled pullback.  The previous version
                # broke here and therefore made the pullback branch
                # unreachable in the forensic replay.
                impulse_setup = impulse_setup or {
                    "family": "BROAD_RISK_ON_MOMENTUM_V1",
                    "setup_time": timestamp,
                    "impulse_high": float(row["high"]),
                    "breadth": market_breadth,
                    "relative_strength_1h": relative_strength,
                }
    if setup is None:
        setup = impulse_setup
    if setup is None:
        return {
            "market": market,
            "status": "NO_CAUSAL_REPLAY_SETUP",
            "actual_blocker": "SETUP_NOT_OBSERVED_UNDER_PREREGISTERED_REPLAY_RULES",
        }
    setup_time = pd.Timestamp(setup["setup_time"])
    five = _features(five_minute) if not five_minute.empty else pd.DataFrame()
    trigger_time: pd.Timestamp | None = None
    trigger_price: float | None = None
    if not five.empty:
        window = five.loc[
            (five.index >= setup_time)
            & (five.index <= setup_time + timedelta(minutes=60))
        ]
        for timestamp, row in window.iterrows():
            if (
                float(row.get("return_15m") or 0.0) > 0.0
                and float(row["close"]) > float(row["ema_20"])
            ):
                trigger_time = timestamp
                trigger_price = float(row["close"])
                break
    if trigger_time is None or trigger_price is None:
        return {
            "market": market,
            "status": "SETUP_VALID_TRIGGER_UNAVAILABLE",
            **setup,
            "microstructure_evidence": "HISTORICAL_ORDERFLOW_NOT_AVAILABLE",
            "actual_blocker": "NO_CAUSAL_5M_TRIGGER_IN_EXECUTABLE_WINDOW",
        }
    history = feature.loc[:setup_time].tail(12)
    atr = float(history["atr_14"].iloc[-1])
    stop = float(history["low"].tail(8).min() - 0.20 * atr)
    risk = trigger_price - stop
    if risk <= 0:
        return {
            "market": market,
            "status": "SETUP_INVALID_STOP",
            **setup,
            "trigger_time": trigger_time,
            "actual_blocker": "NO_DEFENSIBLE_STRUCTURAL_STOP",
        }
    target_1 = trigger_price + 1.5 * risk
    target_2 = trigger_price + 2.5 * risk
    ideal_entry = float(history["close"].iloc[-1])
    maximum_entry = min(
        ideal_entry + 0.35 * atr,
        (target_1 + stop) / 2.0,
        (target_2 + 1.5 * stop) / 2.5,
    )
    gross_target_2 = target_2 / trigger_price - 1.0
    net_target_2 = gross_target_2 - roundtrip_cost_fraction
    net_stop = trigger_price / stop - 1.0 + roundtrip_cost_fraction
    net_rr = net_target_2 / net_stop if net_stop > 0 else 0.0
    cost_ratio = (
        roundtrip_cost_fraction / gross_target_2 if gross_target_2 > 0 else 99.0
    )
    economics_pass = bool(
        trigger_price <= maximum_entry
        and net_target_2 > 0
        and net_rr >= 1.5
        and cost_ratio <= 0.25
    )
    future = feature.loc[(feature.index >= trigger_time) & (feature.index <= end)]
    mfe = (
        float(future["high"].max() / trigger_price - 1.0)
        if not future.empty
        else None
    )
    mae = (
        float(future["low"].min() / trigger_price - 1.0)
        if not future.empty
        else None
    )
    edge_consumed = max(
        0.0,
        min(1.0, (trigger_price - ideal_entry) / max(1e-12, target_2 - ideal_entry)),
    )
    return {
        "market": market,
        "status": "REPLAY_ENTRY_READY" if economics_pass else "REPLAY_ECONOMICS_REJECT",
        **setup,
        "earliest_near_entry": setup_time.isoformat(),
        "earliest_realtime_trigger": trigger_time.isoformat(),
        "entry_price": trigger_price,
        "ideal_entry": ideal_entry,
        "maximum_acceptable_entry": maximum_entry,
        "stop": stop,
        "target_1": target_1,
        "target_2": target_2,
        "roundtrip_cost_fraction": roundtrip_cost_fraction,
        "net_target_2": net_target_2,
        "net_reward_to_risk": net_rr,
        "cost_to_target_ratio": cost_ratio,
        "economics_pass": economics_pass,
        "mfe_after_trigger": mfe,
        "mae_after_trigger": mae,
        "edge_consumed_by_trigger": edge_consumed,
        "microstructure_evidence": "HISTORICAL_ORDERFLOW_NOT_AVAILABLE",
        "actual_runtime_blocker": (
            "KNOWN_STALE_15M_AND_CLOSED_CANDLE_EXECUTION_LATENCY_INCIDENT"
        ),
        "counterfactual_is_not_live_fill": True,
    }


async def run_rally_replay(
    settings: Settings,
    *,
    replay_date: date,
    markets: Sequence[str] = DEFAULT_REPLAY_MARKETS,
) -> dict[str, Any]:
    start = datetime.combine(replay_date, time.min, tzinfo=UTC)
    now = utc_now()
    end = min(now, start + timedelta(days=1) - timedelta(microseconds=1))
    loader = DataLoader(settings)
    semaphore = asyncio.Semaphore(6)
    frames: dict[tuple[str, str], pd.DataFrame] = {}

    async def load_one(market: str, timeframe: str) -> None:
        local = _load_local(settings, market, timeframe, now=now)
        intraday = local.loc[(local.index >= start) & (local.index <= end)]
        if timeframe != "5m" and not intraday.empty:
            frames[(market, timeframe)] = local
            return
        try:
            async with semaphore:
                records = await loader.download_ohlcv(
                    provider="bitvavo",
                    market=market,
                    timeframe=timeframe,
                    start=start - timedelta(hours=6),
                    end=end,
                    resume=False,
                    persist=False,
                )
            frame = _records_to_ohlcv_frame(
                records,
                market=market,
                timeframe=timeframe,
                now=now,
            )
            if not frame.empty:
                frames[(market, timeframe)] = frame
        except Exception:
            if not local.empty:
                frames[(market, timeframe)] = local

    await asyncio.gather(
        *(load_one(market, timeframe) for market in markets for timeframe in ("5m", "15m"))
    )
    feature_frames = {
        market: _features(frames[(market, "15m")])
        for market in markets
        if (market, "15m") in frames and len(frames[(market, "15m")]) >= 30
    }
    return_matrix = pd.DataFrame(
        {
            market: frame["return_1h"]
            for market, frame in feature_frames.items()
        }
    )
    breadth = return_matrix.gt(0).sum(axis=1) / return_matrix.notna().sum(axis=1)
    btc_return = (
        feature_frames.get("BTC-EUR", pd.DataFrame()).get("return_1h", pd.Series(dtype=float))
    )
    roundtrip_cost = float(settings.costs.taker_fee) * 2.0 + 0.001
    rows = [
        _find_replay_candidate(
            market,
            feature,
            frames.get((market, "5m"), pd.DataFrame()),
            breadth,
            btc_return,
            start=start,
            end=end,
            roundtrip_cost_fraction=roundtrip_cost,
        )
        for market, feature in feature_frames.items()
    ]
    rows.sort(
        key=lambda row: (
            row.get("status") == "REPLAY_ENTRY_READY",
            float(row.get("mfe_after_trigger") or -1.0),
        ),
        reverse=True,
    )
    payload = {
        "schema_version": "causal_rally_replay_v1",
        "generated_at": now.isoformat(),
        "replay_start": start.isoformat(),
        "replay_end": end.isoformat(),
        "markets_requested": list(markets),
        "markets_replayed": len(rows),
        "entry_ready_count": sum(row.get("status") == "REPLAY_ENTRY_READY" for row in rows),
        "economics_reject_count": sum(row.get("status") == "REPLAY_ECONOMICS_REJECT" for row in rows),
        "historical_orderflow_available": False,
        "synthetic_data_used": False,
        "orders_generated": 0,
        "orders_submitted": 0,
        "rows": rows,
    }
    output = settings.paths.output_dir / "active_trading"
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"rally_replay_{replay_date.isoformat()}.json"
    md_path = output / f"rally_replay_{replay_date.isoformat()}.md"
    atomic_write_json(json_path, payload)
    lines = [
        f"# Causal rally replay — {replay_date.isoformat()}",
        "",
        f"Replayed markets: {payload['markets_replayed']}",
        f"Economically entry-ready: {payload['entry_ready_count']}",
        "Historical orderflow was unavailable and was not fabricated.",
        "",
        "| Market | Family | Status | Entry | Max entry | Net RR | MFE | Blocker |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {market} | {family} | {status} | {entry} | {maximum} | {rr} | {mfe} | {blocker} |".format(
                market=row.get("market"),
                family=row.get("family", "-"),
                status=row.get("status"),
                entry=row.get("entry_price", "-"),
                maximum=row.get("maximum_acceptable_entry", "-"),
                rr=row.get("net_reward_to_risk", "-"),
                mfe=row.get("mfe_after_trigger", "-"),
                blocker=row.get("actual_blocker") or row.get("actual_runtime_blocker") or "-",
            )
        )
    atomic_write_text(md_path, "\n".join(lines) + "\n")
    payload["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
    return payload


__all__ = ["DEFAULT_REPLAY_MARKETS", "run_rally_replay"]
