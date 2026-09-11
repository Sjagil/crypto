
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from data.bitvavo_market_data_pro import BitvavoMarketDataProManager
from data.orderflow_recorder import HashChainedOrderflowLedger, ProspectiveOrderflowRecorder
from utils.common import atomic_write_json, utc_now


def _secret(value: Any):
    return value if value is not None else None


def _markets_from_file(path: Path | None, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if path is None or not path.is_file():
        return fallback[:25]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback[:25]
    values = payload.get("markets") or (payload.get("universe") or {}).get("markets") or []
    selected = tuple(
        dict.fromkeys(
            str(value).strip().upper().replace("/", "-")
            for value in values
            if str(value).strip().upper().endswith("-EUR")
        )
    )
    return (selected or fallback)[:25]


async def _run(args) -> None:
    settings = get_settings()
    data_key = settings.providers.bitvavo_data_api_key
    data_secret = settings.providers.bitvavo_data_api_secret
    key = data_key or settings.providers.bitvavo_trade_api_key
    secret = data_secret or settings.providers.bitvavo_trade_api_secret
    if key is None or secret is None:
        raise RuntimeError("BITVAVO_MDPRO_DATA_CREDENTIALS_MISSING")

    fallback = tuple(settings.autonomous_live.markets)
    universe_path = Path(args.universe_json).expanduser().resolve() if args.universe_json else None
    markets = _markets_from_file(universe_path, fallback)
    if not markets:
        raise RuntimeError("BITVAVO_MDPRO_UNIVERSE_EMPTY")

    context = settings.paths.context_data_dir
    ledger = HashChainedOrderflowLedger(
        root=context / "orderflow_mdpro_25",
        checkpoint_path=settings.paths.checkpoints_dir / "orderflow_mdpro_25.json",
        maximum_storage_bytes=int(settings.market_data.maximum_storage_gb * 1024**3),
        checkpoint_first_recovery=True,
    )
    recorder = ProspectiveOrderflowRecorder(
        ledger=ledger,
        database=None,
        markets=markets,
        feature_directory=context / "microstructure_mdpro_hourly",
        fifteen_minute_feature_directory=context / "microstructure_mdpro_15m",
        readiness_path=settings.paths.output_dir / "operations/mdpro_readiness.json",
        health_path=settings.paths.output_dir / "operations/mdpro_stream_health.json",
        positioning_directory=context / "prospective_hourly",
        realtime_candle_path=settings.paths.output_dir / "operations/realtime_candles_mdpro.json",
        flush_seconds=0.5,
        batch_size=1000,
    )
    manager = BitvavoMarketDataProManager(
        api_key=key,
        api_secret=secret,
        markets=markets,
        depth=args.depth,
        queue_size=args.queue_size,
    )
    manager.add_seed_callback(recorder.seed_orderbook)

    latest_dir = context / "mdpro"
    latest_path = latest_dir / "latest.json"
    latest_dir.mkdir(parents=True, exist_ok=True)

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signame in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, signame, None)
        if sig is not None:
            try:
                loop.add_signal_handler(sig, stopping.set)
            except NotImplementedError:
                pass

    await manager.start()
    recorder_task = asyncio.create_task(recorder.run(manager), name="mdpro-recorder")
    last_universe_check = 0.0
    try:
        while not stopping.is_set():
            await asyncio.sleep(5)
            now = loop.time()
            if now - last_universe_check >= 60:
                new_markets = _markets_from_file(universe_path, fallback)
                if new_markets != manager.markets:
                    await manager.update_markets(new_markets)
                    recorder.markets = new_markets
                last_universe_check = now
            payload = {
                "schema_version": "round44_mdpro_runtime_v1",
                "generated_at": utc_now().isoformat(),
                "manager": manager.snapshot(),
                "microstructure": recorder.realtime_snapshot(
                    markets=manager.markets,
                    order_notional_eur=10.0,
                ),
                "l3_status": "L3_UNSUPPORTED_BY_EXECUTION_VENUE",
                "automatic_live_authority": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
            atomic_write_json(latest_path, payload)
    finally:
        recorder.stop()
        await manager.stop()
        await asyncio.gather(recorder_task, return_exceptions=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-json")
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--queue-size", type=int, default=20000)
    args = parser.parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
