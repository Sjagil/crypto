#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings
from portfolio.active_allocator import (
    apply_paper_plan,
    build_active_portfolio_plan,
    load_policy,
    wallet_from_mapping,
)
from utils.common import append_jsonl, atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _latest_prices(packet: dict[str, Any]) -> dict[str, float]:
    result = {}
    markets = packet.get("canonical_context", {}).get("market_context", {})
    for market, payload in markets.items():
        states = payload.get("timeframes", {})
        for timeframe in ("15m", "1h", "2h", "4h", "1d"):
            price = states.get(timeframe, {}).get("close")
            if price:
                result[market] = float(price)
                break
    return result


def _initial_wallet(equity: float) -> dict[str, Any]:
    return {
        "schema_version": "active_paper_wallet_v1",
        "cash_eur": equity,
        "equity_eur": equity,
        "positions": {},
        "fill_count": 0,
    }


async def _private_wallet(settings: Settings, packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from core.live_asset_preflight import live_account_health

    markets = tuple(str(value) for value in packet.get("markets") or ())
    health = await live_account_health(settings, markets=markets)
    projection = dict(health.get("balance_projection") or health.get("projection") or {})
    if health.get("status") != "READY":
        return {}, health
    prices = _latest_prices(packet)
    positions = {}
    total_non_eur = 0.0
    for row in projection.get("non_eur_holdings") or []:
        symbol = str(row.get("symbol") or "").upper()
        market = f"{symbol}-EUR"
        price = prices.get(market)
        if price is None:
            continue
        quantity = float(row.get("total") or 0.0)
        positions[market] = {"quantity": quantity, "mark_price": price}
        total_non_eur += quantity * price
    cash = float(projection.get("eur_available") or 0.0)
    return {
        "equity_eur": cash + total_non_eur,
        "cash_eur": cash,
        "positions": positions,
    }, health


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("paper", "live-plan"), default="paper")
    parser.add_argument("--equity-eur", type=float, default=2000.0)
    parser.add_argument("--packet")
    parser.add_argument("--policy")
    parser.add_argument("--refresh-whole-stack", action="store_true")
    parser.add_argument("--sync-data", action="store_true")
    parser.add_argument("--private-read", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    settings = Settings.load(create_directories=True)
    packet_path = (
        Path(args.packet).expanduser().resolve()
        if args.packet
        else root / "output" / "reports" / "whole_stack" / "latest.json"
    )

    if args.sync_data:
        subprocess.run(
            [
                sys.executable,
                "main.py",
                "history",
                "download",
                "--min-years",
                "7",
                "--markets",
                "BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR",
                "--timeframes",
                "15m,1h,2h,4h,1d,1W",
                "--providers",
                "bitvavo,kraken",
                "--resume",
            ],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
        )

    if args.refresh_whole_stack:
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_whole_stack.py"),
                "--markets",
                "BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR",
                "--timeframes",
                "15m,1h,2h,4h,1d,1W",
            ],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
        )

    packet = _read(packet_path)
    policy = load_policy(
        Path(args.policy).expanduser().resolve() if args.policy else None
    )

    output = root / "output" / "portfolio"
    output.mkdir(parents=True, exist_ok=True)
    health: dict[str, Any] | None = None

    if args.mode == "paper":
        wallet_path = root / "output" / "paper" / "active_portfolio_wallet.json"
        wallet_path.parent.mkdir(parents=True, exist_ok=True)
        wallet_payload = (
            _read(wallet_path)
            if wallet_path.is_file()
            else _initial_wallet(args.equity_eur)
        )
    else:
        if not args.private_read:
            payload = {
                "status": "BLOCKED",
                "reason": "LIVE_PLAN_REQUIRES_EXPLICIT_PRIVATE_READ",
                "orders_submitted": 0,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 2
        wallet_payload, health = asyncio.run(_private_wallet(settings, packet))
        if not wallet_payload:
            payload = {
                "status": "BLOCKED",
                "reason": "LIVE_ACCOUNT_HEALTH_NOT_READY",
                "account_health": health,
                "orders_submitted": 0,
            }
            atomic_write_json(output / "active_live_plan.json", payload)
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
            return 2

    wallet = wallet_from_mapping(wallet_payload)
    plan = build_active_portfolio_plan(
        packet,
        wallet,
        settings,
        mode=args.mode,
        policy=policy,
    )

    if args.mode == "paper":
        updated, fills = apply_paper_plan(wallet_payload, plan, settings)
        atomic_write_json(wallet_path, updated)
        for fill in fills:
            append_jsonl(root / "output" / "paper" / "active_portfolio_fills.jsonl", fill)
        plan["paper_execution"] = {
            "fills": fills,
            "wallet_after": updated,
        }
        plan["status"] = "PAPER_CYCLE_EXECUTED"
        artifact = output / "active_paper_plan.json"
    else:
        plan["account_health"] = health
        plan["status"] = (
            "LIVE_TARGETS_READY_FOR_RISK_APPROVAL"
            if plan.get("canonical_live_targets")
            else "LIVE_NO_ELIGIBLE_TARGETS"
        )
        artifact = output / "active_live_plan.json"

    atomic_write_json(artifact, plan)
    plan["artifact"] = str(artifact)
    print(json.dumps(plan, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
