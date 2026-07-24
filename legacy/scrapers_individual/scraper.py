from __future__ import annotations

import asyncio
import os
from typing import Any

import requests

from src.utils import fetch_json_async, now_iso, save_json
from src.scrapers.common import parse_sec_form_ownership_xml

from .scraper_bloomberg import scrape_bloomberg_async
from .scraper_fd import scrape_fd_async
from .scraper_forexfactory import scrape_forexfactory_async
from .scraper_lseg import scrape_lseg_async


def _compute_ofi(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in events:
        try:
            bid_sz = float(e.get("bid_size", 0.0))
            ask_sz = float(e.get("ask_size", 0.0))
            bid_sz_prev = float(e.get("bid_size_prev", bid_sz))
            ask_sz_prev = float(e.get("ask_size_prev", ask_sz))
            ofi = (bid_sz - bid_sz_prev) - (ask_sz - ask_sz_prev)
            mid = 0.5 * (float(e.get("bid", 0.0)) + float(e.get("ask", 0.0)))
            micro_price = (
                float(e.get("ask", 0.0)) * bid_sz + float(e.get("bid", 0.0)) * ask_sz
            ) / max(bid_sz + ask_sz, 1e-9)
            lob_imb = (bid_sz - ask_sz) / max(bid_sz + ask_sz, 1e-9)
            out.append({
                "timestamp": e.get("timestamp"),
                "symbol": e.get("symbol"),
                "ofi": ofi,
                "mid": mid,
                "micro_price": micro_price,
                "lob_imbalance": lob_imb,
            })
        except Exception:
            continue
    return out


async def _polygon_options_snapshot(symbol: str) -> dict[str, Any]:
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return {"status": "SKIPPED_NO_POLYGON_API_KEY", "symbol": symbol, "contracts": []}

    url = "https://api.polygon.io/v3/snapshot/options"
    payload = await fetch_json_async(url, params={"underlying_asset": symbol, "limit": 250, "apiKey": api_key})
    if not isinstance(payload, dict):
        return {"status": "ERROR", "symbol": symbol, "contracts": []}

    contracts = payload.get("results", []) if isinstance(payload.get("results", []), list) else []
    gex_rows: list[dict[str, Any]] = []
    for c in contracts:
        try:
            greeks = c.get("greeks", {}) or {}
            oi = float(c.get("open_interest", 0.0) or 0.0)
            gamma = float(greeks.get("gamma", 0.0) or 0.0)
            strike = float((c.get("details", {}) or {}).get("strike_price", 0.0) or 0.0)
            gex = oi * gamma * strike * strike
            gex_rows.append(
                {
                    "symbol": symbol,
                    "strike": strike,
                    "expiry": (c.get("details", {}) or {}).get("expiration_date"),
                    "option_type": (c.get("details", {}) or {}).get("contract_type"),
                    "open_interest": oi,
                    "gamma": gamma,
                    "gex": gex,
                }
            )
        except Exception:
            continue

    return {
        "status": "OK",
        "symbol": symbol,
        "contracts": gex_rows,
        "gex_total": float(sum(x["gex"] for x in gex_rows)) if gex_rows else 0.0,
    }


async def _openinsider_snapshot(symbols: list[str]) -> dict[str, Any]:
    sec_user_agent = os.getenv("SEC_USER_AGENT", "WorldMonitor/scraper (research@worldmonitor.local)")

    ticker_map_url = "https://www.sec.gov/files/company_tickers.json"
    ticker_map_payload = await fetch_json_async(ticker_map_url, headers={"User-Agent": sec_user_agent})
    ticker_to_cik: dict[str, str] = {}

    if isinstance(ticker_map_payload, dict):
        for node in ticker_map_payload.values():
            if not isinstance(node, dict):
                continue
            ticker = str(node.get("ticker") or "").strip().upper()
            cik = str(node.get("cik_str") or "").strip()
            if ticker and cik:
                ticker_to_cik[ticker] = cik.zfill(10)

    async def _fetch_one(symbol: str) -> dict[str, Any]:
        sym = str(symbol).strip().upper()
        cik = ticker_to_cik.get(sym)
        if not cik:
            return {
                "symbol": sym,
                "status": "MISSING_CIK",
                "insider_buy_ratio_proxy": None,
                "insider_buy_volume_usd_weighted": 0.0,
                "insider_sell_volume_usd_weighted": 0.0,
                "cluster_buy_volume_usd": 0.0,
                "officer_buy_volume_usd": 0.0,
                "planned_10b5_1_sell_volume_usd": 0.0,
            }

        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        sub_payload = await fetch_json_async(submissions_url, headers={"User-Agent": sec_user_agent})
        recent = ((sub_payload or {}).get("filings") or {}).get("recent", {}) if isinstance(sub_payload, dict) else {}
        forms = recent.get("form", []) if isinstance(recent, dict) else []
        accessions = recent.get("accessionNumber", []) if isinstance(recent, dict) else []
        primaries = recent.get("primaryDocument", []) if isinstance(recent, dict) else []
        dates = recent.get("filingDate", []) if isinstance(recent, dict) else []

        xml_payloads: list[str] = []
        for form, accession, primary, filing_date in zip(forms, accessions, primaries, dates):
            form_code = str(form).strip().upper()
            if form_code not in {"3", "3/A", "4", "4/A", "5", "5/A"}:
                continue
            if not accession or not primary:
                continue

            accession_compact = str(accession).replace("-", "")
            primary_doc = str(primary).strip()
            filing_base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_compact}"
            candidate_urls = [
                f"{filing_base}/{primary_doc}",
                f"{filing_base}/xslF345X03/{primary_doc}",
            ]

            for url in candidate_urls:
                try:
                    text = await asyncio.to_thread(
                        lambda: requests.get(url, headers={"User-Agent": sec_user_agent}, timeout=20).text
                    )
                    if "<ownershipDocument" in text:
                        xml_payloads.append(text)
                        break
                except Exception:
                    continue

            if len(xml_payloads) >= 8:
                break

        agg_buy = 0.0
        agg_sell = 0.0
        agg_cluster = 0.0
        agg_officer = 0.0
        agg_10b5 = 0.0
        all_rows: list[dict[str, Any]] = []

        for xml_text in xml_payloads:
            parsed = parse_sec_form_ownership_xml(xml_text, as_of=now_iso(), lookback_days=180)
            agg_buy += float(parsed.get("insider_buy_volume_usd_weighted", 0.0) or 0.0)
            agg_sell += float(parsed.get("insider_sell_volume_usd_weighted", 0.0) or 0.0)
            agg_cluster += float(parsed.get("cluster_buy_volume_usd", 0.0) or 0.0)
            agg_officer += float(parsed.get("officer_buy_volume_usd", 0.0) or 0.0)
            agg_10b5 += float(parsed.get("planned_10b5_1_sell_volume_usd", 0.0) or 0.0)
            parsed_rows = parsed.get("rows", [])
            if isinstance(parsed_rows, list):
                all_rows.extend([row for row in parsed_rows if isinstance(row, dict)])

        total = max(agg_buy + agg_sell, 1e-9)
        ratio = float(agg_buy / total) if total > 0 else None
        return {
            "symbol": sym,
            "status": "OK" if all_rows else "NO_FORM4_ROWS",
            "insider_buy_ratio_proxy": ratio,
            "insider_buy_volume_usd_weighted": float(agg_buy),
            "insider_sell_volume_usd_weighted": float(agg_sell),
            "cluster_buy_volume_usd": float(agg_cluster),
            "officer_buy_volume_usd": float(agg_officer),
            "planned_10b5_1_sell_volume_usd": float(agg_10b5),
            "n_transactions": int(len(all_rows)),
        }

    rows = await asyncio.gather(*[_fetch_one(s) for s in symbols])
    return {
        "status": "OK",
        "rows": rows,
        "source": "SEC_FORM_3_4_5_XML",
        "execution_allowed": False,
        "live_trading": False,
        "paper_broker": False,
    }


async def _finviz_snapshot(symbols: list[str]) -> dict[str, Any]:
    rows = [{"symbol": s, "valuation_snapshot_status": "PENDING_EXTERNAL_SNAPSHOT"} for s in symbols]
    return {"status": "OK_PROXY", "rows": rows}


def _to_futures_symbol(symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    if sym.endswith("USDT"):
        return sym
    crypto_roots = {
        "BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LTC", "LINK", "AVAX",
        "MATIC", "DOT", "SHIB", "TRX", "BCH", "ETC", "ATOM", "UNI", "NEAR", "FIL",
        "APT", "ARB", "OP", "SUI",
    }
    if sym in crypto_roots:
        return f"{sym}USDT"
    return ""


async def _fetch_futures_metrics_one(symbol: str) -> dict[str, Any]:
    fut_symbol = _to_futures_symbol(symbol)
    if not fut_symbol:
        return {
            "symbol": str(symbol).upper(),
            "futures_symbol": None,
            "status": "SKIPPED_NON_DERIVATIVE",
            "futures_open_interest": 0.0,
            "funding_rate": 0.0,
            "futures_open_interest_momentum": 0.0,
            "futures_funding_rate_bias": 0.0,
        }

    oi_now = await fetch_json_async("https://fapi.binance.com/fapi/v1/openInterest", params={"symbol": fut_symbol})
    fr_now = await fetch_json_async("https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": fut_symbol})
    oi_hist = await fetch_json_async(
        "https://fapi.binance.com/futures/data/openInterestHist",
        params={"symbol": fut_symbol, "period": "5m", "limit": 8},
    )

    oi_value = float((oi_now or {}).get("openInterest", 0.0) or 0.0) if isinstance(oi_now, dict) else 0.0
    funding_rate = float((fr_now or {}).get("lastFundingRate", 0.0) or 0.0) if isinstance(fr_now, dict) else 0.0

    oi_series: list[float] = []
    if isinstance(oi_hist, list):
        oi_series = [float((row or {}).get("sumOpenInterest", 0.0) or 0.0) for row in oi_hist if isinstance(row, dict)]
    if not oi_series and oi_value > 0.0:
        oi_series = [oi_value]

    if len(oi_series) >= 2:
        prev = float(oi_series[-2])
        curr = float(oi_series[-1])
        oi_momentum = float((curr - prev) / max(abs(prev), 1e-9))
    else:
        oi_momentum = 0.0

    funding_bias = float(max(min(funding_rate * 250.0, 1.0), -1.0))

    status = "OK" if (oi_value > 0.0 or abs(funding_rate) > 0.0 or len(oi_series) >= 2) else "NO_DATA"
    return {
        "symbol": str(symbol).upper(),
        "futures_symbol": fut_symbol,
        "status": status,
        "futures_open_interest": float(oi_value),
        "funding_rate": float(funding_rate),
        "futures_open_interest_momentum": float(max(min(oi_momentum, 5.0), -5.0)),
        "futures_funding_rate_bias": float(funding_bias),
    }


async def _futures_market_snapshot(symbols: list[str]) -> dict[str, Any]:
    rows = await asyncio.gather(*[_fetch_futures_metrics_one(s) for s in symbols])
    return {
        "status": "OK",
        "source": "BINANCE_FUTURES_PUBLIC",
        "rows": rows,
        "execution_allowed": False,
        "live_trading": False,
        "paper_broker": False,
    }


async def run_all_scrapers_async(out_path: str = "output/source_snapshots/unified_scrape.json") -> dict[str, Any]:
    symbols = ["SPY", "QQQ", "DIA", "IWM", "GLD", "USO", "TLT", "AAPL", "MSFT", "NVDA"]

    bloomberg_task = scrape_bloomberg_async()
    fd_task = scrape_fd_async()
    ff_task = scrape_forexfactory_async()
    lseg_task = scrape_lseg_async()
    polygon_tasks = [asyncio.create_task(_polygon_options_snapshot(s)) for s in symbols]

    bloomberg, fd, forexfactory, lseg = await asyncio.gather(
        bloomberg_task,
        fd_task,
        ff_task,
        lseg_task,
    )
    polygon = await asyncio.gather(*polygon_tasks)
    openinsider, finviz, futures_derivatives = await asyncio.gather(
        _openinsider_snapshot(symbols),
        _finviz_snapshot(symbols),
        _futures_market_snapshot(symbols),
    )

    microstructure_rows: list[dict[str, Any]] = []
    for p in polygon:
        for c in p.get("contracts", []):
            bid_sz = float(c.get("open_interest", 0.0))
            ask_sz = max(0.0, bid_sz * 0.9)
            microstructure_rows.append(
                {
                    "timestamp": now_iso(),
                    "symbol": c.get("symbol"),
                    "bid": c.get("strike", 0.0),
                    "ask": c.get("strike", 0.0),
                    "bid_size": bid_sz,
                    "ask_size": ask_sz,
                    "bid_size_prev": bid_sz,
                    "ask_size_prev": ask_sz,
                }
            )

    ofi_rows = _compute_ofi(microstructure_rows)

    payload = {
        "created_at": now_iso(),
        "status": "OK",
        "sources": {
            "bloomberg": bloomberg,
            "fd": fd,
            "forexfactory": forexfactory,
            "lseg": lseg,
            "polygon_options": polygon,
            "openinsider": openinsider,
            "finviz": finviz,
            "futures_derivatives": futures_derivatives,
        },
        "derived_microstructure": {
            "ofi_rows": ofi_rows,
            "ofi_count": len(ofi_rows),
        },
    }

    save_json(payload, path=__import__("pathlib").Path(out_path))
    return payload


def main() -> None:
    asyncio.run(run_all_scrapers_async())


if __name__ == "__main__":
    main()
