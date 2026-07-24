from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

from src.utils import clean_text, extract_embedded_json, extract_meta, fetch_soup_async, now_iso, parse_table


def base_result(url: str, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "url": url,
        "scraped_at": now_iso(),
        "meta": {},
        "embedded_json": [],
        "records": [],
        "status": "OK",
        "error": None,
    }


async def fetch_with_payload(url: str, source: str, headers: dict[str, str] | None = None) -> tuple[BeautifulSoup | None, dict[str, Any]]:
    res = base_result(url=url, source=source)
    soup = await fetch_soup_async(url, headers=headers)
    if soup is None:
        res["status"] = "ERROR_FETCH"
        res["error"] = "not_accessible_or_js_required"
        return None, res

    res["meta"] = extract_meta(soup)
    res["embedded_json"] = extract_embedded_json(soup)
    return soup, res


def parse_generic_articles(soup: BeautifulSoup, limit: int = 60) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in soup.find_all(["article", "div", "li", "section"]):
        cls = " ".join(item.get("class", []))
        if not re.search(r"story|article|headline|news|card|feed|item", cls, flags=re.I):
            continue

        title_el = item.find(["h1", "h2", "h3", "h4"])
        link_el = item.find("a", href=True)
        time_el = item.find("time")
        desc_el = item.find("p")

        title = clean_text(title_el.get_text() if title_el else "")
        if not title or title in seen or len(title) < 8:
            continue
        seen.add(title)

        out.append(
            {
                "title": title,
                "url": clean_text(link_el["href"]) if link_el else "",
                "published": clean_text(time_el.get("datetime") or time_el.get_text()) if time_el else "",
                "description": clean_text(desc_el.get_text())[:320] if desc_el else "",
            }
        )
        if len(out) >= limit:
            break
    return out


def parse_tables_and_widgets(soup: BeautifulSoup) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for tbl in soup.find_all("table"):
        out.extend(parse_table(tbl))

    for widget in soup.find_all(class_=re.compile(r"security|ticker|index|market|quote", re.I)):
        name = clean_text(widget.get_text(" "))
        if not name:
            continue
        out.append({"raw": name[:400]})
    return out


def logger(name: str) -> logging.Logger:
    return logging.getLogger(f"src.scrapers.{name}")


_BOND_TICKERS = {
    "TLT", "IEF", "IEI", "SHY", "BIL", "TIP", "BND", "AGG", "LQD", "HYG", "EDV", "GOVT",
}
_ETF_TICKERS = {
    "SPY", "QQQ", "VTI", "IWM", "DIA", "EEM", "XLF", "XLK", "XLE", "XLI", "XLP", "XLY", "XLV", "XLB", "XLC", "XLU", "VNQ",
}
_COMMODITY_TICKERS = {
    "GLD", "SLV", "USO", "UNG", "DBC", "SGLN.L", "XLE", "CL=F", "GC=F", "SI=F", "HG=F", "NG=F",
}
_SECTOR_MAP = {
    "TECHNOLOGY": {"AAPL", "MSFT", "NVDA", "AMD", "INTC", "AVGO", "ORCL", "ASML", "QQQ", "DOCN", "PLTR", "META", "GOOGL", "AMZN"},
    "FINANCIALS": {"JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "XLF", "BLND"},
    "ENERGY": {"XOM", "CVX", "XLE", "USO", "CL=F"},
    "CONSUMER_CYCLICAL": {"AMZN", "TSLA", "HD", "LOW", "NKE", "SBUX", "TGT", "ETSY", "CHWY", "ROKU"},
    "HEALTHCARE": {"JNJ", "PFE", "MRK", "ABBV", "UNH", "LLY", "TMO", "CRVO"},
    "INDUSTRIALS": {"GE", "CAT", "HON", "LMT", "BA", "UPS", "FDX", "ARCB", "RXO"},
}


def classify_asset_class(symbol: str, category_hint: str | None = None) -> str:
    ticker = str(symbol).upper().strip()
    hint = str(category_hint or "").lower()

    if ticker.startswith("^"):
        return "Indice"
    if ticker.endswith("=F") or ticker in _COMMODITY_TICKERS:
        return "Commodity"
    if ticker in _BOND_TICKERS or "bond" in hint or "fixed_income" in hint:
        return "Bond"
    if ticker in _ETF_TICKERS or "etf" in hint:
        return "ETF"
    if "indices" in hint or "index" in hint:
        return "Indice"
    if "commodity" in hint or "metals" in hint or "energy" in hint:
        return "Commodity"
    return "Stock"


def classify_sector(symbol: str, category_hint: str | None = None) -> str:
    ticker = str(symbol).upper().strip()
    hint = str(category_hint or "").lower()
    for sector, tickers in _SECTOR_MAP.items():
        if ticker in tickers:
            return sector

    if "health" in hint or "biotech" in hint:
        return "HEALTHCARE"
    if "finance" in hint or "bank" in hint:
        return "FINANCIALS"
    if "energy" in hint or "oil" in hint or "gas" in hint:
        return "ENERGY"
    if "tech" in hint or "ai" in hint or "semiconductor" in hint:
        return "TECHNOLOGY"
    if "consumer" in hint or "retail" in hint:
        return "CONSUMER_CYCLICAL"
    if "industrial" in hint or "transport" in hint:
        return "INDUSTRIALS"
    if classify_asset_class(ticker, category_hint) == "Bond":
        return "RATES"
    if classify_asset_class(ticker, category_hint) == "Commodity":
        return "COMMODITIES"
    return "BROAD_MARKET"


def load_latest_symbol_snapshot_price(repo_root: Path, symbol: str) -> float | None:
    """Best-effort local snapshot lookup for a symbol's latest price.

    This avoids hard failures when live market bars are unavailable.
    """
    target = str(symbol).upper().strip()
    if not target:
        return None

    candidates = [
        repo_root / "output" / "source_snapshots" / "finviz" / "finviz_quote_snapshots.parquet",
        repo_root / "output" / "worldmonitor_plus_enriched" / "current_plus_rows.parquet",
        repo_root / "output" / "worldmonitor_plus" / "current_plus_rows.parquet",
        repo_root / "output" / "source_snapshots" / "finviz" / "finviz_features_v2.csv",
    ]
    price_hints = [
        "price",
        "close",
        "last",
        "last_price",
        "adj_close",
        "prev_close",
        "underlying_price",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        except Exception:
            continue

        if df is None or df.empty:
            continue

        cols_lower = {str(c).lower(): str(c) for c in df.columns}
        symbol_col = cols_lower.get("symbol") or cols_lower.get("ticker")
        if not symbol_col:
            continue

        rows = df[df[symbol_col].astype(str).str.upper().str.strip() == target]
        if rows.empty:
            continue

        price_col = None
        for hint in price_hints:
            if hint in cols_lower:
                price_col = cols_lower[hint]
                break
        if price_col is None:
            for c in rows.columns:
                name = str(c).lower()
                if "price" in name or "close" in name:
                    price_col = str(c)
                    break
        if price_col is None:
            continue

        value = pd.to_numeric(rows[price_col], errors="coerce").dropna()
        if value.empty:
            continue

        out = float(value.iloc[-1])
        if out > 0.0:
            return out

    return None


def build_sentiment_shock_features(
    history_rows: list[dict[str, Any]],
    *,
    as_of_timestamp: str | None = None,
) -> dict[str, float]:
    """Build lagged sentiment features using strictly past observations.

    Expected row fields: timestamp/date, news_count, finbert_pos_mean.
    Missing fields are handled conservatively as zeros.
    """
    if not history_rows:
        return {
            "news_count_1d": 0.0,
            "news_count_7d": 0.0,
            "finbert_pos_mean_7d": 0.0,
            "sentiment_shock_score": 0.0,
        }

    frame = pd.DataFrame(history_rows)
    if frame.empty:
        return {
            "news_count_1d": 0.0,
            "news_count_7d": 0.0,
            "finbert_pos_mean_7d": 0.0,
            "sentiment_shock_score": 0.0,
        }

    ts_col = "timestamp" if "timestamp" in frame.columns else ("date" if "date" in frame.columns else None)
    if ts_col is None:
        return {
            "news_count_1d": 0.0,
            "news_count_7d": 0.0,
            "finbert_pos_mean_7d": 0.0,
            "sentiment_shock_score": 0.0,
        }

    frame["timestamp"] = pd.to_datetime(frame[ts_col], errors="coerce", utc=True)
    frame = frame[~frame["timestamp"].isna()].sort_values("timestamp")
    if frame.empty:
        return {
            "news_count_1d": 0.0,
            "news_count_7d": 0.0,
            "finbert_pos_mean_7d": 0.0,
            "sentiment_shock_score": 0.0,
        }

    if as_of_timestamp:
        as_of = pd.to_datetime(as_of_timestamp, errors="coerce", utc=True)
        if pd.notna(as_of):
            frame = frame[frame["timestamp"] <= as_of]
    if frame.empty:
        return {
            "news_count_1d": 0.0,
            "news_count_7d": 0.0,
            "finbert_pos_mean_7d": 0.0,
            "sentiment_shock_score": 0.0,
        }

    frame["news_count"] = pd.to_numeric(frame.get("news_count", 0.0), errors="coerce").fillna(0.0)
    frame["finbert_pos_mean"] = pd.to_numeric(frame.get("finbert_pos_mean", 0.0), errors="coerce").fillna(0.0)
    daily = (
        frame.set_index("timestamp")
        .resample("1D")
        .agg({"news_count": "sum", "finbert_pos_mean": "mean"})
        .fillna(0.0)
    )

    lag_news = daily["news_count"].shift(1)
    lag_finbert = daily["finbert_pos_mean"].shift(1)

    news_count_1d = float(lag_news.iloc[-1]) if not lag_news.empty else 0.0
    news_count_7d = float(lag_news.tail(7).sum()) if not lag_news.empty else 0.0
    finbert_pos_mean_7d = float(lag_finbert.tail(7).mean()) if not lag_finbert.empty else 0.0

    rolling_mean_7d = lag_news.rolling(7, min_periods=3).mean()
    rolling_std_7d = lag_news.rolling(7, min_periods=3).std().replace(0.0, np.nan)
    shock_z = ((lag_news - rolling_mean_7d) / rolling_std_7d).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sentiment_shock_score = float(np.clip(shock_z.iloc[-1] if not shock_z.empty else 0.0, -5.0, 5.0))

    return {
        "news_count_1d": float(max(news_count_1d, 0.0)),
        "news_count_7d": float(max(news_count_7d, 0.0)),
        "finbert_pos_mean_7d": float(np.clip(finbert_pos_mean_7d, -1.0, 1.0)),
        "sentiment_shock_score": sentiment_shock_score,
    }


def _xml_find_text(node: ET.Element, *tag_suffixes: str) -> str:
    for child in node.iter():
        local = str(child.tag).split("}")[-1].lower()
        if local in {str(x).strip().lower() for x in tag_suffixes}:
            text = str(child.text or "").strip()
            if text:
                return text
    return ""


def parse_sec_form_ownership_xml(
    xml_text: str,
    *,
    as_of: str | None = None,
    lookback_days: int = 180,
) -> dict[str, Any]:
    """Parse SEC ownership XML (Forms 3/4/5) into weighted insider features."""
    raw = str(xml_text or "").strip()
    if not raw:
        return {
            "status": "EMPTY_XML",
            "rows": [],
            "insider_buy_volume_usd_weighted": 0.0,
            "insider_sell_volume_usd_weighted": 0.0,
            "cluster_buy_volume_usd": 0.0,
            "officer_buy_volume_usd": 0.0,
            "planned_10b5_1_sell_volume_usd": 0.0,
        }

    try:
        root = ET.fromstring(raw)
    except Exception:
        return {
            "status": "INVALID_XML",
            "rows": [],
            "insider_buy_volume_usd_weighted": 0.0,
            "insider_sell_volume_usd_weighted": 0.0,
            "cluster_buy_volume_usd": 0.0,
            "officer_buy_volume_usd": 0.0,
            "planned_10b5_1_sell_volume_usd": 0.0,
        }

    now_ts = pd.Timestamp.utcnow().tz_localize(None)
    as_of_ts = pd.to_datetime(as_of, errors="coerce") if as_of else now_ts
    if pd.isna(as_of_ts):
        as_of_ts = now_ts
    if getattr(as_of_ts, "tzinfo", None) is not None:
        as_of_ts = as_of_ts.tz_convert(timezone.utc).tz_localize(None)
    cutoff_ts = pd.Timestamp(as_of_ts) - pd.Timedelta(days=int(max(1, lookback_days)))

    remarks_blob = " ".join(_xml_find_text(root, "remarks", "footnote", "footnotetext").split()).lower()

    rows: list[dict[str, Any]] = []
    for tx in root.iter():
        local = str(tx.tag).split("}")[-1].lower()
        if local not in {"nonderivativetransaction", "derivativetransaction"}:
            continue

        code = _xml_find_text(tx, "transactioncode", "transaction_code").upper()
        shares = pd.to_numeric(_xml_find_text(tx, "transactionshares", "shares"), errors="coerce")
        price = pd.to_numeric(_xml_find_text(tx, "transactionpricepershare", "pricepershare", "price"), errors="coerce")

        amount = float(abs(shares)) if np.isfinite(shares) else 0.0
        px = float(abs(price)) if np.isfinite(price) else 0.0
        volume_usd = float(amount * px) if amount > 0.0 and px > 0.0 else float(amount)
        if volume_usd <= 0.0:
            continue

        owner_is_officer = _xml_find_text(tx, "isofficer", "officer")
        owner_title = _xml_find_text(tx, "officertitle", "title")
        is_officer = str(owner_is_officer).strip().lower() in {"1", "true", "y", "yes"} or bool(owner_title.strip())

        filing_date_raw = _xml_find_text(tx, "transactiondate", "periodofreport", "acceptancedatetime")
        filing_ts = pd.to_datetime(filing_date_raw, errors="coerce")
        if pd.isna(filing_ts):
            filing_ts = as_of_ts
        if getattr(filing_ts, "tzinfo", None) is not None:
            filing_ts = filing_ts.tz_convert(timezone.utc).tz_localize(None)
        if pd.Timestamp(filing_ts) < cutoff_ts:
            continue

        narrative = " ".join(
            [
                str(code or ""),
                str(owner_title or ""),
                remarks_blob,
                _xml_find_text(tx, "transactiontimeliness", "natureofownership", "footnoteid"),
            ]
        ).lower()
        is_10b5_1 = any(token in narrative for token in ["10b5-1", "10b5 1", "rule 10b5"])

        direction = "other"
        if code in {"P", "A"}:
            direction = "buy"
        elif code in {"S", "D"}:
            direction = "sell"

        rows.append(
            {
                "code": code,
                "direction": direction,
                "volume_usd": float(volume_usd),
                "is_officer": bool(is_officer),
                "is_10b5_1": bool(is_10b5_1),
                "filing_date": str(pd.Timestamp(filing_ts).date()),
            }
        )

    if not rows:
        return {
            "status": "NO_TRANSACTIONS_IN_WINDOW",
            "rows": [],
            "insider_buy_volume_usd_weighted": 0.0,
            "insider_sell_volume_usd_weighted": 0.0,
            "cluster_buy_volume_usd": 0.0,
            "officer_buy_volume_usd": 0.0,
            "planned_10b5_1_sell_volume_usd": 0.0,
        }

    by_date_buy_count: dict[str, int] = {}
    for row in rows:
        if row["direction"] == "buy":
            by_date_buy_count[row["filing_date"]] = by_date_buy_count.get(row["filing_date"], 0) + 1

    buy_w = 0.0
    sell_w = 0.0
    cluster_buy = 0.0
    officer_buy = 0.0
    planned_sell = 0.0

    for row in rows:
        usd = float(row["volume_usd"])
        if row["direction"] == "buy":
            weight = 1.0
            if bool(row["is_officer"]):
                officer_buy += usd
                weight *= 1.5
            if int(by_date_buy_count.get(str(row["filing_date"]), 0)) >= 2:
                cluster_buy += usd
                weight *= 1.35
            buy_w += usd * weight
        elif row["direction"] == "sell":
            if bool(row["is_10b5_1"]):
                planned_sell += usd
                sell_w += usd * 0.35
            else:
                sell_w += usd

    return {
        "status": "OK",
        "rows": rows,
        "insider_buy_volume_usd_weighted": float(buy_w),
        "insider_sell_volume_usd_weighted": float(sell_w),
        "cluster_buy_volume_usd": float(cluster_buy),
        "officer_buy_volume_usd": float(officer_buy),
        "planned_10b5_1_sell_volume_usd": float(planned_sell),
    }


def build_event_calendar_feature_matrix(
    date_index: pd.DatetimeIndex,
    *,
    earnings_events: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(date_index, errors="coerce")).dropna().tz_localize(None)
    idx = idx.sort_values()
    if idx.empty:
        return pd.DataFrame(
            columns=[
                "earnings_days_until",
                "earnings_days_since",
                "earnings_surprise_recent",
                "fomc_week_flag",
                "cpi_week_flag",
            ]
        )

    start = pd.Timestamp(idx.min()).normalize()
    end = pd.Timestamp(idx.max()).normalize()

    earnings_ts: list[pd.Timestamp] = []
    surprise_rows: list[tuple[pd.Timestamp, float]] = []
    for node in earnings_events or []:
        raw_date = node.get("date") or node.get("event_date") or node.get("calendar_date")
        dtv = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(dtv):
            continue
        event_ts = pd.Timestamp(dtv).tz_localize(None).normalize()
        earnings_ts.append(event_ts)
        raw_surprise = node.get("surprise")
        if raw_surprise is None:
            raw_surprise = node.get("surprisePercent", node.get("surprise_pct"))
        surprise = pd.to_numeric(raw_surprise, errors="coerce")
        if np.isfinite(surprise):
            surprise_rows.append((event_ts, float(surprise)))

    earnings_days_until = np.full(len(idx), 999.0, dtype=np.float32)
    earnings_days_since = np.full(len(idx), 999.0, dtype=np.float32)
    if earnings_ts:
        event_days = np.asarray(sorted(set(earnings_ts)), dtype="datetime64[D]")
        current_days = np.asarray(idx.values.astype("datetime64[D]"), dtype="datetime64[D]")
        left = np.searchsorted(event_days, current_days, side="left")
        right = np.searchsorted(event_days, current_days, side="right") - 1
        for i in range(len(current_days)):
            if int(left[i]) < len(event_days):
                earnings_days_until[i] = float(max(0, int((event_days[int(left[i])] - current_days[i]).astype("timedelta64[D]").astype(int))))
            if int(right[i]) >= 0:
                earnings_days_since[i] = float(max(0, int((current_days[i] - event_days[int(right[i])]).astype("timedelta64[D]").astype(int))))

    surprise_series = pd.Series(0.0, index=idx, dtype=np.float32)
    if surprise_rows:
        surprise_df = pd.DataFrame(surprise_rows, columns=["date", "surprise"]).groupby("date").mean()
        surprise_df.index = pd.DatetimeIndex(pd.to_datetime(surprise_df.index, errors="coerce")).tz_localize(None)
        surprise_series = pd.to_numeric(surprise_df["surprise"], errors="coerce").reindex(idx).ffill().fillna(0.0).astype(np.float32)

    months = pd.date_range(start=start.replace(day=1), end=end + pd.Timedelta(days=45), freq="MS")
    cpi_events = pd.DatetimeIndex([pd.Timestamp(m) + pd.offsets.BDay(9) for m in months]).tz_localize(None)
    fomc_seed = start + pd.offsets.BDay(3)
    fomc_events = pd.DatetimeIndex(pd.date_range(start=fomc_seed, end=end + pd.Timedelta(days=45), freq="42D")).tz_localize(None)

    fomc_week_flag = np.zeros(len(idx), dtype=np.float32)
    cpi_week_flag = np.zeros(len(idx), dtype=np.float32)
    for i, ts in enumerate(idx):
        week_end = pd.Timestamp(ts) + pd.Timedelta(days=6)
        fomc_week_flag[i] = 1.0 if bool(((fomc_events >= ts) & (fomc_events <= week_end)).any()) else 0.0
        cpi_week_flag[i] = 1.0 if bool(((cpi_events >= ts) & (cpi_events <= week_end)).any()) else 0.0

    return pd.DataFrame(
        {
            "earnings_days_until": earnings_days_until.astype(np.float32),
            "earnings_days_since": earnings_days_since.astype(np.float32),
            "earnings_surprise_recent": surprise_series.to_numpy(dtype=np.float32),
            "fomc_week_flag": fomc_week_flag.astype(np.float32),
            "cpi_week_flag": cpi_week_flag.astype(np.float32),
        },
        index=idx,
    )


def build_cross_asset_macro_sector_features(
    returns_df: pd.DataFrame,
    *,
    market_symbol: str = "SPY",
) -> pd.DataFrame:
    if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
        return pd.DataFrame(
            columns=[
                "gold_momentum",
                "silver_momentum",
                "oil_momentum",
                "dollar_index_proxy",
                "bond_vol_proxy",
                "bitcoin_risk_appetite_proxy",
                "sector_relative_strength",
                "industry_momentum",
                "sector_beta",
            ]
        )

    frame = returns_df.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index().ffill().fillna(0.0)

    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _basket(cols: list[str]) -> pd.Series:
        present = [c for c in cols if c in frame.columns]
        if not present:
            return pd.Series(0.0, index=frame.index)
        return frame[present].mean(axis=1)

    gold = _basket(["GLD", "GC=F", "SGLN.L"]).rolling(20, min_periods=5).mean().fillna(0.0)
    silver = _basket(["SLV", "SI=F"]).rolling(20, min_periods=5).mean().fillna(0.0)
    oil = _basket(["USO", "CL=F", "XLE"]).rolling(20, min_periods=5).mean().fillna(0.0)
    dxy_proxy = _basket(["UUP", "DXY", "DX=F"]).rolling(20, min_periods=5).mean().fillna(0.0)

    bond_proxy = _basket(["TLT", "IEF", "AGG", "BND"]).rolling(20, min_periods=5).std().fillna(0.0)
    btc_proxy = _basket(["BTC-USD", "BITO", "MSTR"]).rolling(20, min_periods=5).mean().fillna(0.0)

    market = _basket([market_symbol]).replace(0.0, np.nan)
    tech = _basket(["XLK", "QQQ", "AAPL", "MSFT", "NVDA"])
    sector_rel = (tech - _basket(["SPY", "VTI"])).fillna(0.0)
    industry_mom = frame.mean(axis=1).rolling(20, min_periods=5).mean().fillna(0.0)
    sector_beta = (tech.rolling(60, min_periods=20).cov(market) / market.rolling(60, min_periods=20).var()).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = pd.DataFrame(
        {
            "gold_momentum": gold,
            "silver_momentum": silver,
            "oil_momentum": oil,
            "dollar_index_proxy": dxy_proxy,
            "bond_vol_proxy": bond_proxy,
            "bitcoin_risk_appetite_proxy": btc_proxy,
            "sector_relative_strength": sector_rel,
            "industry_momentum": industry_mom,
            "sector_beta": sector_beta,
        },
        index=frame.index,
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)