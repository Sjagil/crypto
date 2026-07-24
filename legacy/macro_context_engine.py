from __future__ import annotations

"""Causally aligned crypto macro-context features.

The module is provider-independent. Scrapers/API clients supply historical
DataFrames; this engine aligns rows by their actual availability time, builds
features and classifies regimes.

All percentages use decimal fractions unless a raw provider field is clearly in
percentage points. Source indices must represent ``available_at`` timestamps,
or pass ``available_at_col`` to the individual feature function.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MacroContextConfig:
    short_window: int = 7
    medium_window: int = 20
    long_window: int = 50
    zscore_window: int = 90

    extreme_fear_max: float = 24.0
    fear_max: float = 44.0
    neutral_max: float = 55.0
    greed_max: float = 74.0

    breadth_risk_on: float = 0.55
    breadth_risk_off: float = 0.35
    breadth_thrust_change: float = 0.20

    funding_z_overheated: float = 2.0
    funding_z_crowded_short: float = -2.0
    oi_growth_overheated: float = 0.10
    oi_drop_deleveraging: float = -0.10
    liquidation_z_extreme: float = 2.0
    annual_funding_periods: int = 1095

    vix_risk_off_level: float = 25.0
    high_yield_spread_risk_off: float = 5.0

    high_impact_pre_hours: float = 2.0
    high_impact_post_hours: float = 1.0
    token_unlock_pre_hours: float = 72.0
    material_unlock_pct_float: float = 0.01

    fear_greed_max_age_hours: float = 48.0
    dominance_max_age_hours: float = 48.0
    relative_prices_max_age_hours: float = 8.0
    breadth_max_age_hours: float = 8.0
    derivatives_max_age_hours: float = 4.0
    etf_flows_max_age_hours: float = 96.0
    onchain_max_age_hours: float = 72.0
    global_macro_max_age_hours: float = 48.0

    def __post_init__(self) -> None:
        for name in (
            "short_window", "medium_window", "long_window",
            "zscore_window", "annual_funding_periods",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not (
            self.extreme_fear_max < self.fear_max
            < self.neutral_max < self.greed_max
        ):
            raise ValueError("Fear & Greed thresholds must increase")


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "sentiment": ("fear_greed",),
    "dominance": ("btc_dominance", "stablecoin_dominance", "TOTAL/TOTAL2/TOTAL3"),
    "relative_strength": ("BTC returns", "ETH/BTC", "SOL/BTC", "other asset/BTC"),
    "breadth": ("% above EMA20/50/200", "new highs/lows", "advance-decline"),
    "derivatives": ("funding", "open interest", "basis", "liquidations"),
    "flows": ("BTC ETF flows", "ETH ETF flows"),
    "onchain": ("exchange flows", "MVRV", "SOPR", "NUPL", "network activity"),
    "global_macro": ("DXY", "Nasdaq", "S&P 500", "VIX", "rates", "credit", "liquidity"),
    "events": ("macro releases", "token unlocks", "protocol/regulatory events"),
}


def _utc_index(index: pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(index, utc=True, errors="raise"))


def validate_base_index(base: pd.DataFrame | pd.Index) -> pd.DatetimeIndex:
    index = base.index if isinstance(base, pd.DataFrame) else base
    result = _utc_index(index).sort_values()
    if result.has_duplicates:
        raise ValueError("Base timestamps contain duplicates")
    return result


def validate_source_frame(
    data: pd.DataFrame,
    *,
    available_at_col: str | None = None,
    required_columns: Sequence[str] = (),
) -> pd.DataFrame:
    frame = data.copy()
    missing = set(required_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if available_at_col:
        if available_at_col not in frame:
            raise ValueError(f"Missing availability column: {available_at_col}")
        frame[available_at_col] = pd.to_datetime(
            frame[available_at_col], utc=True, errors="raise"
        )
        frame = frame.set_index(available_at_col)
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Source needs a DatetimeIndex or available_at_col")
    frame.index = _utc_index(frame.index)
    frame = frame.sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def safe_asof_align(
    base_index: pd.DatetimeIndex,
    source: pd.DataFrame,
    *,
    prefix: str,
    max_age_hours: float | None,
    available_at_col: str | None = None,
) -> pd.DataFrame:
    """Backward-only causal alignment with optional staleness tolerance."""
    base_index = validate_base_index(base_index)
    source = validate_source_frame(source, available_at_col=available_at_col)
    left = pd.DataFrame({"timestamp": base_index})
    right = source.reset_index(names="source_available_at")
    tolerance = None if max_age_hours is None else pd.Timedelta(hours=max_age_hours)
    aligned = pd.merge_asof(
        left, right,
        left_on="timestamp", right_on="source_available_at",
        direction="backward", tolerance=tolerance,
    ).set_index("timestamp")
    aligned.index = base_index
    aligned[f"{prefix}source_time"] = aligned["source_available_at"]
    aligned[f"{prefix}age_hours"] = (
        aligned.index.to_series() - aligned["source_available_at"]
    ).dt.total_seconds() / 3600.0
    aligned = aligned.drop(columns="source_available_at")
    metadata_columns = {f"{prefix}source_time", f"{prefix}age_hours"}
    return aligned.rename(columns={
        column: f"{prefix}{column}"
        for column in aligned.columns
        if column not in metadata_columns
    })


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _numeric(data: pd.DataFrame) -> pd.DataFrame:
    return data.apply(pd.to_numeric, errors="coerce")


def _fraction(series: pd.Series) -> pd.Series:
    value = pd.to_numeric(series, errors="coerce")
    median = value.dropna().abs().median()
    return value / 100.0 if pd.notna(median) and median > 1.5 else value


def fear_greed_features(
    base_index: pd.DatetimeIndex,
    data: pd.DataFrame,
    config: MacroContextConfig,
    *,
    value_col: str = "value",
    available_at_col: str | None = None,
) -> pd.DataFrame:
    source = validate_source_frame(
        data, available_at_col=available_at_col, required_columns=(value_col,)
    )
    value = pd.to_numeric(source[value_col], errors="coerce").clip(0, 100)
    f = pd.DataFrame(index=source.index)
    f["value"] = value
    f["change_1"] = value.diff()
    f["change_7"] = value.diff(config.short_window)
    f["sma_7"] = value.rolling(config.short_window).mean()
    f["sma_30"] = value.rolling(30).mean()
    f["zscore"] = rolling_zscore(value, config.zscore_window)
    f["regime"] = np.select(
        [
            value <= config.extreme_fear_max,
            value <= config.fear_max,
            value <= config.neutral_max,
            value <= config.greed_max,
        ],
        ["extreme_fear", "fear", "neutral", "greed"],
        default="extreme_greed",
    )
    f["recovering_from_fear"] = (value <= config.fear_max) & (f["change_7"] > 0)
    f["sentiment_overheated"] = (value > config.greed_max) & (f["zscore"] > 1)
    return safe_asof_align(
        base_index, f, prefix="fear_greed_",
        max_age_hours=config.fear_greed_max_age_hours,
    )


def dominance_features(
    base_index: pd.DatetimeIndex,
    data: pd.DataFrame,
    config: MacroContextConfig,
    *,
    available_at_col: str | None = None,
) -> pd.DataFrame:
    """Raw columns can include BTC/stablecoin dominance and TOTAL market caps."""
    source = _numeric(validate_source_frame(data, available_at_col=available_at_col))
    f = pd.DataFrame(index=source.index)
    for column in ("btc_dominance", "stablecoin_dominance"):
        if column not in source:
            continue
        value = _fraction(source[column])
        f[column] = value
        for period in (1, 7, 30):
            f[f"{column}_change_{period}"] = value.diff(period)
        f[f"{column}_ema20"] = ema(value, 20)
        f[f"{column}_ema50"] = ema(value, 50)
        f[f"{column}_zscore"] = rolling_zscore(value, config.zscore_window)
        f[f"{column}_trend"] = np.select(
            [value > f[f"{column}_ema20"] + 0.0025,
             value < f[f"{column}_ema20"] - 0.0025],
            ["rising", "falling"], default="flat",
        )
    for column in (
        "total_market_cap", "total2_market_cap", "total3_market_cap",
        "stablecoin_market_cap",
    ):
        if column not in source:
            continue
        value = source[column]
        f[column] = value
        for period in (1, 7, 30):
            f[f"{column}_return_{period}"] = value.pct_change(period)
        f[f"{column}_ema200"] = ema(value, 200)
        f[f"{column}_above_ema200"] = value > f[f"{column}_ema200"]
    if {"stablecoin_market_cap", "total_market_cap"}.issubset(source.columns):
        f["calculated_stablecoin_dominance"] = (
            source["stablecoin_market_cap"]
            / source["total_market_cap"].replace(0.0, np.nan)
        )
    return safe_asof_align(
        base_index, f, prefix="dominance_",
        max_age_hours=config.dominance_max_age_hours,
    )


def relative_strength_features(
    base_index: pd.DatetimeIndex,
    prices: pd.DataFrame,
    config: MacroContextConfig,
    *,
    btc_column: str = "btc",
    available_at_col: str | None = None,
) -> pd.DataFrame:
    source = _numeric(validate_source_frame(
        prices, available_at_col=available_at_col, required_columns=(btc_column,)
    ))
    f = pd.DataFrame(index=source.index)
    btc = source[btc_column]
    f["btc_price"] = btc
    for period in (1, 7, 30):
        f[f"btc_return_{period}"] = btc.pct_change(period)
    f["btc_ema20"] = ema(btc, 20)
    f["btc_ema50"] = ema(btc, 50)
    f["btc_distance_ema20"] = btc / f["btc_ema20"] - 1
    for asset in source.columns:
        if asset == btc_column:
            continue
        price = source[asset]
        ratio = price / btc.replace(0.0, np.nan)
        f[f"{asset}_price"] = price
        f[f"{asset}_return_7"] = price.pct_change(7)
        f[f"{asset}_return_30"] = price.pct_change(30)
        f[f"{asset}_btc_ratio"] = ratio
        f[f"{asset}_btc_return_7"] = ratio.pct_change(7)
        f[f"{asset}_btc_return_30"] = ratio.pct_change(30)
        f[f"{asset}_btc_ema20"] = ema(ratio, 20)
        f[f"{asset}_outperforming_btc"] = (
            (ratio > f[f"{asset}_btc_ema20"])
            & (f[f"{asset}_btc_return_30"] > 0)
        )
    return safe_asof_align(
        base_index, f, prefix="relative_",
        max_age_hours=config.relative_prices_max_age_hours,
    )


def breadth_features(
    base_index: pd.DatetimeIndex,
    universe_prices: pd.DataFrame,
    config: MacroContextConfig,
    *,
    available_at_col: str | None = None,
) -> pd.DataFrame:
    """Caller must supply a point-in-time universe to avoid survivorship bias."""
    prices = _numeric(validate_source_frame(
        universe_prices, available_at_col=available_at_col
    )).where(lambda x: x > 0)
    e20, e50, e200 = ema(prices, 20), ema(prices, 50), ema(prices, 200)
    r1, r7, r30 = prices.pct_change(), prices.pct_change(7), prices.pct_change(30)
    valid = prices.notna().sum(axis=1).replace(0, np.nan)
    f = pd.DataFrame(index=prices.index)
    f["universe_size"] = valid
    f["above_ema20"] = (prices > e20).sum(axis=1) / valid
    f["above_ema50"] = (prices > e50).sum(axis=1) / valid
    f["above_ema200"] = (prices > e200).sum(axis=1) / valid
    f["positive_7"] = (r7 > 0).sum(axis=1) / valid
    f["positive_30"] = (r30 > 0).sum(axis=1) / valid
    f["new_highs_30"] = (
        prices >= prices.rolling(30, min_periods=30).max()
    ).sum(axis=1) / valid
    f["new_lows_30"] = (
        prices <= prices.rolling(30, min_periods=30).min()
    ).sum(axis=1) / valid
    f["median_return_1"] = r1.median(axis=1)
    f["median_return_7"] = r7.median(axis=1)
    f["equal_weight_return_1"] = r1.mean(axis=1)
    f["dispersion_1"] = r1.std(axis=1, ddof=0)
    f["advance_decline"] = (
        (r1 > 0).sum(axis=1) - (r1 < 0).sum(axis=1)
    ) / valid
    f["thrust"] = f["above_ema20"].diff(7)
    f["breadth_thrust"] = f["thrust"] >= config.breadth_thrust_change
    f["breadth_risk_on"] = f["above_ema50"] >= config.breadth_risk_on
    f["breadth_risk_off"] = f["above_ema50"] <= config.breadth_risk_off
    return safe_asof_align(
        base_index, f, prefix="breadth_",
        max_age_hours=config.breadth_max_age_hours,
    )


def derivatives_features(
    base_index: pd.DatetimeIndex,
    data: pd.DataFrame,
    config: MacroContextConfig,
    *,
    available_at_col: str | None = None,
) -> pd.DataFrame:
    source = _numeric(validate_source_frame(data, available_at_col=available_at_col))
    f = pd.DataFrame(index=source.index)
    if "funding_rate" in source:
        funding = source["funding_rate"]
        f["funding_rate"] = funding
        f["funding_annualized"] = funding * config.annual_funding_periods
        f["funding_zscore"] = rolling_zscore(funding, config.zscore_window)
        f["funding_overheated"] = f["funding_zscore"] >= config.funding_z_overheated
        f["funding_crowded_short"] = f["funding_zscore"] <= config.funding_z_crowded_short
    if "open_interest" in source:
        oi = source["open_interest"]
        f["open_interest"] = oi
        f["open_interest_change_1"] = oi.pct_change()
        f["open_interest_change_24"] = oi.pct_change(24)
        f["open_interest_zscore"] = rolling_zscore(oi, config.zscore_window)
    if "futures_basis" in source:
        f["futures_basis"] = source["futures_basis"]
        f["futures_basis_zscore"] = rolling_zscore(
            source["futures_basis"], config.zscore_window
        )
    if {"long_liquidations", "short_liquidations"}.issubset(source.columns):
        long_liq = source["long_liquidations"].clip(lower=0)
        short_liq = source["short_liquidations"].clip(lower=0)
        total = long_liq + short_liq
        f["long_liquidations"] = long_liq
        f["short_liquidations"] = short_liq
        f["total_liquidations"] = total
        f["liquidation_imbalance"] = (
            long_liq - short_liq
        ) / total.replace(0.0, np.nan)
        f["liquidation_zscore"] = rolling_zscore(total, config.zscore_window)
    if "price" in source:
        price = source["price"]
        f["price"] = price
        f["price_return_24"] = price.pct_change(24)
        f["price_ema20"] = ema(price, 20)
        f["price_distance_ema20"] = price / f["price_ema20"] - 1
    if {"funding_zscore", "open_interest_change_24"}.issubset(f.columns):
        extended = f.get("price_distance_ema20", pd.Series(0.0, index=f.index)) > 0.05
        f["leverage_overheated"] = (
            (f["funding_zscore"] >= config.funding_z_overheated)
            & (f["open_interest_change_24"] >= config.oi_growth_overheated)
            & extended
        )
    if "open_interest_change_24" in f:
        liq_z = f.get("liquidation_zscore", pd.Series(np.nan, index=f.index))
        f["deleveraging_event"] = (
            (f["open_interest_change_24"] <= config.oi_drop_deleveraging)
            & (liq_z >= config.liquidation_z_extreme)
        )
    return safe_asof_align(
        base_index, f, prefix="derivatives_",
        max_age_hours=config.derivatives_max_age_hours,
    )


def etf_flow_features(
    base_index: pd.DatetimeIndex,
    data: pd.DataFrame,
    config: MacroContextConfig,
    *,
    available_at_col: str | None = None,
) -> pd.DataFrame:
    source = _numeric(validate_source_frame(data, available_at_col=available_at_col))
    f = pd.DataFrame(index=source.index)
    for asset in ("btc", "eth"):
        column = f"{asset}_etf_flow"
        if column not in source:
            continue
        flow = source[column]
        f[column] = flow
        f[f"{asset}_etf_flow_5"] = flow.rolling(5).sum()
        f[f"{asset}_etf_flow_20"] = flow.rolling(20).sum()
        f[f"{asset}_etf_flow_cumulative"] = flow.cumsum()
        f[f"{asset}_etf_flow_zscore"] = rolling_zscore(flow, config.zscore_window)
        f[f"{asset}_positive_flow_regime"] = (
            (f[f"{asset}_etf_flow_5"] > 0)
            & (f[f"{asset}_etf_flow_20"] > 0)
        )
    return safe_asof_align(
        base_index, f, prefix="flows_",
        max_age_hours=config.etf_flows_max_age_hours,
    )


def onchain_features(
    base_index: pd.DatetimeIndex,
    data: pd.DataFrame,
    config: MacroContextConfig,
    *,
    available_at_col: str | None = None,
) -> pd.DataFrame:
    source = _numeric(validate_source_frame(data, available_at_col=available_at_col))
    f = pd.DataFrame(index=source.index)
    recognized = (
        "exchange_netflow", "exchange_reserves", "stablecoin_exchange_inflow",
        "mvrv", "sopr", "nupl", "realized_price", "active_addresses",
        "transaction_volume", "fees", "miner_reserves",
        "long_term_holder_supply", "short_term_holder_supply",
    )
    for column in recognized:
        if column not in source:
            continue
        f[column] = source[column]
        f[f"{column}_change_7"] = source[column].pct_change(7)
        f[f"{column}_zscore"] = rolling_zscore(source[column], config.zscore_window)
    if "exchange_netflow" in f:
        f["exchange_outflow_regime"] = (
            (f["exchange_netflow"] < 0)
            & (f["exchange_netflow_zscore"] < -1)
        )
    if "sopr" in f:
        f["sopr_profit_regime"] = f["sopr"] > 1
    if {"mvrv_zscore", "nupl_zscore"}.issubset(f.columns):
        f["onchain_overheated"] = (
            (f["mvrv_zscore"] > 2) & (f["nupl_zscore"] > 1.5)
        )
    return safe_asof_align(
        base_index, f, prefix="onchain_",
        max_age_hours=config.onchain_max_age_hours,
    )


def global_macro_features(
    base_index: pd.DatetimeIndex,
    data: pd.DataFrame,
    config: MacroContextConfig,
    *,
    available_at_col: str | None = None,
) -> pd.DataFrame:
    source = _numeric(validate_source_frame(data, available_at_col=available_at_col))
    f = pd.DataFrame(index=source.index)
    for column in ("dxy", "nasdaq", "sp500", "vix"):
        if column not in source:
            continue
        value = source[column]
        f[column] = value
        f[f"{column}_return_5"] = value.pct_change(5)
        f[f"{column}_return_20"] = value.pct_change(20)
        f[f"{column}_ema50"] = ema(value, 50)
        f[f"{column}_ema200"] = ema(value, 200)
        f[f"{column}_above_ema200"] = value > f[f"{column}_ema200"]
    for column in (
        "us2y", "us10y", "real_yield", "high_yield_spread",
        "m2", "fed_balance_sheet", "reverse_repo",
    ):
        if column not in source:
            continue
        value = source[column]
        f[column] = value
        f[f"{column}_change_5"] = value.diff(5)
        f[f"{column}_change_20"] = value.diff(20)
        f[f"{column}_zscore"] = rolling_zscore(value, config.zscore_window)
    if {"us10y", "us2y"}.issubset(source.columns):
        f["yield_curve_10y_2y"] = source["us10y"] - source["us2y"]
    nasdaq_up = f.get("nasdaq_above_ema200", pd.Series(False, index=f.index)).fillna(False)
    vix_below = (
        f.get("vix", pd.Series(np.nan, index=f.index))
        < f.get("vix_ema50", pd.Series(np.nan, index=f.index))
    ).fillna(False)
    dxy_falling = (
        f.get("dxy_return_20", pd.Series(np.nan, index=f.index)) < 0
    ).fillna(False)
    f["global_risk_on"] = nasdaq_up & vix_below & dxy_falling
    f["global_risk_off"] = (
        (~nasdaq_up)
        & (
            f.get("vix", pd.Series(np.nan, index=f.index)).ge(config.vix_risk_off_level)
            | f.get("high_yield_spread", pd.Series(np.nan, index=f.index)).ge(
                config.high_yield_spread_risk_off
            )
        )
    ).fillna(False)
    return safe_asof_align(
        base_index, f, prefix="global_",
        max_age_hours=config.global_macro_max_age_hours,
    )


def event_risk_features(
    base_index: pd.DatetimeIndex,
    events: pd.DataFrame,
    config: MacroContextConfig,
    *,
    event_time_col: str = "event_time",
    available_at_col: str = "available_at",
    impact_col: str = "impact",
    event_type_col: str = "event_type",
    unlock_pct_col: str = "unlock_pct_float",
) -> pd.DataFrame:
    """Schedule-based risk only. It never uses the future event outcome."""
    if event_time_col not in events:
        raise ValueError(f"Missing {event_time_col}")
    schedule = events.copy()
    schedule[event_time_col] = pd.to_datetime(schedule[event_time_col], utc=True)
    if available_at_col not in schedule:
        schedule[available_at_col] = schedule[event_time_col]
    schedule[available_at_col] = pd.to_datetime(schedule[available_at_col], utc=True)
    for column, default in (
        (impact_col, "unknown"), (event_type_col, "unknown"),
        (unlock_pct_col, np.nan),
    ):
        if column not in schedule:
            schedule[column] = default
    schedule = schedule.sort_values(event_time_col)
    rows: list[dict[str, Any]] = []
    for timestamp in validate_base_index(base_index):
        known = schedule[schedule[available_at_col] <= timestamp]
        future = known[known[event_time_col] >= timestamp]
        past = known[known[event_time_col] < timestamp]
        row: dict[str, Any] = {
            "next_event_time": pd.NaT,
            "hours_to_next_event": np.nan,
            "next_event_impact": None,
            "next_event_type": None,
            "hours_since_last_event": np.nan,
            "high_impact_event_risk": False,
            "token_unlock_risk": False,
            "next_unlock_pct_float": np.nan,
        }
        if not future.empty:
            event = future.iloc[0]
            row["next_event_time"] = event[event_time_col]
            row["hours_to_next_event"] = (
                event[event_time_col] - timestamp
            ).total_seconds() / 3600
            row["next_event_impact"] = str(event[impact_col]).lower()
            row["next_event_type"] = str(event[event_type_col]).lower()
            row["next_unlock_pct_float"] = pd.to_numeric(
                pd.Series([event[unlock_pct_col]]), errors="coerce"
            ).iloc[0]
        if not past.empty:
            row["hours_since_last_event"] = (
                timestamp - past.iloc[-1][event_time_col]
            ).total_seconds() / 3600
        row["high_impact_event_risk"] = bool(
            (
                row["next_event_impact"] == "high"
                and pd.notna(row["hours_to_next_event"])
                and row["hours_to_next_event"] <= config.high_impact_pre_hours
            )
            or (
                pd.notna(row["hours_since_last_event"])
                and row["hours_since_last_event"] <= config.high_impact_post_hours
            )
        )
        unlock = row["next_unlock_pct_float"]
        row["token_unlock_risk"] = bool(
            row["next_event_type"] == "token_unlock"
            and pd.notna(row["hours_to_next_event"])
            and row["hours_to_next_event"] <= config.token_unlock_pre_hours
            and pd.notna(unlock)
            and float(unlock) >= config.material_unlock_pct_float
        )
        rows.append(row)
    return pd.DataFrame(rows, index=validate_base_index(base_index)).add_prefix("events_")


def _column(features: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    return features[name] if name in features else pd.Series(default, index=features.index)


def classify_macro_regimes(
    features: pd.DataFrame,
    config: MacroContextConfig,
) -> pd.DataFrame:
    """Transparent composite labels. Missing inputs contribute zero, not neutral truth."""
    out = pd.DataFrame(index=features.index)
    btc7 = _column(features, "relative_btc_return_7")
    btc30 = _column(features, "relative_btc_return_30")
    btc_dom7 = _column(features, "dominance_btc_dominance_change_7")
    stable_dom7 = _column(features, "dominance_stablecoin_dominance_change_7")
    total7 = _column(features, "dominance_total_market_cap_return_7")
    total3_7 = _column(features, "dominance_total3_market_cap_return_7")
    breadth_on = _column(features, "breadth_breadth_risk_on", False).fillna(False).astype(bool)
    breadth_off = _column(features, "breadth_breadth_risk_off", False).fillna(False).astype(bool)
    global_on = _column(features, "global_global_risk_on", False).fillna(False).astype(bool)
    global_off = _column(features, "global_global_risk_off", False).fillna(False).astype(bool)
    leverage = _column(features, "derivatives_leverage_overheated", False).fillna(False).astype(bool)
    deleveraging = _column(features, "derivatives_deleveraging_event", False).fillna(False).astype(bool)
    event_risk = _column(features, "events_high_impact_event_risk", False).fillna(False).astype(bool)
    unlock_risk = _column(features, "events_token_unlock_risk", False).fillna(False).astype(bool)

    out["btc_led_market"] = ((btc7 > 0) & (btc_dom7 > 0)).fillna(False)
    out["broad_altcoin_market"] = (
        (btc7 > 0) & (btc_dom7 < 0) & (total3_7 > btc7)
    ).fillna(False)
    out["altcoin_capitulation"] = ((btc7 < 0) & (btc_dom7 > 0)).fillna(False)
    out["stablecoin_rotation"] = ((stable_dom7 > 0) & (total7 < 0)).fillna(False)
    out["leverage_overheated"] = leverage
    out["deleveraging_event"] = deleveraging
    out["global_risk_on"] = global_on
    out["global_risk_off"] = global_off
    out["macro_event_risk"] = event_risk
    out["token_unlock_risk"] = unlock_risk

    components = pd.DataFrame(index=features.index)
    components["btc_momentum"] = np.select([btc30 > 0, btc30 < 0], [1, -1], default=0)
    components["breadth"] = np.select([breadth_on, breadth_off], [1, -1], default=0)
    components["stablecoin"] = np.select([stable_dom7 < 0, stable_dom7 > 0], [1, -1], default=0)
    components["global"] = np.select([global_on, global_off], [1, -1], default=0)
    components["leverage"] = np.select([deleveraging, leverage], [-0.5, -1], default=0)
    components["events"] = np.where(event_risk | unlock_risk, -1, 0)
    for column in components:
        out[f"score_component_{column}"] = components[column]
    out["crypto_risk_score"] = components.sum(axis=1)
    out["crypto_risk_on"] = out["crypto_risk_score"] >= 2
    out["crypto_risk_off"] = out["crypto_risk_score"] <= -2
    out["primary_crypto_regime"] = np.select(
        [
            deleveraging,
            leverage,
            out["altcoin_capitulation"],
            out["broad_altcoin_market"],
            out["btc_led_market"],
            out["crypto_risk_on"],
            out["crypto_risk_off"],
            out["stablecoin_rotation"],
        ],
        [
            "deleveraging_event",
            "leverage_overheated",
            "altcoin_capitulation",
            "broad_altcoin_risk_on",
            "btc_led_risk_on",
            "crypto_risk_on",
            "crypto_risk_off",
            "stablecoin_rotation",
        ],
        default="neutral_or_mixed",
    )
    exposure = pd.Series(0.70, index=features.index)
    exposure.loc[out["crypto_risk_on"]] = 1.00
    exposure.loc[out["crypto_risk_off"]] = 0.35
    exposure.loc[out["altcoin_capitulation"]] = 0.25
    exposure.loc[leverage] = np.minimum(exposure.loc[leverage], 0.60)
    exposure.loc[deleveraging] = np.minimum(exposure.loc[deleveraging], 0.50)
    exposure.loc[event_risk | unlock_risk] = np.minimum(
        exposure.loc[event_risk | unlock_risk], 0.50
    )
    out["research_exposure_multiplier"] = exposure
    out["data_completeness"] = features.notna().mean(axis=1) if len(features.columns) else 0.0
    out["macro_context_usable"] = out["data_completeness"] >= 0.40
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
        availability_columns: Mapping[str, str | None] | None = None,
    ) -> pd.DataFrame:
        index = validate_base_index(base)
        availability = dict(availability_columns or {})
        blocks: list[pd.DataFrame] = []
        if fear_greed is not None:
            blocks.append(fear_greed_features(index, fear_greed, self.config,
                available_at_col=availability.get("fear_greed")))
        if dominance is not None:
            blocks.append(dominance_features(index, dominance, self.config,
                available_at_col=availability.get("dominance")))
        if relative_prices is not None:
            blocks.append(relative_strength_features(index, relative_prices, self.config,
                available_at_col=availability.get("relative_prices")))
        if breadth_prices is not None:
            blocks.append(breadth_features(index, breadth_prices, self.config,
                available_at_col=availability.get("breadth_prices")))
        if derivatives is not None:
            blocks.append(derivatives_features(index, derivatives, self.config,
                available_at_col=availability.get("derivatives")))
        if etf_flows is not None:
            blocks.append(etf_flow_features(index, etf_flows, self.config,
                available_at_col=availability.get("etf_flows")))
        if onchain is not None:
            blocks.append(onchain_features(index, onchain, self.config,
                available_at_col=availability.get("onchain")))
        if global_macro is not None:
            blocks.append(global_macro_features(index, global_macro, self.config,
                available_at_col=availability.get("global_macro")))
        if events is not None:
            blocks.append(event_risk_features(index, events, self.config,
                available_at_col=availability.get("events", "available_at")))
        features = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=index)
        return pd.concat([features, classify_macro_regimes(features, self.config)], axis=1)

    @staticmethod
    def latest_snapshot(features: pd.DataFrame) -> dict[str, Any]:
        if features.empty:
            raise ValueError("features is empty")
        row = features.iloc[-1]
        keys = (
            "primary_crypto_regime", "crypto_risk_score", "crypto_risk_on",
            "crypto_risk_off", "btc_led_market", "broad_altcoin_market",
            "altcoin_capitulation", "stablecoin_rotation",
            "leverage_overheated", "deleveraging_event", "global_risk_on",
            "global_risk_off", "macro_event_risk", "token_unlock_risk",
            "research_exposure_multiplier", "data_completeness",
            "macro_context_usable",
        )
        result: dict[str, Any] = {"timestamp": str(features.index[-1])}
        for key in keys:
            value = row.get(key)
            result[key] = value.item() if isinstance(value, np.generic) else value
        return result


def build_macro_features(base: pd.DataFrame | pd.Index, **datasets: Any) -> pd.DataFrame:
    return MacroContextEngine().build(base, **datasets)


def default_macro_parameters() -> dict[str, Any]:
    return asdict(MacroContextConfig())


def load_csv(path: str | Path, timestamp_col: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if timestamp_col is None:
        candidates = [
            column for column in frame.columns
            if str(column).lower() in {
                "timestamp", "datetime", "date", "time", "available_at"
            }
        ]
        if not candidates:
            raise ValueError(f"No timestamp column found in {path}")
        timestamp_col = candidates[0]
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True)
    return frame.set_index(timestamp_col)


__all__ = [
    "FEATURE_GROUPS", "MacroContextConfig", "MacroContextEngine",
    "breadth_features", "build_macro_features", "classify_macro_regimes",
    "default_macro_parameters", "derivatives_features", "dominance_features",
    "ema", "etf_flow_features", "event_risk_features",
    "fear_greed_features", "global_macro_features", "load_csv",
    "onchain_features", "relative_strength_features", "rolling_zscore",
    "safe_asof_align", "validate_base_index", "validate_source_frame",
]
