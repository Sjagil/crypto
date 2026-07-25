"""Central, validated and secret-safe configuration for the crypto system.

Configuration is deliberately grouped by responsibility.  Research can run
without credentials, while live execution remains blocked unless every static
preflight condition is satisfied.  Runtime health and reconciliation checks
are added by the execution layer.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from core.contracts import EligibilityRecord, EligibilityStatus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKETS = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
SUPPORTED_PROVIDERS = frozenset(
    {
        "bitvavo",
        "kraken",
        "mexc",
        "coinmarketcap",
        "eodhd",
        "sec",
        "fred",
        "alternative_me",
        "defillama",
        "deribit",
        "coinglass",
        "glassnode",
        "cryptoquant",
        "coingecko",
        "polygon",
    }
)
SUPPORTED_TIMEFRAMES = (
    "1m",
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
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1_800,
    "1h": 3_600,
    "2h": 7_200,
    "4h": 14_400,
    "6h": 21_600,
    "8h": 28_800,
    "12h": 43_200,
    "1d": 86_400,
    "1W": 604_800,
    "1M": 2_592_000,
}

CsvList = Annotated[list[str], NoDecode]


class _SettingsBase(BaseSettings):
    """Shared behavior for settings groups loaded from the same environment."""

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
        hide_input_in_errors=True,
    )


def _parse_csv(value: Any) -> Any:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, tuple):
        return list(value)
    return value


def normalize_timeframe(value: str) -> str:
    """Normalize canonical intervals without conflating minute and month."""

    selected = value.strip()
    if selected in {"1W", "1w"}:
        return "1W"
    if selected == "1M":
        return "1M"
    return selected.lower()


def _utc_datetime(value: Any) -> Any:
    if value is None or isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if len(text) == 10:
            text = f"{text}T00:00:00+00:00"
        elif text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    else:
        return value
    if parsed is None:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("datetime values must include a UTC offset")
    return parsed.astimezone(UTC)


class AppSettings(_SettingsBase):
    app_name: str = "crypto-spot-research"
    version: str = "1.0.0"
    environment: Literal["development", "test", "staging", "production"] = Field(
        default="development",
        validation_alias=AliasChoices("ENVIRONMENT", "NODE_ENV"),
    )
    debug: bool = False
    timezone: str = "UTC"
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @field_validator("app_name")
    @classmethod
    def validate_app_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("app_name cannot be empty")
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return value


class PathSettings(_SettingsBase):
    project_root: Path = PROJECT_ROOT
    data_dir: Path = Path("data_store")
    raw_data_dir: Path = Path("data_store/raw")
    processed_data_dir: Path = Path("data_store/normalized")
    cache_dir: Path = Path("data_store/cache")
    context_data_dir: Path = Path("data_store/context")
    intelligence_dir: Path = Path("data_store/intelligence")
    output_dir: Path = Path("output")
    reports_dir: Path = Path("output/reports")
    logs_dir: Path = Path("output/logs")
    checkpoints_dir: Path = Path("output/checkpoints")
    test_runs_dir: Path = Path("output/test_runs")
    lab_dir: Path = Path("output/lab")
    database_path: Path = Path("data_store/crypto.db")

    @model_validator(mode="after")
    def resolve_paths(self) -> "PathSettings":
        root = self.project_root.expanduser().resolve()
        self.project_root = root
        for name in (
            "data_dir",
            "raw_data_dir",
            "processed_data_dir",
            "cache_dir",
            "context_data_dir",
            "intelligence_dir",
            "output_dir",
            "reports_dir",
            "logs_dir",
            "checkpoints_dir",
            "test_runs_dir",
            "lab_dir",
            "database_path",
        ):
            path = getattr(self, name).expanduser()
            if not path.is_absolute():
                path = root / path
            setattr(self, name, path.resolve())
        return self

    def create_directories(self) -> tuple[Path, ...]:
        directories = (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.cache_dir,
            self.context_data_dir,
            self.intelligence_dir,
            self.output_dir,
            self.reports_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.test_runs_dir,
            self.lab_dir,
            self.database_path.parent,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return directories


class MarketDataSettings(_SettingsBase):
    providers: CsvList = Field(default_factory=lambda: ["bitvavo", "kraken", "coinmarketcap"])
    symbols: CsvList = Field(
        default_factory=lambda: list(DEFAULT_MARKETS),
        validation_alias=AliasChoices("SYMBOLS", "MARKETS"),
    )
    quote_currency: str = Field(
        default="EUR",
        validation_alias=AliasChoices("QUOTE_CURRENCY", "PORTFOLIO_BASE_CURRENCY"),
    )
    timeframes: CsvList = Field(default_factory=lambda: ["5m", "15m", "1h", "4h", "1d"])
    base_timeframe: str = "1h"
    start_date: datetime | None = None
    end_date: datetime | None = None
    closed_candles_only: bool = True
    maximum_staleness: timedelta = timedelta(hours=6)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    maximum_retries: int = Field(default=3, ge=1, le=10)
    websocket_queue_size: int = Field(default=2_000, ge=10, le=1_000_000)
    websocket_inactivity_seconds: float = Field(default=45.0, gt=1.0)
    orderbook_maximum_depth: int = Field(default=100, ge=1, le=5_000)
    maximum_concurrent_providers: int = Field(default=3, ge=1, le=16)
    maximum_requests_per_provider: int = Field(default=50_000, ge=1)
    maximum_database_batch_size: int = Field(default=5_000, ge=1, le=100_000)
    maximum_raw_retention_days: int = Field(default=3650, ge=1)
    maximum_orderbook_retention_days: int = Field(default=30, ge=1)
    maximum_trade_retention_days: int = Field(default=90, ge=1)
    maximum_storage_gb: float = Field(default=50.0, gt=0)
    minimum_free_disk_gb: float = Field(default=2.0, ge=0)
    retry_budget: int = Field(default=10, ge=0, le=1_000)
    provider_cooldown_seconds: float = Field(default=0.05, ge=0, le=60)
    candle_close_grace_seconds: float = Field(default=5.0, ge=0, le=600)
    candle_close_grace_seconds_by_timeframe: dict[str, float] = Field(
        default_factory=lambda: {
            "1m": 3.0,
            "5m": 5.0,
            "15m": 8.0,
            "30m": 10.0,
            "1h": 15.0,
            "2h": 20.0,
            "4h": 30.0,
            "6h": 45.0,
            "8h": 60.0,
            "12h": 60.0,
            "1d": 120.0,
            "1w": 300.0,
            "1M": 600.0,
        }
    )

    @field_validator("providers", "symbols", "timeframes", mode="before")
    @classmethod
    def parse_lists(cls, value: Any) -> Any:
        return _parse_csv(value)

    @field_validator("providers")
    @classmethod
    def validate_providers(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(item.strip().lower() for item in values))
        unknown = sorted(set(normalized) - SUPPORTED_PROVIDERS)
        if not normalized or unknown:
            raise ValueError(f"unsupported market-data providers: {unknown}")
        return normalized

    @field_validator("quote_currency")
    @classmethod
    def validate_quote_currency(cls, value: str) -> str:
        value = value.strip().upper()
        if value != "EUR":
            raise ValueError("active execution and accounting require EUR quote currency")
        return value

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(item.strip().upper().replace("/", "-") for item in values))
        if not normalized:
            raise ValueError("at least one market symbol is required")
        for symbol in normalized:
            parts = symbol.split("-")
            if len(parts) != 2 or not all(part.isalnum() for part in parts):
                raise ValueError(f"invalid spot market symbol: {symbol}")
        return normalized

    @field_validator("timeframes")
    @classmethod
    def validate_timeframes(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(normalize_timeframe(item) for item in values))
        unknown = sorted(set(normalized) - set(SUPPORTED_TIMEFRAMES))
        if not normalized or unknown:
            raise ValueError(f"unsupported timeframes: {unknown}")
        return sorted(normalized, key=TIMEFRAME_SECONDS.__getitem__)

    @field_validator("candle_close_grace_seconds_by_timeframe")
    @classmethod
    def validate_candle_close_grace_by_timeframe(
        cls,
        values: dict[str, float],
    ) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for timeframe, seconds in values.items():
            selected = normalize_timeframe(timeframe)
            delay = float(seconds)
            if delay < 0 or delay > 600:
                raise ValueError(
                    "per-timeframe candle close grace must be between 0 and 600 seconds"
                )
            normalized[selected] = delay
        return normalized

    def candle_close_grace_for(self, timeframe: str) -> float:
        """Return the post-boundary delay before a candle is trusted as closed."""

        selected = normalize_timeframe(timeframe)
        return self.candle_close_grace_seconds_by_timeframe.get(
            selected,
            self.candle_close_grace_seconds,
        )

    @field_validator("base_timeframe")
    @classmethod
    def validate_base_timeframe(cls, value: str) -> str:
        value = normalize_timeframe(value)
        if value not in SUPPORTED_TIMEFRAMES:
            raise ValueError("base_timeframe is unsupported")
        return value

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def normalize_dates(cls, value: Any) -> Any:
        return _utc_datetime(value)

    @model_validator(mode="after")
    def validate_consistency(self) -> "MarketDataSettings":
        if self.base_timeframe not in self.timeframes:
            raise ValueError("base_timeframe must be included in timeframes")
        if any(not symbol.endswith(f"-{self.quote_currency}") for symbol in self.symbols):
            raise ValueError("every executable market must use the configured EUR quote")
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date")
        if self.maximum_staleness <= timedelta(0):
            raise ValueError("maximum_staleness must be positive")
        if not self.closed_candles_only:
            raise ValueError("closed_candles_only cannot be disabled")
        return self


class CostSettings(_SettingsBase):
    maker_fee: float = Field(default=0.0015, ge=0.0, le=0.05)
    taker_fee: float = Field(default=0.0025, ge=0.0, le=0.05)
    default_fee: float = Field(default=0.0025, ge=0.0, le=0.05)
    slippage_bps: float = Field(default=8.0, ge=0.0, le=1_000.0)
    spread_bps: float = Field(default=5.0, ge=0.0, le=1_000.0)
    stressed_cost_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)

    @model_validator(mode="after")
    def validate_default_fee(self) -> "CostSettings":
        if self.default_fee < min(self.maker_fee, self.taker_fee):
            raise ValueError("default_fee cannot be below both venue fee assumptions")
        return self


class RiskSettings(_SettingsBase):
    risk_per_trade: float = Field(default=0.005, gt=0.0, le=0.02)
    maximum_risk_per_trade: float = Field(default=0.02, gt=0.0, le=0.02)
    maximum_live_risk_per_trade: float = Field(default=0.01, gt=0.0, le=0.01)
    maximum_research_risk_per_trade: float = Field(default=0.02, gt=0.0, le=0.02)
    maximum_total_open_risk: float = Field(default=0.02, gt=0.0, le=0.10)
    maximum_position_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    maximum_portfolio_exposure: float = Field(default=0.75, gt=0.0, le=1.0)
    maximum_daily_loss: float = Field(default=0.02, gt=0.0, le=0.20)
    maximum_portfolio_drawdown: float = Field(default=0.15, gt=0.0, le=0.50)
    maximum_trades_per_day: int = Field(default=6, ge=1, le=100)
    minimum_stop_distance: float = Field(default=0.0025, gt=0.0, le=0.25)
    reserve_cash_fraction: float = Field(default=0.10, ge=0.0, lt=1.0)

    @property
    def maximum_open_risk(self) -> float:
        return self.maximum_total_open_risk

    @property
    def maximum_drawdown(self) -> float:
        return self.maximum_portfolio_drawdown

    @model_validator(mode="after")
    def validate_risk_hierarchy(self) -> "RiskSettings":
        if self.risk_per_trade > self.maximum_risk_per_trade:
            raise ValueError("risk_per_trade exceeds maximum_risk_per_trade")
        if self.risk_per_trade > self.maximum_research_risk_per_trade:
            raise ValueError("risk_per_trade exceeds the research risk cap")
        if self.maximum_live_risk_per_trade > self.maximum_risk_per_trade:
            raise ValueError("live risk cap exceeds the absolute risk cap")
        if self.maximum_total_open_risk < self.risk_per_trade:
            raise ValueError("maximum_total_open_risk is below per-trade risk")
        if self.maximum_portfolio_exposure + self.reserve_cash_fraction > 1.0:
            raise ValueError("portfolio exposure plus cash reserve cannot exceed equity")
        return self


class ResearchSettings(_SettingsBase):
    minimum_trades: int = Field(default=100, ge=1)
    minimum_effective_sample_size: int = Field(default=60, ge=1)
    minimum_profit_factor: float = Field(default=1.15, gt=0.0)
    minimum_stressed_profit_factor: float = Field(default=1.00, gt=0.0)
    minimum_net_expectancy_r: float = 0.0
    minimum_positive_folds: int = Field(default=5, ge=1)
    minimum_cpcv_path_consistency: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )
    minimum_deflated_sharpe_probability: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
    )
    maximum_probability_of_backtest_overfitting: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
    )
    maximum_white_reality_check_pvalue: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
    )
    maximum_hansen_spa_pvalue: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )
    multiple_testing_bootstrap_samples: int = Field(default=2_000, ge=2_000)
    multiple_testing_block_size: int = Field(default=5, ge=1, le=1_000)
    maximum_drawdown: float = Field(default=0.20, gt=0.0, le=1.0)
    maximum_monte_carlo_probability_of_loss: float = Field(default=0.35, ge=0.0, le=1.0)
    maximum_monte_carlo_probability_of_20pct_drawdown: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
    )
    maximum_probability_of_30pct_drawdown: float = Field(default=0.10, ge=0.0, le=1.0)
    maximum_dirichlet_probability_of_loss: float = Field(default=0.05, ge=0.0, le=1.0)
    minimum_stochastic_p05_total_return: float = Field(default=0.0, gt=-1.0)
    dirichlet_block_count: int = Field(default=12, ge=4, le=100)
    maximum_symbol_profit_concentration: float = Field(default=0.60, gt=0.0, le=1.0)
    maximum_fold_profit_concentration: float = Field(default=0.50, gt=0.0, le=1.0)
    bootstrap_samples: int = Field(default=10_000, ge=100)
    monte_carlo_runs: int = Field(default=10_000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    walk_forward_folds: int = Field(default=6, ge=2)
    stressed_cost_required: bool = True
    parameter_stability_required: bool = True

    @model_validator(mode="after")
    def validate_research_gates(self) -> "ResearchSettings":
        if self.minimum_effective_sample_size > self.minimum_trades:
            raise ValueError("effective sample minimum cannot exceed minimum trades")
        if self.minimum_positive_folds > self.walk_forward_folds:
            raise ValueError("minimum_positive_folds exceeds walk_forward_folds")
        if self.minimum_stressed_profit_factor > self.minimum_profit_factor:
            raise ValueError("stressed profit-factor gate cannot exceed the normal gate")
        return self


class LabSettings(_SettingsBase):
    """Bounded defaults for the continuous combinatorial research service."""

    model_config = SettingsConfigDict(env_prefix="LAB_")

    universe_target_size: int = Field(default=25, ge=20, le=25)
    universe_scan_limit: int = Field(default=100, ge=25, le=5_000)
    minimum_volume_24h_eur: float = Field(default=10_000_000.0, ge=0.0)
    minimum_history_rows: int = Field(default=500, ge=100)
    deep_minimum_history_rows: int = Field(default=2_000, ge=500)
    deep_history_mode: Literal[
        "smoke",
        "bounded",
        "common_full_history",
        "asset_max_history",
    ] = "common_full_history"
    confirmation_job_threshold: int = Field(default=100_000, ge=1)
    maximum_generation_rows: int = Field(default=250_000, ge=100)
    maximum_retries: int = Field(default=3, ge=0, le=20)
    max_workers: int = Field(
        default_factory=lambda: max(1, (os.cpu_count() or 2) - 2),
        ge=1,
    )
    cpu_limit: int | None = Field(default=None, ge=1)
    memory_limit_mb: int = Field(default=4_096, ge=256)
    trial_timeout_seconds: float = Field(default=300.0, gt=0.0)
    combination_timeout_seconds: float = Field(default=3_600.0, gt=0.0)
    heartbeat_seconds: float = Field(default=10.0, gt=0.0)
    idle_poll_seconds: float = Field(default=1.0, gt=0.0)
    universe_refresh_hours: float = Field(default=24.0, gt=0.0)
    liquidity_refresh_hours: float = Field(default=6.0, gt=0.0)
    leaderboard_refresh_minutes: float = Field(default=15.0, gt=0.0)
    leader_retest_hours: float = Field(default=24.0, gt=0.0)
    walk_forward_revalidation_days: float = Field(default=7.0, gt=0.0)
    universe_rebalance_days: float = Field(default=7.0, gt=0.0)
    overfitting_audit_days: float = Field(default=30.0, gt=0.0)
    deterministic_seed: int = 20260317


class OperationalSettings(_SettingsBase):
    """Fail-closed defaults for the practical public-data operating loop."""

    model_config = SettingsConfigDict(env_prefix="OPERATE_")

    profile_name: str = "practical_spot_v1"
    markets: CsvList = Field(default_factory=lambda: ["BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"])
    execution_timeframe: str = "1h"
    trend_timeframe: str = "4h"
    regime_timeframe: str = "1d"
    risk_per_trade: float = Field(default=0.0025, gt=0.0, le=0.005)
    maximum_risk_per_trade: float = Field(default=0.005, gt=0.0, le=0.005)
    maximum_total_open_risk: float = Field(default=0.01, gt=0.0, le=0.01)
    maximum_positions: int = Field(default=2, ge=1, le=2)
    maximum_position_fraction: float = Field(default=0.20, gt=0.0, le=0.20)
    maximum_portfolio_exposure: float = Field(default=0.40, gt=0.0, le=0.40)
    reserve_cash_fraction: float = Field(default=0.20, ge=0.20, lt=1.0)
    maximum_daily_loss: float = Field(default=0.01, gt=0.0, le=0.01)
    drawdown_warning: float = Field(default=0.025, gt=0.0, le=0.025)
    drawdown_block_new_entries: float = Field(default=0.04, gt=0.0, le=0.04)
    drawdown_kill_switch: float = Field(default=0.05, gt=0.0, le=0.05)
    paper_order_cap_eur: float = Field(default=250.0, gt=0.0)
    cycle_seconds: float = Field(default=60.0, gt=0.0, le=3600.0)
    database_read_latency_limit_ms: float = Field(
        default=1_000.0,
        gt=0.0,
        le=30_000.0,
    )
    database_write_latency_limit_ms: float = Field(
        default=1_500.0,
        gt=0.0,
        le=30_000.0,
    )
    control_wait_seconds: float = Field(default=120.0, ge=0.0, le=600.0)
    alert_cooldown_seconds: float = Field(default=300.0, ge=0.0)
    windows_task_name: str = "CryptoPracticalSpotShadow"
    task_start_trigger: Literal["logon", "startup"] = "logon"
    task_restart_count: int = Field(default=3, ge=0, le=10)

    @field_validator("markets", mode="before")
    @classmethod
    def parse_markets(cls, value: Any) -> Any:
        return _parse_csv(value)

    @field_validator("markets")
    @classmethod
    def validate_markets(cls, values: list[str]) -> list[str]:
        normalized = [
            str(value).strip().upper().replace("/", "-").replace("_", "-") for value in values
        ]
        if any(
            len(market.split("-")) != 2 or not all(part.isalnum() for part in market.split("-"))
            for market in normalized
        ):
            raise ValueError("operational markets must use BASE-QUOTE format")
        if not normalized:
            raise ValueError("operational profile requires at least one market")
        return normalized

    @field_validator("execution_timeframe", "trend_timeframe", "regime_timeframe")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        return normalize_timeframe(value)

    @model_validator(mode="after")
    def validate_operational_limits(self) -> "OperationalSettings":
        if self.risk_per_trade > self.maximum_risk_per_trade:
            raise ValueError("operational base risk exceeds its hard maximum")
        if self.maximum_total_open_risk < self.maximum_risk_per_trade:
            raise ValueError("operational open-risk cap is below per-trade cap")
        if (
            self.drawdown_warning >= self.drawdown_block_new_entries
            or self.drawdown_block_new_entries >= self.drawdown_kill_switch
        ):
            raise ValueError("operational drawdown thresholds are not increasing")
        if self.maximum_portfolio_exposure + self.reserve_cash_fraction > 1.0:
            raise ValueError("operational exposure plus reserve exceeds equity")
        return self


class ExecutionSettings(_SettingsBase):
    mode: Literal["research", "backtest", "optimize", "shadow", "paper", "live"] = Field(
        default="research",
        validation_alias=AliasChoices("EXECUTION_MODE", "MODE"),
    )
    live_trading_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("LIVE_TRADING_ENABLED", "LIVE_TRADING_ALLOWED"),
    )
    spot_only: bool = True
    allow_margin: bool = False
    allow_leverage: bool = False
    allow_short_selling: bool = False
    allow_derivatives: bool = False
    allow_futures: bool = False
    allow_options: bool = False
    allow_borrowing: bool = False
    allow_lending: bool = False
    allow_staking: bool = False
    withdrawals_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("WITHDRAWALS_ENABLED", "BITVAVO_WITHDRAWALS_ENABLED"),
    )
    manual_approval_phrase: SecretStr | None = None
    required_manual_approval_phrase: str = "I UNDERSTAND LIVE CRYPTO SPOT RISK"
    maximum_live_order_eur: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def enforce_spot_only(self) -> "ExecutionSettings":
        forbidden = {
            "allow_margin": self.allow_margin,
            "allow_leverage": self.allow_leverage,
            "allow_short_selling": self.allow_short_selling,
            "allow_derivatives": self.allow_derivatives,
            "allow_futures": self.allow_futures,
            "allow_options": self.allow_options,
            "allow_borrowing": self.allow_borrowing,
            "allow_lending": self.allow_lending,
            "allow_staking": self.allow_staking,
            "withdrawals_enabled": self.withdrawals_enabled,
        }
        enabled = sorted(name for name, value in forbidden.items() if value)
        if not self.spot_only or enabled:
            raise ValueError(f"forbidden execution capabilities enabled: {enabled}")
        return self

    def approval_phrase_matches(self) -> bool:
        return bool(
            self.manual_approval_phrase
            and self.manual_approval_phrase.get_secret_value()
            == self.required_manual_approval_phrase
        )


class ProviderSettings(_SettingsBase):
    bitvavo_operator_id: int | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("BITVAVO_OPERATOR_ID", "WM_OPERATOR_ID"),
    )
    bitvavo_data_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("BITVAVO_DATA_API_KEY", "BITVAVO_API_KEY"),
    )
    bitvavo_data_api_secret: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("BITVAVO_DATA_API_SECRET", "BITVAVO_API_SECRET"),
    )
    bitvavo_trade_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("BITVAVO_TRADE_API_KEY", "VENUE_A_API_KEY"),
    )
    bitvavo_trade_api_secret: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("BITVAVO_TRADE_API_SECRET", "VENUE_A_API_SECRET"),
    )
    kraken_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("KRAKEN_API_KEY", "VENUE_B_API_KEY"),
    )
    kraken_api_secret: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("KRAKEN_API_SECRET", "VENUE_B_API_SECRET"),
    )
    coinmarketcap_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("COINMARKETCAP_API_KEY", "CMC_API_KEY"),
    )
    mexc_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("MEXC_API_KEY", "MEXC_DATA_API_KEY"),
    )
    mexc_data_api_secret: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("MEXC_DATA_API_SECRET", "MEXC_API_SECRET"),
    )
    eodhd_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("EODHD_API_KEY", "EOD_API_KEY", "EODHISTORICALDATA_API_KEY"),
    )
    fred_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("FRED_API_KEY"),
    )
    sec_user_agent: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SEC_USER_AGENT"),
    )
    polygon_api_key: SecretStr | None = None
    coinglass_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("COINGLASS_API_KEY"),
    )
    glassnode_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GLASSNODE_API_KEY"),
    )
    cryptoquant_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CRYPTOQUANT_API_KEY"),
    )
    coingecko_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("COINGECKO_API_KEY"),
    )
    dune_api_key: SecretStr | None = None
    coinpaprika_api_key: SecretStr | None = None
    fmp_api_key: SecretStr | None = None
    twelve_data_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("TWELVEDATA_API_KEY"),
    )
    alpha_vantage_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ALPHAVANTAGE_API_KEY"),
    )
    finnhub_api_key: SecretStr | None = None
    marketstack_api_key: SecretStr | None = None
    theta_data_api_key: SecretStr | None = None
    fiscal_api_key: SecretStr | None = None
    current_news_api_key: SecretStr | None = None
    marketaux_api_token: SecretStr | None = None
    openfigi_api_key: SecretStr | None = None
    open_exchange_rates_app_id: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENEXCHANGERATES_APP_ID",
            "OPENEXCHANGE_API_KEY",
        ),
    )
    database_url: SecretStr | None = None
    bitvavo_trade_key_scope: str = Field(
        default="unknown",
        validation_alias=AliasChoices("BITVAVO_TRADE_KEY_SCOPE", "VENUE_A_KEY_SCOPE"),
    )
    bitvavo_data_key_scope: str = Field(
        default="public",
        validation_alias=AliasChoices("BITVAVO_DATA_KEY_SCOPE", "VENUE_A_DATA_KEY_SCOPE"),
    )
    bitvavo_withdrawal_permission: bool = False
    bitvavo_ip_whitelist_confirmed: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "BITVAVO_IP_WHITELIST_CONFIRMED", "EXCHANGE_IP_WHITELIST_CONFIRMED"
        ),
    )

    @property
    def bitvavo_api_key(self) -> SecretStr | None:
        return self.bitvavo_trade_api_key or self.bitvavo_data_api_key

    @property
    def bitvavo_api_secret(self) -> SecretStr | None:
        return self.bitvavo_trade_api_secret or self.bitvavo_data_api_secret

    def has_trade_credentials(self) -> bool:
        return bool(self.bitvavo_trade_api_key and self.bitvavo_trade_api_secret)

    def unsafe_trade_scope(self) -> bool:
        scope = self.bitvavo_trade_key_scope.casefold()
        return self.bitvavo_withdrawal_permission or "withdraw" in scope


class ScraperSettings(_SettingsBase):
    scrapers_enabled: bool = True
    rss_enabled: bool = True
    playwright_fallback_enabled: bool = True
    maximum_concurrency: int = Field(default=5, ge=1, le=20)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    maximum_retries: int = Field(default=3, ge=0, le=10)
    backoff_base_seconds: float = Field(default=0.5, ge=0.0, le=30.0)
    minimum_crypto_relevance_score: float = Field(default=0.50, ge=0.0, le=1.0)
    unknown_publication_time_policy: Literal["forward_only", "reject"] = "forward_only"
    sentiment_as_direct_signal: bool = False
    stale_news_after: timedelta = timedelta(days=2)

    @model_validator(mode="after")
    def enforce_intelligence_safety(self) -> "ScraperSettings":
        if self.sentiment_as_direct_signal:
            raise ValueError("scraped sentiment cannot be a direct trading signal")
        if self.stale_news_after <= timedelta(0):
            raise ValueError("stale_news_after must be positive")
        return self


class ShariahSettings(BaseModel):
    """User-maintained eligibility configuration; it is not certification."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    source_path: Path
    version: int = Field(ge=1)
    markets: dict[str, EligibilityRecord]

    @classmethod
    def load(cls, path: Path) -> "ShariahSettings":
        source = path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Shariah allowlist not found: {source}")
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        raw_markets = raw.get("markets")
        if not isinstance(raw_markets, dict):
            raise ValueError("allowlist must contain a markets mapping")
        records: dict[str, EligibilityRecord] = {}
        for market, details in raw_markets.items():
            normalized = str(market).strip().upper().replace("/", "-")
            if not isinstance(details, dict):
                raise ValueError(f"invalid allowlist record for {normalized}")
            records[normalized] = EligibilityRecord(market=normalized, **details)
        return cls(
            source_path=source,
            version=raw.get("version", 1),
            markets=records,
        )

    def eligibility(self, market: str) -> EligibilityRecord:
        normalized = market.strip().upper().replace("/", "-")
        return self.markets.get(
            normalized,
            EligibilityRecord(
                market=normalized,
                status=EligibilityStatus.REVIEW_REQUIRED,
                reason="UNKNOWN_MARKET_FAIL_CLOSED",
            ),
        )

    def require_allowed(self, market: str) -> EligibilityRecord:
        result = self.eligibility(market)
        if result.status is not EligibilityStatus.ALLOWED:
            raise PermissionError(f"market is not ALLOWED: {result.market} ({result.reason})")
        return result


class Settings(BaseModel):
    """Aggregate configuration used by application composition roots."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    app: AppSettings
    paths: PathSettings
    market_data: MarketDataSettings
    costs: CostSettings
    risk: RiskSettings
    research: ResearchSettings
    lab: LabSettings
    operational: OperationalSettings
    execution: ExecutionSettings
    providers: ProviderSettings
    scrapers: ScraperSettings
    shariah: ShariahSettings

    @classmethod
    def load(
        cls,
        *,
        env_file: Path | str | None = None,
        create_directories: bool = True,
        allowlist_path: Path | str | None = None,
    ) -> "Settings":
        selected_env = Path(env_file).resolve() if env_file else PROJECT_ROOT / ".env"

        def group(settings_type: type[_SettingsBase]) -> _SettingsBase:
            return settings_type(_env_file=selected_env if selected_env.is_file() else None)

        paths = group(PathSettings)
        assert isinstance(paths, PathSettings)
        if create_directories:
            paths.create_directories()
        selected_allowlist = (
            Path(allowlist_path).resolve()
            if allowlist_path
            else paths.project_root / "config" / "shariah_allowlist.yaml"
        )
        return cls(
            app=group(AppSettings),
            paths=paths,
            market_data=group(MarketDataSettings),
            costs=group(CostSettings),
            risk=group(RiskSettings),
            research=group(ResearchSettings),
            lab=group(LabSettings),
            operational=group(OperationalSettings),
            execution=group(ExecutionSettings),
            providers=group(ProviderSettings),
            scrapers=group(ScraperSettings),
            shariah=ShariahSettings.load(selected_allowlist),
        )

    def redacted_dict(self) -> dict[str, Any]:
        def redact(value: Any) -> Any:
            if isinstance(value, SecretStr):
                return "***REDACTED***" if value.get_secret_value() else None
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, timedelta):
                return value.total_seconds()
            if isinstance(value, BaseModel):
                return {name: redact(getattr(value, name)) for name in type(value).model_fields}
            if isinstance(value, dict):
                return {str(key): redact(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [redact(item) for item in value]
            if isinstance(value, StrEnum):
                return value.value
            return value

        return redact(self)

    def static_live_preflight_failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.app.environment != "production":
            failures.append("LIVE_BLOCKED_NOT_PRODUCTION")
        if self.execution.mode != "live":
            failures.append("LIVE_BLOCKED_MODE_NOT_LIVE")
        if not self.execution.live_trading_enabled:
            failures.append("LIVE_BLOCKED_DISABLED")
        if not self.execution.approval_phrase_matches():
            failures.append("LIVE_BLOCKED_MANUAL_APPROVAL")
        if not self.providers.has_trade_credentials():
            failures.append("LIVE_BLOCKED_MISSING_TRADE_CREDENTIALS")
        if self.providers.bitvavo_operator_id is None:
            failures.append("LIVE_BLOCKED_MISSING_OPERATOR_ID")
        if self.providers.unsafe_trade_scope():
            failures.append("LIVE_BLOCKED_UNSAFE_CREDENTIAL_SCOPE")
        if not self.providers.bitvavo_ip_whitelist_confirmed:
            failures.append("LIVE_BLOCKED_IP_WHITELIST_UNCONFIRMED")
        if self.execution.maximum_live_order_eur is None:
            failures.append("LIVE_BLOCKED_MISSING_ORDER_CAP")
        for market in self.market_data.symbols:
            if self.shariah.eligibility(market).status is not EligibilityStatus.ALLOWED:
                failures.append(f"LIVE_BLOCKED_ELIGIBILITY:{market}")
        return tuple(failures)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


__all__ = [
    "AppSettings",
    "CostSettings",
    "DEFAULT_MARKETS",
    "EligibilityRecord",
    "EligibilityStatus",
    "ExecutionSettings",
    "LabSettings",
    "MarketDataSettings",
    "OperationalSettings",
    "PathSettings",
    "ProviderSettings",
    "ResearchSettings",
    "RiskSettings",
    "ScraperSettings",
    "Settings",
    "ShariahSettings",
    "SUPPORTED_PROVIDERS",
    "SUPPORTED_TIMEFRAMES",
    "TIMEFRAME_SECONDS",
    "get_settings",
    "normalize_timeframe",
]
