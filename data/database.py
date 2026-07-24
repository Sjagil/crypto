"""SQLAlchemy persistence with explicit tables, idempotent upserts and exports."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError

from utils.common import stable_hash, stable_json, utc_now

SCHEMA_VERSION = 4
TABLE_NAMES = (
    "candles",
    "trades",
    "ticker_events",
    "orderbook_snapshots",
    "orderbook_statistics",
    "provider_health",
    "provider_capabilities",
    "data_watermarks",
    "data_service_state",
    "raw_manifests",
    "scraper_intelligence",
    "macro_observations",
    "derivatives_context",
    "strategy_signals",
    "backtest_trades",
    "orders",
    "fills",
    "balances",
    "positions",
    "pnl_snapshots",
    "risk_events",
    "kill_switch_events",
    "test_runs",
    "generated_reports",
    "universe_snapshots",
    "universe_members",
    "signal_blocks",
    "strategy_combinations",
    "combination_blocks",
    "parameter_spaces",
    "experiment_jobs",
    "experiment_trials",
    "baseline_results",
    "exact_backtest_results",
    "walk_forward_results",
    "monte_carlo_results",
    "gate_results",
    "leaderboard_entries",
    "leaderboard_snapshots",
    "lab_heartbeats",
    "lab_events",
    "cache_manifests",
    "indicator_registry",
    "indicator_availability",
    "investment_scores",
    "fractal_events",
    "fractal_research_labels",
)
LOGGER = logging.getLogger("crypto.database")


def _utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class Database:
    def __init__(
        self,
        database_url: str | None = None,
        *,
        sqlite_path: Path | str = Path("data_store/crypto.db"),
        retries: int = 3,
    ) -> None:
        if database_url:
            if not database_url.startswith(
                ("sqlite://", "postgresql://", "postgresql+")
            ):
                raise ValueError("DATABASE_URL must be a SQLite or PostgreSQL SQLAlchemy URL")
            url = database_url
        else:
            path = Path(sqlite_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{path.as_posix()}"
        self.engine = create_engine(url, future=True, pool_pre_ping=True)
        self.retries = retries
        self.metadata = MetaData()
        self.version = Table(
            "schema_version",
            self.metadata,
            Column("version", Integer, primary_key=True),
            Column("applied_at", DateTime(timezone=True), nullable=False),
            Column("description", String(200), nullable=False),
        )
        self.tables: dict[str, Table] = {}
        for name in TABLE_NAMES:
            table = Table(
                name,
                self.metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("external_id", String(128), nullable=False),
                Column("provider", String(64), nullable=True),
                Column("market", String(64), nullable=True),
                Column("timeframe", String(32), nullable=True),
                Column("timestamp", DateTime(timezone=True), nullable=True),
                Column("observed_at", DateTime(timezone=True), nullable=True),
                Column("available_at", DateTime(timezone=True), nullable=True),
                Column("status", String(64), nullable=True),
                Column("numeric_value", Float, nullable=True),
                Column("closed", Boolean, nullable=True),
                Column("payload", JSON, nullable=False),
                Column("created_at", DateTime(timezone=True), nullable=False),
                Column("updated_at", DateTime(timezone=True), nullable=False),
                UniqueConstraint("external_id", name=f"uq_{name}_external_id"),
            )
            Index(f"ix_{name}_provider_market_time", table.c.provider, table.c.market, table.c.timestamp)
            Index(f"ix_{name}_available_at", table.c.available_at)
            Index(f"ix_{name}_status_updated", table.c.status, table.c.updated_at)
            self.tables[name] = table
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", self._configure_sqlite)

    @staticmethod
    def _configure_sqlite(dbapi_connection: Any, connection_record: Any) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    def migrate(self) -> int:
        self.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            current = connection.scalar(select(func.max(self.version.c.version)))
            if current is None:
                connection.execute(
                    insert(self.version).values(
                        version=SCHEMA_VERSION,
                        applied_at=utc_now(),
                        description="initial data, operational and lab schemas",
                    )
                )
            elif current > SCHEMA_VERSION:
                raise RuntimeError("database schema is newer than this application")
            elif current < SCHEMA_VERSION:
                connection.execute(
                    insert(self.version).values(
                        version=SCHEMA_VERSION,
                        applied_at=utc_now(),
                        description=(
                            "provider capabilities, watermarks and data-service schemas"
                        ),
                    )
                )
        return SCHEMA_VERSION

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        with self.engine.begin() as connection:
            yield connection

    @staticmethod
    def _external_id(table_name: str, record: dict[str, Any]) -> str:
        if record.get("external_id"):
            return stable_hash([table_name, str(record["external_id"])], length=64)
        for key in (
            "fill_id",
            "order_id",
            "trade_id",
            "event_id",
            "message_id",
            "raw_hash",
            "retrieval_run_id",
            "run_id",
        ):
            if record.get(key):
                prefix = str(record[key])
                qualifiers = (
                    record.get("timestamp"),
                    record.get("canonical_market") or record.get("market"),
                    record.get("data_kind"),
                    record.get("timeframe"),
                )
                return stable_hash([table_name, prefix, qualifiers], length=64)
        return stable_hash([table_name, record], length=64)

    def _row(self, table_name: str, record: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        values = record.get("values") if isinstance(record.get("values"), dict) else {}
        return {
            "external_id": self._external_id(table_name, record),
            "provider": record.get("provider") or record.get("venue"),
            "market": record.get("canonical_market") or record.get("market"),
            "timeframe": record.get("timeframe"),
            "timestamp": _utc(record.get("timestamp") or record.get("filled_at")),
            "observed_at": _utc(record.get("observed_at")),
            "available_at": _utc(record.get("available_at")),
            "status": record.get("status"),
            "numeric_value": record.get("numeric_value") or values.get("value"),
            "closed": record.get("closed"),
            "payload": json.loads(stable_json(record)),
            "created_at": now,
            "updated_at": now,
        }

    def upsert_records(
        self,
        table_name: str,
        records: Iterable[dict[str, Any]],
        *,
        connection: Connection | None = None,
    ) -> int:
        if table_name not in self.tables:
            raise ValueError(f"unknown table: {table_name}")
        rows = [self._row(table_name, dict(record)) for record in records]
        if not rows:
            return 0
        table = self.tables[table_name]

        if table_name == "strategy_signals":
            external_ids = [row["external_id"] for row in rows]

            def preserve_evaluation_history(selected: Connection) -> None:
                existing = {
                    str(record.external_id): record
                    for record in selected.execute(
                        select(
                            table.c.external_id,
                            table.c.payload,
                            table.c.created_at,
                        ).where(table.c.external_id.in_(external_ids))
                    )
                }
                for row in rows:
                    payload = dict(row["payload"])
                    previous = existing.get(str(row["external_id"]))
                    previous_payload = (
                        dict(previous.payload)
                        if previous is not None
                        and isinstance(previous.payload, dict)
                        else {}
                    )
                    first_evaluated_at = (
                        previous_payload.get("first_evaluated_at")
                        or previous_payload.get("evaluated_at")
                        or (
                            previous.created_at.isoformat()
                            if previous is not None
                            and previous.created_at is not None
                            else None
                        )
                        or payload.get("evaluated_at")
                    )
                    payload["first_evaluated_at"] = first_evaluated_at
                    payload["evaluation_count"] = (
                        int(previous_payload.get("evaluation_count") or 1) + 1
                        if previous is not None
                        else 1
                    )
                    row["payload"] = payload

            if connection is not None:
                preserve_evaluation_history(connection)
            else:
                with self.engine.connect() as selected:
                    preserve_evaluation_history(selected)

        def execute(
            selected: Connection,
            selected_rows: list[dict[str, Any]],
        ) -> None:
            dialect = self.engine.dialect.name
            if dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as dialect_insert
            elif dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as dialect_insert
            else:
                for row in selected_rows:
                    existing = selected.scalar(
                        select(table.c.id).where(
                            table.c.external_id == row["external_id"]
                        )
                    )
                    if existing:
                        selected.execute(
                            table.update()
                            .where(table.c.id == existing)
                            .values(**{**row, "created_at": table.c.created_at})
                        )
                    else:
                        selected.execute(insert(table).values(**row))
                return
            batch_size = 200 if dialect == "sqlite" else 2_000
            for offset in range(0, len(selected_rows), batch_size):
                statement = dialect_insert(table).values(
                    selected_rows[offset : offset + batch_size]
                )
                update_columns = {
                    column.name: getattr(statement.excluded, column.name)
                    for column in table.columns
                    if column.name not in {"id", "external_id", "created_at"}
                }
                selected.execute(
                    statement.on_conflict_do_update(
                        index_elements=[table.c.external_id],
                        set_=update_columns,
                    )
                )

        if connection is not None:
            execute(connection, rows)
            return len(rows)
        transaction_batch_size = (
            200 if self.engine.dialect.name == "sqlite" else 2_000
        )
        maximum_retry = 0
        for offset in range(0, len(rows), transaction_batch_size):
            batch = rows[offset : offset + transaction_batch_size]
            for attempt in range(1, self.retries + 1):
                try:
                    with self.engine.begin() as selected:
                        execute(selected, batch)
                    maximum_retry = max(maximum_retry, attempt - 1)
                    break
                except OperationalError:
                    if attempt == self.retries:
                        raise
                    time.sleep(0.05 * 2 ** (attempt - 1))
            else:
                raise AssertionError("database retry loop exhausted")
        LOGGER.debug(
            "database upsert completed",
            extra={
                "component": "database",
                "operation": f"upsert:{table_name}",
                "status": "PASSED",
                "reason_code": "IDEMPOTENT_BATCHED_UPSERT",
                "retry_number": maximum_retry,
            },
        )
        return len(rows)

    def fetch_records(
        self, table_name: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        table = self.tables[table_name]
        statement = select(table).order_by(table.c.id)
        if limit:
            statement = statement.limit(limit)
        with self.engine.connect() as connection:
            return [dict(row._mapping) for row in connection.execute(statement)]

    def fetch_record_by_external_id(
        self,
        table_name: str,
        external_id: str,
    ) -> dict[str, Any] | None:
        """Fetch one idempotent record by its caller-facing external ID."""

        table = self.tables[table_name]
        stored_id = self._external_id(
            table_name,
            {"external_id": external_id},
        )
        statement = select(table).where(table.c.external_id == stored_id).limit(1)
        with self.engine.connect() as connection:
            row = connection.execute(statement).first()
            return dict(row._mapping) if row is not None else None

    def fetch_recent_records(
        self,
        table_name: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("recent-record limit must be positive")
        table = self.tables[table_name]
        statement = select(table).order_by(table.c.id.desc()).limit(limit)
        with self.engine.connect() as connection:
            return [
                dict(row._mapping)
                for row in connection.execute(statement)
            ]

    def latest_closed_candles(
        self,
        *,
        markets: Iterable[str],
        timeframes: Iterable[str],
        provider: str | None = None,
    ) -> dict[str, str]:
        table = self.tables["candles"]
        selected_markets = tuple(markets)
        selected_timeframes = tuple(timeframes)
        statement = (
            select(
                table.c.market,
                table.c.timeframe,
                func.max(table.c.timestamp).label("latest"),
            )
            .where(
                table.c.closed.is_(True),
                table.c.market.in_(selected_markets),
                table.c.timeframe.in_(selected_timeframes),
            )
            .group_by(table.c.market, table.c.timeframe)
        )
        if provider is not None:
            statement = statement.where(table.c.provider == provider)
        with self.engine.connect() as connection:
            return {
                f"{row.market}:{row.timeframe}": row.latest.isoformat()
                for row in connection.execute(statement)
                if row.latest is not None
            }

    def health(self) -> dict[str, Any]:
        total_started = time.perf_counter()
        read_started = time.perf_counter()
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            version = connection.scalar(select(func.max(self.version.c.version)))
            counts = {
                name: int(connection.scalar(select(func.count()).select_from(table)) or 0)
                for name, table in self.tables.items()
            }
            journal_mode = None
            if self.engine.dialect.name == "sqlite":
                journal_mode = connection.scalar(text("PRAGMA journal_mode"))
        read_latency_ms = (time.perf_counter() - read_started) * 1_000

        write_started = time.perf_counter()
        probe = self._row(
            "provider_health",
            {
                "external_id": stable_hash(
                    ["database-health-write-probe", time.perf_counter_ns()],
                    length=64,
                ),
                "status": "ROLLBACK_PROBE",
                "timestamp": utc_now(),
            },
        )
        with self.engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(
                    insert(self.tables["provider_health"]).values(**probe)
                )
            finally:
                transaction.rollback()
        write_latency_ms = (time.perf_counter() - write_started) * 1_000
        return {
            "status": "PASSED",
            "dialect": self.engine.dialect.name,
            "schema_version": version,
            "journal_mode": journal_mode,
            "table_counts": counts,
            "read_latency_ms": read_latency_ms,
            "write_latency_ms": write_latency_ms,
            "latency_ms": (time.perf_counter() - total_started) * 1_000,
        }

    def export(
        self,
        table_name: str,
        path: Path | str,
        *,
        format: str | None = None,
    ) -> Path:
        target = Path(path)
        selected_format = (format or target.suffix.lstrip(".")).casefold()
        frame = pd.DataFrame(self.fetch_records(table_name))
        target.parent.mkdir(parents=True, exist_ok=True)
        if selected_format == "csv":
            frame.to_csv(target, index=False)
        elif selected_format == "parquet":
            frame.to_parquet(target, index=False)
        else:
            raise ValueError("database export format must be CSV or Parquet")
        return target

    def apply_retention(
        self, table_name: str, *, older_than: timedelta
    ) -> int:
        if older_than <= timedelta(0):
            raise ValueError("retention age must be positive")
        table = self.tables[table_name]
        cutoff = utc_now() - older_than
        with self.engine.begin() as connection:
            result = connection.execute(
                delete(table).where(
                    func.coalesce(table.c.timestamp, table.c.created_at) < cutoff
                )
            )
            return int(result.rowcount or 0)

    def close(self) -> None:
        self.engine.dispose()


__all__ = ["Database", "SCHEMA_VERSION", "TABLE_NAMES"]
