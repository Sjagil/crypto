"""Cohesive utilities for time, logging, hashing and atomic persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import logging.handlers
import os
import random
import re
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, SecretStr

T = TypeVar("T")
_SPACE = re.compile(r"\s+")


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_iso(value: datetime | None = None) -> str:
    selected = value or utc_now()
    if selected.tzinfo is None or selected.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return selected.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    return parsed.astimezone(UTC)


def clean_text(value: Any, *, maximum_length: int | None = None) -> str:
    text = _SPACE.sub(" ", str(value or "")).strip()
    if maximum_length is not None and len(text) > maximum_length:
        return f"{text[: max(0, maximum_length - 1)]}…"
    return text


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return "***REDACTED***" if value.get_secret_value() else None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (datetime, date)):
        if isinstance(value, datetime):
            return utc_iso(value)
        return value.isoformat()
    if isinstance(value, (Path, Decimal)):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def stable_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        separators=None if indent else (",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any, *, length: int = 64) -> str:
    if not 8 <= length <= 64:
        raise ValueError("hash length must be between 8 and 64")
    return sha256_text(stable_json(value))[:length]


def atomic_write_bytes(path: Path | str, payload: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                # Windows can transiently deny replacing a status artifact
                # while another process has just opened the old inode.
                # Retrying the same already-fsynced temporary file keeps the
                # operation atomic and bounded.
                time.sleep(min(0.005 * 2**attempt, 0.1))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path | str, value: Any, *, indent: int = 2) -> Path:
    return atomic_write_text(path, f"{stable_json(value, indent=indent)}\n")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: Path | str, value: Any) -> Path:
    """Durably append one record.

    This is intended for single-process ledgers. Cross-process callers must add
    an external lock or use the database-backed ledger.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = f"{stable_json(value)}\n".encode("utf-8")
    with target.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    return target


def resolve_within(root: Path | str, candidate: Path | str) -> Path:
    base = Path(root).expanduser().resolve()
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path is outside configured root: {resolved}") from exc
    return resolved


def redact(value: Any, secrets: Sequence[str] = ()) -> Any:
    """Return a JSON-safe copy with secret-like fields and values removed."""

    secret_values = tuple(secret for secret in secrets if secret)
    secret_names = ("secret", "password", "token", "api_key", "apikey", "credential")
    if isinstance(value, SecretStr):
        return "***REDACTED***"
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, Mapping):
        return {
            str(key): (
                "***REDACTED***"
                if any(part in str(key).casefold() for part in secret_names)
                else redact(item, secret_values)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [redact(item, secret_values) for item in value]
    if isinstance(value, str):
        result = value
        for secret in secret_values:
            result = result.replace(secret, "***REDACTED***")
        return result
    return value


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: Sequence[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in self._secrets:
            message = message.replace(secret, "***REDACTED***")
        record.msg = message
        record.args = ()
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            for secret in self._secrets:
                record.exc_text = record.exc_text.replace(secret, "***REDACTED***")
        return True


class AlertThrottle:
    """Optional, redacted and failure-isolated alert deduplication."""

    def __init__(
        self,
        *,
        state_path: Path,
        audit_path: Path,
        cooldown_seconds: float = 300.0,
        secrets: Sequence[str] = (),
        delivery: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        if cooldown_seconds < 0:
            raise ValueError("alert cooldown cannot be negative")
        self.state_path = state_path
        self.audit_path = audit_path
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.secrets = tuple(value for value in secrets if value)
        self.delivery = delivery
        try:
            self.sent = dict(read_json(state_path)) if state_path.is_file() else {}
        except (OSError, ValueError, TypeError):
            self.sent = {}

    def send(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> bool:
        selected_now = now or utc_now()
        safe_payload = redact(payload, self.secrets)
        fingerprint = stable_hash(
            {"event_type": event_type, "payload": safe_payload},
            length=32,
        )
        previous = self.sent.get(fingerprint)
        if previous:
            try:
                if selected_now - parse_utc(previous) < self.cooldown:
                    return False
            except (TypeError, ValueError):
                pass
        self.sent[fingerprint] = utc_iso(selected_now)
        atomic_write_json(self.state_path, self.sent)
        record = {
            "event_type": clean_text(event_type, maximum_length=80),
            "payload": safe_payload,
            "sent_at": utc_iso(selected_now),
            "fingerprint": fingerprint,
        }
        append_jsonl(self.audit_path, record)
        if self.delivery is not None:
            try:
                self.delivery(str(record["event_type"]), safe_payload)
            except Exception as exc:  # alerts must never stop operations
                append_jsonl(
                    self.audit_path,
                    {
                        "event_type": "ALERT_DELIVERY_FAILED",
                        "reason_code": type(exc).__name__,
                        "recorded_at": utc_iso(),
                    },
                )
        return True


class JsonLogFormatter(logging.Formatter):
    """Stable JSONL formatter with the observability fields used system-wide."""

    fields = (
        "run_id",
        "component",
        "provider",
        "market",
        "timeframe",
        "operation",
        "duration",
        "status",
        "reason_code",
        "exception_type",
        "retry_number",
        "correlation_id",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": utc_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self.fields:
            payload[field] = getattr(record, field, None)
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
            payload["exception"] = record.exc_text or self.formatException(record.exc_info)
        return stable_json(payload)


def configure_logging(
    *,
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    jsonl_file: Path | str | None = None,
    secrets: Sequence[str] = (),
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    logger = logging.getLogger("crypto")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    redactor = RedactingFilter(secrets)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.addFilter(redactor)
    logger.addHandler(stream)
    if log_file is not None:
        target = Path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Windows cannot atomically rename a log file while another research
        # or worker process has it open.  A shared RotatingFileHandler therefore
        # emits noisy rollover tracebacks and can lose operational events.
        # Rotation remains enabled on POSIX; Windows uses an append-only shared
        # file and the existing disk-budget supervisor governs retention.
        file_handler = (
            logging.FileHandler(target, encoding="utf-8")
            if os.name == "nt"
            else logging.handlers.RotatingFileHandler(
                target,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redactor)
        logger.addHandler(file_handler)
    if jsonl_file is not None:
        target = Path(jsonl_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        json_handler = (
            logging.FileHandler(target, encoding="utf-8")
            if os.name == "nt"
            else logging.handlers.RotatingFileHandler(
                target,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        )
        json_handler.setFormatter(JsonLogFormatter())
        json_handler.addFilter(redactor)
        logger.addHandler(json_handler)
    logging.Formatter.converter = __import__("time").gmtime
    return logger


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay_seconds: float,
    maximum_delay_seconds: float = 30.0,
    retryable: tuple[type[BaseException], ...] = (OSError, TimeoutError),
    seed: int = 42,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    randomizer = random.Random(seed)
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except retryable:
            if attempt == attempts:
                raise
            delay = min(maximum_delay_seconds, base_delay_seconds * 2 ** (attempt - 1))
            delay *= 0.75 + randomizer.random() * 0.5
            await asyncio.sleep(delay)
    raise AssertionError("retry loop exhausted unexpectedly")


def chunked(values: Sequence[T], size: int) -> list[Sequence[T]]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    return [values[index : index + size] for index in range(0, len(values), size)]


__all__ = [
    "RedactingFilter",
    "AlertThrottle",
    "JsonLogFormatter",
    "append_jsonl",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "chunked",
    "clean_text",
    "configure_logging",
    "parse_utc",
    "read_json",
    "redact",
    "resolve_within",
    "retry_async",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "stable_hash",
    "stable_json",
    "utc_iso",
    "utc_now",
]
