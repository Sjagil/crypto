from __future__ import annotations

from typing import Any, Mapping


def _safe_message(value: Any) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    return text[:240]


def classify_bitvavo_private_error(
    http_status: int,
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body = dict(payload or {})
    raw_code = body.get("errorCode")
    try:
        error_code = int(raw_code) if raw_code is not None else None
    except (TypeError, ValueError):
        error_code = None
    message = _safe_message(body.get("error"))

    if error_code == 307:
        classification = "IP_WHITELIST_REJECTED"
        retryable = False
        definitive = True
    elif http_status == 401:
        classification = "AUTHENTICATION_REJECTED"
        retryable = False
        definitive = True
    elif http_status == 403:
        classification = "PRIVATE_PERMISSION_REJECTED"
        retryable = False
        definitive = True
    elif http_status == 429:
        classification = "RATE_LIMITED"
        retryable = True
        definitive = False
    elif http_status >= 500:
        classification = "VENUE_TEMPORARILY_UNAVAILABLE"
        retryable = True
        definitive = False
    elif 400 <= http_status < 500:
        classification = "PRIVATE_REQUEST_REJECTED"
        retryable = False
        definitive = True
    else:
        classification = "UNKNOWN_PRIVATE_RESPONSE"
        retryable = False
        definitive = False

    return {
        "provider": "bitvavo",
        "http_status": int(http_status),
        "error_code": error_code,
        "classification": classification,
        "retryable": retryable,
        "definitive": definitive,
        "message": message,
        "secrets_serialized": False,
    }


__all__ = ["classify_bitvavo_private_error"]
