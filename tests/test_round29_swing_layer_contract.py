from __future__ import annotations

import importlib.util
from pathlib import Path

from execution.bitvavo_private_errors import classify_bitvavo_private_error


def test_bitvavo_307_is_structured_ip_whitelist_rejection() -> None:
    result = classify_bitvavo_private_error(
        403,
        {
            "errorCode": 307,
            "error": "This key does not allow access from this IP.",
        },
    )
    assert result["classification"] == "IP_WHITELIST_REJECTED"
    assert result["definitive"] is True
    assert result["retryable"] is False
    assert result["secrets_serialized"] is False


def test_swing_layer_module_is_importable_from_file_spec() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "core" / "swing_layer_live.py"
    spec = importlib.util.spec_from_file_location(
        "round29_swing_layer_live",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "swing_layer_live_preflight",
        "submit_swing_layer_buy",
        "submit_swing_layer_exit",
        "reconcile_swing_layer_live",
        "swing_layer_portfolio",
        "swing_layer_account_snapshot",
        "swing_layer_authority_status",
        "approve_swing_layer_canary",
        "deactivate_swing_layer_canary",
    ):
        assert callable(getattr(module, name))


def test_swing_authority_never_serializes_approval_phrase(
    monkeypatch,
    tmp_path,
) -> None:
    import core.swing_layer_live as swing

    monkeypatch.setattr(
        swing,
        "AUTHORITY_PATH",
        tmp_path / "authority.json",
    )
    payload = swing._persist_authority(
        {
            "active": False,
            "approval_phrase": "SHOULD NEVER BE STORED",
            "approval": "SHOULD NEVER BE STORED",
        }
    )
    text = (tmp_path / "authority.json").read_text(
        encoding="utf-8"
    )
    assert payload["approval_phrase_stored"] is False
    assert "SHOULD NEVER BE STORED" not in text


def test_non_live_buy_is_blocked_before_private_runtime(
    monkeypatch,
) -> None:
    import core.swing_layer_live as swing

    monkeypatch.setattr(
        swing,
        "_authority",
        lambda: {
            **swing._defaults(),
            "active": True,
        },
    )
    result = swing.submit_swing_layer_buy(
        {
            "intent_id": "unit",
            "authority": "PAPER",
            "market": "BTC-EUR",
            "notional_eur": "10",
            "net_edge_bps": 50,
            "stop_pct": 0.02,
            "expires_at": "2099-01-01T00:00:00+00:00",
        },
        execute=True,
    )
    assert result["accepted"] is False
    assert "INTENT_AUTHORITY_NOT_LIVE" in result["blockers"]
    assert result["orders_submitted"] == 0


def test_exception_order_preserves_reconciliation_specificity() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "core" / "live_asset_preflight.py"
    ).read_text(encoding="utf-8")
    expected = (
        'except ReconciliationRequired:\n'
        '            failures.append("PRIVATE_ACCOUNT_RESPONSE_AMBIGUOUS")\n'
        '        except ExecutionBlocked:\n'
        '            failures.append("PRIVATE_ACCOUNT_READ_BLOCKED")'
    )
    assert expected in source
