from __future__ import annotations

import os
from pathlib import Path

import pytest

from utils.common import atomic_write_bytes


def test_atomic_write_retries_transient_windows_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "status.json"
    original_replace = os.replace
    attempts = 0

    def flaky_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise PermissionError("simulated sharing violation")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", flaky_replace)
    monkeypatch.setattr("utils.common.time.sleep", lambda _: None)

    assert atomic_write_bytes(target, b'{"status":"PASSED"}') == target
    assert attempts == 4
    assert target.read_bytes() == b'{"status":"PASSED"}'
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_exhausts_bounded_retries_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "status.json"
    attempts = 0

    def denied_replace(source: str | Path, destination: str | Path) -> None:
        del source, destination
        nonlocal attempts
        attempts += 1
        raise PermissionError("simulated persistent sharing violation")

    monkeypatch.setattr(os, "replace", denied_replace)
    monkeypatch.setattr("utils.common.time.sleep", lambda _: None)

    with pytest.raises(PermissionError):
        atomic_write_bytes(target, b"payload")
    assert attempts == 8
    assert not list(tmp_path.glob("*.tmp"))
