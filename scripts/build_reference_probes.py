"""Build the isolated Lean and NautilusTrader reference probes.

Nautilus is built natively on the current platform. Lean keeps its original
Windows-only C# probe contract; non-Windows hosts can build Nautilus alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


NAUTILUS_RUST_VERSION = (1, 97, 1)


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd, env=env)


def _version_tuple(text: str) -> tuple[int, int, int]:
    match = re.search(r"rustc\s+(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        raise RuntimeError(f"cannot parse rustc version: {text!r}")
    return tuple(int(value) for value in match.groups())


def _native_cargo_command() -> list[str]:
    cargo = shutil.which("cargo")
    rustup = shutil.which("rustup")

    if rustup:
        subprocess.run(
            [rustup, "toolchain", "install", "1.97.1", "--profile", "minimal"],
            check=True,
        )
        return [cargo or "cargo", "+1.97.1"]

    if not cargo:
        raise FileNotFoundError(
            "native cargo/rustup is unavailable. Install Rust with rustup, "
            "then rerun this builder."
        )

    rustc = shutil.which("rustc")
    if not rustc:
        raise FileNotFoundError("cargo exists but rustc is unavailable on PATH")

    version = subprocess.run(
        [rustc, "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if _version_tuple(version) < NAUTILUS_RUST_VERSION:
        raise RuntimeError(
            f"Nautilus requires rustc >= 1.97.1; found {version}. "
            "Install/select Rust 1.97.1 with rustup."
        )
    return [cargo]


def _ensure_full_nautilus_checkout(workspace: Path) -> Path:
    repo = workspace / "crypto-references" / "nautilus_trader"
    if not repo.exists():
        raise FileNotFoundError(f"Nautilus reference checkout missing: {repo}")

    probe = subprocess.run(
        ["git", "-C", str(repo), "sparse-checkout", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        subprocess.run(
            ["git", "-C", str(repo), "sparse-checkout", "disable"],
            check=True,
        )

    expected = "e8be4522cd12a4a65a4d1350f791d414ad246439"
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != expected:
        raise RuntimeError(f"Nautilus reference HEAD drifted: {head} != {expected}")

    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Nautilus reference checkout must remain clean before probe build")

    required = (
        repo / "Cargo.toml",
        repo / "crates" / "model" / "Cargo.toml",
        repo / "crates" / "core" / "Cargo.toml",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("full Nautilus checkout is incomplete: " + ", ".join(missing))
    return repo


def build_nautilus(workspace: Path) -> dict[str, object]:
    repo = _ensure_full_nautilus_checkout(workspace)
    cargo = _native_cargo_command()
    manifest = workspace / "tools" / "reference_probes" / "nautilus" / "Cargo.toml"
    if not manifest.is_file():
        raise FileNotFoundError(f"Nautilus probe manifest missing: {manifest}")

    command = [
        *cargo,
        "build",
        "--release",
        "--manifest-path",
        str(manifest),
    ]
    lockfile = manifest.parent / "Cargo.lock"
    if lockfile.is_file():
        command.insert(len(cargo) + 2, "--locked")

    _run(command, cwd=workspace, env={**os.environ, "NO_COLOR": "1"})

    executable = (
        manifest.parent
        / "target"
        / "release"
        / ("nautilus_reference_probe.exe" if os.name == "nt" else "nautilus_reference_probe")
    )
    if not executable.is_file():
        raise FileNotFoundError(
            f"Nautilus probe build completed but binary is missing: {executable}"
        )

    smoke = subprocess.run(
        [str(executable), "100", "101", "2", "3"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    payload = json.loads(smoke.stdout.strip().splitlines()[-1])
    if float(payload["spread"]) != 1.0 or float(payload["midpoint"]) != 100.5:
        raise RuntimeError(f"Nautilus probe smoke mismatch: {payload}")

    return {
        "status": "READY",
        "binary": str(executable),
        "reference_head": subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "smoke": payload,
    }


def build_lean_windows(workspace: Path) -> dict[str, object]:
    if os.name != "nt":
        return {
            "status": "SKIPPED_PLATFORM_NOT_WINDOWS",
            "reason": "The retained Lean C# probe is Windows-only.",
        }

    lean = workspace / "tools" / "reference_probes" / "lean"
    csc = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
    if not csc.is_file():
        raise FileNotFoundError(f"C# compiler is unavailable: {csc}")

    _run(
        [
            str(csc),
            "/nologo",
            "/optimize+",
            f"/out:{lean / 'lean_reference_probe.exe'}",
            str(lean / "LeanStubs.cs"),
            str(
                workspace
                / "crypto-references"
                / "lean"
                / "Common"
                / "Statistics"
                / "Statistics.cs"
            ),
            str(lean / "Program.cs"),
        ],
        cwd=workspace,
    )
    return {
        "status": "READY",
        "binary": str(lean / "lean_reference_probe.exe"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nautilus-only", action="store_true")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    result: dict[str, object] = {
        "schema_version": "reference_probe_build_v2",
        "platform": sys.platform,
        "nautilus": build_nautilus(workspace),
        "orders_generated": 0,
        "orders_submitted": 0,
        "private_exchange_requests": 0,
    }
    if not args.nautilus_only:
        result["lean"] = build_lean_windows(workspace)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
