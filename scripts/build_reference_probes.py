"""Build the isolated Lean and NautilusTrader reference probes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> int:
    workspace = Path(__file__).resolve().parents[1]
    lean = workspace / "tools" / "reference_probes" / "lean"
    csc = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
    if not csc.is_file():
        raise FileNotFoundError(f"C# compiler is unavailable: {csc}")
    subprocess.run(
        [
            str(csc),
            "/nologo",
            "/optimize+",
            f"/out:{lean / 'lean_reference_probe.exe'}",
            str(lean / "LeanStubs.cs"),
            str(
                workspace / "crypto-references" / "lean" / "Common" / "Statistics" / "Statistics.cs"
            ),
            str(lean / "Program.cs"),
        ],
        check=True,
        cwd=workspace,
    )
    cargo_home = workspace / ".tools" / "cargo"
    rustup_home = workspace / ".tools" / "rustup"
    cargo = cargo_home / "bin" / "cargo.exe"
    if not cargo.is_file():
        raise FileNotFoundError("workspace Rust 1.97.1 gnullvm toolchain is unavailable")
    linker = (
        rustup_home
        / "toolchains"
        / "1.97.1-x86_64-pc-windows-gnullvm"
        / "lib"
        / "rustlib"
        / "x86_64-pc-windows-gnullvm"
        / "bin"
        / "rust-lld.exe"
    )
    if not linker.is_file():
        raise FileNotFoundError(f"workspace Rust linker is unavailable: {linker}")
    subprocess.run(
        [
            str(cargo),
            "build",
            "--release",
            "--manifest-path",
            str(workspace / "tools" / "reference_probes" / "nautilus" / "Cargo.toml"),
        ],
        check=True,
        cwd=workspace,
        env={
            **os.environ,
            "CARGO_HOME": str(cargo_home),
            "RUSTUP_HOME": str(rustup_home),
            "CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER": str(linker),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
