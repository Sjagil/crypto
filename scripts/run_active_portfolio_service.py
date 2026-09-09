#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--equity-eur", type=float, default=2000.0)
    parser.add_argument("--cycles", type=int, default=0)
    parser.add_argument("--refresh-external-every", type=int, default=3)
    args = parser.parse_args()
    if args.interval_seconds < 60:
        raise SystemExit("minimum service interval is 60 seconds")

    root = Path(__file__).resolve().parents[1]
    cycle = 0
    while True:
        cycle += 1
        print(f"[active-portfolio] cycle={cycle} START", flush=True)
        data_sync = [
            sys.executable,
            "main.py",
            "history",
            "download",
            "--min-years",
            "7",
            "--markets",
            "BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR",
            "--timeframes",
            "15m,1h,2h,4h,1d,1W",
            "--providers",
            "bitvavo,kraken",
            "--resume",
        ]
        sync = subprocess.run(
            data_sync,
            cwd=root,
            check=False,
            stdout=subprocess.DEVNULL,
        )
        print(
            f"[active-portfolio] cycle={cycle} data_sync_rc={sync.returncode}",
            flush=True,
        )

        whole = [
            sys.executable,
            str(root / "scripts" / "run_whole_stack.py"),
            "--markets",
            "BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR",
            "--timeframes",
            "15m,1h,2h,4h,1d,1W",
        ]
        if cycle % max(1, args.refresh_external_every) == 0:
            whole.append("--refresh-external")
        subprocess.run(whole, cwd=root, check=False, stdout=subprocess.DEVNULL)
        process = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_active_portfolio_cycle.py"),
                "--mode",
                "paper",
                "--equity-eur",
                str(args.equity_eur),
            ],
            cwd=root,
            check=False,
        )
        print(
            f"[active-portfolio] cycle={cycle} END rc={process.returncode}",
            flush=True,
        )
        if args.cycles and cycle >= args.cycles:
            return process.returncode
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
