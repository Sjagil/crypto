# Risk Management

All live intents pass central execution preflight, institutional canary guard
and risk-manager checks. Strategy code cannot grant itself live authority.

Hard controls include:

- spot-only, long-only execution;
- maximum order and exposure;
- maximum wallet-wide positions;
- daily order and loss caps;
- maximum drawdown;
- stop and exit requirements;
- stale-data and liquidity gates;
- market-rule precision/minimum-order validation;
- reconciliation and unknown-order gates;
- duplicate-order prevention and idempotency;
- persistent kill switch.

No martingale, leverage, margin, shorting, borrowing, derivatives or
withdrawals are supported.

```powershell
.\.venv\Scripts\python.exe .\main.py risk status
.\.venv\Scripts\python.exe .\main.py live emergency-stop --reason "OPERATOR_STOP"
```

A daily profit target is informational and capital-scaled. It cannot enlarge
positions, relax stops, bypass a kill switch or force a trade.
