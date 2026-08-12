# Architecture

`main.py` is the only supported command entry point. The live path is:

```text
public and private Bitvavo state
→ freshness and eligibility
→ regime and strategy signals
→ generated-strategy ranking
→ wallet-wide position limits
→ central preflight and risk controls
→ idempotent Bitvavo spot execution
→ order/fill/position reconciliation
→ Telegram and durable attribution
```

The permanent process is `main.py autonomous-live run`. It owns the live
single-instance lock and supervises public market data, the private account
stream, reconciliation, paper strategies, approved live DNA, Telegram, the
continuous data sync and the orderless strategy factory.

Important state is append-only where an event history is required. Current
snapshots live under `output/live`, `output/portfolio`, `output/operations`
and `output/governance`. Research cannot grant live authority. Exact strategy
DNA and operator authority are separate inputs to execution.

Execution accounting and the legacy-state migration are specified in
[`CANONICAL_EXECUTION_STATE.md`](CANONICAL_EXECUTION_STATE.md).

Inspect the implementation and current gaps with:

```powershell
.\.venv\Scripts\python.exe .\main.py system audit
.\.venv\Scripts\python.exe .\main.py system architecture
```

The system is spot-only, EUR-quoted and long-only. Margin, leverage, shorts,
derivatives and withdrawals are fail-closed.
