# Incident Response

For an unknown order, unexplained position, stale account stream, data
corruption, duplicate intent, excessive loss or reconciliation mismatch:

1. Activate the persistent kill switch.
2. Preserve event and execution ledgers.
3. Reconcile Bitvavo balances and open orders.
4. Determine whether existing positions need a controlled exit.
5. Repair the root cause and rerun tests.
6. Resume only after explicit operator review.

```powershell
.\.venv\Scripts\python.exe .\main.py live emergency-stop --reason "INCIDENT"
.\.venv\Scripts\python.exe .\main.py live reconcile
.\.venv\Scripts\python.exe .\main.py live status
```

Do not delete ledgers, reset order identities or manually edit frozen
strategy hashes. Never paste secrets, request signatures or private headers
into a ticket, Telegram message or report.

Telegram failure is isolated from trading; reconciliation or risk failure is
not. When uncertain, keep new entries blocked while preserving monitoring
and safe position management.
