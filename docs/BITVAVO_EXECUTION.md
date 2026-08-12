# Bitvavo Execution

Bitvavo is the only live execution venue. The live client supports private
account reads, order submission, order status, cancellation, balances and
reconciliation. Prices and quantities use decimal arithmetic and venue
market rules.

Before an entry the runtime requires:

- authenticated private account stream;
- safe trade-only API scope and confirmed IP policy;
- withdrawals disabled;
- fresh closed candles;
- current public order book and liquidity;
- exact frozen strategy identity;
- active operator authority;
- healthy local/remote order reconciliation;
- idempotent client order identity;
- all configured caps.

The audit and status commands never place an order:

```powershell
.\.venv\Scripts\python.exe .\main.py live account-health
.\.venv\Scripts\python.exe .\main.py live reconcile
.\.venv\Scripts\python.exe .\main.py live orders
```

Never include API keys, signatures, headers or complete account identifiers
in logs or incident reports.
