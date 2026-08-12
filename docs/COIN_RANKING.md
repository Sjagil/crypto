# Coin Ranking

The point-in-time top-50 ranking combines:

- 24-hour liquidity and market capacity;
- closed-1h seven-day momentum;
- EMA trend quality;
- realized-volatility fit;
- local data quality and freshness;
- venue and Shariah eligibility;
- token-fundamental coverage;
- a neutral explicit regime component.

Every row stores its subscores and weights. A 75/25 current/previous score
blend provides rank hysteresis. Missing values receive no positive credit.

Generate and inspect:

```powershell
.\.venv\Scripts\python.exe .\main.py ranking build
.\.venv\Scripts\python.exe .\main.py ranking inspect --asset BTC
```

Artifacts:

```text
output/ranking/current.json
output/ranking/history.jsonl
```

`venue_execution_eligible` means an eligible Bitvavo EUR market exists.
`live_execution_eligible` additionally requires adequate fundamental
coverage. An explicit operator exception may authorize a bounded strategy,
but never changes the underlying fundamental status.
