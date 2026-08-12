# Portfolio Allocator

Portfolio decisions are wallet-wide. Existing material holdings count toward
the maximum number of positions even when they predate bot authority.

For the current account below €10,000, the approved generated-strategy sleeve
has:

```text
maximum positions: 3
maximum one position per market: true
maximum one position per strategy DNA: true
maximum order: €10
maximum shared generated exposure: €15
autoscale: false
```

These are upper bounds, not targets. The allocator never fills an empty slot
without a natural signal, sufficient EUR, healthy liquidity, current data and
green risk/reconciliation checks. Multiple strategies cannot independently
claim the same market position.

Inspect:

```powershell
.\.venv\Scripts\python.exe .\main.py portfolio status
.\.venv\Scripts\python.exe .\main.py live positions
.\.venv\Scripts\python.exe .\main.py live strategies
```
