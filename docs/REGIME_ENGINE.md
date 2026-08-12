# Regime Engine

The regime layer is causal and uses closed candles. It may enable, reduce or
block a strategy family; it cannot create a trade without the strategy's own
entry signal.

The live RR route currently combines the daily residual-reversal signal with
market regime eligibility. `RISK_OFF_BLOCK`, stale data or an ineligible
route prevents a new entry but does not stop position management and
reconciliation.

Inspect the current result:

```powershell
.\.venv\Scripts\python.exe .\main.py regime status
.\.venv\Scripts\python.exe .\main.py regime explain
.\.venv\Scripts\python.exe .\main.py hmm status
```

HMM output is observer/context evidence unless a specific frozen strategy DNA
was validated with that HMM policy. HMM state changes never rewrite strategy
parameters and never bypass risk controls.

When regime inputs are stale or contradictory, the correct state is
uncertain/blocked. Missing context is never interpreted as risk-on.
