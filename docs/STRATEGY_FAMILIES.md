# Strategy Families

The audit maps existing strategy DNA to economic families rather than relying
only on display names:

- `time_series_momentum`
- `cross_sectional_momentum`
- `trend_pullback`
- `volatility_compression_breakout`
- `range_mean_reversion`
- `fractal_market_structure`
- `volume_orderflow_confirmation`
- `liquidity_sweep_recovery`
- `regime_conditioned_reversal`
- `defensive_cash_rotation`
- `unclassified_research`

Generated strategies remain bounded compositions: an entry mechanism,
confirmations, an optional regime filter, one stop policy, one exit policy
and explicit sizing. A positive baseline is not enough for live identity.
The live bridge requires exact real-provider evidence, costs, causal
execution, a frozen hash, an executable market and explicit operator
authority.

Current inventory:

```powershell
.\.venv\Scripts\python.exe .\main.py strategies all
.\.venv\Scripts\python.exe .\main.py strategies positive
.\.venv\Scripts\python.exe .\main.py live strategies
```

The canonical classification and duplicate-review evidence is written by
`main.py system audit` to `output/reports/system_audit`.
