# Capital Scaling

Scaling is separate from strategy discovery and live execution validation.
The current generated-strategy authority is Level 1:

```text
€10 maximum order
€15 shared generated exposure
3 wallet-wide positions
no autoscale
```

Any future capital increase requires explicit operator approval and evidence
from real fills, fees, slippage, exits, drawdown and reconciliation. Research,
paper fills or statistical scores cannot silently increase live caps.

Automatic demotion is allowed for negative live expectancy, excessive
slippage, drawdown, stale data, liquidity deterioration, reconciliation
problems or operational incidents.

Inspect current capital state:

```powershell
.\.venv\Scripts\python.exe .\main.py capital status
.\.venv\Scripts\python.exe .\main.py live strategies
```

Do not interpret a daily profit target as a scaling rule.
