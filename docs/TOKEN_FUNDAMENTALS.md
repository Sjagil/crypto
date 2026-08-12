# Token Fundamentals

Token fundamentals are a point-in-time confidence and eligibility layer.
The current implementation reconciles available top-50 fields and explicitly
tracks missing:

- total and maximum supply;
- fully diluted valuation;
- unlock calendar;
- holder concentration;
- protocol revenue;
- token value capture;
- security incidents.

Stablecoins, wrapped tokens, leveraged tokens and staking derivatives are
not automatically executable. Sparse evidence produces `REVIEW_REQUIRED`;
it is never converted into a favorable score.

Commands:

```powershell
.\.venv\Scripts\python.exe .\main.py tokenomics refresh
.\.venv\Scripts\python.exe .\main.py tokenomics inspect --asset TAO
```

Artifact:

```text
output/tokenomics/current.json
```

The module performs no private exchange request and submits no order.
