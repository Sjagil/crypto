# Reconciliation

The exchange is the authority for balances, open orders and fills. Local
state supplies strategy ownership and intent history; it cannot invent a
remote order or position.

Run:

```powershell
.\.venv\Scripts\python.exe .\main.py live reconcile
.\.venv\Scripts\python.exe .\main.py live positions
.\.venv\Scripts\python.exe .\main.py live orders
```

Healthy reconciliation requires:

```text
healthy = true
local open orders = remote open orders
unknown remote orders = 0
unknown material positions = 0 or explicitly baselined
```

Unknown orders, unexplained inventory increases, identity drift or corrupted
baselines block new entries. Existing managed exits remain prioritized when
safe. Reconciliation commands perform private reads but submit zero orders.
