# Round 35: active portfolio-driven live execution

Round 35 does not create a second broker. Actual exchange authority remains in
`main.py autonomous-live` and the canonical Bitvavo execution client.

The Round-34 active portfolio planner now acts as an additional live constraint:

- `ENTRY` / `ADD` caps the existing canonical entry budget;
- `REDUCE` requests only the excess managed quantity above the wallet target;
- `FULL_EXIT` requests the entire canonically managed position;
- `HOLD` does not manufacture an order;
- stale or unavailable private-account plans block new portfolio-driven entries
  while existing reconciliation and protective exits remain operable.

For event-driven managed positions the live path uses:

- atomic Level-2 BUY submission through the existing canonical target/risk chain;
- venue-native Bitvavo `stopLoss` after a verified entry fill;
- actual TP1 partial SELL when both the realized leg and remaining protected leg
  satisfy current venue minimums;
- breakeven-stop-only TP1 fallback when a safe partial cannot be represented;
- actual full SELL at TP2;
- native trailing-stop replacement after TP1;
- automatic native-stop re-protection after any partial strategy/portfolio exit;
- existing soft/time/structure exits plus active-portfolio REDUCE/FULL_EXIT;
- append-only ledger, idempotency, private reconciliation and owned-unit checks.

## Audit

```bash
python scripts/active_live_contract_audit.py
```

## Active live orchestration

Status:

```bash
python scripts/run_active_live_system.py status
```

Start the sidecar and canonical supervisor:

```bash
python scripts/run_active_live_system.py start
```

Keep refreshing the private wallet/portfolio plan every five minutes while the
canonical autonomous supervisor remains the only order owner:

```bash
python scripts/run_active_live_system.py run --interval-seconds 300
```

Stop:

```bash
python scripts/run_active_live_system.py shutdown
```

This sidecar never writes live strategy authority, never stores an approval
phrase and never calls `submit_order` itself. Existing Level-2/service/strategy
approval, private-account health, IP whitelist, data, economics and risk gates
remain authoritative.
