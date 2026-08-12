# Canonical execution state

The append-only live execution ledger is the financial source of truth. A
deterministic reducer in `execution/canonical_state.py` rebuilds orders, fills,
FIFO lots, positions, ownership, confirmed protection, reserved capital, fees,
open risk and realized PnL. Files under `output/live` are rebuildable read
models; they never overwrite ledger truth.

## Accounting contract

- Fill identity is the economic deduplication key. Repeated REST, WebSocket or
  private/public observations cannot apply the same fill twice.
- Event timestamps determine economic order; recorded timestamps and event
  priority provide deterministic tie breaking.
- Buy fees are capitalized into FIFO lot cost. Exit fees reduce realized PnL.
- A sell with no matching historical lot records an evidence gap. It never
  fabricates a zero cost basis or a profit.
- Protection is `CONFIRMED_ACTIVE` only after an acknowledged active exchange
  stop. A local stop intention is not protection.
- Open risk includes stop distance for protected quantity and the full known
  cost basis for unprotected quantity. Missing cost evidence produces an
  incomplete value, never fictitious zero risk.
- Strategy ownership is exact, mixed or unknown and follows actual fills.
- A closed position has zero quantity, zero protected quantity and zero open
  risk. Negative quantities and negative reservations are invariant failures.

## Replay and migration

`DurableLedger.canonical_state()` replays the entire ledger. The supervisor
persists four read-only migration artifacts:

- `canonical_execution_state.json`
- `canonical_execution_replay.json`
- `canonical_execution_divergence.json`
- `execution_state_migration_status.json`

The old position tracker is rebuilt from canonical state and is a read model
only. Divergences are classified as expected schema differences, rounding
differences, missing historical evidence or real defects. Legacy state cannot
write back into canonical state.

### State-owner verification

Before migration, the writers/readers classified as follows:

| State | Prior classification | Post-migration role |
|---|---|---|
| `output/checkpoints/live_execution.jsonl` | `CANONICAL_CANDIDATE` | immutable canonical input log |
| `output/live/position_tracker.json` | `DUPLICATE` | canonical derived read model |
| `output/live/generated_strategy_live_state.json` | `LEGACY` | temporary strategy-runtime state; cannot overwrite canonical economics |
| `output/live/event_driven_execution_state.json` | `DUPLICATE` / `LEGACY` | temporary engine-runtime state; cannot overwrite canonical economics |
| `output/reports/current_position.json` | `REPORT_ONLY` | reporting only; no recovery authority |
| Bitvavo reconciliation observations | `CANONICAL_CANDIDATE` | immutable reconciliation events reduced through the same aggregate |

The verified contradiction concerned BTC, ETH and LINK. The legacy tracker had
the correct quantities but blank owners, no stops, no protected quantity and
zero open risk; generated-strategy runtime state held DNA and planned stops;
the event-driven state reported no positions. Canonical replay instead derives
the exact strategy owner and acknowledged native stop from the execution
ledger. The dual-read report classifies lost ownership, confirmed-stop and
open-risk differences as real defects and the absent legacy protected-quantity
field as an expected schema difference.

## Reference concepts used

The design was informed by concepts in the local read-only reference mirrors;
no source code was copied.

| Reference repository | File | Class/function | Concept adopted natively |
|---|---|---|---|
| NautilusTrader | `crates/execution/src/engine/mod.rs` | `ExecutionEngine`, `reconcile_execution_report`, `reconcile_fill_report` | one event-application boundary and reconciliation through the same state transitions |
| NautilusTrader | `crates/execution/src/reconciliation/orders.rs` | `reconcile_order_report`, `reconcile_fill_report` | idempotent cumulative order/fill reconciliation |
| NautilusTrader | `crates/live/src/execution/manager.rs` | `reconcile_execution_mass_status`, `reconcile_open_order_reports`, `reconcile_position_reports` | venue observations become reconciliation evidence rather than a second ledger |
| NautilusTrader | `crates/common/src/cache/mod.rs` | `Cache` | derived query state has one authoritative event source |
| NautilusTrader | `crates/portfolio/src/portfolio.rs` | `Portfolio` | positions and risk are projections of execution facts |
| LEAN | `Common/Algorithm/Framework/Portfolio/PortfolioTarget.cs` | `PortfolioTarget` | portfolio intent is distinct from executed holdings |
| LEAN | `Common/Algorithm/Framework/Portfolio/PortfolioTargetCollection.cs` | `PortfolioTargetCollection` | intent collection does not own fill economics |
| LEAN | `Engine/TransactionHandlers/BrokerageTransactionHandler.cs` | `BrokerageTransactionHandler` | brokerage events form the execution-truth boundary |

Nautilus Trader concepts:

- execution-engine event application and cache ownership:
  `crypto-references/nautilus_trader/crates/execution/src/engine/mod.rs`
- order reconciliation and cumulative state:
  `crypto-references/nautilus_trader/crates/execution/src/reconciliation/orders.rs`
- live execution reconciliation sequencing:
  `crypto-references/nautilus_trader/crates/live/src/execution/manager.rs`
- cache consistency:
  `crypto-references/nautilus_trader/crates/common/src/cache/mod.rs`
- portfolio state derived from execution events:
  `crypto-references/nautilus_trader/crates/portfolio/src/portfolio.rs`

LEAN concepts:

- target quantity as intent, separate from holdings:
  `crypto-references/lean/Common/Algorithm/Framework/Portfolio/PortfolioTarget.cs`
- collection and fulfillment of target intent:
  `crypto-references/lean/Common/Algorithm/Framework/Portfolio/PortfolioTargetCollection.cs`
- brokerage transaction processing as the boundary for order and fill truth:
  `crypto-references/lean/Engine/TransactionHandlers/BrokerageTransactionHandler.cs`

These references support the architectural separation between intent,
execution events and derived portfolio state. The native implementation keeps
the repository's Bitvavo spot-only, long-only, EUR accounting and existing live
authority unchanged.
