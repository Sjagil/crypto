# Canonical architecture ownership

Phase B establishes one writer/owner per financial responsibility. “Derived read model” means the component may observe and render canonical state but may not mutate it.

| responsibility | canonical owner | permitted writers | derived/read-only consumers | reference concept | migration status |
|---|---|---|---|---|---|
| market observations | `data.multi_source_platform` | configured collectors and normalization adapters | features, research, dashboards | Freqtrade lifecycle | existing owner; consolidate schemas |
| point-in-time features | `data.multi_source_platform.PointInTimeFeatureStore` | causal feature builders | research, ML shadow, dashboards | Qlib/FreqAI | existing owner; reconcile legacy tensor store |
| research dataset freeze | `research.research_factory.DatasetIdentity` and `ml.contracts.CanonicalDatasetManifest` | research factory or offline ML registry only | exact validation and ML research | Qlib | canonical contracts added; legacy ML artifact not migrated without PIT proof |
| strategy output | `portfolio.contracts.InvestmentIntent` | deterministic strategies and SHADOW ML filters | portfolio construction | LEAN Insight/Alpha boundary | native contract added |
| desired holdings | `portfolio.contracts.PortfolioTarget` | portfolio construction only | risk and dashboards | LEAN PortfolioTarget | native contract added |
| risk approval | `portfolio.contracts.RiskApproval` plus `risk.risk_manager` | deterministic risk engine only | execution and audit | LEAN RiskManagement | native contract and BUY adapter enforced |
| executable delta | `portfolio.contracts.ExecutionIntent` | target-to-execution adapter after valid risk approval | Bitvavo execution and audit | LEAN ExecutionModel | native contract and BUY adapter enforced |
| order submission/state | `execution.execution` and append-only execution evidence | Bitvavo execution client only | canonical reducer | Nautilus ExecutionEngine | canonical chain enforced for all four BUY producers; protective SELL remains available |
| fills/positions/lots/fees/PnL | `execution.canonical_state` | deterministic ledger replay only | risk, accounting, reporting | Nautilus reconciliation/cache | existing canonical owner |
| transaction costs | `core.economics.CanonicalCostModel` | calibrated cost pipeline | research, portfolio, execution | cross-cutting | canonical assumption owner added; exact backtester uses a native adapter |
| model registry/promotion | `ml.registry` | offline training/governance only | SHADOW inference and dashboards | Qlib Recorder/FreqAI lifecycle | prospective PIT builder and five-fold purged trainer added; 2,966 legacy rows remain excluded and the current canonical model is `DATA_PENDING` |
| live authority | `core.execution_authority` and runtime control state | explicit operator-controlled policy only | all live modules | native safety | unchanged; highest gate with risk |

## Enforced contract chain

`InvestmentIntent -> PortfolioTarget -> RiskApproval -> ExecutionIntent -> OrderIntent -> exchange`

The immutable contracts and provenance fields are enforced at the Bitvavo BUY boundary. Autonomous RR entries, event-driven entries, generated-strategy entries/reprices, and manual live CLI entries all construct the same canonical chain before submission. Any future caller that omits it is blocked before network or ledger mutation; risk-reducing SELL/protective paths deliberately remain available as a safety exception.

The opportunity ML compatibility dataset remains readable, but it is not canonical evidence. New snapshots record `event_time`, `available_at`, finality, and provenance prospectively. The canonical builder never backfills those fields and registers only causally valid rows. Model registration additionally requires 500 rows, at least 100 examples per class, a 24-hour purge, five exact walk-forward folds, isolated validation/test ranges, and calibration metrics. All resulting authority remains `SHADOW_ONLY`.

Research libraries and local references never own live authority, risk, orders, fills, positions, or accounting.
