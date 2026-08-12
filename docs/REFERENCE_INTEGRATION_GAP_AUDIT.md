# Reference integration Phase A gap audit

This audit traces the current composition root (`main.py -> core.cli.main`) and treats the existing repository as the foundation.

| capability | current implementation | reference | current gap | action | owner after migration |
|---|---|---|---|---|---|
| canonical execution ledger and replay | `execution/canonical_state.py` | NautilusTrader | strong reducer exists; reconciliation semantics span multiple live modules | KEEP_AND_HARDEN | `execution.canonical_state` |
| strategy intent to target boundary | canonical `portfolio.contracts` and `portfolio.targets`; some legacy direct callers remain | LEAN | migration debt remains outside the canonical buy chain | KEEP_CANONICAL_OWNER_AND_MIGRATE_CALLERS | `portfolio.targets` |
| approximate Stage 0 | `research/research_factory.py` | vectorbt | native vectorization exists; agreement/false-negative benchmark needs consolidation | KEEP_NATIVE_AND_BENCHMARK | `research.research_factory` |
| purged walk-forward | `WalkForwardManifest`, `research/optimization.py` | PyBroker | strong primitives are split across validation paths | CONSOLIDATE_SHARED_VALIDATION_CONTRACT | `research.validation` |
| ML dataset/model lifecycle | two feature stores, `DatasetIdentity`, `ExperimentContract`, distributed SHADOW markers | Qlib | no one canonical label store or model registry | CONSOLIDATE_DATASET_CONTRACT_AND_BUILD_REGISTRY | `ml.registry` |
| crypto bias lifecycle | causality, static lookahead, recursive warmup, shadow inference | Freqtrade/FreqAI | not yet one mandatory lifecycle contract | CONSOLIDATE_FAIL_CLOSED_LIFECYCLE | `ml.lifecycle` |
| position-management RL | native dependency-free spot transition model and baselines | FinRL | insufficient prospective episodes and Gymnasium/PyTorch/SB3 absent | KEEP_SHADOW_AND_COLLECT_EPISODES | `rl.position_management` |
| autonomous R&D loop | immutable native hypothesis/experiment/feedback trace | RD-Agent | deliberately no live promotion authority | KEEP_BOUNDED_RESEARCH_ONLY | `research.autonomous_rd` |
| structured intelligence | native bull/bear/risk evidence and SHADOW AI decision | TradingAgents | no approved model/provider and no live decision authority | KEEP_STRUCTURED_SHADOW_EVIDENCE | `core.structured_market_intelligence` |
| costs and expectancy | `SharedCostModel` and `research.backtest.CostModel` | cross-cutting | representations can drift | CONSOLIDATE_CANONICAL_COST_CONTRACT | `core.economics` |
| reference health | aggregate local probes | all | probe failure aborts aggregate; old probe roles differ from new architecture roles | ISOLATE_AND_RECLASSIFY | `reporting.reference_health` |

## Active chains observed

- Live/data: Bitvavo sources -> normalization/runtime -> live decision modules -> risk/authority checks -> `execution.execution` -> execution events -> `execution.canonical_state`.
- Research: processed point-in-time data -> `research.features` -> Stage 0 -> exact native backtest -> purged walk-forward/stress -> immutable research artifact.
- ML: feature and shadow-scoring pieces exist, but a canonical dataset/label/model registry chain is not yet complete.

## Phase B entry decision

Proceed only when the generated Phase A artifact reports all nine clones clean, exact commits/tree hashes/licenses verified, named source evidence present, positive usage counts, and unique primary responsibilities. No reference integration may change production execution authority.
