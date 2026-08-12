# P1 research-factory reference map

P1 uses the repositories under `crypto-references/` as read-only design references. No
reference framework is imported into the runtime and no source code was copied. The native
research and execution boundaries remain authoritative.

| Reference | Inspected concept | Native P1 implementation | Boundary |
|---|---|---|---|
| vectorbt `vectorbt/portfolio/base.py` — `Portfolio.from_signals` | Broadcasting signals and parameter arrays for fast screening | `generate_parameter_grid` and `simulate_stage0` | Approximate rejection only; never execution evidence |
| PyBroker `src/pybroker/strategy.py` — `WalkforwardMixin.walkforward_split` | Chronological walk-forward windows | `WalkForwardManifest` plus the existing native chronological split | Purge, embargo, selection rule and final test are recorded before validation |
| Freqtrade `freqtrade/optimize/analysis/lookahead.py` — `LookaheadAnalysis` | Compare full and truncated calculations | `stage0_causality_check` and `static_lookahead_audit` | A detected future-data dependency is a hard rejection |
| Freqtrade `freqtrade/optimize/analysis/recursive.py` — `RecursiveAnalysis` | Indicator sensitivity to startup history | `recursive_warmup_stability` | Unstable warmup is reported and cannot promote |
| Qlib `qlib/workflow/recorder.py` — `Recorder` | Versioned experiments, parameters, metrics and artifacts | `DatasetIdentity`, `ExperimentContract`, `ResearchCache`, immutable run artifact | Dataset bytes, strategy version, cost model and validation manifest enter identity |
| LEAN `Algorithm.Framework/` | Alpha, portfolio and execution separation | Stage-0 authority invariant and exact-engine routing | Stage 0 cannot create paper/live authority or orders |
| NautilusTrader `python/nautilus_trader/backtest/__init__.pyi` — `BacktestRunConfig` / `BacktestEngineConfig` | Explicit reproducible run configuration | `ExperimentContract` and `WalkForwardManifest` | The existing `research.backtest.BacktestEngine` remains the exact authority |

## Live-safety patterns adopted after the LINK inventory incident

These are design patterns, not imported execution code. The local Bitvavo
adapter and its canonical state remain authoritative.

| Reference pattern | Local implementation | Fail-closed boundary |
|---|---|---|
| NautilusTrader reconciliation keeps venue state, external orders and deterministic identities distinct | `core/account_inventory.py`, `core/cash_balance_guard.py`, `execution/canonical_state.py` | An exchange fill without a canonical client identity changes cash evidence, but never grants position ownership or exit authority |
| NautilusTrader separates position reconciliation from strategy ownership | `core/live_asset_preflight.py`, `core/execution_authority.py` | Canonical managed protection may continue for its exact quantity while unrelated external inventory blocks every new entry |
| vectorbt and PyBroker document explicit same-bar stop/target assumptions | `reporting/telegram_signal_evidence.py` | A candle touching TP2 and stop is conservatively classified as ambiguous/failure; rounded legacy alerts cannot promote |
| LEAN separates alpha, portfolio construction, risk and execution | roadmap implementation certification plus independent `operational_readiness` | Passing code/tests never enables live authority and never proves profitability |
| Freqtrade lookahead and recursive analysis patterns | native causality and warm-up gates in the research factory | A future-data or unstable-warmup finding is a hard research rejection |
| Qlib experiment recording | hashed reference probes and immutable run artifacts | Every reference repo remains pinned, read-only and outside credential/exchange boundaries |

Bitvavo venue behavior was rechecked against the official API documentation on
2026-08-11. Spot stop-loss orders are native trigger orders with an
`awaitingTrigger` state, open orders are available through an authenticated
reconciliation endpoint, and account WebSocket updates expose both order and
fill events. The implementation therefore uses exact spot quantities and
client identities; it does not assume a futures-style `reduce_only` flag.

- <https://docs.bitvavo.com/docs/rest-api/create-order/>
- <https://docs.bitvavo.com/docs/rest-api/get-open-orders/>
- <https://docs.bitvavo.com/docs/websocket-api/track-your-orders/>

The selected P0.5 branch is `ALPHA_RESEARCH_RESET_REQUIRED_WITH_BOUNDED_PROMISING_EXCEPTION`.
Consequently, P1 does not perform a broad parameter rescue of gross-negative families. Its one
bounded structural exception is the failed-breakdown reversal research adapter, derived from the
small positive P0.5 implementation while explicitly not treating those eight episodes as proof.

Run the factory with:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_research_factory
```

Use `--stage0-only` for a zero-authority screening benchmark. Artifacts are written under
`output/research_factory/runs/<run_id>/`; `output/research_factory/latest.json` is only a pointer
and operator summary. Unscreened P0.5 inventory stays visible and is never relabelled as tested.
