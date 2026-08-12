# Reference integration policy

Status: Phase A engineering boundary. The local license classification below is not legal advice.

The nine repositories under `crypto-references/` are immutable evidence sources. Production and research code must adopt concepts through clean native implementations, observable-behavior tests, and explicit provenance. Reference code receives no credentials, exchange client, order authority, risk authority, or canonical-state write access.

| repo | commit | license | role | integration_mode | used_concepts | forbidden_copying | runtime_dependency | fallback | tests |
|---|---|---|---|---|---|---|---:|---|---|
| nautilus_trader | `e8be4522cd12a4a65a4d1350f791d414ad246439` | LGPL-3.0 | execution state and reconciliation invariants | C concept/reference only; native implementation | idempotent fills, event reconciliation, replay | no upstream implementation copying | false | `execution.canonical_state` | canonical replay and Phase A contract tests |
| lean | `c6cc3b743ed7b65d5e0b9fa2bfc18b7d3ac2aea0` | Apache-2.0 | strategy intent to portfolio target boundary | C concept/reference only; native implementation | PortfolioTarget, target delta, risk-before-execution | no LEAN class copying | false | block without native target and risk approval | Phase A plus future target contract tests |
| vectorbt | `34b6d5935e3ea3eccd549e2592bc0f455b8045f5` | Apache-2.0 with Commons Clause | approximate vectorized Stage-0 screening | C concept/reference only; native implementation | broadcast grids, signal matrices, cheap cost screen | Commons Clause source is not copied | false | native `research.research_factory` Stage 0 | research-factory and Phase A tests |
| pybroker | `e0e7b08886343274efb05b96f7399ca3de280aa5` | Apache-2.0 with Commons Clause | chronological walk-forward validation | C concept/reference only; native implementation | ordered windows, retraining boundaries | Commons Clause source is not copied | false | native validation manifest and exact engine | research validation and Phase A tests |
| qlib | `79633dd9506ea689e5400dea0197717b5b3d74b7` | MIT | immutable ML dataset, experiment, and model lifecycle | C concept/reference only; native implementation | Dataset, Recorder, prediction provenance | no Qlib implementation copying | false | native content-addressed artifacts | feature-store and Phase A tests |
| freqtrade | `89d469fe638eaf116d45a8f92598aeed4d9f6dde` | GPL-3.0 | crypto ML lifecycle and bias controls | C concept/reference only; native implementation | lookahead slices, recursive warmup, expiry/retrain | no GPL implementation import/copy into production | false | native fail-closed checks and SHADOW | research causality and Phase A tests |
| finrl-trading | `e65d6f0483ead7d2ef4a5fc940cdf960392a25c1` | Apache-2.0 | bounded position-management RL contract | C concept/reference only; native implementation | state/action/reward, deterministic evaluation, baselines | no agent/environment/execution copying | false | deterministic HOLD/REDUCE/EXIT baselines; SHADOW only | RL environment and Phase A tests |
| rd-agent | `6762f84f9bc0f5c6486c50a00e128a57ac6c3683` | MIT | autonomous research lifecycle and memory | C concept/reference only; native implementation | hypothesis, experiment, feedback, trace | no orchestration/LLM implementation copying | false | immutable native research trace | research-loop and Phase A tests |
| tradingagents | `a33fd4c0f134485a43553a2c23a63cb14adbd88f` | Apache-2.0 | structured multi-perspective intelligence | C concept/reference only; native implementation | bull/bear/risk separation, typed hand-off | no graph/prompt/agent copying | false | neutral NO_TRADE snapshot | intelligence and Phase A tests |

## Hard boundaries

- One primary responsibility is assigned to each reference.
- Exact native validation remains the only economic-evidence authority.
- Bitvavo plus the native execution stack remain the only exchange execution path.
- Canonical accounting remains owned by `execution.canonical_state`.
- Risk and live-authority gates always outrank strategy and ML output.
- PyBroker and vectorbt are especially restricted to concept-only use because their local licenses include the Commons Clause.
- Freqtrade/FreqAI is not an execution owner and GPL source is not imported into production.
- FinRL, RD-Agent and TradingAgents remain research/SHADOW references and receive no model-promotion or order authority.
- Existing probes in `research/reference_integrations.py` are non-authoritative diagnostics; they do not establish architectural integration and must remain failure-isolated.

## Required provenance sequence

`reference source -> named invariant/design -> clean native implementation -> native tests -> hash-addressed evidence`

Any unclear license or provenance state is classified as `E_UNCLEAR_DO_NOT_COPY` and blocks copying or runtime coupling.
