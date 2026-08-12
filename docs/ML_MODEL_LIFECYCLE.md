# Canonical ML and model lifecycle

ML remains evidence-gated and has no live decision authority.

The canonical lifecycle is:

`point-in-time feature rows -> immutable dataset manifest -> separated frozen labels -> chronological purged validation -> immutable model artifact -> registry status -> monitored SHADOW inference`

Canonical owners:

- `ml.contracts` defines dataset, label, model, range, provenance and lifecycle schemas.
- `ml.labels` creates conservative triple-barrier labels after the feature cutoff and freezes them immutably.
- `ml.registry` content-addresses datasets/models and rejects promotion when causal, economic, calibration, sample, manual or live evidence is missing.
- `ml.lifecycle` checks `event_time`, `available_at`, closed/final inputs, warmup, expiry and deterministic fallback.
- `core.opportunity_intelligence.build_canonical_ml_dataset` accepts only prospectively recorded causal timestamps, immutable finality, feature provenance and separated label windows.
- `core.opportunity_intelligence.train_canonical_shadow_models` requires 500 rows, 100 examples per class, a 24-hour purge, five exact walk-forward folds, an isolated validation range, an untouched test range and calibration metrics before registering a model.

The existing opportunity-intelligence bundle is intentionally not registered. Its current 2,966 rows do not carry explicit `event_time`, `available_at`, finality and provenance fields, and its five chronological folds do not record exact train/validation/test ranges or purged walk-forward provenance. It remains a legacy SHADOW observation with zero live influence. The current critical-drift state is an additional promotion blocker.

The canonical pipeline currently reports `DATA_PENDING`: all 2,966 legacy rows were excluded without backfill, so no canonical dataset or model was registered from historical evidence. Newly frozen snapshots contain the required fields and accumulate prospectively. A successful canonical model is still `SHADOW_ONLY`, expires after 30 days and cannot create orders or raise risk.

When required provenance is absent, the system must report `NOT_EVALUABLE` or `BLOCKED_LEGACY_PROVENANCE`; it must never manufacture ranges, timestamps, calibration, economic evidence or model authority.

The sanitized dashboard projection is `reference_integration.model_state` in `ui.server.build_ui_snapshot`.
