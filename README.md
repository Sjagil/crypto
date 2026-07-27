# Crypto Spot Research System

This repository is a compact, fail-closed research, backtest, paper-trading, and
capability-gated live execution system for centralized-exchange crypto spot
markets quoted in EUR.

It is research software, not investment advice, a Shariah ruling, a
profitability claim, or an assurance that live trading is safe. The default
mode is research. Unknown market eligibility is blocked, and no strategy can
short, borrow, use leverage, use derivatives, stake, lend, or withdraw funds.

## Practical shadow and paper runbook

Run these commands from `C:\Users\alhar\Documents\crypto` in PowerShell.
`practical_spot_v1` observes BTC-EUR, ETH-EUR, SOL-EUR and LINK-EUR on 1h,
with 4h trend and 1d regime confirmation. It uses public Bitvavo data only.

1. Check that everything is healthy:

   ```powershell
   .\.venv\Scripts\python.exe main.py operate preflight --mode shadow --profile practical_spot_v1
   .\.venv\Scripts\python.exe main.py operate health --mode shadow --profile practical_spot_v1
   ```

2. Start shadow mode:

   ```powershell
   .\.venv\Scripts\python.exe main.py operate start --mode shadow --profile practical_spot_v1 --continuous --resume
   ```

3. View status:

   ```powershell
   .\.venv\Scripts\python.exe main.py operate status --mode shadow --profile practical_spot_v1
   .\.venv\Scripts\python.exe main.py operate report --mode shadow --profile practical_spot_v1
   ```

4. Stop safely:

   ```powershell
   .\.venv\Scripts\python.exe main.py operate drain --mode shadow --wait --timeout 60 --poll-seconds 0.2
   .\.venv\Scripts\python.exe main.py operate stop --mode shadow --wait --timeout 60 --poll-seconds 0.2
   .\.venv\Scripts\python.exe main.py operate lock-status
   # Only after lock-status proves the owner process is stale:
   .\.venv\Scripts\python.exe main.py operate recover-stale-lock
   ```

5. Start paper mode only after a genuine candidate has passed the research and
   shadow gates and has been manually approved:

   ```powershell
   .\.venv\Scripts\python.exe main.py operate activate-paper --id CANDIDATE_ID --yes
   .\.venv\Scripts\python.exe main.py operate start --mode paper --profile practical_spot_v1 --continuous --resume
   ```

6. Inspect the active or available candidate:

   ```powershell
   .\.venv\Scripts\python.exe main.py operate candidates
   .\.venv\Scripts\python.exe main.py operate candidate-inspect --id CANDIDATE_ID
   ```

7. Reconcile paper state:

   ```powershell
   .\.venv\Scripts\python.exe main.py operate reconcile --mode paper
   ```

8. A kill switch never resets automatically. Inspect it first, resolve the
   reported health checks, then use the existing explicit risk reset workflow
   with confirmation and a recorded reason:

   ```powershell
   .\.venv\Scripts\python.exe main.py risk kill-switch-status
   .\.venv\Scripts\python.exe main.py operate reset-kill-switch --mode shadow --reason "resolved checks and reconciled state" --yes
   ```

9. Preview and install the Windows scheduled task:

   ```powershell
   .\.venv\Scripts\python.exe main.py operate task-install --mode shadow --profile practical_spot_v1 --dry-run
   .\.venv\Scripts\python.exe main.py operate task-install --mode shadow --profile practical_spot_v1
   .\.venv\Scripts\python.exe main.py operate task-status
   ```

10. Logs are in `output/logs`, reports and current state are in
    `output/operations`, and durable checkpoints are in `output/checkpoints`.

11. When no approved candidate exists, the service remains healthy in
    `IDLE_NO_APPROVED_CANDIDATE`. It still collects public ticker, trade,
    order-book and provider-health data and never invents a strategy.

12. Live mode remains blocked: this phase authorizes no live acceptance order,
    no automatic promotion and no bypass of the multi-gate live preflight.
    Shadow uses no private endpoint; paper sends no exchange order.

## Supported scope

- Markets: long-only CEX spot pairs quoted in EUR.
- Default allowlist: `BTC-EUR`, `ETH-EUR`, `SOL-EUR`, and `LINK-EUR`.
- Execution venue: Bitvavo only, behind the existing capability preflight.
- Historical/public exchange data: Bitvavo, Kraken, and MEXC spot.
- Public live streams: normalized Bitvavo, Kraken, and MEXC ticker, trade, and
  Level 2 events; MEXC's current protobuf messages are decoded locally.
- Context-only providers: CoinMarketCap, EODHD, SEC, FRED, MEXC derivatives,
  and public Deribit crypto option chains. None can become executable
  instruments.
- Storage: SQLAlchemy with SQLite WAL by default and PostgreSQL when a valid
  `DATABASE_URL` is configured.
- Intelligence: bounded RSS and web ingestion with relevance filtering,
  per-source status, deduplication, and explicit publication-time knowability.
- Accounting: EUR cash plus owned base-asset units; no synthetic short balance.
- Research: a formal item-level indicator registry, causal features, 14
  registered strategy families, next-open simulation, transaction costs,
  walk-forward analysis, stress tests, parameter stability, bootstrap and
  Monte Carlo uncertainty.

The allowlist in `config/shariah_allowlist.yaml` is an operational control, not
a religious opinion. Review it independently and keep uncertain assets in
`REVIEW_REQUIRED`.

## Windows PowerShell setup

Run these commands from PowerShell:

```powershell
Set-Location 'C:\Users\alhar\Documents\crypto'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m playwright install chromium
Copy-Item .env.example .env
python main.py config validate
python main.py doctor
python main.py self-test
python -m pytest -q
```

If script activation is prohibited, keep PowerShell unchanged and replace
`python` in every example with `.\.venv\Scripts\python.exe`.

`Copy-Item` is optional for research-only use. Never commit `.env`. Use
different Bitvavo credentials for public data and trading. A live key must be
trade-only, IP-restricted, and incapable of withdrawals.

## Quick research workflow

The synthetic workflow validates orchestration without pretending that
synthetic results establish an edge:

```powershell
python main.py strategies list
python main.py backtest --market BTC-EUR --strategy ema_trend_pullback --rows 900 --monte-carlo-runs 100
python main.py optimize --market BTC-EUR --strategy ema_trend_pullback --rows 900 --method random --trials 10
python main.py walk-forward --market BTC-EUR --strategy ema_trend_pullback --rows 900 --folds 6
python main.py research --market BTC-EUR --strategy ema_trend_pullback --rows 900 --method random --trials 10 --monte-carlo-runs 100
```

Research output is placed under `output/reports`. A rejected research gate is a
valid pipeline result. It means the evidence did not satisfy the configured
acceptance criteria.

To backtest local normalized candles:

```powershell
python main.py data validate '.\data_store\normalized\BTC-EUR_1h.parquet' --market BTC-EUR --timeframe 1h
python main.py backtest --data '.\data_store\normalized\BTC-EUR_1h.parquet' --market BTC-EUR --strategy ema_trend_pullback --monte-carlo-runs 10000
```

## Public data

Inspect provider limitations, then download only allowlisted markets:

```powershell
python main.py data providers
python main.py data download --markets BTC-EUR ETH-EUR SOL-EUR LINK-EUR --timeframes 1h 4h 1d --providers bitvavo kraken coinmarketcap --start '2020-01-01T00:00:00Z'
python main.py data inspect '.\data_store\normalized\BTC-EUR_1h.parquet'
python main.py providers list
python main.py providers capabilities
python main.py providers test --public-only
python main.py data historical --provider mexc --market BTC-USDT --timeframe 1h
python main.py data estimate --providers all --universe-size 25 --history-profile maximum --timeframes all
python main.py data fetch --providers bitvavo,kraken,mexc,coinmarketcap --universe-size 10 --history-profile standard --timeframes 5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,1W --resume --yes
python main.py data context-fetch --providers coinmarketcap,eodhd,fred,sec,alternative_me,defillama,deribit --history-profile standard --resume
python main.py data status
python main.py data coverage
python main.py data gaps
python main.py data freshness
python main.py data database-health
```

Downloaded candles are normalized once, restricted to closed candles, checked
for duplicates and impossible OHLC values, stored per provider without
cross-provider candle synthesis, written atomically, and accompanied by raw,
normalized, watermark, lineage, and hash provenance. Canonical non-native
intervals are resampled from one finer provider series only.

Relevant official references:

- [Bitvavo candlestick data](https://docs.bitvavo.com/docs/rest-api/get-candlestick-data/)
- [Kraken OHLC data](https://docs.kraken.com/api-reference/market-data/get-ohlc-data)
- [CoinMarketCap API](https://coinmarketcap.com/api/documentation/pro-api-reference/cryptocurrency)
- [MEXC public market data](https://www.mexc.com/api-docs/spot-v3/market-data-endpoints/)
- [FRED observations and vintages](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

## Live data, order books, and context

These commands are public-data only and cannot submit orders:

```powershell
python main.py websocket run --provider bitvavo --market BTC-EUR --duration 60
python main.py websocket status
python main.py orderbook snapshot --provider kraken --market BTC-EUR --depth 100
python main.py orderbook inspect --provider mexc --market BTC-USDT --depth 100
python main.py macro build --timeframes 1h,4h,12h,1d
python main.py macro inspect
python main.py gex collect --underlying BTC
python main.py gex inspect
```

The local book uses `Decimal` prices and quantities, validates sequences and
checksums where supplied, invalidates on gaps, requests a fresh snapshot, and
reports spread, microprice, imbalance, depth, slippage, impact, pressure, and
resiliency. These features remain inputs to strategy and risk gates; they never
bypass them.

Macro inputs require an explicit provider, cadence, maximum age, window
interpretation, and unit declaration. Alignment is backward-only by
`available_at` or `observed_at`. FRED observation dates and revision vintages
remain separate. Funding annualization uses each record's real funding
interval. GEX publishes unsigned gross exposure and a clearly labeled signed
heuristic; open interest does not establish dealer direction.

## Intelligence ingestion

Run sources, inspect their individual status, and audit causal timing:

```powershell
python main.py scrape run
python main.py scrape status
python main.py scrape inspect --limit 10
python main.py scrape audit
```

The scraper honors `robots.txt`, uses bounded concurrency and payload sizes, and
does not bypass access controls. Playwright is only a non-evasive rendering
fallback. Missing or unverifiable publication times are usable from
`observed_at` forward only; the system never reconstructs earlier availability.
Sentiment is a feature or filter, never an automatic standalone trade signal.

## Strategies and research controls

```powershell
python main.py strategies describe ema_trend_pullback
python main.py features build --data '.\data_store\normalized\BTC-EUR_1h.parquet' --market BTC-EUR
python main.py monte-carlo --r-multiples '1.5,-1,2,-1,0.8,-1,1.2' --runs 10000 --trades 100
```

The feature registry includes causal trend, momentum, volatility, volume,
candlestick, confirmed-fractal market structure, BTC-relative strength, and
time-aware intelligence features. A pivot is unavailable until its right-side
confirmation bars have closed.

The formal registry preserves all 1,149 supplied masterlist occurrences as
1,147 unique canonical definitions. Every definition has one explicit coverage
status, typed parameter metadata, causal/warm-up fields, provider and missing
data policies, strategy roles, redundancy grouping, and a deterministic
configuration hash. Registry presence is not an implementation or profitability
claim.

```powershell
python main.py indicators coverage --output '.\output\reports\indicator_coverage.json'
python main.py indicators list --family FRACTAL
python main.py indicators list --provider-availability required
python main.py indicators describe fractal.5_candle_confirmed_fractal
python main.py indicators fractal-audit --rows 500
python main.py features audit --rows 500 --market BTC-EUR
python main.py investing score --asset BTC --input '.\normalized-investing-input.json'
```

Confirmed 3-, 5-, and 7-candle fractals expose separate pivot and confirmation
timestamps. Tradable frames contain confirmation events only. Post-fractal MFE,
MAE, and efficiency are generated by a separate research-label function and
are never early strategy inputs. Higher-timeframe fractals are aligned
backward-only from already confirmed higher-timeframe candles.

The investing scorer is separate from trading. It reports valuation, adoption,
usage, tokenomics, dilution risk, liquidity, decentralization, security, holder
concentration, economic sustainability, macro sensitivity, and data-quality
subscores. Missing inputs lower confidence and are never filled as positive or
negative evidence.

The backtester executes a signal at the following candle's open. It models
fees, spread, slippage, position precision, minimum order value, cash reserve,
owned units, stops, targets, trailing exits, partial reductions, gaps, and a
conservative stop-first rule when stop and target occur in the same bar.

Research promotion requires all configured gates, including:

- valid data and intelligence timing;
- no lookahead or repainting;
- allowlisted markets;
- sufficient trades and effective sample size;
- positive holdout expectancy and profit factor;
- stressed-cost survival;
- valid walk-forward folds with limited concentration;
- acceptable drawdown and Monte Carlo loss probabilities;
- parameter-neighborhood stability.

`--promote-to-paper` can label a passing result `PAPER_CANDIDATE`; it cannot
authorize live trading.

## Paper trading

Paper trading uses a durable JSONL event ledger, market rules, fees, spread,
slippage, latency, reservations, partial fills, cancellation, idempotency, and
reconciliation checks.

```powershell
python main.py paper status
python main.py paper run --market BTC-EUR --strategy manual-paper-check --capital 2000 --price 20000 --quantity 0.001 --idempotency-key paper-check-001
python main.py paper reconcile
```

Paper trading is still simulation. It does not prove fill quality, exchange
availability, or profitability.

## Live trading is blocked by default

Inspect blockers without placing an order:

```powershell
python main.py live status
python main.py live preflight --research-report '.\output\reports\research\research_summary.json' --data '.\data_store\normalized\BTC-EUR_1h.parquet' --market BTC-EUR --timeframe 1h
```

Live preflight requires, at minimum:

- `ENVIRONMENT=production` and `EXECUTION_MODE=live`;
- `LIVE_TRADING_ENABLED=true`;
- the exact configured manual approval phrase;
- a finite `MAXIMUM_LIVE_ORDER_EUR`;
- a separate Bitvavo trade key and secret;
- a positive Bitvavo operator ID;
- declared trade-only scope, no withdrawal permission, and confirmed IP
  whitelist;
- `PAPER_CANDIDATE` research status;
- fresh valid local candles;
- exchange connectivity, balance reads, and successful reconciliation;
- healthy risk controls, an inactive kill switch, and an `ALLOWED` market.

Only after every gate passes can this command submit a real spot order:

```powershell
python main.py live run --research-report '.\output\reports\research\research_summary.json' --data '.\data_store\normalized\BTC-EUR_1h.parquet' --market BTC-EUR --timeframe 1h --strategy approved-strategy --side BUY --price 20000 --quantity 0.001 --stop-fraction 0.05 --idempotency-key 'unique-reviewed-live-intent'
```

That command has real financial consequences. Review the research report,
account balances, order cap, market price, quantity, and stop assumption before
running it. The client never retries an ambiguous order submission; it requires
reconciliation by client order ID instead. The active code contains no
withdrawal, transfer, margin, borrowing, or derivative endpoint.

Official execution references:

- [Bitvavo REST authentication](https://docs.bitvavo.com/docs/rest-api/introduction/)
- [Bitvavo create order](https://docs.bitvavo.com/docs/rest-api/create-order/)
- [Bitvavo balances](https://docs.bitvavo.com/docs/rest-api/get-account-balance/)

## Configuration

`config/settings.py` is the only configuration model. It validates environment
values and redacts every `SecretStr` in configuration output and generated
reports. Existing aliases such as `VENUE_A_API_KEY`, `VENUE_A_API_SECRET`,
`VENUE_B_API_KEY`, `VENUE_B_API_SECRET`, `CMC_API_KEY`, and `WM_OPERATOR_ID`
remain supported.

Useful inspection commands:

```powershell
python main.py config show
python main.py eligibility list
python main.py eligibility check BTC-EUR
python main.py report '.\output\reports\research\research_summary.json'
python main.py positions status
python main.py positions pnl
python main.py risk correlation
python main.py risk drawdown
python main.py risk kill-switch-status
python main.py report charts
```

`config show` emits only redacted secrets.

## Continuous combinatorial research lab

The lab composes registered, causal `SignalBlock` building blocks and delegates
all exact fills, optimization, walk-forward validation, Monte Carlo analysis,
and acceptance gates to the existing research stack. It is spot-only,
long-only, never places an order, and never promotes a result to live trading.
Fast-screen output is always labeled `SCREENING_ONLY`.

CoinMarketCap snapshots preserve a discovery universe of 25 technically
suitable ordinary crypto assets, scanning below rank 25 when stablecoins,
wrapped representations, duplicates, insufficient history, or illiquidity are
excluded. Discovery is exploratory only. The allowed-research and execution
universes remain separate and fail-closed: only explicitly `ALLOWED` assets can
enter executable research, and execution additionally requires a usable
Bitvavo EUR spot market. `REVIEW_REQUIRED` assets can appear only in a
separately requested research-only scope and can never become paper or live
candidates. Current-universe historical tests remain labeled
`CURRENT_UNIVERSE_RETROSPECTIVE`.

`--data-mode real` is the default. It requires normalized local data with
immutable `REAL_PROVIDER_DATA` provenance and sufficient closed candles. A
missing, stale, incomplete, or provider-failed dataset blocks the run; the lab
never substitutes synthetic candles. Synthetic mode exists only for explicit
offline smoke testing and its rows are excluded from the default leaderboard.

Useful commands:

```powershell
python main.py lab universe refresh --size 25 --scan-limit 100
python main.py lab universe coverage
python main.py lab blocks list
python main.py lab blocks validate
python main.py lab data status --universe-size 5 --timeframes 1h,4h --minimum-rows 2000
python main.py lab data prepare --markets BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR --allowed-universe --timeframes 5m,15m,1h,4h,1d --history-profile maximum --minimum-rows 500 --force
python main.py lab data validate --universe-size 5 --timeframes 1h,4h --minimum-rows 2000
python main.py lab indicators coverage
python main.py lab indicators describe --id rsi
python main.py lab indicators parameters --id rsi
python main.py lab indicators test --id rsi
python main.py lab combinations estimate --profile quick --universe-size 5 --combination-sizes 1,2,3
python main.py lab combinations generate --profile quick --combination-sizes 1,2 --resume --yes
python main.py lab combinations status
python main.py lab run --data-mode real --profile quick --universe-size 5 --combination-sizes 1,2 --timeframes 1h,4h --workers 2 --only-missing --resume --once
python main.py lab run --data-mode synthetic --profile quick --blocks rsi_threshold --combination-sizes 1 --parameter rsi_threshold.value=13:15:0.5 --once
python main.py lab run --continuous --data-mode real --profile quick --universe-size 5 --combination-sizes 1,2 --workers 2 --only-missing --resume
python main.py lab status
python main.py lab pause
python main.py lab resume
python main.py lab drain
python main.py lab stop
python main.py lab queue
python main.py lab leaderboard --top 25
python main.py lab leaderboard export
python main.py lab report
python main.py lab campaign plan --name microstructure-5m15m
python main.py lab campaign run --name microstructure-5m15m --workers 4 --max-trials 20 --yes
python main.py lab campaign plan --name formal-five-family
python main.py lab campaign run --name formal-five-family --workers 4 --max-trials 20 --yes
python main.py lab campaign plan --name cross-sectional-rotation
python main.py lab campaign run --name cross-sectional-rotation --yes
python main.py lab campaign plan --name cross-sectional-ensemble
python main.py lab campaign run --name cross-sectional-ensemble --yes
python main.py lab campaign report --name cross-sectional-ensemble
python main.py lab campaign external --name cross-sectional-ensemble
python main.py lab campaign forward --name cross-sectional-ensemble
python main.py lab campaign audit --name cross-sectional-ensemble
python main.py lab campaign observe --name cross-sectional-ensemble
python main.py lab campaign package --name cross-sectional-ensemble
python main.py lab campaign plan --name institutional-rotation-v2
python main.py lab campaign run --name institutional-rotation-v2 --yes
python main.py lab campaign report --name institutional-rotation-v2
python main.py lab campaign plan --name capital-utilization-v1
python main.py lab campaign run --name capital-utilization-v1 --yes
python main.py lab campaign report --name capital-utilization-v1
python main.py lab campaign observe --name capital-utilization-v1
python main.py lab campaign plan --name diversified-rotation-v1
python main.py lab campaign run --name diversified-rotation-v1 --yes
python main.py lab campaign report --name diversified-rotation-v1
python main.py lab campaign observe --name diversified-rotation-v1
python main.py lab campaign plan --name portfolio-breakout-v1
python main.py lab campaign run --name portfolio-breakout-v1 --yes
python main.py lab campaign report --name portfolio-breakout-v1
python main.py lab campaign observe --name portfolio-breakout-v1
python main.py lab campaign plan --name portfolio-storm-v1 --storm-trials 5000
python main.py lab campaign run --name portfolio-storm-v1 --storm-trials 5000 --yes
python main.py lab campaign status --name portfolio-storm-v1
python main.py lab campaign report --name portfolio-storm-v1
python main.py lab campaign plan --name signal-synthesis-storm-v1 --storm-trials 5000
python main.py lab campaign run --name signal-synthesis-storm-v1 --storm-trials 5000 --yes
python main.py lab campaign status --name signal-synthesis-storm-v1
python main.py lab campaign report --name signal-synthesis-storm-v1
python main.py lab campaign plan --name residual-reversal-v1
python main.py lab campaign run --name residual-reversal-v1 --yes
python main.py lab campaign report --name residual-reversal-v1
python main.py lab campaign observe --name residual-reversal-v1
python main.py lab campaign autopilot
python main.py lab campaign autopilot --mode status
python main.py lab campaign autopilot --run-research --refresh-data
python main.py lab campaign autopilot --mode continuous --run-research --refresh-data --max-cycles 7
python main.py lab campaign autopilot --mode task-install --dry-run
python main.py lab campaign autopilot --mode task-install --yes
python main.py lab campaign autopilot --mode task-status
python main.py lab state
python main.py lab state --apply
```

The microstructure campaign screens every valid registered single/pair DNA on
real common-history 5m and 15m candles before exact survivor tests. The formal
campaign deliberately narrows the next stage to five economic hypotheses:
trend breakout, pullback in an uptrend, range mean reversion, volatility
expansion after prior contraction, and BTC-relative strength. It freezes data,
feature, software, cost, gate, seed, and DNA hashes in a stage-0 plan before
screening. Higher-timeframe 4h/1d state becomes visible only after the source
candle closes. Zero accepted candidates is a valid result; no gate is weakened
to manufacture profitability.

The cross-sectional campaigns rank the allowed daily assets, execute the
decision at the next open, hold at most two assets, and support cash as an
explicit allocation. Every one-way weight change, including terminal
liquidation, pays fees, slippage, and half-spread. Assets join the panel only
after their own real point-in-time history provides the required warmup.
`cross-sectional-ensemble` combines declared momentum horizons with continuous
BTC-trend and market-breadth exposure scaling. Baseline, joint-parameter,
sensitivity, and exact results are separate result types; sensitivity rows
cannot become screening survivors.

The institutional audit keeps the frozen signal DNA unchanged but applies a
separately hashed fail-closed portfolio policy: BTC-EUR, ETH-EUR, SOL-EUR and
LINK-EUR only, at most 40% total exposure, at most 20% per asset, at least 60%
cash, and at least 90 real daily observations before an asset can rank. Reports
separate scheduled rebalances, changed portfolios, buy/sell fills, holding
episodes, weekly effective sample size, daily/weekly returns, and asset-trade,
closed-position, portfolio-period and rebalance-episode profit factors. Exact
asset PnL attribution is reconciled to final equity. The audit also rebuilds
all 160 original family paths and reports DSR for daily raw, daily HAC, weekly
raw and weekly ESS samples while retaining all 1,240 known trials. PBO is
reported both by nominal DNA and by exact unique return path with midrank tie
handling. The formal value is always the worst valid result.

The economic comparison uses the strategy's actual time-varying daily
exposure, a point-in-time equal-weight allowed universe and the same calendar
and costs. Validation, confirmation and double-cost confirmation each require
a positive paired block-bootstrap lower bound. PnL concentration is reported
by asset, year, causal BTC/volatility/breadth regime and top-five position
episodes. These diagnostics cannot be used to select a new threshold
retroactively.

```bash
python main.py lab campaign audit --name cross-sectional-ensemble
```

`institutional-rotation-v2` is a separate 48-combination continuation family;
it cannot overwrite the original frozen lead. Its DSR accounts for all prior
known trials. Benchmark evidence includes cash, BTC buy-and-hold, point-in-time
equal-weight weekly portfolios, a volatility-matched BTC view, and predeclared
regime/momentum ablations. These are diagnostics, not retroactive promotion
evidence.

`capital-utilization-v1` keeps the frozen momentum, ranking, universe,
weekly timing, filters and next-open execution unchanged. It compares an exact
frozen control with defensive 40%, balanced 60%, semi-aggressive 80% and
piecewise semi-aggressive 80% allocation policies. Each decision records
eligible/excluded assets, exclusion reasons, ranks, regime components, budgets
before and after caps, cash reason attribution, turnover and expected costs.
The report adds Sortino, Omega, CVaR, drawdown duration, exposure buckets,
40/60/80% equal-weight and exposure-matched benchmarks, plus paired block
bootstrap differences. Every policy is counted as a known trial. Policies
above configured operational exposure remain research-only; all observers
generate and submit zero orders. Their forward ledgers are append-only:
historical source truncation, policy/execution identity drift, or any changed
realized record fails closed. Formal performance gates remain dormant until
365 realized daily intervals, 30 changed portfolios and the declared
BTC-trend, volatility and breadth coverage are all present. The scheduled
autopilot updates these capital-utilization observers alongside the breakout
observers without granting paper or live authority.

`diversified-rotation-v1` is a separate six-trial continuation family. It keeps
the frozen 20/90-day momentum horizons, weekly next-open timing, EMA50 asset
filter, BTC/breadth regime and ALLOWED universe, while declaring new strategy
DNA for top-3/top-4 selection and inverse-volatility or equal-risk-contribution
weighting. A 60-day backward-only covariance matrix supplies ex-ante 15% or 20%
annualized volatility targeting. ERC solver failures, risk-model cash, expected
costs and decision reasons are fail-closed and audited. The family cannot
overwrite or promote the original frozen lead and all six variants count toward
DSR, White, SPA and PBO.

`portfolio-breakout-v1` is a separate eight-trial economic alpha family, not an
allocation continuation. It evaluates classic prior-channel Turtle 20/10 and
55/20 rules with EMA50/EMA200 trend filters and equal/inverse-volatility
weights. Entries and daily exits use completed closes and execute next open.
The strict policy remains BTC/ETH/SOL/LINK only, 40% maximum total exposure,
20% per asset and 60% minimum cash. The report detects identical return paths
caused by hard-cap saturation but still counts every declared variant in the
1,312-trial multiple-testing universe.

`portfolio-storm-v1` preregisters a deterministic 5,000-DNA search before any
objective is evaluated. It varies only declared momentum, EMA, top-1/top-2,
rebalance, regime-mapping, weighting, strict exposure and hysteresis choices.
Every trial stays below 40% total, 20% per asset and above 60% cash. Selection
uses only the development split and constructs a Pareto front by maximizing
portfolio-period profit factor while minimizing Ulcer Index and turnover
efficiency. Validation and confirmation are reported only after survivor DNA
is frozen. White Reality Check, Hansen SPA and PBO use all 5,000 development
return paths aggregated to causal W-SUN periods; their 2,000-sample circular
block bootstrap runs in bounded batches. DSR uses the full known-trial
denominator. Failed statistics or confirmation therefore prevent a research
pass and the storm can never create a paper/live candidate directly.

`signal-synthesis-storm-v1` reuses the canonical 134-block registry rather
than inventing a second indicator engine. One unavailable high-impact-event
block remains explicitly blocked until a timestamped point-in-time event feed
provides its required source column; the other 133 blocks are covered across
all 11 implemented families and seven roles. Exactly 5,000 immutable DNA paths
are stratified over 1h/4h/1d, all six pairs from BTC/ETH/SOL/LINK, layered/all/
majority/weighted-vote logic, min/default/max parameter alleles and fixed-R,
trailing-trend or time-regime exits. Every path holds at most two 20% positions,
keeps at least 60% cash, shifts signals to next open and pays every fill cost.
The mark-to-market screen uses development-only PF/Ulcer/turnover objectives.
A pre-Pareto gate rejects inactive fronts (at least 12 active development
weeks, positive development return and PF above one), while all inactive and
failed paths still remain in White, SPA, PBO and DSR accounting. Positive
confirmation paths receive a second audit through the canonical event-driven
backtester under normal and doubled costs. Screening or exact economic success
cannot bypass the unchanged statistical, forward, manual-approval or live
gates.

`campaign autopilot` runs one bounded, orderless cycle by default. It audits a
deterministic data fingerprint, schedules only the already preregistered
breakout, portfolio-storm and signal-synthesis families when `--run-research`
is enabled, verifies every frozen observer manifest, and persists a cycle
record below `output/lab/autopilot/`. Research
is skipped when data is unchanged or the seven-day research interval has not
elapsed. `--refresh-data` refreshes only the strict ALLOWED universe;
`--mode continuous` is explicit and can be bounded with `--max-cycles`.
When new data and the weekly interval are both present, the research stage also
creates or reuses immutable 5,000-DNA epochs below
`output/lab/storm_epochs/` and `output/lab/signal_storm_epochs/`. Portfolio
epochs fingerprint daily data; signal epochs fingerprint 1h, 4h and 1d data.
Repeated access to either unchanged fingerprint reuses its existing epoch; a
genuinely new selection epoch conservatively adds all 5,000 trials to the
cumulative DSR denominator. Epochs remain research-only.
The general point-in-time research feature snapshot is disabled by default.
`--build-feature-store` explicitly builds or reuses
`output/lab/feature_store/portfolio_daily_v1/latest.npz`; this does not
authorize AI or model development. Its 22 dimensionless/log-scaled features
include returns, realized volatility, shifted Donchian distances, EMA
distances, volume state, BTC-relative momentum, cross-sectional rank and
breadth. The separate target is causally aligned from the next executable open
to the following open. Immutable snapshots live below their dataset ID and an
atomic pointer identifies the latest snapshot. No full-sample normalization is
permitted.
The observer stage then reconstructs append-only open-to-open forward returns
for all eight frozen breakout DNA variants and all five capital-utilization
policies. Every record hashes its source
prices, weights, turnover, costs, regime and realized hypothetical return.
The ordered observation hashes are also protected by a deterministic hash
chain. Revised, deleted, reordered or truncated historical evidence fails
closed. Performance gates remain
disabled until 365 realized daily intervals, 30 rebalances and five decisions
in every required trend/volatility/breadth state are all present; only then are
normal/stressed return, PF, drawdown, ESS and block-bootstrap CI evaluated.
`--mode task-install --yes` installs a least-privilege Windows task that runs
the orderless refresh/research cycle daily at 03:15 local time, starts missed
runs when available, ignores overlapping instances and records all evidence in
the same persistent state. A UTC data watermark requires all four assets to
contain exactly the expected last closed daily candle; partial data produces
`WAITING_FOR_COMPLETE_DAILY_SNAPSHOT` without research or observer updates.
Inspect installation first with `--dry-run`; status and removal use
`task-status` and `task-remove --yes`.

Forward-degradation evidence may be supplied with `--degradation-input` as JSON
containing `live_return`, `cv_mean`, `cv_std` and `observation_count`. Fewer
than 30 observations remains `INSUFFICIENT_FORWARD_DATA`. Checkpoints at 30,
90 and 180 days are diagnostic only and cannot change lifecycle state. From
365 observations, an undefined metric or z-score below -2.0 may activate the
persistent `SYSTEM_DEGRADED` kill-switch. Stage failures and order/promotion
invariant violations remain immediate operational failures. Reset requires
`--mode reset --yes --reason "..."`. The autopilot cannot set paper/live ready,
cannot submit orders and cannot overwrite the frozen candidate.

Formal promotion now also requires two independent stochastic robustness gates.
The stationary-bootstrap Monte Carlo uses 10,000 dependent-path simulations
with geometric block restarts and must keep both the terminal-loss probability
below 5% and the probability of breaching a 20% drawdown below 1%; its 5th
percentile total return must remain non-negative. The Dirichlet
time-concentration stress runs 10,000 simulations for concentration alphas
0.5, 1.0 and 5.0 over chronological market blocks. Every profile must retain
at least 95% probability of a positive terminal result and a non-negative 5th
percentile. Both checks run on normal and stressed-cost net return paths.
Probability gates use the one-sided 95% Wilson upper confidence bound rather
than accepting the raw simulation frequency at face value.
They never alter strategy DNA, DSR, White Reality Check, Hansen SPA, PBO or the
known-trial denominator. Missing, invalid or insufficient paths fail closed.
The generic acceptance gate and the rotation, capital-utilization,
diversification and breakout campaign reports all enforce and persist this
evidence.

`python main.py lab campaign run --name absolute-momentum-v1 --yes` evaluates
the accounted classical absolute-time-series-momentum family. It uses three
fixed momentum horizons, per-asset EMA200 eligibility, a BTC EMA200 regime,
inverse-volatility weights, Sunday-close decisions and Monday-next-open
execution. The primary 5% volatility-budget path is capped at 20% total and
20% per asset with at least 80% cash. All 16,715 known formal, storm,
development, ablation and risk-budget trials remain in the DSR denominator.
Its observers are `FROZEN_FORWARD_RESEARCH`, generate no orders and cannot
promote while PBO or the untouched-forward gate fails. The daily orderless
autopilot observer stage updates these ledgers independently of the slower
research-search interval. Each append-only ledger reports read-only 30, 90 and
180-day diagnostic milestones plus the mandatory 365-day formal sample gate;
diagnostic milestones never authorize promotion.

`python main.py lab campaign run --name absolute-momentum-plateau-v1 --yes`
runs a separately versioned robustness family around the fixed 20/60/120-day
absolute-momentum hypothesis. It predeclares all 117 horizon-shift,
volatility-lookback and volatility-budget trials, evaluates the complete
Gaussian N±2 horizon neighbourhood, and selects only on development data.
Every trial is written once to a content-addressed, hash-chained registry under
`output/lab/strategy_registry/absolute_momentum_plateau_v1/`. Repeating the
same data and DNA is idempotent; changed data creates a new immutable
data-epoch record for the same strategy DNA. Strategy-DNA counts determine the
multiple-testing denominator. Data-epoch records are reported separately and
never inflate that denominator.
Plateau smoothing is a robustness diagnostic, never a way to override DSR,
White Reality Check, Hansen SPA, standard PBO, plateau-selection PBO or the
untouched-forward requirement. The campaign is orderless, uses at most 20%
total exposure and remains blocked whenever any formal gate fails.

`python main.py lab campaign run --name volatility-contraction-v1 --yes`
evaluates a separate 16-DNA classical alpha family. A long signal requires a
20- or 40-day volatility estimate below its causal 20th/30th percentile over
strictly older 252-day history, followed within five closed candles by a
20/55-day prior-channel breakout. EMA200 asset and BTC regime filters, 10/20
day exits, 10%/15% volatility targets, inverse-volatility allocation and
next-open execution are fixed before the run. Exposure is capped at 40% total
and 20% per asset with at least 60% cash. All trials are content-addressed and
included in the cumulative multiple-testing denominator. The historical v1
development winner fails confirmation, DSR, PBO and stochastic gates; it is
therefore retained only as `FROZEN_FORWARD_RESEARCH`, with zero order or
promotion authority.

`python main.py lab campaign run --name multi-alpha-ensemble-v1 --yes`
evaluates one fixed, preregistered portfolio of three frozen classical sleeves:
5% absolute momentum, canonical Turtle 20/10 with EMA200, and the frozen
volatility-contraction primary. Sleeve weights are equal and are not optimized.
A causal 60-day diagonal volatility target, top-2 selection, next-open
execution and normal/stressed costs are applied under the same 40% total,
20% per-asset and 60% minimum-cash limits. Its v1 history remained positive in
development, validation and confirmation, but validation profit factor,
inherited component-selection bias, untouched holdout and both stochastic
gates failed. It is therefore a balanced research lead only: the immutable
registry and orderless observer are retained, while paper/live promotion stays
fail-closed.

`python main.py lab campaign run --name trend-pullback-v1 --yes` evaluates a
separate 12-DNA long-only mean-reversion family. Entries require a 10/20/40-day
log-price z-score of -1.5 or -2.0 while both the asset and BTC remain above
predeclared long-horizon EMAs; exits occur near the rolling mean or when either
trend filter fails. Signals use completed daily closes and execute at the next
open under normal and stressed costs, with the same 40% total, 20% per-asset
and 60% minimum-cash limits. The development winner lost money in validation
and under full-sample stressed costs, while DSR, White Reality Check, Hansen
SPA, PBO and both stochastic gates failed. All 12 negative trials remain in
the immutable cumulative denominator; no post-hoc variant is promoted.

`python main.py lab campaign run --name range-expansion-4h-v1-1 --yes`
evaluates a separate 16-DNA, four-hour range-expansion family on BTC-EUR,
ETH-EUR, SOL-EUR and LINK-EUR. A completed-bar Donchian breakout must exceed a
causal ATR threshold, pass relative-volume confirmation and agree with
asset/BTC trend filters; execution occurs at the next common four-hour open.
The engine uses the strict intersection of observed asset timestamps and never
fills missing bars. Version 1.0 stopped before trial registration when it found
that a held asset could lack a next-open valuation. Its failed preflight plan
is retained for audit. Version 1.1 made the calendar policy explicit and
started a new immutable campaign rather than rewriting that failed attempt.

The preregistered v1.1 development winner was
`RE4H_E60_X30_R15_V15_EMA600`. It lost 2.9% net over the full sample and 45.2%
under stressed costs, recorded a 20.3% normal-cost drawdown, failed
confirmation, and failed DSR, White Reality Check, Hansen SPA, PBO and both
stochastic gates. All 16 trials remain in the cumulative 16,877-trial
denominator. The observer is orderless, forward progress is 0/2,190 closed
four-hour bars, and no research, paper or live promotion is permitted.

`python main.py lab campaign run --name sentiment-recovery-v1 --yes`
evaluates a separate preregistered eight-DNA daily family on BTC-EUR and
ETH-EUR. Entries require a timestamped recovery from external extreme fear
while both BTC and the selected asset remain above a fixed 100/200-day EMA.
Signals use completed daily observations and execute at the next open, with
normal and stressed costs, at most 40% total exposure, 20% per asset and at
least 60% cash. The Alternative.me history is backward-only aligned and never
backfilled before inception. Its historical values have source-day timestamps
but no revision vintages; that retrospective-source limitation is explicit
and makes genuine forward evidence mandatory.

The development-selected v1 primary, `SR_F20_D5_EMA100`, returned 10.3% over
the full sample with a 6.7% maximum drawdown and only 1.6% average exposure.
PBO passed at 0.071, but validation and confirmation produced no trades, DSR
was 0.002, White Reality Check was 0.315 and Hansen SPA was 0.296. Economic,
stochastic and untouched-holdout gates therefore failed. All eight trials are
stored in the immutable 21,320-trial accounting history; the observer remains
orderless at 0/365 forward days and no promotion is permitted.

`python main.py lab campaign run --name residual-momentum-v1 --yes`
evaluates a separate eight-DNA factor-residual family. A fixed 20% BTC core is
held only above BTC EMA200. At each Sunday close, one additional 20% altcoin
satellite may be selected from ETH-EUR, SOL-EUR and LINK-EUR when its
20/60-day log return minus its backward-looking 90/180-day BTC beta times BTC
return is positive and its own price is above EMA100/200. Execution occurs at
the following open under normal and stressed costs; exposure remains capped at
40% total, 20% per asset and at least 60% cash.

The development-selected primary `RM_R60_B180_EMA200` returned 158.1% over the
full sample with 14.9% CAGR, 0.85 Sharpe and 19.4% average exposure. DSR
(0.973), White Reality Check (0.017) and Hansen SPA (0.017) passed, but PBO
(0.429), 28.7% drawdown, only 69 actual rebalances, validation PF 1.042,
negative confirmation and both stochastic gates failed. It also underperformed
the simpler BTC/ETH trend benchmark. All eight trials remain in the immutable
21,328-trial denominator; its 0/365-day observer is orderless and cannot
promote.

`python main.py lab campaign run --name dual-asset-trend-v1 --yes` evaluates
one fixed, discovery-informed BTC/ETH strategy rather than selecting another
historical winner. BTC and ETH require EMA200 trends; a backward-only 60-day
full covariance matrix scales equal active-asset weights toward 15% annualized
volatility. Entries rebalance after the Sunday close, daily trend exits execute
at the next open, and exposure remains capped at 40% total and 20% per asset.

The fixed `DAT_EMA200_COV60_VOL15` path returned 86.3% with 9.5% CAGR, 0.87
Sharpe, 14.8% average exposure and 213 rebalances. Development, validation,
confirmation and their double-cost counterparts were all positive. DSR
(0.987), White Reality Check (0.018) and Hansen SPA (0.018) passed. It still
failed the 20% drawdown limit at 22.2%, validation PF at 1.136, both stochastic
gates and the uncontaminated-selection/untouched-holdout gates. PBO is
explicitly not applicable to a single fixed DNA, but that is not recorded as a
pass. Because the simple benchmark was already observed before this policy was
declared, only genuine forward evidence can resolve its discovery
contamination. The orderless observer starts at 0/365 days.
This family contributes exactly one strategy trial to the cumulative
21,329-trial denominator; repeated evaluation on later data epochs remains
separately hash-chained without becoming a new strategy trial.

`python main.py lab campaign run --name liquidity-sweep-v1 --yes` runs the
separately preregistered eight-DNA confirmed-fractal liquidity-sweep family.
Signals use only fractal levels that became knowable after their full two- or
three-bar right-side confirmation. A daily low must sweep the prior confirmed
low and close back above it on sufficient spot volume while both the asset and
BTC are above EMA200; execution is delayed to the next open. Positions exit on
a bearish sweep, trend loss, recovery to the prior confirmed high, or the fixed
10/20-day holding limit. Exposure remains capped at 40% total and 20% per asset
with at least 60% cash.

The development-selected `LS_F3_V15_H10` path was positive in development,
validation, confirmation and double-cost counterparts, returning 14.4% with a
1.175 period profit factor and 10.3% maximum drawdown. It nevertheless failed
the minimum 100-rebalance gate with 74 events and failed DSR (0.213), White
Reality Check (0.654), Hansen SPA (0.578), PBO (0.514) and both stochastic
gates. The family is therefore retained only as an orderless positive
diagnostic lead in the cumulative 21,337-strategy-trial history; all eight
observers start at 0/365 forward days and paper/live remain blocked.

`python main.py lab campaign run --name residual-reversal-v1 --yes` runs the
separately preregistered eight-DNA BTC-beta residual mean-reversion family.
The strictly prior 30/60-day rolling BTC beta is removed from each altcoin
return. Three- or five-day residual shocks are standardized against a strictly
prior 90-day distribution; only unusually negative ETH/SOL/LINK residuals are
bought while BTC remains above its causal EMA200. Signals execute at the next
daily open, BTC itself is never traded, and exposure remains capped at 40%
total and 20% per asset.

The development-selected `RR_B60_H5_Z20` path returned 45.8% net with 5.7%
CAGR, 1.09 Sharpe, 1.979 period profit factor and 4.3% maximum drawdown.
Development, validation, confirmation and every double-cost counterpart were
positive. PBO (0.029), Hansen SPA (0.029), White Reality Check (0.065) and
normal/stressed Monte Carlo plus Dirichlet gates passed. Promotion remains
blocked because DSR is 0.402, only 65 rebalance events exist versus the fixed
100-event minimum, and no globally untouched historical holdout remains.
All eight DNA rows are retained in the cumulative 21,345-strategy-trial
history with separate 0/365 orderless forward observers.

Before each daily autopilot cycle, all active forward-observer ledgers are
cryptographically audited. Observation checksums, the hash chain, immutable
identity fields and zero-order/live-permission invariants must all match before
data refresh or research is allowed to start. Corruption trips the existing
persistent kill switch and is reported in
`output/lab/reports/forward_ledger_preflight_v1.json`.

AI, neural-network and other machine-learning development is explicitly
embargoed. `python main.py lab ai status` reports the fail-closed policy.
Eligibility requires an economically and statistically passed immutable
strategy, completed forward/shadow/paper validation, at least 180 profitable
live calendar days after costs, 30 closed live trades, two live regimes,
drawdown within mandate, no unresolved incident, hashed evidence and separate
manual authorization. Until every check passes, only classical rule-based
research, deterministic backtests, lifecycle hardening and forward observation
are permitted.

`campaign observe` writes `FROZEN_FORWARD_RESEARCH` rankings and hypothetical
next-open weights with zero generated or submitted orders. Forward promotion
requires 365 new closed daily observations, 30 valid rebalances, and coverage
of rising/falling BTC trend, high/low volatility and broad/narrow breadth.
`campaign package` requires a clean committed worktree, reruns Ruff and the
non-network test suite, cross-checks candidate identities, archives the exact
Git source, and writes a SHA-256 manifest and zip checksum.

Single-market combinations also carry explicit exit DNA. `FIXED_R` uses a
hard ATR stop and finite R-target, `TRAILING_TREND` uses a distant safety target
plus a prior-bar ATR trail, and `TIME_REGIME` uses a hard stop with time or
causal regime/signal exits. Swing searches include wider stops and holding
windows up to 720 bars; the exit-model version is part of the experiment hash.

The ensemble run may write
`output/lab/candidates/rotation_research_lead_v1.json`. This is an immutable
forward-validation registration, not an approved trading candidate. It remains
blocked from paper and live modes until every statistical and multiple-testing
gate passes on genuinely new data.

`QUICK` runs static checks, canonical baselines, a bounded fast screen, and
exact survivor backtests. `STANDARD`, `DEEP`, and `EXHAUSTIVE` progressively
add canonical hyperparameter and robustness work; the deepest profiles create
a durable queue and do not claim the entire search is instantly complete.
Large generation estimates require `--yes`. CPU and memory bounds can be set
with `--workers`, `--cpu-limit`, `--memory-limit-mb`, `--trial-timeout`, and
`--combination-timeout`.

Runtime state, checkpoints, immutable universe manifests, failures,
leaderboards, reports, and charts are written below `output/lab/`. Restarting
with `--resume` recovers stale work and deterministic experiment hashes prevent
completed experiments from being duplicated.

Storm data epochs distinguish new DNA from new selection opportunities. An
exact retry of the same 5,000-DNA search space on the same data fingerprint is
reused and does not increment anything. Evaluating and selecting from that
space on a genuinely new closed-data fingerprint creates 5,000 new evaluation
trials, while its unique-DNA count remains unchanged. The reconciled indexes
use `STRATEGY_DNA_X_CLOSED_DATA_EPOCH_EVALUATIONS`.

All campaign evidence is reconciled by the fail-closed global trial audit:

```bash
python main.py lab trials audit
```

It writes a content-addressed snapshot, append-only index, and
`output/lab/reports/global_trial_accounting_v1.json`. Future autonomous storm
runs use this global evaluation denominator before adding their new trials.
Historical campaign reports remain immutable and retain their at-birth
denominators. Gaussian neighborhood smoothing is treated as a robustness
selector only; it cannot guarantee a PBO pass.

`MULTI_ALPHA_ENSEMBLE_V2` is one preregistered classical meta-strategy that
combines the frozen absolute-momentum and beta-residual-reversal sleeves. It
uses equal fixed sleeves, causal volatility targeting, an additional next-open
meta-execution lag, at most 40% total exposure and 20% per asset. The family is
historically discovery-contaminated, remains orderless, and requires 365 new
daily observations plus 30 forward rebalances; it cannot authorize paper or
live trading from its historical backtest.

`PEER_RESIDUAL_REVERSAL_V1` is a four-DNA classical relative-value
family. Each ETH/SOL/LINK return is adjusted by a strictly prior rolling beta
to the equal-weight return of the other two altcoins; only unusually negative
peer-relative shocks are eligible while BTC is above its causal EMA200. The
family is long-only, next-open, capped at 40% total and 20% per asset, and is
registered as forward research only. Its historical selection failed PBO,
White Reality Check, Hansen SPA and DSR, so no parameter repair or promotion is
permitted.

`BTC_SHOCK_DIFFUSION_V1` is a four-DNA classical cross-asset lead-lag
family. It estimates each ETH/SOL/LINK beta only from observations preceding a
large completed-close BTC shock, then holds at most two positive
beta-implied underreactors for three or five days. Signals require causal
EMA200 trend confirmation, execute at the next open, retain at least 60% cash,
and have no order authority. The development-selected path was historically
positive, but validation weakened, doubled-cost validation turned negative and
PBO remained far above the gate. It is therefore frozen forward research, not
a paper or live candidate.

`MACRO_LIQUIDITY_ROTATION_V1` is a two-DNA classical FRED-only campaign.
It accepts only `SOURCE_AVAILABLE_AT` observations for WALCL, M2SL, and NFCI;
derivatives, EODHD equity/VIX, stablecoin, on-chain, options, and GEX history
are rejected when the available files are snapshot-only or `FORWARD_ONLY`.
The fixed Sunday-close/Monday-open variants require two or all three liquidity
votes, allocate 10% to each allowed asset above its causal EMA200, cap total
exposure at 40%, and retain at least 60% zero-yield cash.

```bash
python main.py lab campaign plan --name macro-liquidity-v1
python main.py lab campaign run --name macro-liquidity-v1 --yes
python main.py lab campaign observe --name macro-liquidity-v1
```

The real campaign rejected both variants. The development-selected 3-of-3
path returned -4.70%, changed the portfolio only twice, and had PBO 0.743; the
2-of-3 path returned -33.57% with a -35.54% drawdown. White Reality Check and
Hansen SPA were both 1.0. It has no paper/live permission or order authority.

`python main.py lab campaign run --name multi-horizon-trend-v1 --yes`
executes one preregistered classical CTA-style DNA over BTC/ETH/SOL/LINK.
Positive 240-day momentum is mandatory and 20/60/120/240-day votes scale
daily next-open weights within 40% total exposure, 20% per asset, and 60%
minimum cash. Its immutable plan predates the result run.

The real campaign returned +161.17% net at 24.98% average exposure, but was
rejected: validation was only +3.82%, confirmation was -17.59%, double-cost
confirmation was -21.30%, and maximum drawdown was -34.51%. The confirmation
expectancy confidence lower bound and all stochastic robustness gates failed.
DSR, White Reality Check, and Hansen SPA were positive under the authoritative
32,101-trial denominator, but cannot override those failures or the absence of
an untouched holdout and forward sample. The DNA remains registered and
orderlessly observable; it has no paper/live permission.

`VOLUME_STRATEGY_CATALOG_V1` is a preregistered 1,275-trial classical
campaign over 51 real Bitvavo EUR asset/timeframe datasets. It covers 11
assets, 5m/15m/1h/4h/1d, and five distinct families: Donchian plus RVOL,
trend pullback dry-up/recovery, volume-contraction breakout, OBV plus CMF,
and VWAP plus MFI reclaim. Every market/timeframe/family has a fixed
five-point parameter neighborhood scored with the immutable Gaussian
`[0.05, 0.25, 0.40, 0.25, 0.05]` kernel. All 1,275 trials count globally;
only the 500 BTC/ETH/SOL/LINK trials can ever be considered for promotion.

```bash
python main.py lab campaign plan --name volume-strategy-catalog-v1
python main.py lab campaign run --name volume-strategy-catalog-v1 --yes
python main.py lab campaign report --name volume-strategy-catalog-v1
```

The real campaign rejected its allowed-universe diagnostic winner. The
LINK-EUR 4h VWAP/MFI-reclaim path earned +16.31% in development but lost
5.06% in validation, 3.38% in confirmation, and 6.09% in double-cost
confirmation. Its complete parameter plateau failed, DSR was 0, plateau
selection PBO was 0.767, and White Reality Check/Hansen SPA were
approximately 0.996/0.984. Timeframe-specific diagnostics likewise rejected
promotion: 4h was the closest cohort (White 0.064) but failed SPA (0.384),
PBO (0.292), and confirmation.

The existing shadow microstructure recorder stores real Bitvavo trades,
nonce-valid L2 snapshots, spread, depth, microprice, book imbalance and taker
imbalance. Its current history spans only hours, not enough for a causal
order-flow backtest. CVD, footprint, stacked imbalance, absorption, VPIN and
POC/VAH/VAL/HVN/LVN strategies therefore remain fail-closed until at least
90 days and 100,000 real observations exist; candle-direction volume is
explicitly labelled as a proxy and is never presented as trade delta.

AI, neural networks, reinforcement learning, and other learned-model work
remain explicitly embargoed until a classical strategy passes the historical
statistical gates, completes its frozen forward sample, and demonstrates
acceptable paper/live execution evidence.

### Prospective positioning evidence and disabled €5 canary

The eventual Bitvavo live sleeve is registered but disabled by default:
maximum €5 per order, maximum €5 total exposure, one open position, spot-only,
1x leverage, no shorts and no autoscaling. Registration is not order
permission. An order below the venue minimum is rejected rather than enlarged,
and unknown account exposure fails closed.

```bash
python main.py live canary-policy
python main.py live status
```

Historical evaluation trials, historical selection events, forward
observations and forward reselections are separate concepts. Frozen forward
observations do not increase the multiple-testing denominator. The audit
reports this distinction explicitly:

```bash
python main.py lab trials audit
```

The shadow collector now performs one hourly prospective context cycle for the
most recently closed UTC hour. It stores an immutable current CoinMarketCap
top-50 snapshot and MEXC funding, open interest, mark/index premium and basis
for BTC/ETH/SOL/LINK. Every record preserves event time, arrival time,
`available_at` and its raw payload hash. Missing liquidation data remains
explicitly unavailable. No historical top-50 membership or order flow is
backfilled from today's data.

Task Scheduler remains the preferred Windows service mechanism. Where local
policy blocks task creation, a least-privilege hidden per-user logon launcher
can be installed. The launcher starts a separate least-privilege supervisor:
it monitors the single-instance collector lock, archives only provably stale
locks and restarts an unexpectedly exited shadow collector after 30 seconds.
An explicit `operate stop --mode shadow` writes a durable disable marker, so
an intentional stop is never automatically undone.

```bash
python main.py operate startup-install --mode shadow --profile practical_spot_v1
python main.py operate startup-status --mode shadow --profile practical_spot_v1
python main.py operate supervisor-status
```

The first new positioning family is deliberately small rather than another
parameter storm. `CROWDING_AVOIDANCE_V1` contains four immutable long-blocker
DNA variants and cannot be evaluated until the prospective dataset reaches its
predeclared 90/180/365-day checkpoints.

```bash
python main.py microstructure plan
python main.py microstructure status
python main.py microstructure data-status
python main.py microstructure storage-status
python main.py microstructure observe
python main.py microstructure observer-audit
python main.py microstructure audit
python main.py microstructure readiness-report
python main.py microstructure gate-check --stage technical_feature_validation
```

The continuous shadow service additionally records Bitvavo public trade,
24-hour ticker, and order-book WebSocket events. The 24-hour channel is used
for spot base- and quote-volume facts; the best-bid/ask-only ticker is not
misrepresented as a volume source. Records are segmented by UTC hour and
joined by a cross-segment SHA-256 chain. Trades preserve exchange time,
collector arrival time, aggressor side, base and quote quantity, sequence, and
raw-payload hash. Closed-hour snapshots report delta/CVD inputs, order-book
imbalance, microprice, spread, funding, open interest, basis, and
perpetual/spot volume. The cross-venue ratio uses MEXC `amount24` in USDT
divided by Bitvavo `volumeQuote` after causally converting that EUR amount to
USDT with the same asset's contemporaneous MEXC index price and Bitvavo spot
price. The conversion rate, converted quote volume and method are retained in
every v3 snapshot. MEXC `volume24` is contract count and is explicitly never
compared with BTC/ETH/SOL/LINK spot-base units. A
mid-hour start, sequence gap, reconnect, dropped
message, or missing stream is recorded as `DATA_GAP`; nothing is interpolated.
Snapshots are finalized only after a five-minute late-arrival grace period.
When the matching positioning snapshot is still in flight, finalization is
deferred instead of permanently manufacturing a gap; after ten minutes the
hour is sealed with the explicit `POSITIONING_CONTEXT_TIMEOUT` reason. Raw
hourly segments remain writable for another full hour before lossless
sealing. Readiness is rebuilt from immutable snapshot hashes and
source-record hashes; a latest gap resets the consecutive counter to zero.
The default order-flow storage ceiling is 250 GB. At the observed normal
losslessly compressed segment size of roughly 21 MB per hour, this covers the
preregistered 365-day window with headroom; collection still fails closed
before the configured ceiling or free-disk reserve is breached.

Every sealed hour is also appended once to the orderless
`CROWDING_AVOIDANCE_V1` observer. Its four preregistered DNA variants are
evaluated without ranking or threshold changes only after funding, open
interest, perpetual/spot volume and real spot-CVD history are all available.
Earlier hours are retained as `DATA_GAP_NOT_EVALUATED` or `FEATURE_WARMUP`,
never as negative signals. Observation records form a source-linked SHA-256
chain, preserve the AI embargo and cannot generate targets or orders.
Backtesting remains forbidden until `microstructure gate-check` proves at
least 90 consecutive complete days. Preliminary research and formal regime
assessment remain separately blocked until 180 and 365 consecutive days.
None of these milestones can activate paper or live trading.

The classical regime router is governance-first and orderless. It classifies
only completed daily candles using BTC EMA200 trend and slope, 14-day
choppiness, 30-day volatility versus its 252-day baseline, and allowlisted
market breadth. Risk-on changes require three consecutive closed-candle
confirmations; risk-off or uncertain states move immediately to cash. A sleeve
can receive risk only after its research, forward, and requested lifecycle
gates have already passed. Allocation is capped at 40% total and 20% per
sleeve with at least 60% cash. With no approved sleeve—or in a blocked
regime—the router is deterministically 100% cash and cannot generate orders.
Its decisions form a verified append-only SHA-256 chain in
`output/lab/reports/regime_router_status_v1.json`.

Every generated baseline is paired with deterministic one-dimensional
sensitivity work for each tunable block parameter. Large grids use a documented
deterministic non-default sample in QUICK mode; joint parameter work is handled
by the typed optimizer. `WALK_FORWARD_FIXED` remains a diagnostic, while
`WALK_FORWARD_OPTIMIZED` performs a separate train-only optimization in every
fold before freezing parameters on its next validation window.

## Architecture

```text
config/       validated settings and user-maintained eligibility
core/         immutable contracts, domain errors, and CLI command orchestration
data/         unified providers, WebSockets, L2 books, database, derivatives context
scrapers/     RSS/web intelligence, relevance, timing, and source status
research/     features, strategies, backtest, optimization, and mathematics
risk/         portfolio gates, correlation analysis, and drawdown protection
execution/    paper/live Bitvavo controls, positions, PnL, and report-only dust
reporting/    reports, statistical charts, redaction, and hash manifests
tests/        deterministic unit and integration tests
legacy/       archived inherited code; never imported by active modules
main.py       thin executable composition root
```

The active system is intentionally small. `legacy/` exists only for provenance
and is excluded from linting and active imports.

## Verification

```powershell
python -m compileall -q config core data execution reporting research risk scrapers utils main.py
python -m ruff check config core data execution reporting research risk scrapers utils main.py tests
python -m pytest -q
python main.py doctor
python main.py self-test
python main.py test offline
python main.py test providers
python main.py test reporting
python main.py test full
```

Network access is not required for the default test suite. Network-dependent
tests carry the `network` marker and are excluded from plain `pytest -q`.
Complete test runs write explicit `PASSED`, `FAILED`, `SKIPPED`, `PARTIAL`, or
`BLOCKED` statuses under `output/test_runs/<run_id>/`; missing credentials are
reported as `SKIPPED_MISSING_CREDENTIALS`, never as passes.

## Known limitations

- Public endpoints, HTML layouts, fee schedules, and exchange rules can change.
- Kraken's OHLC endpoint exposes only a bounded recent window; it is a fallback,
  not a complete historical archive.
- CoinMarketCap enrichment needs its own key and is not an execution source.
- EODHD, CoinMarketCap, and FRED checks are skipped when their credentials are
  absent. SEC access requires a descriptive `SEC_USER_AGENT`.
- MEXC public WebSockets use provider-owned protobuf schemas that can change;
  schema changes fail visibly and require an adapter update.
- Public liquidation context is recorded as unavailable when MEXC does not
  expose a suitable public endpoint; it is not fabricated.
- Signed crypto GEX is a heuristic because public option open interest does not
  reveal dealer inventory direction.
- Masterlist entries marked `DATA_PROVIDER_REQUIRED`,
  `DATA_CURRENTLY_UNAVAILABLE`, `MANUAL_REVIEW_REQUIRED`, or `RESEARCH_ONLY`
  are coverage records, not fabricated features. Missing external observations
  remain NaN with availability metadata.
- Jurik Moving Average is registered as unavailable; no proprietary or
  unverifiable approximation is substituted.
- RSS and web coverage is incomplete and may be delayed, blocked, or
  forward-only.
- Paper fills cannot reproduce queue position, exchange outages, or all market
  impact.
- Backtest and Monte Carlo results depend on the supplied data and assumptions.
- Operational allowlisting does not replace independent Shariah review.
- Passing research is not evidence that future returns will be positive.
