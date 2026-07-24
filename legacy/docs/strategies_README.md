
# Simple Half-Step Backtest Engine

## What it does

- Long-only, one-position-at-a-time backtesting
- Signals are calculated on closed candles and executed on the next candle open
- Fees and slippage on both entry and exit
- Stop-loss and take-profit handling
- Conservative same-bar stop/target handling
- Exact 0.5 parameter steps using `Decimal`
- Integer-only parameters where half periods make no canonical sense
- Fractional EMA, Wilder smoothing, RSI, and ATR
- Optional explicitly interpolated fractional SMA
- Coordinate search by default, avoiding billions of Cartesian combinations
- Optional full grid search for deliberately small spaces
- Train/validation optimization
- CSV output with all tested candidates

## CSV format

The CSV must contain:

```text
timestamp,open,high,low,close,volume
```

The timestamp may also be called `datetime`, `date`, or `time`.

## Install

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python simple_backtest_engine.py BTC_EUR_4h.csv \
  --initial-cash 2000 \
  --fee-pct 0.25 \
  --slippage-pct 0.08 \
  --train-fraction 0.70 \
  --top-n 20
```

PowerShell:

```powershell
python .\simple_backtest_engine.py .\BTC_EUR_4h.csv `
  --initial-cash 2000 `
  --fee-pct 0.25 `
  --slippage-pct 0.08 `
  --train-fraction 0.70 `
  --top-n 20
```

## Important parameter rule

Do not make every parameter a half-step parameter.

Good candidates for half steps:

- EMA period, if using the generalized EMA definition
- RSI period, if using Wilder alpha `1 / period`
- ATR period, under the same generalized definition
- RSI thresholds
- Stop-loss percentage
- Take-profit percentage
- ATR multiples

Keep these integer unless you deliberately define interpolation:

- SMA windows
- rolling highs/lows
- Donchian windows
- candle counts
- minimum trade counts
- time stops measured in bars

## Add another strategy

Create a function with this signature:

```python
def my_strategy(data, parameters):
    return StrategySignals(
        entry=entry_boolean_series,
        exit=exit_boolean_series,
        stop_pct=0.025,
        target_pct=0.05,
    )
```

Then define a parameter grid using `ParameterSpec`.

## This is still a research engine

Before live use, add:

- walk-forward folds instead of one split
- multi-asset portfolio accounting
- corporate actions for stocks
- session calendars and opening auctions
- borrow/currency handling if relevant
- strategy-specific trailing exits
- parameter stability analysis
- bootstrap or Monte Carlo resampling
- double-cost stress testing


## Why coordinate search is the default

Seven broad parameter ranges at 0.5 increments can create billions of full-grid
combinations. Coordinate search sweeps one parameter at a time, retains the best
candidate, and repeats the process. This is far cheaper and is suitable for a
first research pass.

It can still find a local rather than global optimum. Use the winning region to
create a smaller full grid, then verify it on untouched data and walk-forward
folds.

# Market structure module

The package now includes:

```text
market_structure_patterns.py
```

It generates deterministic features for:

- candle geometry and candlestick patterns
- confirmed fractal highs and lows
- bullish and bearish BOS
- bullish and bearish CHoCH
- liquidity sweeps and reclaimed swing levels
- equal highs and equal lows
- displacement candles
- bullish and bearish fair value gaps
- active and mitigated FVG zones
- bullish and bearish order-block proxies
- order-block mitigation and invalidation
- premium, discount, and equilibrium zones

## Basic feature generation

```python
import pandas as pd

from market_structure_patterns import (
    MarketStructureConfig,
    build_market_structure_features,
    summarize_events,
)

data = pd.read_csv("BTC_EUR_4h.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
data = data.set_index("timestamp")

config = MarketStructureConfig(
    fractal_left=2,
    fractal_right=2,
    atr_period=14,
    break_basis="close",
    sweep_buffer_atr=0.05,
    displacement_atr_multiple=1.25,
    order_block_lookback=12,
)

features = build_market_structure_features(data, config)
print(summarize_events(features))
features.to_csv("BTC_EUR_4h_market_structure.csv")
```

## Use it with the backtest engine

```python
from market_structure_patterns import (
    default_market_structure_parameter_specs,
    default_market_structure_parameters,
    liquidity_sweep_reversal_strategy,
    market_structure_parameter_constraint,
)
from simple_backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    CostModel,
    optimize_coordinate_train_validate,
)

engine = BacktestEngine(
    BacktestConfig(
        initial_cash=2000,
        risk_fraction_per_trade=0.005,
        costs=CostModel(
            fee_pct_per_side=0.0025,
            slippage_pct_per_side=0.0008,
        ),
    )
)

best, train_results, validation_results = optimize_coordinate_train_validate(
    data=data,
    strategy=liquidity_sweep_reversal_strategy,
    specs=default_market_structure_parameter_specs(),
    initial_parameters=default_market_structure_parameters(),
    engine=engine,
    train_fraction=0.70,
    top_n=20,
    rounds=3,
    constraint=market_structure_parameter_constraint,
)
```

## Anti-look-ahead rule

A five-candle fractal with two candles on the right is only confirmed two
candles after the pivot. The tradable columns are therefore:

```text
confirmed_fractal_high
confirmed_fractal_low
confirmed_fractal_high_price
confirmed_fractal_low_price
```

The `raw_fractal_*` columns exist for chart inspection only. Using them as
same-candle signals introduces future information and invalidates the backtest.

## Order-block warning

OHLCV candles cannot prove where institutions placed orders. This module uses a
transparent proxy: the most recent opposite-colour candle before a
BOS-or-CHoCH candle that also satisfies the displacement filter. Test it as a
feature. Do not treat the label itself as evidence that institutional demand or
supply existed there.
