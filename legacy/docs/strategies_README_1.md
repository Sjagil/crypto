
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

# Candles Engine

The standalone `candles_engine.py` classifies each closed OHLCV candle by geometry, context, volume, volatility, dominant type, bias, strength, confidence and multi-candle patterns.

```bash
python candles_engine.py BTC_EUR_4h.csv --output BTC_EUR_4h_candles.csv --summary-json candles_summary.json --last 20
```

Use `CandleEngine().analyze(data)` in Python. Pattern context uses prior candles only, and signals must be executed no earlier than the next candle open.
