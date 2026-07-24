# Trading Math Engine

```python
from trading_math_engine import (
    expectancy,
    breakeven_win_rate,
    calculate_position_size,
    drawdown_recovery,
    probability_at_least_one_losing_streak,
    bootstrap_expectancy,
    monte_carlo_binary_system,
    empirical_risk_of_ruin,
)
```

## Expectancy

```python
result = expectancy(
    win_rate=0.45,
    average_win_r=2.0,
    average_loss_r=1.0,
    cost_r=0.05,
)
print(result.net_expectancy_r)
```

## Position sizing

```python
size = calculate_position_size(
    account_equity=10_000,
    risk_fraction=0.01,
    entry_price=100,
    stop_price=95,
    fee_fraction_per_side=0.001,
    slippage_fraction_per_side=0.0005,
    max_position_fraction=0.25,
)
print(size.units, size.actual_risk)
```

## Empirical risk of ruin

```python
trade_r = [1.8, -1.0, -0.7, 2.4, -1.0, 1.2]

result = empirical_risk_of_ruin(
    trade_r,
    risk_fraction=0.005,
    trades_per_simulation=500,
    simulations=10_000,
    block_size=3,
    seed=42,
)
print(result.risk_of_ruin)
```

Use block_size above one when trades cluster by regime or repeated signals.
