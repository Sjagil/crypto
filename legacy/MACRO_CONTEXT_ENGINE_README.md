# Macro Context Engine

`macro_context_engine.py` zet historische crypto- en macrodata om in één causale featuretabel voor backtests.

## Hoofdinterface

```python
from macro_context_engine import MacroContextEngine, MacroContextConfig

engine = MacroContextEngine(
    MacroContextConfig(
        funding_z_overheated=2.0,
        breadth_risk_on=0.55,
        high_impact_pre_hours=2.0,
    )
)

features = engine.build(
    candles,
    fear_greed=fear_greed_df,
    dominance=dominance_df,
    relative_prices=relative_prices_df,
    breadth_prices=breadth_prices_df,
    derivatives=derivatives_df,
    etf_flows=etf_flows_df,
    onchain=onchain_df,
    global_macro=global_macro_df,
    events=events_df,
)
```

Alle datasets zijn optioneel. Ontbrekende groepen worden niet stilzwijgend als neutraal behandeld.

## Losse functies importeren

```python
from macro_context_engine import (
    fear_greed_features,
    dominance_features,
    relative_strength_features,
    breadth_features,
    derivatives_features,
    etf_flow_features,
    onchain_features,
    global_macro_features,
    event_risk_features,
    classify_macro_regimes,
)
```

## Beschikbaarheidstijd en look-ahead

De index van elke bron moet het tijdstip zijn waarop de rij daadwerkelijk beschikbaar werd. Wanneer de bron een aparte kolom heeft:

```python
features = engine.build(
    candles,
    etf_flows=etf_flow_df,
    availability_columns={
        "etf_flows": "available_at",
    },
)
```

De engine gebruikt uitsluitend backward as-of joins:

```text
source_available_at <= candle_timestamp
```

## Invoerschema per groep

### Fear & Greed

```text
value
```

### Dominance en marktcapitalisatie

```text
btc_dominance
stablecoin_dominance
total_market_cap
total2_market_cap
total3_market_cap
stablecoin_market_cap
```

### Relative prices

```text
btc
eth
sol
link
...
```

Elke niet-BTC-kolom krijgt automatisch een verhouding en relative-strengthfeatures tegenover BTC.

### Breadth

Een brede matrix met één prijsreeks per asset:

```text
BTC, ETH, SOL, LINK, ...
```

Gebruik een point-in-time universum. De huidige top 100 terugvullen door het verleden veroorzaakt survivorship bias.

### Derivatives

```text
funding_rate
open_interest
futures_basis
long_liquidations
short_liquidations
price
```

Deze data mag als context voor spot-only strategieën worden gebruikt. De engine opent zelf geen futurespositie.

### ETF flows

```text
btc_etf_flow
eth_etf_flow
```

### On-chain

```text
exchange_netflow
exchange_reserves
stablecoin_exchange_inflow
mvrv
sopr
nupl
realized_price
active_addresses
transaction_volume
fees
miner_reserves
long_term_holder_supply
short_term_holder_supply
```

### Global macro

```text
dxy
nasdaq
sp500
vix
us2y
us10y
real_yield
high_yield_spread
m2
fed_balance_sheet
reverse_repo
```

### Events

```text
available_at
event_time
impact
event_type
unlock_pct_float
```

## Belangrijkste eindkolommen

```text
primary_crypto_regime
crypto_risk_score
crypto_risk_on
crypto_risk_off
btc_led_market
broad_altcoin_market
altcoin_capitulation
stablecoin_rotation
leverage_overheated
deleveraging_event
global_risk_on
global_risk_off
macro_event_risk
token_unlock_risk
research_exposure_multiplier
data_completeness
macro_context_usable
```

## Strategievoorbeeld

```python
entry = (
    technical_entry
    & features["crypto_risk_on"]
    & ~features["leverage_overheated"]
    & ~features["macro_event_risk"]
)

position_fraction = (
    base_position_fraction
    * features["research_exposure_multiplier"]
)
```

`research_exposure_multiplier` is een researchfeature, geen bewezen live sizingregel.
