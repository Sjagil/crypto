from __future__ import annotations

import pandas as pd
from arch import arch_model


def fit_garch(close: pd.Series):
    returns = close.pct_change().dropna() * 100

    model = arch_model(
        returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
    )

    result = model.fit(disp="off")
    return result


def main() -> None:
    df = pd.read_parquet("output/worldmonitor_59/canonical_long.parquet")
    msft = df[df["symbol"] == "MSFT"].copy()

    result = fit_garch(msft["close"])
    print(result.summary())

    cond_vol = result.conditional_volatility
    print("\nConditional volatility tail:")
    print(cond_vol.tail())


if __name__ == "__main__":
    main()
