from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def fit_markov_regression(close: pd.Series):
    returns = close.pct_change().dropna() * 100

    model = sm.tsa.MarkovRegression(
        returns,
        k_regimes=2,
        trend="c",
        switching_variance=True,
    )

    result = model.fit(disp=False)
    return result


def main() -> None:
    df = pd.read_parquet("output/worldmonitor_59/canonical_long.parquet")
    msft = df[df["symbol"] == "MSFT"].copy()

    result = fit_markov_regression(msft["close"])

    print(result.summary())
    print("\nSmoothed probabilities tail:")
    print(result.smoothed_marginal_probabilities.tail())


if __name__ == "__main__":
    main()
