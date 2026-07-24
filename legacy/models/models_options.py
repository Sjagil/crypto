from __future__ import annotations

from math import log, sqrt, exp
from statistics import NormalDist

N = NormalDist().cdf


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    S = underlying price
    K = strike
    T = time to expiry in years
    r = risk-free rate as decimal
    sigma = implied/realized volatility as decimal
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, T and sigma must be positive")

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type.lower() == "call":
        return S * N(d1) - K * exp(-r * T) * N(d2)

    if option_type.lower() == "put":
        return K * exp(-r * T) * N(-d2) - S * N(-d1)

    raise ValueError("option_type must be 'call' or 'put'")


def main() -> None:
    price = black_scholes_price(
        S=100,
        K=100,
        T=30 / 365,
        r=0.05,
        sigma=0.20,
        option_type="call",
    )
    print("Black-Scholes call price:", price)


if __name__ == "__main__":
    main()
