# src/bs_greeks.py

import numpy as np
from scipy.stats import norm

def bs_greeks(S, K, r, T, sigma, option_type='call'):
    """
    Computes Black-Scholes price and Greeks (delta, gamma, vega) for European options.

    Parameters:
    - S: Spot price
    - K: Strike price
    - r: Risk-free rate
    - T: Time to maturity (in years)
    - sigma: Volatility (annualized)
    - option_type: 'call' or 'put'

    Returns:
    - price, delta, gamma, vega
    """

    if T <= 0:
        # At expiry: option is worth intrinsic value, delta is binary
        if option_type == 'call':
            price = max(S - K, 0)
            delta = 1.0 if S > K else 0.0
        elif option_type == 'put':
            price = max(K - S, 0)
            delta = -1.0 if S < K else 0.0
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        gamma = 0.0
        vega = 0.0
        return price, delta, gamma, vega

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return price, delta, gamma, vega
