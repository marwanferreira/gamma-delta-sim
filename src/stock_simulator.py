# src/stock_simulator.py

import numpy as np

def simulate_stock_path(S0, r, sigma, T, days, seed=None):
    """
    Simulates a daily stock price path using Geometric Brownian Motion.

    Parameters:
    - S0: Initial stock price
    - r: Risk-free interest rate (annualized)
    - sigma: Volatility (annualized)
    - T: Total time horizon (in years)
    - days: Number of time steps (trading days)
    - seed: Random seed for reproducibility (optional)

    Returns:
    - Array of simulated stock prices (length = days + 1)
    """

    if seed is not None:
        np.random.seed(seed)

    dt = T / days
    prices = [S0]

    for _ in range(days):
        dW = np.random.normal(0, np.sqrt(dt))
        S_next = prices[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
        prices.append(S_next)

    return np.array(prices)
