# src/hedging_engine.py

import numpy as np
from src.bs_greeks import bs_greeks

def run_delta_hedging(S_path, K, r, sigma, T, option_type='call'):
    """
    Runs a daily delta-hedging simulation over a given stock price path.

    Parameters:
    - S_path: Array of stock prices (length = days + 1)
    - K: Strike price
    - r: Risk-free rate
    - sigma: Volatility
    - T: Total time horizon (years)
    - option_type: 'call' or 'put'

    Returns:
    - Dictionary with time series of:
      stock_positions, option_values, deltas, gammas, vegas, daily_PnLs, cum_PnL
    """

    days = len(S_path) - 1
    dt = T / days
    stock_positions = []
    option_values = []
    deltas = []
    gammas = []
    vegas = []
    daily_PnLs = []

    cash = 0.0
    position = 0.0

    for day in range(days + 1):
        S = S_path[day]
        T_remaining = T - day * dt
        price, delta, gamma, vega = bs_greeks(S, K, r, T_remaining, sigma, option_type)

        option_values.append(price)
        deltas.append(delta)
        gammas.append(gamma)
        vegas.append(vega)

        if day == 0:
            # Initial hedge
            position = -delta
            cash = price + position * S  # Sell option + hedge
            stock_positions.append(position)
            daily_PnLs.append(0.0)
        else:
            # Re-hedge
            prev_position = position
            position = -delta
            d_position = position - prev_position
            cash *= np.exp(r * dt)  # Cash earns risk-free rate
            cash -= d_position * S  # Cost of adjusting hedge
            stock_positions.append(position)

            # PnL = Change in option value + stock PnL + cash interest
            pnl = (option_values[-1] - option_values[-2]) \
                  + prev_position * (S - S_path[day - 1]) \
                  + (cash - cash / np.exp(r * dt))  # Interest earned
            daily_PnLs.append(pnl)

    cum_PnL = np.cumsum(daily_PnLs)
    return {
        "stock_positions": np.array(stock_positions),
        "option_values": np.array(option_values),
        "deltas": np.array(deltas),
        "gammas": np.array(gammas),
        "vegas": np.array(vegas),
        "daily_PnLs": np.array(daily_PnLs),
        "cum_PnL": cum_PnL
    }
