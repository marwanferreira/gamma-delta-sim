# main.py

import matplotlib.pyplot as plt
from src.stock_simulator import simulate_stock_path
from src.hedging_engine import run_delta_hedging
import os

# Simulation parameters
S0 = 100        # Initial stock price
K = 100         # Strike price (ATM)
r = 0.02        # Risk-free rate
sigma = 0.20    # Volatility
T = 1.0         # Time to maturity (1 year)
days = 21       # Trading days in simulation (1 month)
option_type = 'call'

# Simulate stock price path
S_path = simulate_stock_path(S0, r, sigma, T, days, seed=42)

# Run delta hedging simulation
results = run_delta_hedging(S_path, K, r, sigma, T, option_type)

# Create plots directory
os.makedirs("plots", exist_ok=True)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(S_path, label='Stock Price')
plt.title('Simulated Stock Path')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(results['cum_PnL'], label='Cumulative PnL')
plt.title('Cumulative Hedging PnL')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(results['deltas'], label='Delta')
plt.title('Delta Over Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(results['gammas'], label='Gamma')
plt.title('Gamma Over Time')
plt.grid(True)
plt.legend()

plt.tight_layout()

# Save before showing
plt.savefig("plots/hedging_results.png", dpi=300)
plt.show()
