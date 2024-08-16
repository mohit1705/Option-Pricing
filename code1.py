import numpy as np

# Function to calculate the option price using Monte Carlo simulation
def monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations, num_steps):
    """
    S0: Initial stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate (annualized)
    sigma: Volatility of the underlying asset (annualized)
    num_simulations: Number of Monte Carlo simulations
    num_steps: Number of time steps within each simulation
    """

    # Time increment
    dt = T / num_steps
    
    # Simulating the asset price paths
    asset_price_paths = np.zeros((num_simulations, num_steps + 1))
    asset_price_paths[:, 0] = S0
    
    for t in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_simulations)  # Generating random numbers for normal distribution
        asset_price_paths[:, t] = asset_price_paths[:, t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Calculating the payoff for a European call option
    payoff = np.maximum(asset_price_paths[:, -1] - K, 0)
    
    # Discounting the payoff back to present value
    option_price = np.exp(-r * T) * np.mean(payoff)
    
    return option_price

# Parameters specific to the Indian market (example values)
S0 = 18000  # Initial stock price (e.g., Nifty 50 index)
K = 18500   # Strike price
T = 0.25    # Time to maturity (3 months)
r = 0.06    # Risk-free rate (approx. 6% annualized)
sigma = 0.20  # Volatility (approx. 20% annualized)
num_simulations = 10000  # Number of simulations
num_steps = 252  # Number of time steps (daily steps for 1 year)

# Calculating the option price
option_price = monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations, num_steps)

print(f"The estimated European call option price is: INR {option_price:.2f}")
