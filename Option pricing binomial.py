# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 01:15:13 2025

@author: 2103020
"""

#Step 01, get the input
S0 = float(input("Please enter the initial Stock price: "))
K = float(input("Please enter the exercise/strike price: "))
T = int(input("Please enter the time to maturity in years of your option: "))
N = int(input("Please enter number of steps (N): "))
r = float(input("Please nter risk-free rate as a decimal number: "))
sigma = float(input("Please enter your asset volatility as a decimal number: "))
option_type = input("Please enter your option type (call/put): ").strip().lower()
option_style = input("Please enter your option style (European/American): ").strip().lower()

import numpy as np

#compute the delta t
delta_t = T/N
#up and down factors
u = np.exp(sigma*np.sqrt(delta_t))
d = 1/u

#Risk neutral probabilities
Pu = (np.exp(r*delta_t)-d)/(u-d)
Pd = 1-Pu

#Now, let's build the binomial trees

stock_tree = [[0 for j in range (i+1)] for i in range(N+1)]

for i in range(N+1):
    for j in range(i+1):
        stock_tree[i][j] = S0*(u**j)*(d**(i-j))
        
print("\nStock Price Tree:")
for row in stock_tree:
    print(row)
    
#We know that at the last step, i = N, so it'll help to find the value of the option at expiry

option_tree = [[0 for j in range (i+1)] for i in range(N+1)]

for j in range(N+1):
        if option_type == "call":
            option_tree[N][j] = max(stock_tree[N][j]-K,0)
        elif option_type == "put":
            option_tree[N][j] = max(K-stock_tree[N][j],0)

#print the different payoffs
print("Option values at expiration:", option_tree[N])

#Now, let's go backward
for i in range(N-1, -1, -1):
    for j in range(i+1):
        expected_value = np.exp(-r * delta_t) * (Pu * option_tree[i+1][j+1] + Pd * option_tree[i+1][j])

        # If American, we have to check for early exercise
        if option_style == "american":
            if option_type == "call":
                exercise_value = max(stock_tree[i][j] - K, 0)
            elif option_type == "put":
                exercise_value = max(K - stock_tree[i][j], 0)
            option_tree[i][j] = max(expected_value, exercise_value)  # Take the max
        else:
            option_tree[i][j] = expected_value  # European options only follows risk-neutral valuation

#Let's print the Option price at time T = 0

print(f"The option price is: {option_tree[0][0]:.4f}")



import numpy as np

def binomial_option_pricing(S0, K, T, N, r, sigma, option_type, option_style):
    """
    Compute the price of a European or American option using a binomial tree.
    
    Parameters:
    S0 : float, Initial stock price
    K  : float, Strike price
    T  : float, Time to maturity (years)
    N  : int, Number of steps in the binomial tree
    r  : float, Risk-free interest rate (decimal)
    sigma : float, Volatility of the underlying asset (decimal)
    option_type : str, "call" or "put"
    option_style : str, european" or "american"
    
    Returns:
    float, The option price at time 0
    """
    
    # Step 1: Compute binomial model parameters
    delta_t = T / N
    u = np.exp(sigma * np.sqrt(delta_t))  # Up factor
    d = 1 / u  # Down factor
    Pu = (np.exp(r * delta_t) - d) / (u - d)  # Probability up
    Pd = 1 - Pu  # Probability down

    # Step 2: Build stock price tree
    stock_tree = [[0 for _ in range(i+1)] for i in range(N+1)]
    for i in range(N+1):
        for j in range(i+1):
            stock_tree[i][j] = S0 * (u ** j) * (d ** (i - j))

    # Step 3: Compute option payoff at expiration
    option_tree = [[0 for _ in range(i+1)] for i in range(N+1)]
    for j in range(N+1):
        if option_type == "call":
            option_tree[N][j] = max(stock_tree[N][j] - K, 0)
        elif option_type == "put":
            option_tree[N][j] = max(K - stock_tree[N][j], 0)

    # Step 4: Work backwards to compute option price
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            expected_value = np.exp(-r * delta_t) * (Pu * option_tree[i+1][j+1] + Pd * option_tree[i+1][j])

            # If American, check for early exercise
            if option_style == "american":
                if option_type == "call":
                    exercise_value = max(stock_tree[i][j] - K, 0)
                elif option_type == "put":
                    exercise_value = max(K - stock_tree[i][j], 0)
                option_tree[i][j] = max(expected_value, exercise_value)  # Max of hold vs exercise
            else:
                option_tree[i][j] = expected_value  # European follows only risk-neutral valuation

    # Return final option price
    return option_tree[0][0]

# Example Usage
option_price = binomial_option_pricing(
    S0=50, K=50, T=10, N=1, r=0.02, sigma=0.2, option_type="call", option_style="european"
)

print(f"\nThe option price is: {option_price:.4f}")

import matplotlib.pyplot as plt




import numpy as np
import matplotlib.pyplot as plt

# Binomial Option Pricing Function
def binomial_option_pricing(S0, K, T, N, r, sigma, option_type, option_style):
    delta_t = T / N
    u = np.exp(sigma * np.sqrt(delta_t))
    d = 1 / u
    Pu = (np.exp(r * delta_t) - d) / (u - d)
    Pd = 1 - Pu

    # Stock Price Tree
    stock_tree = [[0 for _ in range(i+1)] for i in range(N+1)]
    for i in range(N+1):
        for j in range(i+1):
            stock_tree[i][j] = S0 * (u ** j) * (d ** (i - j))

    # Option Price Tree
    option_tree = [[0 for _ in range(i+1)] for i in range(N+1)]
    for j in range(N+1):
        if option_type == "call":
            option_tree[N][j] = max(stock_tree[N][j] - K, 0)
        elif option_type == "put":
            option_tree[N][j] = max(K - stock_tree[N][j], 0)

    # Backward Induction
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            expected_value = np.exp(-r * delta_t) * (Pu * option_tree[i+1][j+1] + Pd * option_tree[i+1][j])
            if option_style == "american":
                if option_type == "call":
                    exercise_value = max(stock_tree[i][j] - K, 0)
                elif option_type == "put":
                    exercise_value = max(K - stock_tree[i][j], 0)
                option_tree[i][j] = max(expected_value, exercise_value)
            else:
                option_tree[i][j] = expected_value

    return option_tree[0][0]

# Compute Greeks Function
def compute_greeks(S0, K, T, N, r, sigma, option_type, option_style, h=0.01):
    C = binomial_option_pricing(S0, K, T, N, r, sigma, option_type, option_style)

    # Delta
    C_plus = binomial_option_pricing(S0 + h, K, T, N, r, sigma, option_type, option_style)
    C_minus = binomial_option_pricing(S0 - h, K, T, N, r, sigma, option_type, option_style)
    delta = (C_plus - C_minus) / (2 * h)

    # Gamma
    C_plus_plus = binomial_option_pricing(S0 + 2*h, K, T, N, r, sigma, option_type, option_style)
    C_minus_minus = binomial_option_pricing(S0 - 2*h, K, T, N, r, sigma, option_type, option_style)
    gamma = (C_plus - 2*C + C_minus) / (h ** 2)

    # Theta
    C_T_minus = binomial_option_pricing(S0, K, T - h, N, r, sigma, option_type, option_style)
    theta = (C_T_minus - C) / h

    # Rho
    C_r_plus = binomial_option_pricing(S0, K, T, N, r + h, sigma, option_type, option_style)
    C_r_minus = binomial_option_pricing(S0, K, T, N, r - h, sigma, option_type, option_style)
    rho = (C_r_plus - C_r_minus) / (2 * h)

    return delta, gamma, theta, rho

# Parameters
S0 = 50  
K = 50   
T = 10   
N_values = [1, 10, 50, 100, 500, 1000]
r = 0.02  
sigma = 0.2  
option_types = ["call", "put"]
option_styles = ["european", "american"]

# Store Results
results = {opt_type: {"european": {}, "american": {}} for opt_type in option_types}

for opt_type in option_types:
    for opt_style in option_styles:
        prices, deltas, gammas, thetas, rhos = [], [], [], [], []
        for N in N_values:
            price = binomial_option_pricing(S0, K, T, N, r, sigma, opt_type, opt_style)
            delta, gamma, theta, rho = compute_greeks(S0, K, T, N, r, sigma, opt_type, opt_style)

            prices.append(price)
            deltas.append(delta)
            gammas.append(gamma)
            thetas.append(theta)
            rhos.append(rho)
        
        results[opt_type][opt_style]["prices"] = prices
        results[opt_type][opt_style]["deltas"] = deltas
        results[opt_type][opt_style]["gammas"] = gammas
        results[opt_type][opt_style]["thetas"] = thetas
        results[opt_type][opt_style]["rhos"] = rhos


print(results["call"]["european"])  # Check if data exists
print(results["put"]["european"])   # Check put values


# Slight horizontal offset to separate overlapping curves
offset = 2  

def plot_metric(metric, ylabel, title):
    plt.figure(figsize=(7,5))
    
    # Plot CALL Options (European vs. American)
    plt.plot([n - offset for n in N_values], results["call"]["european"][metric], marker="d", linestyle="dashed", linewidth=2.5, label="European Call", color="navy")
    plt.plot([n + offset for n in N_values], results["call"]["american"][metric], marker="o", linestyle="solid", linewidth=1.5, label="American Call", color="cyan")
    
    # Plot PUT Options (European vs. American)
    plt.plot([n - offset for n in N_values], results["put"]["european"][metric], marker="s", linestyle="dashed", linewidth=2.5, label="European Put", color="darkred")
    plt.plot([n + offset for n in N_values], results["put"]["american"][metric], marker="p", linestyle="solid", linewidth=1.5, label="American Put", color="orange")
    
    plt.xlabel("Number of Steps (N)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Generate Plots
plot_metric("prices", "Option Price", "Option Price vs. N (European vs. American)")
plot_metric("deltas", "Delta", "Delta vs. N (European vs. American)")
plot_metric("gammas", "Gamma", "Gamma vs. N (European vs. American)")
plot_metric("thetas", "Theta", "Theta vs. N (European vs. American)")
plot_metric("rhos", "Rho", "Rho vs. N (European vs. American)")
