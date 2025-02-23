import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from fredapi import Fred
from numba import njit, prange

#------------------------------------------------------------------------------
# Setup directories for stored data and for saving the plot
data_dir = "./industrials_hist"
plot_dir = "./alpha_analysis"
os.makedirs(plot_dir, exist_ok=True)

#------------------------------------------------------------------------------
# Input: specify the ticker of interest (e.g., "AAPL" or any other)
target_ticker = "CTAS"  # <-- Replace with your desired ticker symbol

#------------------------------------------------------------------------------
# Define the analysis period: from 2020-01-01 to today
start_date = "2020-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

#------------------------------------------------------------------------------
# Load historical data for the target ticker and for SPY from CSV;
# then restrict both to the period 2020-01-01 to today.
df_target = pd.read_csv(os.path.join(data_dir, f"{target_ticker}.csv"), parse_dates=["date"], index_col="date")
df_target.sort_index(inplace=True)
df_target = df_target.loc[start_date:end_date]
df_target['return'] = df_target['close'].pct_change()

df_spy = pd.read_csv(os.path.join(data_dir, "SPY.csv"), parse_dates=["date"], index_col="date")
df_spy.sort_index(inplace=True)
df_spy = df_spy.loc[start_date:end_date]
df_spy['return'] = df_spy['close'].pct_change()

#------------------------------------------------------------------------------
# Autocorrelation Calculation
lags = 50  # Number of lags to compute autocorrelation

# Compute autocorrelation for the target ticker and SPY
acf_target = acf(df_target['return'].dropna(), nlags=lags)
acf_spy = acf(df_spy['return'].dropna(), nlags=lags)

#------------------------------------------------------------------------------
# Plotting the Autocorrelation Function (ACF) using a stem plot for the target ticker
fig, ax = plt.subplots(figsize=(6, 6))
ax.stem(range(lags + 1), acf_target, basefmt=" ", linefmt="orange", markerfmt="o", label=f"{target_ticker} ACF")
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.set_title(f"Autocorrelation for {target_ticker}")
ax.grid(True)
ax.legend()

# Save the plot in the specified folder
plot_file_target = os.path.join(plot_dir, f"acf_{target_ticker}_2020_today.png")
plt.savefig(plot_file_target)
plt.close()

print(f"ACF plot for {target_ticker} saved to: {plot_file_target}")

#------------------------------------------------------------------------------
# Plotting the Autocorrelation Function (ACF) using a stem plot for SPY
fig, ax = plt.subplots(figsize=(6, 6))
ax.stem(range(lags + 1), acf_spy, basefmt=" ", linefmt="purple", markerfmt="o", label="SPY ACF")
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.set_title("Autocorrelation for SPY")
ax.grid(True)
ax.legend()

# Save the plot in the specified folder
plot_file_spy = os.path.join(plot_dir, f"acf_SPY_2020_today.png")
plt.savefig(plot_file_spy)
plt.close()

print(f"ACF plot for SPY saved to: {plot_file_spy}")
