import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred

#------------------------------------------------------------------------------ 
# Set up directories for storing data and saving the plot.
data_dir = "./industrials_hist"
plot_dir = "./alpha_analysis"
os.makedirs(plot_dir, exist_ok=True)

#------------------------------------------------------------------------------ 
# Input: specify the ticker of interest (e.g., "AAPL" or any other)
target_ticker = "CTAS"  # <-- Replace with your desired ticker symbol

#------------------------------------------------------------------------------ 
# Load the historical data for the target ticker
ticker_file = os.path.join(data_dir, f"{target_ticker}.csv")
if not os.path.exists(ticker_file):
    raise FileNotFoundError(f"Data for ticker {target_ticker} not found in {data_dir}.")
    
df_ticker = pd.read_csv(ticker_file, parse_dates=["date"], index_col="date")
df_ticker.sort_index(inplace=True)

# Filter the data to the period 2020-01-01 to today
df_ticker = df_ticker.loc["2020-01-01":]
df_ticker["return"] = df_ticker["close"].pct_change()

#------------------------------------------------------------------------------ 
# Load SPY data as the market proxy
spy_file = os.path.join(data_dir, "SPY.csv")
if not os.path.exists(spy_file):
    raise FileNotFoundError("Data for SPY not found in ./industrials_hist.")

df_spy = pd.read_csv(spy_file, parse_dates=["date"], index_col="date")
df_spy.sort_index(inplace=True)
df_spy = df_spy.loc["2020-01-01":]
df_spy["return"] = df_spy["close"].pct_change()

#------------------------------------------------------------------------------ 
# Get the risk-free rate from FRED using the 10-Year T-Bond yield.
fred = Fred(api_key="apikey")
rf_series = fred.get_series('DGS10', observation_start='2020-01-01', observation_end='2025-12-31')
rf_series = rf_series.dropna().sort_index()

# Convert annualized yield (in %) to a daily rate (assuming 252 trading days)
rf_daily = rf_series / 100 / 252

#------------------------------------------------------------------------------ 
# Combine asset and market returns on common dates
combined_index = df_ticker.index.intersection(df_spy.index)
combined = pd.DataFrame({
    "asset_return": df_ticker.loc[combined_index, "return"],
    "market_return": df_spy.loc[combined_index, "return"]
})

# Align the risk-free rate to these dates (using forward-fill)
combined["rf"] = rf_daily.reindex(combined.index, method="ffill")
combined.dropna(inplace=True)

# Compute excess returns: asset and market returns minus the risk-free rate
combined["asset_excess"] = combined["asset_return"] - combined["rf"]
combined["market_excess"] = combined["market_return"] - combined["rf"]

#------------------------------------------------------------------------------ 
# Define the rolling window size (63 trading days ~ one quarter)
window = 63

# Compute the rolling covariance and variance for excess returns
combined["rolling_cov"] = combined["asset_excess"].rolling(window).cov(combined["market_excess"])
combined["rolling_var"] = combined["market_excess"].rolling(window).var()

# Compute rolling betas
combined["rolling_beta"] = combined["rolling_cov"] / combined["rolling_var"]

# Compute rolling means of excess returns for alpha estimation
combined["rolling_asset_mean"] = combined["asset_excess"].rolling(window).mean()
combined["rolling_market_mean"] = combined["market_excess"].rolling(window).mean()

# Estimate rolling alpha
combined["rolling_alpha"] = combined["rolling_asset_mean"] - combined["rolling_beta"] * combined["rolling_market_mean"]

#------------------------------------------------------------------------------ 
# Calculate the standard error of the rolling alpha
combined["rolling_alpha_residuals"] = combined["asset_excess"] - combined["rolling_alpha"] - combined["rolling_beta"] * (combined["market_excess"] - combined["rolling_market_mean"])

# Standard deviation of residuals over rolling window
combined["rolling_alpha_se"] = combined["rolling_alpha_residuals"].rolling(window).std() / np.sqrt(window)

# Calculate the t-statistic for rolling alpha
combined["t_alpha"] = combined["rolling_alpha"] / combined["rolling_alpha_se"]

#------------------------------------------------------------------------------ 
# Plot the t-statistic for alpha
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(combined.index, combined["t_alpha"], color="blue", label="t-statistic")
ax.axhline(y=2, color="red", linestyle="--", label="(95% confidence)")
ax.axhline(y=-2, color="red", linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("t-statistic")
ax.set_title(f"Alpha Statistical Significance")
ax.legend(loc="upper left")
ax.grid(True)

# Save the plot in the specified folder
plot_file = os.path.join(plot_dir, f"t_statistic_rolling_alpha_{target_ticker}_2020_present.png")
plt.savefig(plot_file)
plt.close()

print(f"t-statistic plot for rolling alpha for {target_ticker} (2020-present) saved as: {plot_file}")
