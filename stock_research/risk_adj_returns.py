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
df_ticker = df_ticker.loc["2020-01-01":"2025-12-31"]
df_ticker["return"] = df_ticker["close"].pct_change()

#------------------------------------------------------------------------------ 
# Load SPY data as the market proxy
spy_file = os.path.join(data_dir, "SPY.csv")
if not os.path.exists(spy_file):
    raise FileNotFoundError("Data for SPY not found in ./industrials_hist.")

df_spy = pd.read_csv(spy_file, parse_dates=["date"], index_col="date")
df_spy.sort_index(inplace=True)
df_spy = df_spy.loc["2020-01-01":"2025-12-31"]
df_spy["return"] = df_spy["close"].pct_change()

#------------------------------------------------------------------------------ 
# Get the risk-free rate from FRED using the 10-Year T-Bond yield.
fred = Fred(api_key="apikey")
rf_series = fred.get_series('DGS10', observation_start='2020-01-01', observation_end='2025-12-31')
rf_series = rf_series.dropna().sort_index()
rf_daily = rf_series / 100 / 252  # Convert to daily risk-free rate assuming 252 trading days

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

# Compute rolling means (for Sharpe, Sortino, and Information ratio)
rolling_mean_asset = combined["asset_excess"].rolling(window).mean()
rolling_mean_market = combined["market_excess"].rolling(window).mean()

# Compute rolling standard deviation (for Sharpe ratio)
rolling_std_asset = combined["asset_excess"].rolling(window).std()

# Compute rolling downside deviation (for Sortino ratio)
rolling_downside_dev_asset = combined["asset_excess"].rolling(window).apply(lambda x: np.sqrt(np.mean(np.minimum(x, 0)**2)), raw=False)

# Compute rolling tracking error (for Information ratio)
rolling_tracking_error = (combined["asset_excess"] - combined["market_excess"]).rolling(window).std()

# Compute the rolling Sharpe ratio
rolling_sharpe_asset = rolling_mean_asset / rolling_std_asset
rolling_sharpe_market = rolling_mean_market / rolling_std_asset  # Market Sharpe ratio

# Compute the rolling Sortino ratio
rolling_sortino_asset = rolling_mean_asset / rolling_downside_dev_asset
rolling_sortino_market = rolling_mean_market / rolling_downside_dev_asset  # Market Sortino ratio

# Compute the rolling Information ratio
rolling_information_ratio = (rolling_mean_asset - rolling_mean_market) / rolling_tracking_error

#------------------------------------------------------------------------------ 
# Plot Rolling Sharpe Ratio
plt.figure(figsize=(12, 6))
plt.plot(combined.index, rolling_sharpe_asset, label=f"Rolling Sharpe Ratio ({target_ticker})", color="blue")
plt.plot(combined.index, rolling_sharpe_market, label="Rolling Sharpe Ratio (SPY)", color="orange")
plt.title(f"Rolling Sharpe Ratio: {target_ticker} vs. SPY (2020-2025)")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"rolling_sharpe_{target_ticker}_vs_SPY.png"))
plt.close()

#------------------------------------------------------------------------------ 
# Plot Rolling Sortino Ratio
plt.figure(figsize=(12, 6))
plt.plot(combined.index, rolling_sortino_asset, label=f"Rolling Sortino Ratio ({target_ticker})", color="blue")
plt.plot(combined.index, rolling_sortino_market, label="Rolling Sortino Ratio (SPY)", color="orange")
plt.title(f"Rolling Sortino Ratio: {target_ticker} vs. SPY (2020-2025)")
plt.xlabel("Date")
plt.ylabel("Sortino Ratio")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"rolling_sortino_{target_ticker}_vs_SPY.png"))
plt.close()

#------------------------------------------------------------------------------ 
# Plot Rolling Information Ratio
plt.figure(figsize=(12, 6))
plt.plot(combined.index, rolling_information_ratio, label=f"Rolling Information Ratio ({target_ticker})", color="blue")
plt.title(f"Rolling Information Ratio: {target_ticker} vs. SPY (2020-2025)")
plt.xlabel("Date")
plt.ylabel("Information Ratio")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"rolling_information_ratio_{target_ticker}.png"))
plt.close()

print(f"Rolling Sharpe, Sortino, and Information ratio plots for {target_ticker} (2020-2025) saved in: {plot_dir}")
