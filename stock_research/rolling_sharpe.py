import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# Get the risk-free rate from FRED using the 10-Year T-Bond yield
fred = Fred(api_key="apikey")
rf_series = fred.get_series('DGS10', observation_start=start_date, observation_end=end_date)
rf_series = rf_series.dropna().sort_index()
# Convert annualized percentage yield to a daily rate (assuming 252 trading days)
rf_daily = rf_series / 100 / 252

# Align the risk-free rate with each dataset (forward fill)
rf_target = rf_daily.reindex(df_target.index, method='ffill')
rf_spy = rf_daily.reindex(df_spy.index, method='ffill')

# Compute daily excess returns (return minus risk-free rate)
df_target['excess'] = df_target['return'] - rf_target
df_spy['excess'] = df_spy['return'] - rf_spy

#------------------------------------------------------------------------------
# Define rolling window size (e.g., 63 trading days ~ one quarter)
window = 63

#------------------------------------------------------------------------------
# Define numba-accelerated function to compute moments (mean, std, skew, kurtosis)
@njit(parallel=True)
def compute_moments(arr):
    n = arr.shape[0]
    mean_val = 0.0
    for i in prange(n):
        mean_val += arr[i]
    mean_val /= n
    var_val = 0.0
    for i in prange(n):
        var_val += (arr[i] - mean_val)**2
    if n > 1:
        var_val /= (n - 1)
    else:
        var_val = 0.0
    std_val = np.sqrt(var_val)
    skew = 0.0
    kurt = 0.0
    for i in prange(n):
        skew += ((arr[i] - mean_val) / std_val)**3
        kurt += ((arr[i] - mean_val) / std_val)**4
    skew /= n
    kurt /= n
    return mean_val, std_val, skew, kurt

# Standard error for the Sharpe ratio estimator (adjusted for non-normality)
@njit
def compute_sharpe_se(sr, skew, kurt, T):
    # Using a variant with a (1/12) factor
    return np.sqrt((1 - sr * skew + (1/12) * sr**2 * (kurt - 1)) / T)

#------------------------------------------------------------------------------
# Define functions to compute the rolling Sharpe ratio and its standard error
def rolling_sharpe(x):
    # x: window of daily excess returns
    if np.std(x) == 0:
        return np.nan
    daily_sr = np.mean(x) / np.std(x)
    # Annualize the Sharpe ratio (multiply by sqrt(252))
    return daily_sr * np.sqrt(252)

def rolling_sharpe_error(x):
    T = len(x)
    if np.std(x) == 0:
        return np.nan
    daily_sr = np.mean(x) / np.std(x)
    mean_val, std_val, skew, kurt = compute_moments(x)
    se_daily = compute_sharpe_se(daily_sr, skew, kurt, T)
    # Annualize the error
    return se_daily * np.sqrt(252)

#------------------------------------------------------------------------------
# Compute rolling Sharpe ratios and error series using the defined functions
rolling_sharpe_target = df_target['excess'].rolling(window).apply(rolling_sharpe, raw=True)
rolling_error_target = df_target['excess'].rolling(window).apply(rolling_sharpe_error, raw=True)

rolling_sharpe_spy = df_spy['excess'].rolling(window).apply(rolling_sharpe, raw=True)
rolling_error_spy = df_spy['excess'].rolling(window).apply(rolling_sharpe_error, raw=True)

#------------------------------------------------------------------------------
# Drop NaN values from the rolling series for plotting
target_sharpe = rolling_sharpe_target.dropna()
target_error = rolling_error_target.loc[target_sharpe.index]
spy_sharpe = rolling_sharpe_spy.dropna()
spy_error = rolling_error_spy.loc[spy_sharpe.index]

#------------------------------------------------------------------------------
# Sample 5 evenly spaced error bar points from each series
def sample_five(series):
    if len(series) < 15:
        return series.index, series.values
    indices = np.linspace(0, len(series)-1, 15, dtype=int)
    return series.index[indices], series.values[indices]

target_err_dates, target_sr_samples = sample_five(target_sharpe)
target_err_values = target_error.loc[target_err_dates].values  # corresponding errors

spy_err_dates, spy_sr_samples = sample_five(spy_sharpe)
spy_err_values = spy_error.loc[spy_err_dates].values

#------------------------------------------------------------------------------
# Plot rolling Sharpe ratios for the target ticker and SPY on the same y-axis.
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the target ticker rolling Sharpe (blue line) with error bars
ax.plot(target_sharpe.index, target_sharpe, color='blue', label=f"{target_ticker} Rolling Sharpe")
ax.errorbar(target_err_dates, target_sharpe.loc[target_err_dates], yerr=target_err_values,
            fmt='o', color='blue', capsize=4, label=f"{target_ticker} Error")

# Plot the SPY rolling Sharpe (red dashed line) with error bars
ax.plot(spy_sharpe.index, spy_sharpe, color='red', linestyle='--', label="SPY Rolling Sharpe")
ax.errorbar(spy_err_dates, spy_sharpe.loc[spy_err_dates], yerr=spy_err_values,
            fmt='o', color='red', capsize=4, label="SPY Error")

# Set labels and title using a common y-axis
ax.set_xlabel("Date")
ax.set_ylabel("Rolling Sharpe Ratio")
plt.title(f"Rolling Sharpe Ratio ({target_ticker} vs SPY)")

ax.legend(loc='upper left')
plt.grid(True)

# Save the plot in the specified folder
plot_file = os.path.join(plot_dir, f"rolling_sharpe_{target_ticker}_vs_SPY_2020_today.png")
plt.savefig(plot_file)
plt.close()

print(f"Rolling Sharpe Ratio comparison plot saved to: {plot_file}")
