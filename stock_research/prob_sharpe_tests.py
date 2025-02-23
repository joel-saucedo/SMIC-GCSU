import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from numba import njit, prange
import requests

#------------------------------------------------------------------------------
# Setup directories for data and plots
data_dir = "./industrials_hist"
plot_dir = "./industrials_hist/80_prob_sharpe_plots"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

#------------------------------------------------------------------------------
# List of S&P 500 Industrial tickers and extra SPY ticker (using XLI as proxy)
tickers = [
    "GE", "UNP", "LHX", "CAT", "HON", "BA", "DE", "ITW", "ADP", "ETN", "GD", "LMT",
    "UPS", "WM", "PCAR", "CTRA", "TT", "AMCR", "ITW", "TDG", "RSG", "WM", "DAR", "AXP",
    "FIS", "JCI", "NSC", "CARR", "CPRT", "PCAR", "IR", "AXON", "RBC", "CHRW", "GWW",
    "IR", "GPC", "FBHS", "AME", "PWR", "DAL", "ROP", "CTAS", "LUV", "ROK", "UAL", "XYL",
    "EPX", "ETN", "DOV", "EMR", "VLT", "LEA", "HUBB", "SNA", "LUV", "JBHT", "LDOS", "CPB",
    "EFX", "MAS", "J", "PH", "ICX", "SWK", "TXT", "NDSN", "PKG", "CHD", "AOS", "AAL",
    "GHC", "HI"
]
# Extra ticker representing industrial SPY performance (for comparison) note: using SPY until figure out how to pull SPY data
SPY_ticker = "SPY"

#------------------------------------------------------------------------------
# Your FMP API key and function to pull and store historical data if not already saved
API_KEY = "apikey"  # replace with your own FMP API key

def fetch_and_save_stock_data(ticker):
    file_path = os.path.join(data_dir, f"{ticker}.csv")
    if os.path.exists(file_path):
        # Data already stored, just load it
        df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
        df.sort_index(inplace=True)
        return df
    else:
        # Pull from FMP API and save to CSV
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?serietype=line&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        if "historical" in data:
            df = pd.DataFrame(data["historical"])[["date", "close"]]
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            df.to_csv(file_path)
            return df
        else:
            return None

#------------------------------------------------------------------------------
# Get the historical risk free rate from FRED using the 10-Year T-bond series.
fred = Fred(api_key="apikey")
rf_series = fred.get_series('DGS10', observation_start='1990-01-01')
rf_series = rf_series.dropna().sort_index()
# Convert annualized percentage yield to a daily rate (assuming 252 trading days)
rf_daily = rf_series / 100 / 252

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

# Standard error for the Sharpe ratio estimator (adjusting for non-normality)
@njit
def compute_sharpe_se(overall_sr, skew, kurt, T):
    return np.sqrt((1 - overall_sr * skew + (1/12) * overall_sr**2 * (kurt - 1)) / T)

#------------------------------------------------------------------------------
# Function to process a ticker: load data, compute daily returns, align risk-free rate,
# resample to quarterly frequency, and compute quarterly Sharpe ratios.
def process_ticker(ticker):
    df = fetch_and_save_stock_data(ticker)
    if df is None or df.empty:
        return None
    # Compute daily returns
    df['return'] = df['close'].pct_change()
    df = df.dropna(subset=['return'])
    
    # Align risk free rate to dates (forward fill missing values)
    rf_aligned = rf_daily.reindex(df.index, method='ffill')
    
    # Resample daily returns to quarterly frequency:
    quarterly_return = df['return'].resample('Q').sum()
    trading_days = df['return'].resample('Q').count()
    daily_std = df['return'].resample('Q').std()
    quarterly_std = daily_std * np.sqrt(trading_days)
    
    # Quarterly risk free return (sum of daily risk-free rates)
    quarterly_rf = rf_aligned.resample('Q').sum()
    
    # Compute quarterly Sharpe ratios
    quarterly_sharpe = (quarterly_return - quarterly_rf) / quarterly_std
    quarterly_sharpe = quarterly_sharpe.dropna()
    
    # Need sufficient data points
    if len(quarterly_sharpe) < 5:
        return None
    
    # Overall Sharpe ratio and standard error estimation:
    overall_sr = quarterly_sharpe.mean()
    T = len(quarterly_sharpe)
    qs_np = quarterly_sharpe.values.astype(np.float64)
    mean_q, std_q, skew_q, kurt_q = compute_moments(qs_np)
    sigma_sr = compute_sharpe_se(overall_sr, skew_q, kurt_q, T)
    ranking_score = overall_sr / sigma_sr if sigma_sr != 0 else 0
    lower_bound = overall_sr - 1.80 * sigma_sr

    return {
        "df": df,
        "quarterly_sharpe": quarterly_sharpe,
        "overall_sr": overall_sr,
        "sigma_sr": sigma_sr,
        "ranking_score": ranking_score,
        "lower_bound": lower_bound,
        "daily_returns": df['return']
    }

#------------------------------------------------------------------------------
# Process each stock ticker and store results in a dictionary.
ticker_results = {}
for ticker in tickers:
    print(f"Processing {ticker} ...")
    result = process_ticker(ticker)
    if result is not None:
        ticker_results[ticker] = result

# Process the extra (SPY) ticker for comparison.
print(f"Processing SPY ticker {SPY_ticker} ...")
SPY_result = process_ticker(SPY_ticker)
if SPY_result is not None:
    ticker_results[SPY_ticker] = SPY_result

#------------------------------------------------------------------------------
# Build a summary DataFrame of metrics (exclude the SPY ticker for ranking top 5)
summary = {
    "ticker": [],
    "overall_sr": [],
    "sigma_sr": [],
    "ranking_score": [],
    "lower_bound": []
}
for t, metrics in ticker_results.items():
    if t == SPY_ticker:
        continue  # skip SPY for ranking top 5 stocks
    summary["ticker"].append(t)
    summary["overall_sr"].append(metrics["overall_sr"])
    summary["sigma_sr"].append(metrics["sigma_sr"])
    summary["ranking_score"].append(metrics["ranking_score"])
    summary["lower_bound"].append(metrics["lower_bound"])

results_df = pd.DataFrame(summary)

# Select top 5 stocks based on the ranking score
top5 = results_df.sort_values(by="ranking_score", ascending=False).head(5)
top5_tickers = top5["ticker"].tolist()
print("Top 5 stocks based on ranking score:", top5_tickers)

#------------------------------------------------------------------------------
# PLOTTING

# 1. Plot time series (price history) for top 5 stocks and the SPY ticker.
plt.figure(figsize=(12, 6))
for ticker in top5_tickers:
    df = ticker_results[ticker]["df"]
    plt.plot(df.index, df["close"], label=ticker)
# Add the SPY ticker
if SPY_ticker in ticker_results:
    df_SPY = ticker_results[SPY_ticker]["df"]
    plt.plot(df_SPY.index, df_SPY["close"], label=f"{SPY_ticker}", linestyle="--", linewidth=2)
plt.title("Top 5 Stocks vs SPY - Price Time Series")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "top5_SPY_time_series.png"))
plt.close()

# 2. Plot overall Sharpe ratios with 80% CI error bars for the top 5 stocks plus the SPY.
plt.figure(figsize=(10, 6))
tickers_to_plot = top5_tickers + [SPY_ticker]
bars = []
errors = []
for ticker in tickers_to_plot:
    overall_sr = ticker_results[ticker]["overall_sr"]
    sigma_sr = ticker_results[ticker]["sigma_sr"]
    bars.append(overall_sr)
    errors.append(1.96 * sigma_sr)
plt.errorbar(tickers_to_plot, bars, yerr=errors, fmt='o', capsize=5, markersize=8)
plt.title("Overall Sharpe Ratio with 80% CI (Top 5 Stocks and SPY)")
plt.xlabel("Ticker")
plt.ylabel("Overall Sharpe Ratio")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "top5_SPY_sharpe_confidence.png"))
plt.close()

# 3. Plot daily return distributions for each top ticker and the SPY (separate plots).
for ticker in top5_tickers + [SPY_ticker]:
    plt.figure(figsize=(8, 5))
    returns = ticker_results[ticker]["daily_returns"]
    plt.hist(returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title(f"{ticker} - Daily Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{ticker}_daily_return_distribution.png"))
    plt.close()

# 4. Plot quarterly Sharpe ratio distributions for each top ticker and the SPY.
for ticker in top5_tickers + [SPY_ticker]:
    plt.figure(figsize=(8, 5))
    qs = ticker_results[ticker]["quarterly_sharpe"]
    plt.hist(qs, bins=30, alpha=0.75, color='green', edgecolor='black')
    plt.title(f"{ticker} - Quarterly Sharpe Ratio Distribution")
    plt.xlabel("Quarterly Sharpe Ratio")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{ticker}_quarterly_sharpe_distribution.png"))
    plt.close()

# 5. Combined plot: Histogram of quarterly Sharpe ratios for top 5 stocks plus SPY.
plt.figure(figsize=(10, 6))
for ticker in top5_tickers + [SPY_ticker]:
    qs = ticker_results[ticker]["quarterly_sharpe"]
    plt.hist(qs, bins=30, alpha=0.5, label=ticker)
plt.title("Combined Quarterly Sharpe Ratio Distribution (Top 5 Stocks + SPY)")
plt.xlabel("Quarterly Sharpe Ratio")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "combined_quarterly_sharpe_distribution.png"))
plt.close()

print("All plots have been saved in:", plot_dir)
