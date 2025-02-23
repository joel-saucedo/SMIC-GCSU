import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from numba import njit
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from fredapi import Fred

# Create directory to save plots and CSV files
plot_dir = "./portfolio_plots"
os.makedirs(plot_dir, exist_ok=True)

# FMP API key
API_KEY = "apikey"

# Initialize Fred and get the risk-free rate from the 10-Year Treasury (DGS10)
fred = Fred(api_key="apikey")  # Replace with your actual FRED API key
today_str = datetime.today().strftime("%Y-%m-%d")
rf_series = fred.get_series("DGS10", observation_start="2020-01-01", observation_end=today_str)
rf = rf_series.iloc[-1] / 100.0  # Convert from percentage to decimal
print("Risk Free Rate (annual):", rf)

###############################################################################
# Monte Carlo Simulation and Markowitz Efficient Frontier
###############################################################################

# --- Data Acquisition ---
def fetch_time_series(ticker):
    """
    Fetch historical time series data for a given ticker from the FMP API.
    Returns a list of historical price records.
    """
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching historical data for {ticker}: {response.text}")
        return None
    data = response.json()
    return data.get("historical", [])

# Load data for VIS and other tickers
vis_data = fetch_time_series("VIS")
tickers = ["GE", "CAT", "DAC", "EMR", "POWL", "CTAS"]
time_series_data = {ticker: fetch_time_series(ticker) for ticker in tickers}

# --- Returns Preparation ---
def prepare_returns(data, ticker):
    """
    Convert historical price data into a DataFrame indexed by date.
    Compute daily returns as: r = (close_t - close_{t-1})/close_{t-1}.
    """
    df = pd.DataFrame(data)
    if df.empty:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df[f'{ticker}_return'] = df['close'].pct_change()
    return df[[f'{ticker}_return']].dropna()

vis_returns = prepare_returns(vis_data, "VIS")
returns_dfs = [vis_returns]
for ticker in tickers:
    r_df = prepare_returns(time_series_data[ticker], ticker)
    if r_df is not None:
        returns_dfs.append(r_df)

# Merge all returns by inner join on dates
returns_df = pd.concat(returns_dfs, axis=1, join='inner')
returns_df.dropna(inplace=True)

# Compute mean daily returns and covariance matrix
mu = returns_df.mean()
cov_matrix = returns_df.cov()
cov_matrix.to_csv(os.path.join(plot_dir, "covariance_matrix.csv"))

# --- Eigenvalue Decomposition ---
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
print("Eigenvalues of the covariance matrix:")
print(eigenvalues)
pd.DataFrame(eigenvalues, columns=["eigenvalue"]).to_csv(os.path.join(plot_dir, "eigenvalues.csv"), index=False)
eigvecs_df = pd.DataFrame(eigenvectors, index=cov_matrix.index, columns=[f"eig_{i}" for i in range(len(eigenvalues))])
eigvecs_df.to_csv(os.path.join(plot_dir, "eigenvectors.csv"))

# --- Portfolio Constraints ---
# We assume order: VIS, GE, CAT, DAC, EMR, POWL, CTAS.
# Constraints:
#   sum(w) = 1, w_i >= ε, w_i <= 0.40 for i ≥ 2, so that ∑_{i=2}^{n} w_i ≤ (1 - ε)
n_assets = returns_df.shape[1]
epsilon = 0.01

# --- Monte Carlo Simulation ---
@njit
def simulate_portfolios_numba(N, n_non_vis, epsilon):
    # Generate candidate portfolios for non-VIS assets uniformly in [ε, 0.40],
    # then set VIS weight = 1 - sum(non-VIS). Only keep portfolios with sum(non-VIS) ≤ (1-ε).
    n_assets = n_non_vis + 1
    portfolios = np.empty((N, n_assets))
    count = 0
    for k in range(N):
        non_vis = np.empty(n_non_vis)
        s = 0.0
        for i in range(n_non_vis):
            non_vis[i] = np.random.uniform(epsilon, 0.40)
            s += non_vis[i]
        if s <= (1 - epsilon):
            vis_weight = 1 - s
            portfolios[count, 0] = vis_weight
            for i in range(n_non_vis):
                portfolios[count, i+1] = non_vis[i]
            count += 1
    return portfolios[:count]

n_non_vis = n_assets - 1
N_sim = 100000  # Increase candidate count for better resolution if needed
simulated_weights = simulate_portfolios_numba(N_sim, n_non_vis, epsilon)
print(f"Monte Carlo Simulation: Generated {simulated_weights.shape[0]} portfolios.")

# --- Compute Portfolio Metrics ---
# Daily metrics:
sim_returns_daily = simulated_weights.dot(mu.values)
sim_variances_daily = np.einsum('ij,jk,ik->i', simulated_weights, cov_matrix.values, simulated_weights)
sim_risks_daily = np.sqrt(sim_variances_daily)
# Subtract daily rf approximated as rf/252 from return
sim_sharpes_daily = (sim_returns_daily - rf/252) / sim_risks_daily

# Annualize:
sim_returns_ann = (1 + sim_returns_daily)**252 - 1
sim_risks_ann = sim_risks_daily * np.sqrt(252)
sim_sharpes_ann = (sim_returns_ann - rf) / sim_risks_ann

# Select portfolios:
idx_min_var = np.argmin(sim_risks_ann)
idx_max_sharpe = np.argmax(sim_sharpes_ann)
idx_max_return = np.argmax(sim_returns_ann)

portfolio_min_var = simulated_weights[idx_min_var]
portfolio_max_sharpe = simulated_weights[idx_max_sharpe]
portfolio_max_return = simulated_weights[idx_max_return]

print("Selected Portfolio (Minimal Variance) Weights:")
print(portfolio_min_var)
print("Selected Portfolio (Maximal Sharpe) Weights:")
print(portfolio_max_sharpe)
print("Selected Portfolio (Maximal Returns) Weights:")
print(portfolio_max_return)

# --- Efficient Frontier via Convex Hull ---
# Build annualized (risk, return) points
points_ann = np.column_stack((sim_risks_ann, sim_returns_ann))
hull = ConvexHull(points_ann)
hull_points = points_ann[hull.vertices]
# Sort by risk
sorted_order = np.argsort(hull_points[:, 0])
hull_points_sorted = hull_points[sorted_order]
# Extract upper envelope (Pareto frontier)
efficient_points = []
current_max = -1e10
for pt in hull_points_sorted:
    if pt[1] > current_max:
        efficient_points.append(pt)
        current_max = pt[1]
efficient_points = np.array(efficient_points)
if efficient_points.size > 0:
    risk_interp = np.linspace(efficient_points[:,0].min(), efficient_points[:,0].max(), 100)
    returns_interp = np.interp(risk_interp, efficient_points[:,0], efficient_points[:,1])
else:
    risk_interp, returns_interp = np.array([]), np.array([])

# Annualized metrics for selected portfolios:
def annualize_portfolio(port_weights, mu, cov_matrix):
    daily_ret = port_weights.dot(mu.values)
    annual_ret = (1 + daily_ret)**252 - 1
    daily_var = port_weights.dot(cov_matrix.values).dot(port_weights)
    annual_risk = np.sqrt(daily_var) * np.sqrt(252)
    return annual_ret, annual_risk

min_var_ann_return, min_var_ann_risk = annualize_portfolio(portfolio_min_var, mu, cov_matrix)
max_sharpe_ann_return, max_sharpe_ann_risk = annualize_portfolio(portfolio_max_sharpe, mu, cov_matrix)
max_return_ann_return, max_return_ann_risk = annualize_portfolio(portfolio_max_return, mu, cov_matrix)

# --- Backtesting with Rolling-Window Analysis ---
last_date = returns_df.index.max()
start_date = last_date - DateOffset(years=5)
returns_last5 = returns_df.loc[returns_df.index >= start_date]

@njit
def rolling_performance_metrics(daily_returns, window_size, rf=0.0):
    n = daily_returns.shape[0]
    num_windows = n - window_size + 1
    cum_return_arr = np.empty(num_windows)
    ann_return_arr = np.empty(num_windows)
    ann_vol_arr = np.empty(num_windows)
    sharpe_arr = np.empty(num_windows)
    max_dd_arr = np.empty(num_windows)
    for i in range(num_windows):
        window = daily_returns[i:i+window_size]
        cum = np.prod(1 + window)
        cum_ret = cum - 1.0
        cum_return_arr[i] = cum_ret
        ann_ret = (1 + cum_ret)**(252 / window_size) - 1
        ann_return_arr[i] = ann_ret
        mean_val = np.sum(window) / window_size
        var = 0.0
        for j in range(window_size):
            var += (window[j] - mean_val)**2
        var /= (window_size - 1)
        std_val = np.sqrt(var)
        ann_vol = std_val * np.sqrt(252)
        ann_vol_arr[i] = ann_vol
        sharpe_arr[i] = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
        cum_series = np.empty(window_size)
        cum_series[0] = 1 + window[0]
        for j in range(1, window_size):
            cum_series[j] = cum_series[j-1] * (1 + window[j])
        running_max = cum_series[0]
        max_dd = 0.0
        for j in range(window_size):
            if cum_series[j] > running_max:
                running_max = cum_series[j]
            dd = (cum_series[j] - running_max) / running_max
            if dd < max_dd:
                max_dd = dd
        max_dd_arr[i] = max_dd
    return cum_return_arr, ann_return_arr, ann_vol_arr, sharpe_arr, max_dd_arr

window_size = 252  # ~1 year
daily_returns_array = returns_last5.dot(portfolio_min_var).values
cum_ret_arr, ann_ret_arr, ann_vol_arr, roll_sharpe_arr, max_dd_arr = rolling_performance_metrics(daily_returns_array, window_size, rf)

dates = returns_last5.index[window_size-1:]
metrics_df = pd.DataFrame({
    "Cumulative Return": cum_ret_arr,
    "Annualized Return": ann_ret_arr,
    "Annualized Volatility": ann_vol_arr,
    "Sharpe Ratio": roll_sharpe_arr,
    "Max Drawdown": max_dd_arr
}, index=dates)

# --- Visualization and Saving Plots ---
# Efficient Frontier Plot (Annualized)
plt.figure(figsize=(12, 8))
sc = plt.scatter(sim_risks_ann, sim_returns_ann, c=sim_sharpes_ann, cmap='viridis', alpha=0.4, label="Monte Carlo Portfolios")
plt.colorbar(sc, label="Annualized Sharpe Ratio")
plt.plot(risk_interp, returns_interp, 'r-', linewidth=2, label="Efficient Frontier")
plt.scatter(min_var_ann_risk, min_var_ann_return, color='blue', marker='*', s=200, 
            label=f"Min Var: Return {min_var_ann_return:.2%}, Risk {min_var_ann_risk:.2%}")
plt.scatter(max_sharpe_ann_risk, max_sharpe_ann_return, color='green', marker='*', s=200, 
            label=f"Max Sharpe: Return {max_sharpe_ann_return:.2%}, Risk {max_sharpe_ann_risk:.2%}")
plt.scatter(max_return_ann_risk, max_return_ann_return, color='purple', marker='*', s=200, 
            label=f"Max Return: Return {max_return_ann_return:.2%}, Risk {max_return_ann_risk:.2%}")
plt.xlabel("Annualized Risk (Std. Dev.)")
plt.ylabel("Annualized Return")
plt.title("Efficient Frontier (Annualized) via Monte Carlo Simulation")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "efficient_frontier.png"))
plt.close()

# Cumulative Returns Plot with Final Return in Legend
plt.figure(figsize=(12, 6))
def plot_cumulative(portfolio_weights, label):
    daily_ret = returns_last5.dot(portfolio_weights).values
    cum_ret = np.cumprod(1 + daily_ret)
    plt.plot(returns_last5.index, cum_ret, label=f"{label} (Final: {cum_ret[-1]:.2f})")

plot_cumulative(portfolio_min_var, "Minimal Variance")
plot_cumulative(portfolio_max_sharpe, "Maximal Sharpe")
plot_cumulative(portfolio_max_return, "Maximal Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Backtesting: Cumulative Returns Over Last 5 Years")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "cumulative_returns.png"))
plt.close()

# Rolling Performance Metrics Plot for Minimal Variance Portfolio (as example)
plt.figure(figsize=(12, 6))
for metric in metrics_df.columns:
    plt.plot(metrics_df.index, metrics_df[metric], label=metric)
plt.xlabel("Date")
plt.title("Rolling Window Performance Metrics (Minimal Variance Portfolio)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "rolling_performance_metrics.png"))
plt.close()
