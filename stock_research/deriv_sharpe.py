import os
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Setup directories for stored data and for saving the plot
data_dir = "./industrials_hist"
plot_dir = "./alpha_analysis"
os.makedirs(plot_dir, exist_ok=True)

#------------------------------------------------------------------------------
# Input: specify the ticker of interest (e.g., "YOUR_TICKER")
target_ticker = "CTAS"  # <-- Replace with your desired ticker symbol

#------------------------------------------------------------------------------
# Load historical data for the target ticker and SPY from CSV
df_target = pd.read_csv(os.path.join(data_dir, f"{target_ticker}.csv"), parse_dates=["date"], index_col="date")
df_spy = pd.read_csv(os.path.join(data_dir, "SPY.csv"), parse_dates=["date"], index_col="date")

# Sort the data by date
df_target.sort_index(inplace=True)
df_spy.sort_index(inplace=True)

# Restrict both datasets to their common date range
common_index = df_target.index.intersection(df_spy.index)
df_target = df_target.loc[common_index]
df_spy = df_spy.loc[common_index]

#------------------------------------------------------------------------------
# Compute daily returns
df_target['return'] = df_target['close'].pct_change()
df_spy['return'] = df_spy['close'].pct_change()

# Compute cumulative returns (starting at 0% cumulative return)
df_target['cum_return'] = (1 + df_target['return']).cumprod() - 1
df_spy['cum_return'] = (1 + df_spy['return']).cumprod() - 1

# Extract the final cumulative return values
final_return_target = df_target['cum_return'].iloc[-1]
final_return_spy = df_spy['cum_return'].iloc[-1]

#------------------------------------------------------------------------------
# Plot cumulative returns for the target ticker and SPY
plt.figure(figsize=(12, 6))
plt.plot(df_target.index, df_target['cum_return'], label=f"{target_ticker} (Final Return: {final_return_target:.2%})", color='blue')
plt.plot(df_spy.index, df_spy['cum_return'], label=f"SPY (Final Return: {final_return_spy:.2%})", linestyle="--", color='red')

plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title(f"Cumulative Return: {target_ticker} vs SPY")
plt.legend(loc="upper left")
plt.grid(True)

# Save the plot
plot_file = os.path.join(plot_dir, f"cumulative_returns_{target_ticker}_vs_SPY.png")
plt.savefig(plot_file)
plt.close()

print(f"Cumulative returns plot saved to: {plot_file}")



import os
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Setup directories for stored data and for saving the plot
data_dir = "./industrials_hist"
plot_dir = "./alpha_analysis"
os.makedirs(plot_dir, exist_ok=True)

#------------------------------------------------------------------------------
# Input: specify the ticker of interest (e.g., "YOUR_TICKER")
target_ticker = "CTAS"  # <-- Replace with your desired ticker symbol

#------------------------------------------------------------------------------
# Define analysis period: from 2020-01-01 to today
start_date = "2020-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

#------------------------------------------------------------------------------
# Load historical data for the target ticker and for SPY from CSV
df_target = pd.read_csv(os.path.join(data_dir, f"{target_ticker}.csv"), parse_dates=["date"], index_col="date")
df_spy = pd.read_csv(os.path.join(data_dir, "SPY.csv"), parse_dates=["date"], index_col="date")

# Sort the data by date
df_target.sort_index(inplace=True)
df_spy.sort_index(inplace=True)

# Restrict both datasets to the period from start_date to today
df_target = df_target.loc[start_date:end_date]
df_spy = df_spy.loc[start_date:end_date]

# Restrict to common dates if needed
common_index = df_target.index.intersection(df_spy.index)
df_target = df_target.loc[common_index]
df_spy = df_spy.loc[common_index]

#------------------------------------------------------------------------------
# Compute daily returns
df_target['return'] = df_target['close'].pct_change()
df_spy['return'] = df_spy['close'].pct_change()

#------------------------------------------------------------------------------
# Compute cumulative returns:
# Cumulative return = (product of (1 + daily return)) - 1
df_target['cum_return'] = (1 + df_target['return']).cumprod() - 1
df_spy['cum_return'] = (1 + df_spy['return']).cumprod() - 1

# Extract the final cumulative return values
final_return_target = df_target['cum_return'].iloc[-1]
final_return_spy = df_spy['cum_return'].iloc[-1]

#------------------------------------------------------------------------------
# Plot cumulative returns for the target ticker and SPY
plt.figure(figsize=(12, 6))
plt.plot(df_target.index, df_target['cum_return'], label=f"{target_ticker} (Final Return: {final_return_target:.2%})", color='blue')
plt.plot(df_spy.index, df_spy['cum_return'], label=f"SPY (Final Return: {final_return_spy:.2%})", linestyle="--", color='red')

plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title(f"Cumulative Return (2020 to Today): {target_ticker} vs SPY")
plt.legend(loc="upper left")
plt.grid(True)

# Save the plot
plot_file = os.path.join(plot_dir, f"cumulative_returns_{target_ticker}_vs_SPY_2020_today.png")
plt.savefig(plot_file, dpi=300)
plt.close()

print(f"Cumulative returns plot saved to: {plot_file}")
