"""
CCA (Comparable Company Analysis) Script
-------------------------------------------
This script performs a Comparable Company Analysis (CCA) by retrieving financial data for a target company and its peers from the Financial Modeling Prep (FMP) API.
It calculates key metrics and valuation multiples, aggregates peer summary statistics, and computes an implied valuation for the target company.
Finally, all results are written into a single CSV file with clearly defined sections.

Main Components:
------------------
1. Configuration & API Setup:
   - Sets the FMP API key and creates an output directory (CCA_DIR) to store the results.

2. FMP API Fetch Functions:
   - fetch_profile(ticker): Retrieves the company's profile data.
   - fetch_quote(ticker): Retrieves current market data (stock price, shares outstanding, etc.).
   - fetch_income_statement(ticker, limit): Retrieves the most recent income statement.
   - fetch_balance_sheet(ticker, limit): Retrieves the most recent balance sheet.
   - fetch_enterprise_value(ticker, limit): Retrieves the most recent enterprise value data.

3. Data Conversion:
   - convert_df(df, thousand_cols, ratio_cols): Converts numeric values by scaling columns (dividing by 1,000) and rounding numbers based on specified column types.

4. Metric Aggregation:
   - get_company_metrics(ticker): Aggregates key financial metrics (e.g., market cap, debt, revenue, net income) and calculates valuation multiples (EV/EBITDA, EV/Sales, P/E Ratio) for a given ticker.

5. Main CCA Logic:
   - run_cca():
       • Prompts the user to enter a target ticker and a list of comparable peer tickers.
       • Retrieves financial metrics for the target and peer companies.
       • Computes summary statistics (min, mean, median, max) for the peer valuation multiples.
       • Calculates the implied valuation for the target company using the median multiples from peers.
       • Determines a final recommendation (BUY, SELL, or HOLD) based on how the target's actual share price compares to the implied share price range.
       • Writes all results (company metrics, valuation multiples, peer summary, implied valuation, and final recommendation) into a single CSV file with sections.

Usage:
------
Run the script, input the target ticker and peer tickers when prompted, and review the output CSV file generated in the CCA directory.
"""



import os
import requests
import datetime
import numpy as np
import pandas as pd

# ----------------------------
# CONFIGURATION & API KEY
# ----------------------------
API_KEY = "FMP_API_KEY"  # FMP API key
CCA_DIR = "./CCA"
os.makedirs(CCA_DIR, exist_ok=True)

# ----------------------------
# FMP API FETCH FUNCTIONS
# ----------------------------
def fetch_profile(ticker):
    """Fetch company profile data from FMP."""
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Error fetching profile for {ticker}: {r.text}")
        return {}
    data = r.json()
    if not data:
        print(f"No profile data returned for {ticker}")
        return {}
    return data[0]

def fetch_quote(ticker):
    """Fetch quote data (including current price, shares outstanding) from FMP."""
    url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Error fetching quote for {ticker}: {r.text}")
        return {}
    data = r.json()
    if not data:
        print(f"No quote data returned for {ticker}")
        return {}
    return data[0]

def fetch_income_statement(ticker, limit=1):
    """Fetch the most recent income statement from FMP."""
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit={limit}&apikey={API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Error fetching income statement for {ticker}: {r.text}")
        return {}
    data = r.json()
    if not data:
        print(f"No income statement data for {ticker}")
        return {}
    return data[0]

def fetch_balance_sheet(ticker, limit=1):
    """Fetch the most recent balance sheet from FMP."""
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit={limit}&apikey={API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Error fetching balance sheet for {ticker}: {r.text}")
        return {}
    data = r.json()
    if not data:
        print(f"No balance sheet data for {ticker}")
        return {}
    return data[0]

def fetch_enterprise_value(ticker, limit=1):
    """Fetch the most recent enterprise value data from FMP."""
    url = f"https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}?limit={limit}&apikey={API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Error fetching enterprise value for {ticker}: {r.text}")
        return {}
    data = r.json()
    if not data:
        print(f"No enterprise value data for {ticker}")
        return {}
    return data[0]

# ----------------------------
# CONVERSION HELPER FUNCTION
# ----------------------------
def convert_df(df, thousand_cols, ratio_cols):
    """
    Returns a new DataFrame with numeric values converted:
      - For columns in thousand_cols: divide by 1,000 and round to 2 decimals.
      - For columns in ratio_cols: round to 2 decimals.
      - Other numeric columns are rounded to 2 decimals.
    """
    new_df = df.copy()
    for col in new_df.columns:
        try:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
        except:
            pass
        if col in thousand_cols:
            new_df[col] = new_df[col].apply(lambda x: round(x/1000, 2) if pd.notna(x) else x)
        elif col in ratio_cols:
            new_df[col] = new_df[col].apply(lambda x: round(x, 2) if pd.notna(x) else x)
        else:
            if pd.api.types.is_numeric_dtype(new_df[col]):
                new_df[col] = new_df[col].apply(lambda x: round(x, 2) if pd.notna(x) else x)
    return new_df

# ----------------------------
# METRIC AGGREGATION FOR CCA
# ----------------------------
def get_company_metrics(ticker):
    """
    Retrieves key metrics for the given ticker via FMP API.
    Returns a dict with:
      - Stock Price, Shares Outstanding, Market Cap (from profile/quote)
      - Debt, Minority Interest, Cash (from balance sheet)
      - Enterprise Value, EBITDA, Revenue, Net Income (from income statement & EV endpoint)
      - Calculated multiples: EV/EBITDA, EV/Sales, P/E Ratio
    """
    profile = fetch_profile(ticker)
    quote = fetch_quote(ticker)
    income = fetch_income_statement(ticker)
    balance = fetch_balance_sheet(ticker)
    ev_data = fetch_enterprise_value(ticker)
    
    try:
        stock_price = float(quote.get("price", np.nan))
    except:
        stock_price = np.nan
    try:
        shares_out = float(quote.get("sharesOutstanding", np.nan))
    except:
        shares_out = np.nan
    try:
        market_cap = float(profile.get("mktCap", np.nan))
    except:
        market_cap = np.nan

    try:
        debt = float(balance.get("totalDebt", np.nan))
    except:
        debt = np.nan
    try:
        minority = float(balance.get("minorityInterest", 0))
    except:
        minority = 0.0
    try:
        cash = float(balance.get("cashAndCashEquivalents", np.nan))
    except:
        cash = np.nan

    try:
        revenue = float(income.get("revenue", np.nan))
    except:
        revenue = np.nan
    try:
        net_income = float(income.get("netIncome", np.nan))
    except:
        net_income = np.nan
    try:
        ebitda = float(income.get("ebitda", np.nan))
    except:
        ebitda = np.nan

    try:
        enterprise_value = float(ev_data.get("enterpriseValue", np.nan))
    except:
        enterprise_value = np.nan

    ev_ebitda = enterprise_value / ebitda if ebitda and ebitda != 0 else np.nan
    ev_sales = enterprise_value / revenue if revenue and revenue != 0 else np.nan
    eps = net_income / shares_out if shares_out and shares_out != 0 else np.nan
    pe_ratio = stock_price / eps if eps and eps != 0 else np.nan

    return {
        "Ticker": ticker,
        "Stock Price": stock_price,
        "Shares Outstanding": shares_out,
        "Market Cap": market_cap,
        "Debt": debt,
        "Minority Interest": minority,
        "Cash": cash,
        "Enterprise Value": enterprise_value,
        "EBITDA": ebitda,
        "Revenue": revenue,
        "Net Income": net_income,
        "EV/EBITDA": ev_ebitda,
        "EV/Sales": ev_sales,
        "P/E Ratio": pe_ratio
    }

# ----------------------------
# MAIN CCA LOGIC
# ----------------------------
def run_cca():
    # Step 0: Input
    target_ticker = input("Enter target ticker (e.g. CTAS): ").strip().upper()
    peers_input = input("Enter comparable company tickers (space separated, e.g. UNF ARMK MGRC): ")
    peer_tickers = [t.strip().upper() for t in peers_input.split() if t.strip()]
    
    companies = [target_ticker] + peer_tickers
    metrics_list = []
    for ticker in companies:
        data = get_company_metrics(ticker)
        metrics_list.append(data)
    
    df_metrics = pd.DataFrame(metrics_list).set_index("Ticker")
    company_thousand_cols = ["Shares Outstanding", "Market Cap", "Debt", "Cash", "Enterprise Value", "EBITDA", "Revenue", "Net Income"]
    company_ratio_cols = ["EV/EBITDA", "EV/Sales", "P/E Ratio"]
    df_metrics_converted = convert_df(df_metrics, company_thousand_cols, company_ratio_cols)
    
    multiples_df = df_metrics[["EV/EBITDA", "EV/Sales", "P/E Ratio"]]
    multiples_df_converted = convert_df(multiples_df, [], company_ratio_cols)
    
    # Step 3: Compute summary statistics for peers (exclude the target)
    peers_df = multiples_df.loc[peer_tickers]
    stats = {
        "Min": [peers_df["EV/EBITDA"].min(), peers_df["EV/Sales"].min(), peers_df["P/E Ratio"].min()],
        "Mean": [peers_df["EV/EBITDA"].mean(), peers_df["EV/Sales"].mean(), peers_df["P/E Ratio"].mean()],
        "Median": [peers_df["EV/EBITDA"].median(), peers_df["EV/Sales"].median(), peers_df["P/E Ratio"].median()],
        "Max": [peers_df["EV/EBITDA"].max(), peers_df["EV/Sales"].max(), peers_df["P/E Ratio"].max()]
    }
    stats_df = pd.DataFrame(stats, index=["EV/EBITDA", "EV/Sales", "P/E Ratio"])
    stats_df = convert_df(stats_df, [], company_ratio_cols)
    
    # Transpose Peer Multiples Summary so multiples are columns:
    stats_df_transposed = stats_df.T
    
    # Step 4: Implied Valuation for Target
    target = df_metrics.loc[target_ticker]
    median_ev_ebitda = stats_df.loc["EV/EBITDA", "Median"]
    median_ev_sales = stats_df.loc["EV/Sales", "Median"]
    median_pe = stats_df.loc["P/E Ratio", "Median"]

    implied_EV_EBITDA = median_ev_ebitda * target["EBITDA"] if pd.notna(target["EBITDA"]) else np.nan
    implied_equity_EBITDA = implied_EV_EBITDA - target["Debt"] - target["Minority Interest"] + target["Cash"]
    implied_share_price_EBITDA = implied_equity_EBITDA / target["Shares Outstanding"] if target["Shares Outstanding"] else np.nan

    implied_EV_Sales = median_ev_sales * target["Revenue"] if pd.notna(target["Revenue"]) else np.nan
    implied_equity_Sales = implied_EV_Sales - target["Debt"] - target["Minority Interest"] + target["Cash"]
    implied_share_price_Sales = implied_equity_Sales / target["Shares Outstanding"] if target["Shares Outstanding"] else np.nan

    implied_equity_PE = median_pe * target["Net Income"] if pd.notna(target["Net Income"]) else np.nan
    implied_EV_PE = implied_equity_PE + target["Debt"] + target["Minority Interest"] - target["Cash"]
    implied_share_price_PE = implied_equity_PE / target["Shares Outstanding"] if target["Shares Outstanding"] else np.nan

    implied_data = {
        "Median Multiple": [median_ev_ebitda, median_ev_sales, median_pe],
        "Target Metric": [target["EBITDA"], target["Revenue"], target["Net Income"]],
        "Implied EV": [implied_EV_EBITDA, implied_EV_Sales, implied_EV_PE],
        "Debt": [target["Debt"]]*3,
        "Minority Interest": [target["Minority Interest"]]*3,
        "Cash": [target["Cash"]]*3,
        "Implied Equity": [implied_equity_EBITDA, implied_equity_Sales, implied_equity_PE],
        "Shares Outstanding": [target["Shares Outstanding"]]*3,
        "Implied Share Price": [implied_share_price_EBITDA, implied_share_price_Sales, implied_share_price_PE]
    }
    implied_df = pd.DataFrame(implied_data, index=["EV/EBITDA", "EV/Sales", "P/E"])
    # Convert for output: absolute metrics in thousands, ratios as is.
    implied_thousand_cols = ["Target Metric", "Implied EV", "Debt", "Cash", "Implied Equity", "Shares Outstanding"]
    implied_ratio_cols = ["Median Multiple", "Implied Share Price"]
    implied_df_converted = convert_df(implied_df, implied_thousand_cols, implied_ratio_cols)
    
    # Transpose Implied Valuation so that methods become columns:
    implied_df_transposed = implied_df_converted.T

    implied_range_lower = np.nanmin(implied_df["Implied Share Price"])
    implied_range_upper = np.nanmax(implied_df["Implied Share Price"])
    actual_share_price = target["Stock Price"]
    
    if actual_share_price < implied_range_lower:
        recommendation = "BUY"
    elif actual_share_price > implied_range_upper:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    
    summary_df = pd.DataFrame({
        "Actual Share Price": [round(actual_share_price, 2)],
        "Implied Range Lower": [round(implied_range_lower, 2)],
        "Implied Range Upper": [round(implied_range_upper, 2)],
        "Recommendation": [recommendation]
    })
    
    # ----------------------------
    # OUTPUT: Write all results into one CSV file with sections
    # ----------------------------
    output_lines = []
    output_lines.append("=== Company Metrics ===")
    output_lines.append(df_metrics_converted.to_csv())
    output_lines.append("\n=== Valuation Multiples ===")
    output_lines.append(multiples_df_converted.to_csv())
    output_lines.append("\n=== Peer Multiples Summary Statistics ===")
    output_lines.append(stats_df_transposed.to_csv())
    output_lines.append("\n=== Implied Valuation for Target ===")
    output_lines.append(implied_df_transposed.to_csv())
    output_lines.append("\n=== Final Valuation Summary ===")
    output_lines.append(summary_df.to_csv(index=False))
    
    output_file = os.path.join(CCA_DIR, f"{target_ticker}_CCA.csv")
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    
    print(f"CCA results saved to {output_file}")

if __name__ == "__main__":
    run_cca()
