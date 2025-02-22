"""
DCF Valuation and Sensitivity Analysis Script
----------------------------------------------
This script performs a discounted cash flow (DCF) valuation for a given ticker symbol by combining historical
financial data with projected metrics. It also conducts a sensitivity analysis on the valuation by varying
the terminal growth rate and the weighted average cost of capital (WACC).

Key Components:
---------------
1. Data Retrieval:
   - fetch_profile(ticker): Retrieves company profile data from the FinancialModelingPrep (FMP) API.
   - fetch_quote(ticker): Retrieves current quote data (e.g., share price, shares outstanding) from the FMP API.

2. Financial Data Parsing:
   - parse_financials_file(filepath): Reads a CSV file containing three sections (Income Statement, Balance Sheet,
     Cash Flow Statement), splits these sections, and returns each as a pandas DataFrame.
   - preprocess_df(df): A placeholder function for DataFrame preprocessing.
   - extract_historical(income_df, balance_df, cashflow_df): Extracts key financial metrics (such as revenue, net income,
     and working capital) from the historical data.

3. Historical Metrics and Projections:
   - compute_historical_metrics(historical): Computes historical averages, growth rates, and other metrics from the
     extracted financial data.
   - project_future(last_hist_data, avgs, last_year, num_years=5): Projects future financial data over a specified number
     of years using the historical averages.
   - combine_hist_and_proj(hist_results, proj_results, hist_years): Combines historical and projected data into a single
     structure including free cash flow calculations, memo data, and key assumptions.

4. DCF Valuation:
   - write_DCF_csv(fiscal_years, free_cf, memo, assumptions, output_path): Writes the combined historical and projected
     data into a CSV file with clearly separated sections.
   - compute_wacc_lines(ticker, fred_api_key): Computes the Weighted Average Cost of Capital (WACC) using company data
     (debt, equity, interest expense) and external data (10-year treasury rate from FRED).
   - dcf_valuation_details(ticker, growth_rate, fiscal_years, free_cf, inc_df, bal_df): Calculates the DCF valuation,
     including present values of projected free cash flows, terminal value, enterprise value, and the implied share price.
   - sensitivity_analysis_csv(ticker, growth_rate, fiscal_years, free_cf, inc_df, bal_df): Performs a sensitivity analysis
     by varying the terminal growth rate and WACC, and outputs the results as CSV-formatted data.

5. Main Workflow:
   - main(): Orchestrates the entire process by:
       • Prompting the user for a ticker symbol and terminal growth rate (only once).
       • Reading historical financial statements from a CSV file.
       • Computing historical metrics and projecting future data.
       • Running the DCF valuation and sensitivity analysis.
       • Combining all outputs into a single CSV file with sections separated by header lines.

Usage:
------
Run the script and input the required ticker symbol and terminal growth rate when prompted.
Ensure that a financials CSV file (with Income Statement, Balance Sheet, and Cash Flow Statement sections) exists
in the "data" directory. The final DCF valuation and sensitivity analysis output will be saved in the "DCF" directory.
"""









import os
import datetime
import numpy as np
import pandas as pd
import requests
from fredapi import Fred
from io import StringIO

# FMP API key (provided)
API_KEY = "FMP_API_KEY"
fred_api_key = "FRED_API_KEY"  # Your FRED API key

# -------------------------------
# Helper functions to fetch data
# -------------------------------
def fetch_profile(ticker):
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching profile for {ticker}: {response.text}")
        return {}
    data = response.json()
    if not data:
        print(f"No profile data returned for {ticker}")
        return {}
    return data[0]

def fetch_quote(ticker):
    url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching quote for {ticker}: {response.text}")
        return {}
    data = response.json()
    if not data:
        print(f"No quote data returned for {ticker}")
        return {}
    return data[0]

# -------------------------------
# Functions for parsing financials
# -------------------------------
def parse_financials_file(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    section_lines = {
        "Income Statement:": [],
        "Balance Sheet:": [],
        "Cash Flow Statement:": []
    }
    current_section = None
    for line in lines:
        if line in section_lines:
            current_section = line
            continue
        if not line:
            continue
        if current_section:
            section_lines[current_section].append(line)

    income_df = pd.read_csv(StringIO("\n".join(section_lines["Income Statement:"])))
    balance_df = pd.read_csv(StringIO("\n".join(section_lines["Balance Sheet:"])))
    cashflow_df = pd.read_csv(StringIO("\n".join(section_lines["Cash Flow Statement:"])))

    for df in [income_df, balance_df, cashflow_df]:
        if "calendarYear" in df.columns:
            df.set_index("calendarYear", inplace=True)
            df.index = [f"{int(year)}A" if str(year).isdigit() else str(year) for year in df.index]

    return income_df, balance_df, cashflow_df

def preprocess_df(df):
    return df

def extract_historical(income_df, balance_df, cashflow_df):
    income_df = preprocess_df(income_df)
    balance_df = preprocess_df(balance_df)
    cashflow_df = preprocess_df(cashflow_df)

    income_df = income_df.apply(pd.to_numeric, errors="coerce")
    balance_df = balance_df.apply(pd.to_numeric, errors="coerce")
    cashflow_df = cashflow_df.apply(pd.to_numeric, errors="coerce")

    hist_revenue    = income_df["revenue"]
    hist_cogs       = income_df["costOfRevenue"]
    hist_net_inc    = income_df["netIncome"]
    hist_int_exp    = income_df["interestExpense"]
    hist_tax_exp    = income_df["incomeTaxExpense"]
    hist_da         = income_df["depreciationAndAmortization"]

    hist_capex = cashflow_df["capitalExpenditure"]

    hist_current_assets = balance_df["totalCurrentAssets"]
    hist_ar           = balance_df["netReceivables"]
    hist_inv          = balance_df["inventory"]
    hist_current_liab = balance_df["totalCurrentLiabilities"]
    hist_ap           = balance_df["accountPayables"]

    hist_other_ca = hist_current_assets - (hist_ar + hist_inv)
    hist_other_cl = hist_current_liab - hist_ap
    hist_nwc      = hist_current_assets - hist_current_liab

    historical = {}
    for fy in hist_revenue.index:
        historical[fy] = {
            "Revenue": hist_revenue.get(fy, np.nan),
            "COGS": hist_cogs.get(fy, np.nan),
            "Net Income": hist_net_inc.get(fy, np.nan),
            "Interest Expense": hist_int_exp.get(fy, np.nan),
            "Tax Expense": hist_tax_exp.get(fy, np.nan),
            "D&A": hist_da.get(fy, np.nan),
            "CapEx": -1 * hist_capex.get(fy, np.nan),
            "Current Assets": hist_current_assets.get(fy, np.nan),
            "Accounts Receivable": hist_ar.get(fy, np.nan),
            "Inventory": hist_inv.get(fy, np.nan),
            "Other Current Assets": hist_other_ca.get(fy, np.nan),
            "Current Liabilities": hist_current_liab.get(fy, np.nan),
            "Accounts Payable": hist_ap.get(fy, np.nan),
            "Other Current Liabilities": hist_other_cl.get(fy, np.nan),
            "NWC": hist_nwc.get(fy, np.nan),
        }
    return historical

def compute_historical_metrics(historical):
    years = sorted(historical.keys())
    results = {}
    rev_growths = []
    net_inc_pct = []
    int_exp_pct = []
    tax_pct = []
    cogs_pct = []
    da_pct = []
    capex_pct = []
    dso_list = []
    dio_list = []
    dpo_list = []
    other_ca_pct = []
    other_cl_pct = []

    for i, fy in enumerate(years):
        data = historical[fy]
        EBIT = data["Net Income"] + data["Interest Expense"] + data["Tax Expense"]
        EBIAT = EBIT - data["Tax Expense"]
        if i == 0:
            nwc_change = 0
        else:
            prev_nwc = historical[years[i-1]]["NWC"]
            nwc_change = data["NWC"] - prev_nwc
        UFCF = EBIT - data["Tax Expense"] + data["D&A"] - data["CapEx"] - nwc_change
        data["EBIT"] = EBIT
        data["EBIAT"] = EBIAT
        data["Change NWC"] = nwc_change
        data["UFCF"] = UFCF

        if i > 0:
            prev_rev = historical[years[i-1]]["Revenue"]
            if prev_rev and prev_rev != 0:
                rev_growths.append(data["Revenue"] / prev_rev - 1)
        if data["Revenue"]:
            net_inc_pct.append(data["Net Income"] / data["Revenue"])
            int_exp_pct.append(data["Interest Expense"] / data["Revenue"])
            cogs_pct.append(data["COGS"] / data["Revenue"])
            da_pct.append(data["D&A"] / data["Revenue"])
            capex_pct.append(data["CapEx"] / data["Revenue"])
            other_ca_pct.append(data["Other Current Assets"] / data["Revenue"])
            other_cl_pct.append(data["Other Current Liabilities"] / data["Revenue"])
            dso_list.append(data["Accounts Receivable"] / data["Revenue"] * 365)
            if data["COGS"] and data["COGS"] != 0:
                dio_list.append(data["Inventory"] / data["COGS"] * 365)
                dpo_list.append(data["Accounts Payable"] / data["COGS"] * 365)
        if data["Net Income"]:
            tax_pct.append(data["Tax Expense"] / data["Net Income"])

        results[fy] = data

    averages = {
        "avg_rev_growth": np.mean(rev_growths) if rev_growths else 0,
        "avg_net_inc_pct": np.mean(net_inc_pct) if net_inc_pct else 0,
        "avg_int_exp_pct": np.mean(int_exp_pct) if int_exp_pct else 0,
        "avg_tax_pct": np.mean(tax_pct) if tax_pct else 0,
        "avg_cogs_pct": np.mean(cogs_pct) if cogs_pct else 0,
        "avg_da_pct": np.mean(da_pct) if da_pct else 0,
        "avg_capex_pct": np.mean(capex_pct) if capex_pct else 0,
        "avg_dso": np.mean(dso_list) if dso_list else 0,
        "avg_dio": np.mean(dio_list) if dio_list else 0,
        "avg_dpo": np.mean(dpo_list) if dpo_list else 0,
        "avg_other_ca_pct": np.mean(other_ca_pct) if other_ca_pct else 0,
        "avg_other_cl_pct": np.mean(other_cl_pct) if other_cl_pct else 0,
    }

    return results, averages, years

def project_future(last_hist_data, avgs, last_year, num_years=5):
    proj = {}
    base_revenue = last_hist_data["Revenue"]
    rev_list = []
    current_rev = base_revenue
    for i in range(num_years):
        current_rev *= (1 + avgs["avg_rev_growth"])
        rev_list.append(current_rev)

    prev_nwc = last_hist_data["NWC"]
    for i in range(num_years):
        fy_label = f"{last_year + i + 1}E"
        rev = rev_list[i]
        cogs = rev * avgs["avg_cogs_pct"]
        net_income = rev * avgs["avg_net_inc_pct"]
        int_exp = rev * avgs["avg_int_exp_pct"]
        tax_exp = net_income * avgs["avg_tax_pct"]
        ebit = net_income + int_exp + tax_exp
        ebiat = ebit - tax_exp
        da = rev * avgs["avg_da_pct"]
        capex = rev * avgs["avg_capex_pct"]
        ar = (avgs["avg_dso"] * rev) / 365
        inv = (avgs["avg_dio"] * cogs) / 365
        other_ca = rev * avgs["avg_other_ca_pct"]
        cur_assets = ar + inv + other_ca
        ap = (avgs["avg_dpo"] * cogs) / 365
        other_cl = rev * avgs["avg_other_cl_pct"]
        cur_liab = ap + other_cl
        nwc = cur_assets - cur_liab
        change_nwc = nwc - prev_nwc
        prev_nwc = nwc
        ufcf = ebit - tax_exp + da - capex - change_nwc
        proj[fy_label] = {
            "Revenue": rev,
            "COGS": cogs,
            "Net Income": net_income,
            "Interest Expense": int_exp,
            "Tax Expense": tax_exp,
            "EBIT": ebit,
            "EBIAT": ebiat,
            "D&A": da,
            "CapEx": capex,
            "Accounts Receivable": ar,
            "Inventory": inv,
            "Other Current Assets": other_ca,
            "Current Assets": cur_assets,
            "Accounts Payable": ap,
            "Other Current Liabilities": other_cl,
            "Current Liabilities": cur_liab,
            "NWC": nwc,
            "Change NWC": change_nwc,
            "UFCF": ufcf
        }
    return proj

def combine_hist_and_proj(hist_results, proj_results, hist_years):
    hist_years_sorted = sorted(hist_years)
    proj_years_sorted = sorted(proj_results.keys(), key=lambda x: int(x[:-1]))
    all_years = hist_years_sorted + proj_years_sorted

    def get_val(metric, fy):
        if fy in hist_results:
            return hist_results[fy].get(metric, np.nan)
        else:
            return proj_results.get(fy, {}).get(metric, np.nan)

    free_cf = {
        "Net Income": [get_val("Net Income", fy) for fy in all_years],
        "Tax Expense": [get_val("Tax Expense", fy) for fy in all_years],
        "Operating Profit (EBIT)": [get_val("EBIT", fy) for fy in all_years],
        "EBIAT (NOPAT)": [get_val("EBIAT", fy) for fy in all_years],
        "(+) Depreciation & Amortization": [get_val("D&A", fy) for fy in all_years],
        "(-) Capital Expenditures": [get_val("CapEx", fy) for fy in all_years],
        "(-) Change in NWC": [get_val("Change NWC", fy) for fy in all_years],
        "Unlevered Free Cash Flow": [get_val("UFCF", fy) for fy in all_years],
    }

    memo = {
        "Accounts Receivable": [get_val("Accounts Receivable", fy) for fy in all_years],
        "Inventory": [get_val("Inventory", fy) for fy in all_years],
        "Other Current Assets": [get_val("Other Current Assets", fy) for fy in all_years],
        "Total Current Assets": [get_val("Current Assets", fy) for fy in all_years],
        "Accounts Payable": [get_val("Accounts Payable", fy) for fy in all_years],
        "Other Current Liabilities": [get_val("Other Current Liabilities", fy) for fy in all_years],
        "Total Current Liabilities": [get_val("Current Liabilities", fy) for fy in all_years],
        "Net Working Capital (NWC)": [get_val("NWC", fy) for fy in all_years],
    }

    assumptions = {
        "Revenue": [get_val("Revenue", fy) for fy in all_years],
        "COGS": [get_val("COGS", fy) for fy in all_years],
        "Revenue Growth": []
    }
    assumptions["Revenue Growth"].append(np.nan)
    for i in range(1, len(hist_years_sorted)):
        prev_rev = hist_results[hist_years_sorted[i-1]]["Revenue"]
        curr_rev = hist_results[hist_years_sorted[i]]["Revenue"]
        if prev_rev and prev_rev != 0:
            assumptions["Revenue Growth"].append(curr_rev / prev_rev - 1)
        else:
            assumptions["Revenue Growth"].append(np.nan)
    num_proj = len(proj_years_sorted)
    for i in range(num_proj):
        if i == 0:
            prev_rev = hist_results[hist_years_sorted[-1]]["Revenue"]
            curr_rev = proj_results[proj_years_sorted[i]]["Revenue"]
            if prev_rev and prev_rev != 0:
                assumptions["Revenue Growth"].append(curr_rev / prev_rev - 1)
            else:
                assumptions["Revenue Growth"].append(np.nan)
        else:
            prev_rev = proj_results[proj_years_sorted[i-1]]["Revenue"]
            curr_rev = proj_results[proj_years_sorted[i]]["Revenue"]
            if prev_rev and prev_rev != 0:
                assumptions["Revenue Growth"].append(curr_rev / prev_rev - 1)
            else:
                assumptions["Revenue Growth"].append(np.nan)

    assumptions["Net Income % of Revenue"] = []
    assumptions["Interest Expense % of Revenue"] = []
    assumptions["Tax Expense % of Net Income"] = []
    assumptions["COGS % of Revenue"] = []
    assumptions["D&A % of Revenue"] = []
    assumptions["CapEx % of Revenue"] = []

    for fy in all_years:
        rev = get_val("Revenue", fy)
        net_inc = get_val("Net Income", fy)
        int_exp = get_val("Interest Expense", fy)
        tax_exp = get_val("Tax Expense", fy)
        cogs_val = get_val("COGS", fy)
        da_val = get_val("D&A", fy)
        capex_val = get_val("CapEx", fy)

        if rev and rev != 0:
            assumptions["Net Income % of Revenue"].append(net_inc / rev)
            assumptions["Interest Expense % of Revenue"].append(int_exp / rev)
            assumptions["COGS % of Revenue"].append(cogs_val / rev)
            assumptions["D&A % of Revenue"].append(da_val / rev)
            assumptions["CapEx % of Revenue"].append(capex_val / rev)
        else:
            assumptions["Net Income % of Revenue"].append(np.nan)
            assumptions["Interest Expense % of Revenue"].append(np.nan)
            assumptions["COGS % of Revenue"].append(np.nan)
            assumptions["D&A % of Revenue"].append(np.nan)
            assumptions["CapEx % of Revenue"].append(np.nan)

        if net_inc and net_inc != 0:
            assumptions["Tax Expense % of Net Income"].append(tax_exp / net_inc)
        else:
            assumptions["Tax Expense % of Net Income"].append(np.nan)

    return all_years, free_cf, memo, assumptions

def write_DCF_csv(fiscal_years, free_cf, memo, assumptions, output_path):
    lines = []
    lines.append("Free Cash Flow:")
    header = "Metric," + ",".join(fiscal_years)
    lines.append(header)
    for metric, vals in free_cf.items():
        row = [metric] + [str(round(v,2)) if pd.notna(v) else "" for v in vals]
        lines.append(",".join(row))
    lines.append("")
    lines.append("Memo:")
    header = "Metric," + ",".join(fiscal_years)
    lines.append(header)
    for metric, vals in memo.items():
        row = [metric] + [str(round(v,2)) if pd.notna(v) else "" for v in vals]
        lines.append(",".join(row))
    lines.append("")
    lines.append("Assumptions:")
    lines.append("Fiscal Year," + ",".join(fiscal_years))
    for metric in ["Revenue", "COGS", "Revenue Growth"]:
        row = [metric] + [str(round(v,4)) if pd.notna(v) else "" for v in assumptions[metric]]
        lines.append(",".join(row))
    lines.append("")
    for metric in [
        "Net Income % of Revenue",
        "Interest Expense % of Revenue",
        "Tax Expense % of Net Income",
        "COGS % of Revenue",
        "D&A % of Revenue",
        "CapEx % of Revenue",
    ]:
        row = [metric]
        for val in assumptions[metric]:
            row.append(str(round(val*100,2)) + "%" if pd.notna(val) else "")
        lines.append(",".join(row))
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Free cash flows output written to {output_path}")

# -------------------------------
# WACC Calculation helper
# -------------------------------
def get_last_period_value(df, column_label):
    try:
        val = df.iloc[-1][column_label]
    except Exception as e:
        val = np.nan
    return float(val) if pd.notna(val) else 0.0

def compute_wacc_lines(ticker, fred_api_key):
    stmt_path = os.path.join("data", f"{ticker}_financials.csv")
    if not os.path.exists(stmt_path):
        raise FileNotFoundError(f"{stmt_path} not found.")
    inc_df, bal_df, _ = parse_financials_file(stmt_path)
    debt = get_last_period_value(bal_df, "netDebt")
    interest_expense = get_last_period_value(inc_df, "interestExpense")
    cost_of_debt = (interest_expense / debt) if debt else 0.0
    tax_rate = 0.21
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
    profile = fetch_profile(ticker)
    if not profile:
        raise ValueError("Profile not found.")
    equity = profile.get("mktCap", 0.0)
    beta = profile.get("beta", 1.0)
    total_cap = equity + debt
    weight_debt = debt / total_cap if total_cap else 0.0
    weight_equity = equity / total_cap if total_cap else 0.0
    fred = Fred(api_key=fred_api_key)
    today = datetime.datetime.today()
    start_date = today - datetime.timedelta(days=30)
    dgs10 = fred.get_series("DGS10", observation_start=start_date, observation_end=today)
    risk_free_rate = dgs10.iloc[-1] / 100.0 if not dgs10.empty else 0.0
    expected_market_return = 0.1
    market_risk_premium = expected_market_return - risk_free_rate
    cost_of_equity = risk_free_rate + beta * market_risk_premium
    wacc = weight_debt * after_tax_cost_of_debt + weight_equity * cost_of_equity
    lines = []
    lines.append("Weighted Average Cost of Capital (WACC):")
    lines.append(f"Equity,{equity}")
    lines.append(f"Debt,{debt}")
    lines.append("")
    lines.append(f"Cost of Debt,{cost_of_debt}")
    lines.append(f"Tax Rate,{tax_rate}")
    lines.append(f"Weight of Debt,{weight_debt}")
    lines.append(f"After Tax Cost of Debt,{after_tax_cost_of_debt}")
    lines.append("")
    lines.append(f"Risk Free Rate,{risk_free_rate}")
    lines.append(f"Expected Market Return,{expected_market_return}")
    lines.append(f"Market Risk Premium,{market_risk_premium}")
    lines.append(f"Levered Beta,{beta}")
    lines.append(f"Weight of Equity,{weight_equity}")
    lines.append(f"Cost of Equity,{cost_of_equity}")
    lines.append("")
    lines.append(f"WACC,{wacc}")
    return lines

# -------------------------------
# DCF Valuation details (returns string content)
# -------------------------------
def dcf_valuation_details(ticker, growth_rate, fiscal_years, free_cf, inc_df, bal_df):
    # Compute WACC (using similar logic as above)
    debt = get_last_period_value(bal_df, "netDebt")
    interest_expense = get_last_period_value(inc_df, "interestExpense")
    cost_of_debt = (interest_expense / debt) if debt else 0.0
    tax_rate = 0.21
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
    profile = fetch_profile(ticker)
    equity = profile.get("mktCap", 0.0)
    beta = profile.get("beta", 1.0)
    total_cap = equity + debt
    weight_debt = debt / total_cap if total_cap else 0.0
    weight_equity = equity / total_cap if total_cap else 0.0
    fred = Fred(api_key=fred_api_key)
    today = datetime.datetime.today()
    start_date = today - datetime.timedelta(days=30)
    dgs10 = fred.get_series("DGS10", observation_start=start_date, observation_end=today)
    risk_free_rate = dgs10.iloc[-1] / 100.0 if not dgs10.empty else 0.0
    expected_market_return = 0.1
    market_risk_premium = expected_market_return - risk_free_rate
    cost_of_equity = risk_free_rate + beta * market_risk_premium
    WACC = weight_debt * after_tax_cost_of_debt + weight_equity * cost_of_equity

    proj_cols = [col for col in fiscal_years if col.endswith("E")]
    if not proj_cols:
        raise ValueError("No projected (ending with 'E') columns found.")
    proj_year_list = sorted([int(col[:-1]) for col in proj_cols if col[:-1].isdigit()])
    first_proj_year = proj_year_list[0]
    unlevered_fcf = free_cf.get("Unlevered Free Cash Flow")
    sum_pv_fcf = 0.0
    pv_dict = {}
    for col in proj_cols:
        try:
            year_int = int(col[:-1])
        except:
            continue
        n = year_int - first_proj_year + 1
        try:
            fcf_val = float(unlevered_fcf[fiscal_years.index(col)])
        except:
            fcf_val = 0.0
        dfactor = (1.0 + WACC) ** n
        pv = fcf_val / dfactor
        pv_dict[col] = pv
        sum_pv_fcf += pv
    last_proj_col = sorted(proj_cols, key=lambda x: int(x[:-1]))[-1]
    try:
        last_fcf = float(unlevered_fcf[fiscal_years.index(last_proj_col)])
    except:
        last_fcf = 0.0
    TV = (last_fcf * (1.01 + growth_rate )) / (WACC - growth_rate - 0.01)
    n_terminal = int(last_proj_col[:-1]) - first_proj_year + 1
    pv_terminal = TV / ((1.0 + WACC) ** n_terminal)
    enterprise_value = sum_pv_fcf + pv_terminal

    cash = get_last_period_value(bal_df, "cashAndCashEquivalents")
    debt_val = get_last_period_value(bal_df, "netDebt")
    minority = get_last_period_value(bal_df, "minorityInterest")

    quote = fetch_quote(ticker)
    shares_outstanding = quote.get("sharesOutstanding", 0)
    share_price = quote.get("price", 0.0)
    equity_value = enterprise_value + cash - debt_val - minority
    implied_share_price = equity_value / shares_outstanding if shares_outstanding else 0.0
    upside_downside = ((implied_share_price - share_price) / share_price) * 100.0 if share_price else 0.0

    lines = []
    lines.append("Unlevered Free Cashflow:")
    lines.append("Fiscal year," + ",".join(fiscal_years))
    row1 = "Unlevered Free Cash Flow," + ",".join(str(unlevered_fcf[fiscal_years.index(c)]) for c in fiscal_years)
    lines.append(row1)
    pv_list = []
    for c in fiscal_years:
        if c in proj_cols:
            pv_list.append(str(round(pv_dict.get(c, 0.0), 2)))
        else:
            pv_list.append("")
    lines.append("Present value of Free Cash Flow," + ",".join(pv_list))
    lines.append("")
    lines.append("Implied Share Price Calculation:")
    lines.append("Metric,Value")
    lines.append(f"Sum of PV of FCF,{round(sum_pv_fcf,2)}")
    lines.append(f"Growth Rate,{growth_rate}")
    lines.append(f"WACC,{round(WACC,4)}")
    lines.append(f"Terminal Value,{round(TV,2)}")
    lines.append(f"PV of Terminal Value,{round(pv_terminal,2)}")
    lines.append(f"Enterprise Value,{round(enterprise_value,2)}")
    lines.append(f"(+) Cash,{round(cash,2)}")
    lines.append(f"(-) Debt,{round(debt_val,2)}")
    lines.append(f"(-) Minority Interest,{round(minority,2)}")
    lines.append(f"Equity Value,{round(equity_value,2)}")
    lines.append(f"Diluted Shares Outstanding (mm),{round(shares_outstanding/1e6,2)}")
    lines.append(f"Implied Share Price,{round(implied_share_price,2)}")
    lines.append(f"{ticker} Share Price,{share_price}")
    lines.append("")
    lines.append(f"Upside/Downside,{round(upside_downside,2)}%")
    return "\n".join(lines)

# -------------------------------
# Sensitivity Analysis (returns CSV string)
# -------------------------------
def sensitivity_analysis_csv(ticker, growth_rate, fiscal_years, free_cf, inc_df, bal_df):
    debt = get_last_period_value(bal_df, "netDebt")
    interest_expense = get_last_period_value(inc_df, "interestExpense")
    cost_of_debt = (interest_expense / debt) if debt else 0.0
    tax_rate = 0.21
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
    profile = fetch_profile(ticker)
    equity = profile.get("mktCap", 0.0)
    beta = profile.get("beta", 1.0)
    total_cap = equity + debt
    weight_debt = debt / total_cap if total_cap else 0.0
    weight_equity = equity / total_cap if total_cap else 0.0
    fred = Fred(api_key=fred_api_key)
    today = datetime.datetime.today()
    start_date = today - datetime.timedelta(days=30)
    dgs10 = fred.get_series("DGS10", observation_start=start_date, observation_end=today)
    risk_free_rate = dgs10.iloc[-1] / 100.0 if not dgs10.empty else 0.0
    expected_market_return = 0.1
    market_risk_premium = expected_market_return - risk_free_rate
    cost_of_equity = risk_free_rate + beta * market_risk_premium
    base_wacc = weight_debt * after_tax_cost_of_debt + weight_equity * cost_of_equity

    proj_cols = [col for col in fiscal_years if col.endswith("E")]
    proj_year_list = sorted([int(col[:-1]) for col in proj_cols if col[:-1].isdigit()])
    first_proj_year = proj_year_list[0]
    unlevered_fcf = free_cf.get("Unlevered Free Cash Flow")
    last_proj_col = sorted(proj_cols, key=lambda x: int(x[:-1]))[-1]
    try:
        last_fcf = float(unlevered_fcf[fiscal_years.index(last_proj_col)])
    except:
        last_fcf = 0.0

    # Generate a growth range based on the input terminal growth rate.
    # For example, if growth_rate = 0.03, then the range is from 0.03 to 0.05.
    growth_range = np.linspace(growth_rate - 0.01, growth_rate + 0.01, 5)
    wacc_range = np.linspace(base_wacc - 0.03, base_wacc + 0.01, 5)

    sensitivity_table = pd.DataFrame(index=["{:.2f}".format(w) for w in wacc_range],
                                     columns=["{:.3f}".format(g) for g in growth_range])
    for w in wacc_range:
        for g in growth_range:
            sum_pv_fcf = 0.0
            for col in proj_cols:
                try:
                    year_int = int(col[:-1])
                except:
                    continue
                n = year_int - first_proj_year + 1
                try:
                    fcf_val = float(unlevered_fcf[fiscal_years.index(col)])
                except:
                    fcf_val = 0.0
                discount_factor = (1 + w) ** n
                sum_pv_fcf += fcf_val / discount_factor
            n_terminal = int(last_proj_col[:-1]) - first_proj_year + 1
            TV = 0.0 if (w - g) == 0 else (last_fcf * (1 + g)) / (w - g)
            pv_terminal = TV / ((1 + w) ** n_terminal)
            enterprise_value = sum_pv_fcf + pv_terminal
            cash = get_last_period_value(bal_df, "cashAndCashEquivalents")
            debt_val = get_last_period_value(bal_df, "netDebt")
            minority = get_last_period_value(bal_df, "minorityInterest")
            quote = fetch_quote(ticker)
            shares_outstanding = quote.get("sharesOutstanding", 0)
            equity_value = enterprise_value + cash - debt_val - minority
            implied_price = equity_value / shares_outstanding if shares_outstanding else 0.0
            sensitivity_table.loc["{:.2f}".format(w), "{:.3f}".format(g)] = implied_price

    return sensitivity_table.to_csv()

# -------------------------------
# Main function: Combine all steps & outputs
# -------------------------------
def main():
    # Ask once for the ticker and terminal growth rate
    ticker = input("Enter ticker symbol (e.g. GE): ").strip().upper()
    try:
        growth_rate = float(input("Enter the terminal growth rate (e.g. 0.03 for 3%): "))
    except:
        growth_rate = 0.03

    # 1) Process the historical statements
    statement_file = os.path.join("data", f"{ticker}_financials.csv")
    if not os.path.exists(statement_file):
        print(f"File {statement_file} not found.")
        return

    inc_df, bal_df, cf_df = parse_financials_file(statement_file)
    historical_dict = extract_historical(inc_df, bal_df, cf_df)
    hist_results, avgs, hist_years = compute_historical_metrics(historical_dict)
    last_fy = hist_years[-1]
    last_year_int = int(last_fy[:-1])  # remove the "A" suffix
    proj_dict = project_future(hist_results[last_fy], avgs, last_year_int, num_years=5)
    fiscal_years, free_cf, memo, assumptions = combine_hist_and_proj(hist_results, proj_dict, hist_years)

    # 2) Create DCF free cash flow and assumptions CSV content (as a string)
    dcf_lines = []
    dcf_lines.append("Free Cash Flow:")
    header = "Metric," + ",".join(fiscal_years)
    dcf_lines.append(header)
    for metric, vals in free_cf.items():
        row = [metric] + [str(round(v,2)) if pd.notna(v) else "" for v in vals]
        dcf_lines.append(",".join(row))
    dcf_lines.append("")
    dcf_lines.append("Memo:")
    header = "Metric," + ",".join(fiscal_years)
    dcf_lines.append(header)
    for metric, vals in memo.items():
        row = [metric] + [str(round(v,2)) if pd.notna(v) else "" for v in vals]
        dcf_lines.append(",".join(row))
    dcf_lines.append("")
    dcf_lines.append("Assumptions:")
    dcf_lines.append("Fiscal Year," + ",".join(fiscal_years))
    for metric in ["Revenue", "COGS", "Revenue Growth"]:
        row = [metric] + [str(round(v,4)) if pd.notna(v) else "" for v in assumptions[metric]]
        dcf_lines.append(",".join(row))
    dcf_lines.append("")
    for metric in ["Net Income % of Revenue",
                   "Interest Expense % of Revenue",
                   "Tax Expense % of Net Income",
                   "COGS % of Revenue",
                   "D&A % of Revenue",
                   "CapEx % of Revenue"]:
        row = [metric]
        for val in assumptions[metric]:
            row.append(str(round(val*100,2)) + "%" if pd.notna(val) else "")
        dcf_lines.append(",".join(row))
    dcf_content = "\n".join(dcf_lines)

    # 3) Get WACC output (as string)
    wacc_lines = compute_wacc_lines(ticker, fred_api_key)
    wacc_content = "\n".join(wacc_lines)

    # 4) Compute DCF valuation details (as string)
    dcf_valuation_content = dcf_valuation_details(ticker, growth_rate, fiscal_years, free_cf, inc_df, bal_df)

    # 5) Run sensitivity analysis using the input growth rate and adjusted linear space (as CSV string)
    sensitivity_csv_content = sensitivity_analysis_csv(ticker, growth_rate, fiscal_years, free_cf, inc_df, bal_df)

    # 6) Combine all outputs into a single CSV file with sections separated by headers
    combined_sections = []
    combined_sections.append("=== DCF Free Cash Flow and Assumptions ===")
    combined_sections.append(dcf_content)
    combined_sections.append("=== WACC Calculation ===")
    combined_sections.append(wacc_content)
    combined_sections.append("=== DCF Valuation ===")
    combined_sections.append(dcf_valuation_content)
    combined_sections.append("=== Sensitivity Analysis ===")
    combined_sections.append(sensitivity_csv_content)
    combined_output = "\n".join(combined_sections)

    out_dir = "DCF"
    os.makedirs(out_dir, exist_ok=True)
    combined_file = os.path.join(out_dir, f"{ticker}_DCF.csv")
    with open(combined_file, "w") as f:
        f.write(combined_output)
    print(f"DCF written to {combined_file}")

if __name__ == "__main__":
    main()
