"""
This script retrieves financial statements for a given company ticker from the FinancialModelingPrep (FMP) API,
organizes the data into pandas DataFrames, and then saves all the statements into a single CSV file with clear
section headers.

Key Components:
----------------
1. API Configuration:
   - Sets up the FMP API key and base URL.
   - Defines an output directory to store the CSV file.

2. Functions:
   - fetch_financials(endpoint: str, ticker: str) -> pd.DataFrame:
       * Constructs the API request URL for a specific financial statement endpoint (e.g., income statement,
         balance sheet, or cash flow statement) and ticker.
       * Sends a GET request to the FMP API.
       * Converts the JSON response into a pandas DataFrame.
       * Sets the "date" column as the index (if available) to organize the data chronologically.
       
   - save_to_csv_sections(dataframes: dict, ticker: str):
       * Receives a dictionary of DataFrames keyed by the statement type.
       * Writes each DataFrame into one CSV file, structuring the file into distinct sections
         (Income Statement, Balance Sheet, and Cash Flow Statement).
       * Ensures each section is separated by headers and blank lines for clarity.
       
   - main():
       * Prompts the user for a ticker symbol.
       * Calls fetch_financials for the income statement, balance sheet, and cash flow statement.
       * Organizes the resulting DataFrames into a dictionary.
       * Invokes save_to_csv_sections to output the combined CSV file.
       
Usage:
------
Run the script and enter the desired ticker symbol (e.g., TDG) when prompted.
The script will then download the financial data from the FMP API and save it into a CSV file
located in the "./data" directory.
"""





import requests
import pandas as pd
import os

# FinancialModelingPrep API Key (Replace with your own API key)
API_KEY = "ENTER_FINANCIALMODELINGPREP_API_KEY"

# Base URL for FMP API
BASE_URL = "https://financialmodelingprep.com/api/v3"

# Output Directory
OUTPUT_DIR = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_financials(endpoint: str, ticker: str) -> pd.DataFrame:
    """
    Fetches financial data from FMP API for the given endpoint and ticker,
    and returns it as a DataFrame.
    """
    url = f"{BASE_URL}/{endpoint}/{ticker}?apikey={API_KEY}&limit=10"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching {endpoint} for {ticker}: {response.text}")
        return pd.DataFrame()

    data = response.json()
    if not data:
        print(f"No data returned for {endpoint} for {ticker}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df.set_index("date", inplace=True)
    
    return df


def save_to_csv_sections(dataframes: dict, ticker: str):
    """
    Saves all financial statements into one CSV file.
    The file is structured into sections:
      - Income Statement:
      - Balance Sheet:
      - Cash Flow Statement:
    """
    filename = os.path.join(OUTPUT_DIR, f"{ticker}_financials.csv")
    with open(filename, "w") as f:
        # Define the order of sections
        sections = [
            "Income Statement",
            "Balance Sheet",
            "Cash Flow Statement"
        ]
        for section in sections:
            f.write(f"{section}:\n")
            df = dataframes.get(section, pd.DataFrame())
            if not df.empty:
                f.write(df.to_csv())
            else:
                f.write("No data available\n")
            f.write("\n")  # Blank line between sections
    print(f"Financials saved to {filename}")


def main():
    # Prompt for ticker symbol
    ticker = input("Enter ticker symbol (e.g. GE): ").strip().upper()
    
    # Fetch each financial statement
    income_statement = fetch_financials("income-statement", ticker)
    balance_sheet = fetch_financials("balance-sheet-statement", ticker)
    cashflow_statement = fetch_financials("cash-flow-statement", ticker)

    # Organize data by section name
    financials = {
        "Income Statement": income_statement,
        "Balance Sheet": balance_sheet,
        "Cash Flow Statement": cashflow_statement,
    }

    # Save all statements into one CSV file with sections
    save_to_csv_sections(financials, ticker)


if __name__ == "__main__":
    main()
