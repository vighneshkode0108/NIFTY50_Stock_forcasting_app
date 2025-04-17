import os
import configparser
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from utils import nifty50_companies

config = configparser.ConfigParser()
config.read("config.ini")

no_of_data_days = int(config['data']['no_of_data_days'])

def fetch_and_append_data(company, ticker):
    filename = f"data/{ticker}.csv"
    
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=no_of_data_days)

    if os.path.exists(filename):
        existing_data = pd.read_csv(filename, parse_dates=['Date'])
        last_date = existing_data['Date'].max().date()
        start_date = last_date + timedelta(days=1)

    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if not stock_data.empty:
        stock_data.reset_index(inplace=True)  # Makes 'Date' a column

        if os.path.exists(filename):
            existing_data = pd.read_csv(filename, parse_dates=['Date'])
            combined = pd.concat([existing_data, stock_data], ignore_index=True)
            combined.drop_duplicates(subset='Date', keep='last', inplace=True)
            combined.to_csv(filename, index=False)
            print(f"Updated data for {company}.")
        else:
            stock_data.to_csv(filename, index=False)
            print(f"Saved new data for {company} from {start_date} to {end_date}.")
    else:
        print(f"No new data available for {company}.")

def main():
    for company, ticker in nifty50_companies.items():
        fetch_and_append_data(company, ticker)

if __name__ == "__main__":
    main()
