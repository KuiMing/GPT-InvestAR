"""
Get stock price from yahoo finance and save data in sqlite
"""

import os
import glob
import json
import argparse
from datetime import datetime, timedelta
import sqlite3
from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd

# yf.pdr_override()


def is_after_splits(symbol):
    stock = yf.Ticker(symbol)
    splits = stock.splits
    if len(splits) == 0:
        return False
    today = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    return today in splits.index.strftime("%Y-%m-%d")


def update_after_split(symbol, conn):
    sql = f"SELECT * FROM price_table where symbol == '{symbol}'"
    history = pd.read_sql(sql, con=conn)
    today = datetime.today().strftime("%Y-%m-%d")
    price_data = pdr.get_data_yahoo(symbol, start=history.Date.min()[:10], end=today)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM price_table WHERE symbol = ?", (symbol,))
    price_data.to_sql("price_table", con=conn, if_exists="append")


def get_stock_price(ticker, end=None, start="2020-01-01"):
    if end == None:
        end = datetime.now().strftime("%Y-%m-%d")
    output = yf.download(ticker, start=start, end=end)
    output.columns = ["Close", "High", "Low", "Open", "Volume"]
    output["Adj Close"] = output["Close"]
    output = output[["High", "Low", "Open", "Close", "Adj Close", "Volume"]]
    return output


def main(args):
    """
    main function
    """
    config_dict = json.load(open(args.config_path, encoding="utf8"))
    conn = sqlite3.connect(args.sqlite)
    start_date = args.start
    if args.start is None:
        # start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        sp500 = get_stock_price("^GSPC")
        sp500.reset_index(inplace=True)
        max_date = pd.read_sql("SELECT MAX(Date) as Date FROM price_table", con=conn)
        data = sp500[sp500.Date > max_date.Date.values[-1]]
        #         max_date = '2025-02-14'
        #         data = sp500[sp500.Date > max_date]
        if not data.empty:
            data.reset_index(inplace=True, drop=True)
            start_date = pd.to_datetime(data.Date)[0].strftime("%Y-%m-%d")
    if args.end is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    symbol_names = [
        os.path.basename(folder)
        for folder in glob.glob(
            os.path.join(config_dict["annual_reports_pdf_save_directory"], "*")
        )
        if os.path.isdir(folder)
    ]

    for symbol in symbol_names:
        try:
            price_data = get_stock_price(symbol, start=start_date, end=end_date)
        except:
            print(f"{symbol} is not found")
        price_data["symbol"] = symbol
        price_data.to_sql("price_table", con=conn, if_exists="append")
        if is_after_splits(symbol):
            update_after_split(symbol, conn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        dest="config_path",
        type=str,
        required=True,
        help="""Full path of config.json""",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="end date",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="start date",
    )
    parser.add_argument(
        "--sqlite",
        type=str,
        default="price.sqlite",
        help="sqlite database of price",
    )
    main(args=parser.parse_args())
