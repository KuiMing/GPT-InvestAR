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

yf.pdr_override()


def main(args):
    """
    main function
    """
    config_dict = json.load(open(args.config_path, encoding="utf8"))
    conn = sqlite3.connect(args.sqlite)
    start_date = args.start
    if args.start is None:
        start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    symbol_names = [
        os.path.basename(folder)
        for folder in glob.glob(
            os.path.join(config_dict["annual_reports_pdf_save_directory"], "*")
        )
        if os.path.isdir(folder)
    ]

    for symbol in symbol_names:
        price_data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
        price_data["symbol"] = symbol
        price_data.to_sql("price_table", con=conn, if_exists="append")


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
        help="start date",
    )
    parser.add_argument(
        "--sqlite",
        type=str,
        default="price.sqlite",
        help="sqlite database of price",
    )
    main(args=parser.parse_args())
