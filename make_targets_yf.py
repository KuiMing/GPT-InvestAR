"""
Make targets from yahoo finance
"""

import os
import glob

# import pickle
import json
import sys
import argparse
from datetime import datetime, timedelta
from scipy import stats
import numpy as np
import pandas as pd
import sqlite3

# from openbb_terminal.sdk import openbb
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

CONN = sqlite3.connect("price.sqlite")
DATE = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
try:
    ALL_DATA = pd.read_sql(f"SELECT * FROM price_table where Date>='{DATE}'", CONN)
# pylint: disable=broad-exception-caught
except Exception:
    pass
# OLD_DATA = pickle.load(open("df_20240228.pkl", "rb"))


def get_symbol_price(symbol):
    data = ALL_DATA[ALL_DATA.symbol == symbol]
    data["Date"] = pd.to_datetime(data.Date)
    data.set_index(keys="Date", inplace=True)
    return data


def get_stock_price(symbol, start, end, conn):
    sql = f"SELECT * FROM price_table where symbol='{symbol}' and Date>='{start}' and Date<='{end}'"
    data = pd.read_sql(sql, conn)
    data["Date"] = pd.to_datetime(data.Date)
    data.set_index(keys="Date", inplace=True)
    return data


def get_ar_dates(symbol, config_dict):
    """
    Returns: The annual report dates for each symbol.
    """
    symbol_path = os.path.join(config_dict["annual_reports_pdf_save_directory"], symbol)
    folder_names = [
        os.path.basename(folder)
        for folder in glob.glob(os.path.join(symbol_path, "*"))
        if os.path.isdir(folder)
    ]
    return sorted(folder_names)


def get_pct_returns_defined_date(price_data, start_date, end_date, tolerance_days=7):
    """
    Function to give percentage returns over a defined time range.
    Args:
        price_data: A pandas DF containing the overall price history of a stock symbol
        start_date: The start date of computing pct returns
        end_date: The end date of computing pct returns
        tolerance_days: If the price data is missing for more than this then return nan.
                        Some data can be missing due to Weekends or Holidays
    Returns:
        Percentage price change between start and end date.
    """
    price_data_range = price_data.sort_index().loc[
        lambda x: (x.index >= start_date) & (x.index <= end_date)
    ]
    num_days_diff_start = (price_data_range.index[0] - start_date).days
    num_days_diff_end = (end_date - price_data_range.index[-1]).days
    if (num_days_diff_start > tolerance_days) | (num_days_diff_end >= tolerance_days):
        return np.nan
    start_price = price_data_range.iloc[0]["Adj Close"]
    end_price = price_data_range.iloc[-1]["Adj Close"]
    price_pct_diff = ((end_price - start_price) / start_price) * 100.0
    return price_pct_diff


def get_pct_returns_range(price_data, start_date, end_date, quantile, tolerance_days=7):
    """
    Function to give quantile based percentage returns between a defined time range.
    For example to get the maximum returns achieved from start_date and before the end_date
    Args:
        price_data: A pandas DF containing the overall price history of a stock symbol
        start_date: The start date of computing pct returns
        end_date: The end date of computing pct returns
        tolerance_days: If the price data is missing for more than this then return nan.
                        Some data can be missing due to Weekends or Holidays
    Returns:
        Specified quantile of percentage price change between start_date and before the end_date
    """
    price_data_range = price_data.sort_index().loc[
        lambda x: (x.index >= start_date) & (x.index <= end_date)
    ]
    num_days_diff_start = (price_data_range.index[0] - start_date).days
    num_days_diff_end = (end_date - price_data_range.index[-1]).days
    if (num_days_diff_start > tolerance_days) | (num_days_diff_end >= tolerance_days):
        return np.nan
    start_price = price_data_range.iloc[0]["Adj Close"]
    end_price = price_data_range["Adj Close"].quantile(quantile)
    price_pct_diff = ((end_price - start_price) / start_price) * 100.0
    return price_pct_diff


def get_all_targets(price_data, start_date, num_days_12m, prepend_string):
    """
    Function to return a dictionary of targets which contain percentage returns over different
    time ranges and quantiles
    Args:
        price_data: A pandas DF containing the overall price history of a stock symbol
        start_date: The start date of computing pct returns
        num_days_12m: Num of Days between start_date and successive annual report date
        prepend_string: Denoting the category of price_data. One of 'target' and 'sp500'
    """
    target_returns_dict = {}
    try:
        # 98th percentile is proxy for max returns
        target_returns_dict[f"{prepend_string}_max"] = get_pct_returns_range(
            price_data, start_date, start_date + timedelta(days=num_days_12m), 0.98
        )
        # 2nd percentile is proxy for min returns
        target_returns_dict[f"{prepend_string}_min"] = get_pct_returns_range(
            price_data, start_date, start_date + timedelta(days=num_days_12m), 0.02
        )
    # pylint: disable=broad-exception-caught
    except Exception:
        target_returns_dict[f"{prepend_string}_max"] = np.nan
        target_returns_dict[f"{prepend_string}_min"] = np.nan

    # Get returns for 3, 6, 9 and 12 month duration. Some alteration is done in time duration
    # as the annual reports may be release in an interval of less or more than 12 months.
    for period in [3, 6, 9, 12]:
        num_days = int(num_days_12m * (period / 12))
        end_date = start_date + timedelta(days=num_days)
        try:
            pct_returns = get_pct_returns_defined_date(price_data, start_date, end_date)
        # pylint: disable=broad-exception-caught
        except Exception:
            pct_returns = np.nan
        target_returns_dict[f"{prepend_string}_{str(period)}m"] = pct_returns
    return target_returns_dict


def make_targets(
    symbol,
    start_date,
    end_date,
    price_data_sp500,
    config_dict,
    conn=None,
    date="2023-02-27",
):
    """
    Function to generate target return information for each symbol based on
    annual report dates
    Args:
        symbol: stock ticker
        start_date: overall historical start date from where price data is to be fetched
        end_date: overall end date upto which price data is to be fetched
        price_data_sp500: Prefetched dataframe for ticker ^GSPC which gives price data for S&P500
    Returns:
        Pandas DF containing percentage returns between annual report dates for the symbol
    """
    # price_data = openbb.stocks.load(symbol, start_date=start_date, end_date=end_date,
    #                                 verbose=False)
    if conn is None:
        price_data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    else:
        # price_data = get_stock_price(symbol, start=start_date, end=end_date, conn=conn)
        price_data = get_symbol_price(symbol)

    ar_dates = pd.DataFrame({"date": get_ar_dates(symbol, config_dict)})
    ar_dates = ar_dates[ar_dates.date > date].date.values.tolist()
    df = pd.DataFrame()
    for i in range(len(ar_dates) - 1):
        curr_report_date = datetime.strptime(ar_dates[i], "%Y-%m-%d")
        # Start and end dates are offset by 2 days to be conservative and allowing the price to settle.
        curr_start_date = datetime.strptime(ar_dates[i], "%Y-%m-%d") + timedelta(days=2)
        curr_end_date_12m = datetime.strptime(ar_dates[i + 1], "%Y-%m-%d") - timedelta(
            days=2
        )
        num_days_12m = (curr_end_date_12m - curr_start_date).days
        if num_days_12m < 200:
            continue
        target_dict = get_all_targets(
            price_data, curr_start_date, num_days_12m, "target"
        )
        sp500_dict = get_all_targets(
            price_data_sp500, curr_start_date, num_days_12m, "sp500"
        )
        target_dict.update(sp500_dict)
        target_df = pd.DataFrame.from_dict(target_dict, orient="index").T
        target_df["report_date"] = curr_report_date
        target_df["start_date"] = curr_start_date
        target_df["end_date"] = curr_end_date_12m
        df = pd.concat([df, target_df], ignore_index=True)
    df["symbol"] = symbol
    return df


def make_targets_all_symbols(start_date, end_date, config_dict, conn):
    """
    Function to return the complete dataframe for all symbols and all annual report date periods
    """
    symbol_names = [
        os.path.basename(folder)
        for folder in glob.glob(
            os.path.join(config_dict["annual_reports_pdf_save_directory"], "*")
        )
        if os.path.isdir(folder)
    ]
    symbol_names.sort()
    price_data_sp500 = pdr.get_data_yahoo("^GSPC", start=start_date, end=end_date)
    connect = sqlite3.connect("target.sqlite")
    date = pd.read_sql("SELECT MAX(report_date) as date FROM target", connect)
    date = date.date.values[0]
    connect.close()
    full_df = pd.DataFrame()
    # Iterate over all symbols in the directory
    for i, symbol in enumerate(symbol_names):

        df = make_targets(
            symbol, start_date, end_date, price_data_sp500, config_dict, conn, date
        )
        full_df = pd.concat([full_df, df], ignore_index=True)
        print(f"Completed: {i + 1}/{len(symbol_names)}, {datetime.now()}")
    return full_df


def get_normalized_column(df, col):
    """
    Function to rank and then normalise a column in the df.
    Returns:
        Pandas DF with additional normalised column
    """
    new_col = col + "_normalised"
    preds = df.loc[:, col]
    ranked_preds = (preds.rank(method="average").values - 0.5) / preds.count()
    gauss_ranked_preds = stats.norm.ppf(ranked_preds)
    df.loc[:, new_col] = gauss_ranked_preds
    return df


def bin_targets(df, input_col, output_col, percentile_list, label_list):
    """
    Function for binning target columns according to percentiles
    Args:
        input_col: target column to normalise
        output_col: Name of new normalised column
        percentile_list: Percentiles for binning
        label_list: labels aka bins
    Returns:
        Pandas DF with binned targets. Used for final ML model building
    """
    s = df.loc[:, input_col]
    binned_series = pd.qcut(s, q=percentile_list, labels=label_list)
    label_list_float = [np.float32(x) for x in label_list]
    binned_series.replace(to_replace=label_list, value=label_list_float, inplace=True)
    df.loc[:, output_col] = binned_series.astype("float32")
    return df


def main(args):
    """
    main function
    """
    with open(args.config_path, encoding="utf8") as json_file:
        config_dict = json.load(json_file)
    start_date = args.start
    if args.start is None:
        start_date = "2002-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    if args.sqlite:
        conn = sqlite3.connect(args.sqlite)
    else:
        conn = None
    targets_df = make_targets_all_symbols(start_date, end_date, config_dict, conn)
    targets_df_filtered = targets_df.loc[lambda x: ~(x.isnull().any(axis=1))]
    # Create a column called era which denotes the year of annual report filing
    targets_df_filtered["era"] = targets_df_filtered["report_date"].apply(
        lambda x: x.year
    )
    # Drop duplicates if they exist.
    # Could be if consecutive annual reports are published in same year.
    targets_df_filtered_dedup = targets_df_filtered.drop_duplicates(
        subset=["era", "symbol"]
    ).reset_index(drop=True)
    target_cols = [
        c for c in targets_df_filtered_dedup.columns if c.startswith("target")
    ]
    # Generate normalised target columns
    for target in target_cols:
        targets_df_filtered_dedup = targets_df_filtered_dedup.groupby(
            "era", group_keys=False
        ).apply(lambda df, target=target: get_normalized_column(df, target))

    # Create final target column for Machine Learning model building
    input_col_target = "target_12m_normalised"
    output_col_target = "target_ml"
    targets_df_filtered_dedup = targets_df_filtered_dedup.groupby(
        "era", group_keys=False
    ).apply(
        lambda df: bin_targets(
            df,
            input_col_target,
            output_col_target,
            [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ["0.0", "0.25", "0.5", "0.75", "1.0"],
        )
    )
    connect = sqlite3.connect("target.sqlite")
    targets_df_filtered_dedup.to_sql(
        "target", con=connect, index=False, if_exists="append"
    )
    connect.close()
    # with open(config_dict["targets_df_path"], "wb") as handle:
    #     pickle.dump(targets_df_filtered_dedup, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
        "--sqlite",
        type=str,
        default=None,
        help="sqlite database of price, it will get the price from yahoo if it's None",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="start date",
    )
    main(args=parser.parse_args())
    sys.exit(0)
