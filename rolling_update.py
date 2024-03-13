import pandas as pd
from sklearn.linear_model import LinearRegression


def rolling_prod_profit(df, price):
    """
    Calculate the cumulative product of investment returns for a set of stocks over time using a rolling window approach.

    Parameters:
    - df (DataFrame): A Pandas DataFrame containing stock prediction data. Expected columns:
        - 'report_date': The date of the report, indicating the start of the prediction.
        - 'end_date': The end date of the prediction.
        - 'pred_reg_12m': The predicted return for the next 12 months.
        - 'symbol': The identifier symbol of the stock.
    - price (DataFrame): A Pandas DataFrame containing stock price data. Expected columns:
        - 'date': The date of the trade.
        - 'Open': Opening price of the stock for the day.
        - 'Close': Closing price of the stock for the day.
        - 'symbol': The identifier symbol of the stock.

    Returns:
    - cumulative_return (DataFrame): A Pandas DataFrame containing the months and their corresponding cumulative returns.

    This function performs the following steps:
    1. Converts the date columns in 'df' and 'price' to datetime format and adds a month column to 'df' based on 'report_date'.
    2. Initializes an empty DataFrame 'selected_stocks' to store the stocks selected each month and a variable 'total_return' to track the cumulative return.
    3. For each month in 'df', selects the top five stocks with the highest predicted return.
    4. Calculates the average stock return for the month based on the opening and closing prices of these stocks.
    5. Updates the cumulative return and adds it to the 'cumulative_return' list.
    6. Finally, converts the 'cumulative_return' list to a DataFrame and returns it.

    Note:
    - The function assumes that the 'symbol' column in 'df' and 'price' can be used to match stocks.
    - The function does not account for transaction costs and other factors that might affect actual returns.
    - This function is suitable for analyzing return rates based on specific prediction strategies.
    """

    df_test = df.copy()
    print(df_test.shape)
    df_test["report_date"] = pd.to_datetime(df_test["report_date"])
    df_test["end_date"] = pd.to_datetime(df_test["end_date"])
    price["date"] = pd.to_datetime(price["date"])

    df_test["month"] = df_test["report_date"].dt.to_period("M")

    selected_stocks = pd.DataFrame()
    total_return = 1

    cumulative_return = [1]

    months = list(df_test.sort_values("month").month.unique())
    for month in df_test.sort_values("month").month.unique():

        current_month_predictions = df_test[
            (df_test["month"] == month) & (df_test["end_date"] >= month.start_time)
        ]

        combined = pd.concat([selected_stocks, current_month_predictions])
        combined = combined[combined["end_date"] >= month.start_time]

        top5 = combined.nlargest(5, "pred_reg_12m")
        selected_stocks = top5.drop_duplicates("symbol")

        stock_prices = price[price["symbol"].isin(selected_stocks.symbol)]
        buy_price = (
            stock_prices[stock_prices["date"].dt.to_period("M") == (month + 1)]
            .groupby("symbol")
            .agg({"Open": "first"})
            .reset_index()
        )
        sell_price = (
            stock_prices[stock_prices["date"].dt.to_period("M") == (month + 1)]
            .groupby("symbol")
            .agg({"Close": "last"})
            .reset_index()
        )
        stock_return = ((sell_price.Close - buy_price.Open) / buy_price.Open).mean()
        total_return = cumulative_return[-1] * (1 + stock_return)
        cumulative_return.append(total_return)

    months.append((month + 1))
    cumulative_return = pd.DataFrame(
        dict(month=months, cumulative_return=cumulative_return)
    )
    cumulative_return["month"] = cumulative_return.month.astype(str)
    print("Total Return:", total_return)
    return cumulative_return


def rolling_fit_predict(data, year, feature_cols):
    df = data.copy()
    df = df[~df.target_ml.isna()]
    reg_12m = LinearRegression(positive=True).fit(
        df.loc[df.year <= year, feature_cols], df.loc[df.year <= year, "target_ml"]
    )
    # df_origin.loc[df_origin.year == i + 1, 'pred_reg_12m'] = \
    # reg_12m.predict(df_origin.loc[df_origin.year == i + 1, feature_cols])
    return reg_12m
