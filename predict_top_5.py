import json
import sqlite3
from datetime import datetime, timedelta
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import requests


def rolling_gptprod_profit(df_test, price):

    # 轉換日期格式
    df_test["report_date"] = pd.to_datetime(df_test["report_date"])
    df_test["end_date"] = pd.to_datetime(df_test["end_date"])
    try:
        price["date"] = pd.to_datetime(price["date"])
    except KeyError:
        price.rename(columns={"Date": "date"}, inplace=True)
        price["date"] = pd.to_datetime(price["date"])

    # 按月分組
    df_test["month"] = df_test["report_date"].dt.to_period("M")
    print(df_test.month.max())
    # 初始化選中股票列表和累計收益
    selected_stocks = pd.DataFrame()
    total_return = 1
    cumulative_return = [1]
    stop_trading = False
    consecutive_profit_months = 0
    months = list(df_test.sort_values("month").month.unique())
    stock_list = pd.DataFrame()
    # 遍歷每個月

    for month in df_test.sort_values("month").month.unique():
        # 獲取當月的預測
        current_month_predictions = df_test[(df_test["month"] == month)]

        # 結合上個月選出的股票，同時檢查是否過期
        combined = pd.concat([selected_stocks, current_month_predictions])
        combined = combined[combined["report_date"] >= (month - 12).start_time]

        # 選擇預測值最高的五檔股票
        top5 = combined.nlargest(5, "pred_reg_12m")
        selected_stocks = top5.drop_duplicates("symbol")
        tmp = selected_stocks[["symbol", "pred_reg_12m"]]
        tmp["month"] = month

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
        monthly_return = ((sell_price.Close - buy_price.Open) / buy_price.Open).mean()

        # 檢查是否停止買賣
        if stop_trading:
            if monthly_return > 0:
                consecutive_profit_months += 1
                if consecutive_profit_months >= 1:
                    stop_trading = False
                    consecutive_profit_months = 0
        else:
            total_return = cumulative_return[-1] * (1 + monthly_return)
            if monthly_return < -0.1:
                stop_trading = True
                consecutive_profit_months = 0

        tmp["stop_trading"] = stop_trading
        stock_list = stock_list.append(tmp, ignore_index=True)
        cumulative_return.append(total_return)

    # 打印累計收益
    months.append((month + 1))
    cumulative_return = pd.DataFrame(
        dict(month=months, cumulative_return=cumulative_return)
    )
    cumulative_return["month"] = cumulative_return.month.astype(str)
    # 打印累計收益
    print("Total Return:", total_return)
    return cumulative_return, stock_list


def rolling_fit_predict(data, year, feature_cols):
    df = data.copy()
    df = df[~df.target_ml.isna()]
    df_origin = data.copy()
    for i in year:
        print(i)
        reg_12m = LinearRegression(positive=True).fit(
            df.loc[df.year <= i, feature_cols], df.loc[df.year <= i, "target_ml"]
        )
        df_origin.loc[df_origin.year == i + 1, "pred_reg_12m"] = reg_12m.predict(
            df_origin.loc[df_origin.year == i + 1, feature_cols]
        )
    return df_origin


def line_broadcast_flex(data):
    path = "line_config.json"
    with open(path, "r", encoding="utf8") as f:
        config = json.load(f)

    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(config["token"]),
    }
    contents = []
    if data[0]["stop_trading"]:

        contents.append(
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {
                        "type": "text",
                        "text": "Stop Trading",
                        "size": "sm",
                        "color": "#555555",
                        "flex": 0,
                    }
                ],
            }
        )
    else:
        for item in data:
            contents.append(
                {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                        {
                            "type": "text",
                            "text": item["symbol"],
                            "size": "sm",
                            "color": "#555555",
                            "flex": 0,
                        },
                        {
                            "type": "text",
                            "text": "{:.4f}".format(item["pred_reg_12m"]),
                            "size": "sm",
                            "color": "#111111",
                            "align": "end",
                        },
                    ],
                }
            )

    flex_message = {
        "type": "flex",
        "altText": "Stock Information",
        "contents": {
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "STOCK INFO",
                        "weight": "bold",
                        "color": "#1DB446",
                        "size": "sm",
                    },
                    {"type": "separator", "margin": "xxl"},
                    {
                        "type": "box",
                        "layout": "vertical",
                        "margin": "xxl",
                        "spacing": "sm",
                        "contents": contents,
                    },
                    {"type": "separator", "margin": "xxl"},
                ],
            },
            "styles": {"footer": {"separator": True}},
        },
    }

    payload = json.dumps({"messages": [flex_message]})
    response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
    print(response.text)


def main():
    connect = sqlite3.connect("feature.sqlite")
    feature = pd.read_sql("SELECT * FROM feature", con=connect)
    connect.close()

    connect = sqlite3.connect("target.sqlite")
    target = pd.read_sql("SELECT * FROM target", con=connect)
    connect.close()

    connect = sqlite3.connect("price.sqlite")
    price = pd.read_sql("SELECT * FROM price_table", con=connect)
    connect.close()

    year = int(datetime.now().strftime("%Y"))
    df = feature.merge(
        target[["symbol", "report_date", "target_ml", "end_date"]],
        on=["symbol", "report_date"],
        how="left",
    )
    model = pickle.load(open("model.pkl", "rb"))
    df["report_date"] = pd.to_datetime(df.report_date)
    df.loc[df.end_date.isna(), "end_date"] = df.loc[
        df.end_date.isna(), "report_date"
    ] + timedelta(days=365)
    years = [str(year - 1), str(year)]
    output = rolling_fit_predict(df, years, feature_cols=model["feature"])
    _, stock_list = rolling_gptprod_profit(output[output.year.isin(years)], price)
    line_broadcast_flex(stock_list.tail(5).to_dict("records"))


if __name__ == "__main__":
    main()
