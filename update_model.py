from datetime import datetime, timedelta
import pandas as pd
from rolling_update import rolling_fit_predict
import sqlite3
import pickle


def main():
    models = pickle.load(open("model", "rb"))
    feature_col = models["feature"]
    connect = sqlite3.connect("feature.sqlite")
    feature = pd.read_sql("SELECT * FROM feature", con=connect)
    connect.close()

    connect = sqlite3.connect("target.sqlite")
    target = pd.read_sql("SELECT * FROM target", con=connect)
    connect.close()

    df = feature.merge(
        target[["symbol", "report_date", "target_ml", "end_date"]],
        on=["symbol", "report_date"],
        how="left",
    )
    df["report_date"] = pd.to_datetime(df.report_date)
    df.loc[df.end_date.isna(), "end_date"] = df.loc[
        df.end_date.isna(), "report_date"
    ] + timedelta(days=365)
    model = rolling_fit_predict(data=df, year=datetime.year, feature_cols=feature_col)

    models["model"] = model


if __name__ == "__main__":
    main()
