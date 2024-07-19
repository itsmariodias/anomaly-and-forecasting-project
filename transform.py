import numpy as np
import pandas as pd


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    # Extract time features
    df["IS_WEEKEND"] = df["TX_DATETIME"].apply(lambda x: int(x.weekday() >= 5))
    df["IS_NIGHT"] = df["TX_DATETIME"].apply(lambda x: int(x.hour <= 6))

    df = df.groupby("CUSTOMER_ID").apply(
        lambda x: get_customer_features(x, windows_size_in_days=[1, 7, 30])
    )
    df.sort_values("TX_DATETIME").reset_index(drop=True)
    df = df.groupby("TERMINAL_ID").apply(
        lambda x: get_terminal_features(
            x, delay_period=7, windows_size_in_days=[1, 7, 30], feature="TERMINAL_ID"
        )
    )
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)

    df["TX_DATETIME"] = df["TX_DATETIME"].astype("int64") // 1e9

    return df


def extract_forecast_features(data: pd.DataFrame) -> pd.DataFrame:
    # Extract lag features for Machine Learning models

    data["Date"] = pd.to_datetime(data["Date"], unit="ms")
    data["year"] = data["Date"].dt.year
    data["year"] = data["year"] - min(data["year"])
    data["month"] = data["Date"].dt.month
    data["week"] = data["Date"].dt.isocalendar().week
    data["day"] = data["Date"].dt.day
    data["dayofweek"] = data["Date"].dt.day_of_week

    data["lag_1"] = data["Volume"].shift(1)
    data["lag_3"] = data["Volume"].shift(3)
    data["lag_7"] = data["Volume"].shift(7)
    data["lag_14"] = data["Volume"].shift(14)
    data["lag_28"] = data["Volume"].shift(28)
    data["lag_90"] = data["Volume"].shift(90)
    data["lag_365"] = data["Volume"].shift(365)

    data["rolling_mean_7"] = data["Volume"].rolling(window=7).mean().shift()
    data["rolling_mean_14"] = data["Volume"].rolling(window=14).mean().shift()
    data["rolling_mean_28"] = data["Volume"].rolling(window=28).mean().shift()

    data["rolling_std_7"] = data["Volume"].rolling(window=7).std().shift()
    data["rolling_std_14"] = data["Volume"].rolling(window=14).std().shift()
    data["rolling_std_28"] = data["Volume"].rolling(window=28).std().shift()

    data["rolling_skew_7"] = data["Volume"].rolling(window=7).skew().shift()
    data["rolling_skew_14"] = data["Volume"].rolling(window=14).skew().shift()
    data["rolling_skew_28"] = data["Volume"].rolling(window=28).skew().shift()

    data["rolling_max_7"] = data["Volume"].rolling(window=7).max().shift()
    data["rolling_max_14"] = data["Volume"].rolling(window=14).max().shift()
    data["rolling_max_28"] = data["Volume"].rolling(window=28).max().shift()

    data["Date"] = data["Date"].astype("int64") // 1e9

    return data

def get_customer_features(df: pd.DataFrame, windows_size_in_days=[1, 7, 30]):
    # Let us first order transactions chronologically
    df = df.sort_values("TX_DATETIME")

    # The transaction date and time is set as the index, which will allow the use of the rolling function
    df.index = df.TX_DATETIME

    # For each window size
    for window_size in windows_size_in_days:

        # Compute the sum of the transaction amounts and the number of transactions for the given window size
        SUM_AMOUNT_TX_WINDOW = df["TX_AMOUNT"].rolling(str(window_size) + "d").sum()
        NB_TX_WINDOW = df["TX_AMOUNT"].rolling(str(window_size) + "d").count()

        # Compute the average transaction amount for the given window size
        # NB_TX_WINDOW is always >0 since current transaction is always included
        AVG_AMOUNT_TX_WINDOW = SUM_AMOUNT_TX_WINDOW / NB_TX_WINDOW

        # Save feature values
        df["CUSTOMER_ID_NB_TX_" + str(window_size) + "DAY_WINDOW"] = list(NB_TX_WINDOW)
        df["CUSTOMER_ID_AVG_AMOUNT_" + str(window_size) + "DAY_WINDOW"] = list(
            AVG_AMOUNT_TX_WINDOW
        )

    # Reindex according to transaction IDs
    df.index = df.TRANSACTION_ID

    return df


def get_terminal_features(
    df: pd.DataFrame,
    delay_period=7,
    windows_size_in_days=[1, 7, 30],
    feature="TERMINAL_ID",
):

    df = df.sort_values("TX_DATETIME")

    df.index = df.TX_DATETIME

    NB_FRAUD_DELAY = df["TX_FRAUD"].rolling(str(delay_period) + "d").sum()
    NB_TX_DELAY = df["TX_FRAUD"].rolling(str(delay_period) + "d").count()

    for window_size in windows_size_in_days:

        NB_FRAUD_DELAY_WINDOW = (
            df["TX_FRAUD"].rolling(str(delay_period + window_size) + "d").sum()
        )
        NB_TX_DELAY_WINDOW = (
            df["TX_FRAUD"].rolling(str(delay_period + window_size) + "d").count()
        )

        NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
        NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY

        RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW

        df[feature + "_NB_TX_" + str(window_size) + "DAY_WINDOW"] = list(NB_TX_WINDOW)
        df[feature + "_RISK_" + str(window_size) + "DAY_WINDOW"] = list(RISK_WINDOW)

    df.index = df.TRANSACTION_ID

    # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0)
    df.fillna(0, inplace=True)

    return df
