from data_processor import config
import talib
import pandas as pd


def add_tech_indicator(df: pd.DataFrame) -> pd.DataFrame:
    tics = df["tic"].unique().tolist()
    df_list = []
    for tic in tics:
        tic_df = df[df["tic"] == tic].reset_index(drop=True)
        open = tic_df["open"]
        high = tic_df["high"]
        low = tic_df["low"]
        close = tic_df["close"]
        volume = tic_df["amount"]

        tic_df["SMA_5"] = talib.SMA(close, timeperiod=5)
        tic_df["SMA_20"] = talib.SMA(close, timeperiod=20)
        tic_df["SMA_60"] = talib.SMA(close, timeperiod=60)
        tic_df["SMA_120"] = talib.SMA(close, timeperiod=120)
        tic_df["BBANDS_UPP"], tic_df["BBANDS_MID"], tic_df["BBANDS_LOW"] = talib.BBANDS(
            close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0
        )
        _, _, tic_df["macdhist"] = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        tic_df["RSI"] = talib.RSI(close, timeperiod=14)
        tic_df["NATR"] = talib.NATR(high, low, close, timeperiod=14)
        tic_df["ADOSC"] = talib.ADOSC(
            high, low, close, volume, fastperiod=3, slowperiod=10
        )

        df_list.append(tic_df)

    result_df = pd.concat(df_list).sort_values(["date", "tic"], ignore_index=True)

    return result_df


def add_fundamental_data(price_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    ;