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
        volume = tic_df["volume"]

        tic_df["SMA_20"] = talib.SMA(close, timeperiod=20)
        tic_df["SMA_60"] = talib.SMA(close, timeperiod=60)
        tic_df["SMA_120"] = talib.SMA(close, timeperiod=120)
        # tic_df["BBANDS_UPP"], tic_df["BBANDS_MID"], tic_df["BBANDS_LOW"] = talib.BBANDS(
        #     close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0
        # )
        _, _, tic_df["macdhist"] = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        tic_df["CCI"] = talib.CCI(high, low, close, timeperiod=14)
        tic_df["RSI"] = talib.RSI(close, timeperiod=14)
        tic_df["NATR"] = talib.NATR(high, low, close, timeperiod=14)
        tic_df["ADOSC"] = talib.ADOSC(
            high, low, close, volume, fastperiod=3, slowperiod=10
        )

        df_list.append(tic_df)

    result_df = pd.concat(df_list).sort_values(["date", "tic"], ignore_index=True)
    result_df["date"] = pd.to_datetime(result_df["date"])

    return result_df


def add_fundamental_data(price_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    tics = price_df["tic"].unique().tolist()
    df_list = []
    for tic in tics:
        tic_price = price_df[price_df["tic"] == tic].reset_index(drop=True)
        tic_fund = fund_df[fund_df["tic"] == tic].reset_index(drop=True)
        tic_price["date"] = pd.to_datetime(tic_price["date"])
        tic_fund["date"] = pd.to_datetime(tic_fund["date"])

        # sometimes 2 diffferent reports at the same date.
        tic_fund = tic_fund.drop_duplicates("date", keep="last", ignore_index=True)

        list_date = list(
            pd.date_range(tic_price["date"].min(), tic_price["date"].max())
        )
        process = pd.DataFrame({"date": list_date, "tic": tic}).merge(
            tic_price, on=["date", "tic"], how="left"
        )
        process = process.merge(tic_fund, how="left", on=["date", "tic"])

        for col in ["ROE", "AssetStoEquity", "Pnitoni", "Nitogr", "TaxBurden"]:
            process[col] = process[col].ffill()
            process[col] = pd.to_numeric(process[col])

        process.dropna(how="any", inplace=True)
        df_list.append(process)

    result_df = pd.concat(df_list).sort_values(["date", "tic"], ignore_index=True)

    return result_df


def add_all_indicators(price_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    price_df = add_tech_indicator(price_df)
    result = add_fundamental_data(price_df, fund_df)
    return result


def tech_indicator_only(price_df: pd.DataFrame) -> pd.DataFrame:
    price_df = price_df.drop(columns=["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"])
    price_df["date"] = pd.to_datetime(price_df["date"])
    return add_tech_indicator(price_df)
