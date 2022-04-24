import datetime

QUERY_FROM_BAOSTOCK_INDEX = ["sz50", "hs300", "zz500"]

APPENDED_INDEX = ["SZ", "GEM", "TEST", "trainable_sz50_2009"]

TEST = ["sh.600000", "sh.600028", "sh.600030", "sh.600031"]

FIRST_DAY_OF_2009 = datetime.datetime(2009, 1, 5)
FIRST_DAY_OF_2011 = datetime.datetime(2011, 1, 4)
FIRST_DAY_OF_2014 = datetime.datetime(2014, 1, 2)

PRICE_DF_COLUMNS = [
    "date",
    "tic",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
]

INDICATORS = [
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "SMA_20",
    "SMA_60",
    "SMA_120",
    "macdhist",
    "CCI",
    "RSI",
    "NATR",
    "ADOSC",
    "ROE",
    "AssetStoEquity",
    "Pnitoni",
    "Nitogr",
    "TaxBurden",
]

TECH_INDICATORS = [
    "SMA_20",
    "SMA_60",
    "SMA_120",
    "macdhist",
    "CCI",
    "RSI",
    "NATR",
    "ADOSC",
]
