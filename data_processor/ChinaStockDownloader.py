from datetime import datetime
import baostock as bs
import pandas as pd
from tqdm import tqdm
from data_processor import config
import os


class ChinaStockDownloader(object):
    """Download Chine stock data"""

    def __init__(self, start_date: str, end_date: str, stock=[], index=None):
        """Download China Stock Data

        Args:
            start_date (str): "YYYY-mm-dd"
            end_date (str): "YYYY-mm-dd"
            stock (list, optional): Stock list to be downloaded. Defaults to [].
            index (str, ['sz50', 'hs300', 'zz500']): Stock index to be downloaded. Defaults to None.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.start_year = datetime.strptime(start_date, "%Y-%m-%d").year
        self.end_year = datetime.strptime(end_date, "%Y-%m-%d").year
        self.index = index

        self.stock_list = []
        if self.index is None:
            self.stock_list = stock
        elif self.index in config.QUERY_FROM_BAOSTOCK_INDEX:
            self.stock_list = self._query_constituent_list()
        elif self.index in config.APPENDED_INDEX:
            self.stock_list = getattr(config, self.index)

    def download_price(
        self,
        to_csv=True,
        path=None,
        force_download=False,
        frequency="d",
        adjust_flag="3",
    ) -> pd.DataFrame:
        """Download price data

        Args:
            to_csv (bool, optional): Whether save data to csv file. Defaults to True.
            path (str, optional): The path to the csv file. Defaults to None.
            force_download (bool, optional): Download even if data already exists. Defaults to False.
            frequency (str, optional): The frequency of price data. Defaults to "d".
                "d": day, "w": week, "m": month, "5": 5 min, "15": 15 min, "30": 30 min, "60": 60 min
            adjust_flag (int, optional): Adjust flag.
        Returns:
            pandas.DataFrame: Price df
        """
        if path is None:
            path = f"./data/{self.index}_{self.start_date}~{self.end_date}.csv"

        if os.path.exists(path) and not force_download:
            print("Load from downloaded data")
            return pd.read_csv(path)

        fields = "date,code,open,high,low,close,amount"
        columns = config.PRICE_DF_COLUMNS

        lg = bs.login()
        result_list = []

        stocks_iter = tqdm(self.stock_list)
        for code in stocks_iter:
            stocks_iter.set_description("Downloading price data of " + code)

            rs = bs.query_history_k_data_plus(
                code,
                fields,
                self.start_date,
                self.end_date,
                frequency=frequency,
                adjustflag=adjust_flag,
            )
            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())
            result = pd.DataFrame(data_list, columns=columns)
            result_list.append(result)

        bs.logout()
        df = pd.concat(result_list, ignore_index=True).sort_values(
            ["date", "tic"], ignore_index=True
        )
        for col in ["open", "high", "low", "close", "amount"]:
            df[col] = df[col].astype(float)

        if to_csv:
            df.to_csv(path, index=False)

        return df

    def download_fundament(self) -> pd.DataFrame:
        """Download fundamental data
         Reference: http://baostock.com/baostock/index.php/%E5%AD%A3%E9%A2%91%E6%9D%9C%E9%82%A6%E6%8C%87%E6%95%B0

         Returns:
             pd.DataFrame: columns = ['tic', 'date', 'ROE', 'AssetStoEquity', 'Pnitoni', 'Nitogr',
        'TaxBurden']
        """
        drop_list = ["statDate", "dupontAssetTurn", "dupontIntburden", "dupontEbittogr"]
        rename_dict = {
            "code": "tic",
            "pubDate": "date",
            "dupontROE": "ROE",
            "dupontAssetStoEquity": "AssetStoEquity",
            "dupontPnitoni": "Pnitoni",
            "dupontNitogr": "Nitogr",
            "dupontTaxBurden": "TaxBurden",
        }

        lg = bs.login()
        stocks_iter = tqdm(self.stock_list)
        result_list = []
        for code in stocks_iter:
            stocks_iter.set_description("Downloading fundamental data of " + code)
            code_dupont_list = []
            for year in range(self.start_year, self.end_year + 1):
                for quarter in range(1, 5):
                    rs = bs.query_dupont_data(code=code, year=year, quarter=quarter)
                    while (rs.error_code == "0") & rs.next():
                        code_dupont_list.append(rs.get_row_data())
            code_result = pd.DataFrame(code_dupont_list, columns=rs.fields)
            result_list.append(code_result)
        bs.logout()

        df = (
            pd.concat(result_list, ignore_index=True)
            .rename(columns=rename_dict)
            .drop(columns=drop_list)
        )

        return df

    def _query_constituent_list(self) -> list:
        lg = bs.login()

        rs = getattr(bs, f"query_{self.index}_stocks")()
        consituents = []
        while (rs.error_code == "0") & rs.next():
            consituents.append(rs.get_row_data())
        result = pd.DataFrame(consituents, columns=rs.fields)["code"].to_list()

        bs.logout()
        return result


if __name__ == "__main__":
    downloader = ChinaStockDownloader(
        start_date="2019-01-01", end_date="2020-05-01", index="TEST"
    )
    df = downloader.download_fundament()
    df.to_csv("f.csv")
