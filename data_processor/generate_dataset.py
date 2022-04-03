import pandas as pd
import os
from data_processor.ChinaStockDownloader import ChinaStockDownloader


def generate_parquet_dataset(
    start_date="2008-01-01",
    end_date="2022-01-01",
    folder_path="./data",
    index=["sz50", "hs300"],
):
    """_summary_

    Args:
        start_date (str, optional): _description_. Defaults to "2008-01-01".
        end_date (str, optional): _description_. Defaults to "2020-01-01".
        folder_path (str, optional): _description_. Defaults to "./data".
        index (list, optional): _description_. Defaults to ["sz50", "hs300"].
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    for id in index:
        dl = ChinaStockDownloader(start_date, end_date, index=id)
        dl.download_price().to_parquet(os.path.join(folder_path, f"{id}_price.parquet"))
        dl.download_fundament().to_parquet(
            os.path.join(folder_path, f"{id}_fundament.parquet")
        )
