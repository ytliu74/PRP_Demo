import pandas as pd
from data_processor import ChinaStockDownloader
from data_processor import generate_dataset

start_date = "2010-01-01"
end_date = "2022-05-01"
folder_path = "./data"
index = ["TICKER_LIST_590"]

generate_dataset.generate_parquet_dataset(start_date, end_date, folder_path, index)
