import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Agg')
import datetime
import itertools
import sys

from finrl import config
from finrl import config_tickers
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.finrl_meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

from data_processor.indicator import add_all_indicators
import data_processor.config
from data_processor.ChinaStockDownloader import single_stock_query
import quantstats as qs

if __name__ == "__main__":
    algo = sys.argv[1]

    price_df = pd.read_parquet("./data/sz50_price.parquet")
    fund_df = pd.read_parquet("./data/sz50_fundament.parquet")

    df = add_all_indicators(price_df, fund_df)
    tmp_list = []
    for tic in data_processor.config.trainable_sz50_2009:
        tmp_df = df[df["tic"] == tic]
        tmp_list.append(tmp_df)
    df = pd.concat(tmp_list, ignore_index=True)

    train = data_split(df, "2009-01-01", "2020-01-01")
    trade = data_split(df, "2020-01-01", "2022-01-01")

    indicators = data_processor.config.INDICATORS
    n_indicators = len(indicators)
    stock_dimension = len(df.tic.unique())
    state_space = 1 + (2 + n_indicators) * stock_dimension
    print(f"stock_dimension: {stock_dimension}, state_space: {state_space}")

    env_kwargs = {
    "hmax": 100,
    "initial_amount" : 1000000,
    # "initial_list": [10000000] + [0 for i in range(stock_dimension)],
    # buy and sell cost for each stock
    "num_stock_shares" : [0] * stock_dimension,
    "buy_cost_pct": [0.001] * stock_dimension,
    "sell_cost_pct": [0.001] * stock_dimension,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": indicators,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    }
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()

    # SAC
    if algo == "sac":
        agent = DRLAgent(env=env_train)
        SAC_PARAMS = {
            "batch_size": 128,
            "buffer_size": 1000000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        }

        model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
        trained_sac = agent.train_model(
            model=model_sac, tb_log_name="sac", total_timesteps=200000
        )
        model = trained_sac

    # DDPG
    if algo == "ddpg":
        DDPG_PARAMS = {
            "batch_size": 128,
            "buffer_size": 100000,
            "learning_rate": 0.0001,
        }

        agent = DRLAgent(env=env_train)
        model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)

        trained_ddpg = agent.train_model(
            model=model_ddpg, tb_log_name="ddpg", total_timesteps=100000
        )
        model = trained_ddpg

    # Backtest
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model, environment=e_trade_gym
    )

    baseline_df = single_stock_query("2020-01-01", "2022-01-01", "sh.000016")

    test_ret = get_daily_return(df_account_value)
    base_ret = get_daily_return(baseline_df, "close")
    test_ret.index = pd.DatetimeIndex(test_ret.index.date)
    base_ret.index = pd.DatetimeIndex(base_ret.index.date)

    qs.reports.html(
        test_ret, base_ret, title="sz50", download_filename=f"sz50_{algo}.html"
    )
