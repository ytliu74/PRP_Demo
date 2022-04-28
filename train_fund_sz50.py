from re import A
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Agg')
import datetime
import itertools
import sys
import os

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

# if __name__ == "__main__":
#     algo = sys.argv[1]
#     cuda = sys.argv[2]


def train_fund_sz50(algo, tensorboard_log=False, cuda=0, timesteps=100000):

    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + "results"):
        os.makedirs("./" + "results")

    if tensorboard_log:
        log = config.TENSORBOARD_LOG_DIR
    else:
        log = None

    price_df = pd.read_parquet("./data/sz50_price.parquet")
    fund_df = pd.read_parquet("./data/sz50_fundament.parquet")

    df = add_all_indicators(price_df, fund_df)

    available_tics = (
        df.set_index("date").loc[data_processor.config.FIRST_DAY_OF_2009].tic.tolist()
    )

    tmp_list = []
    for tic in available_tics:
        tmp_df = df[df["tic"] == tic]
        tmp_list.append(tmp_df)

    df = pd.concat(tmp_list, ignore_index=True)

    train = data_split(df, "2014-01-01", "2020-01-01")
    trade = data_split(df, "2020-01-01", "2022-01-01")

    indicators = data_processor.config.INDICATORS
    n_indicators = len(indicators)
    stock_dimension = len(df.tic.unique())
    state_space = 1 + (2 + n_indicators) * stock_dimension
    print(f"stock_dimension: {stock_dimension}, state_space: {state_space}")

    env_kwargs = {
        "hmax": 1000,
        "initial_amount": 10000000,
        # "initial_list": [10000000] + [0 for i in range(stock_dimension)],
        # buy and sell cost for each stock
        "num_stock_shares": [0] * stock_dimension,
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
            "batch_size": 256,
            "buffer_size": 1000000,
            "learning_rate": 0.00005,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
            "device": f"cuda:{cuda}",
        }

        model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS, tensorboard_log=log)
        trained_sac = agent.train_model(
            model=model_sac, tb_log_name="sac", total_timesteps=timesteps
        )
        model = trained_sac

    # DDPG
    if algo == "ddpg":
        DDPG_PARAMS = {
            "batch_size": 256,
            "buffer_size": 1000000,
            "learning_rate": 0.0001,
            "device": f"cuda:{cuda}",
        }

        agent = DRLAgent(env=env_train)
        model_ddpg = agent.get_model(
            "ddpg", model_kwargs=DDPG_PARAMS, tensorboard_log=log
        )

        trained_ddpg = agent.train_model(
            model=model_ddpg, tb_log_name="ddpg", total_timesteps=timesteps
        )
        model = trained_ddpg

    # Backtest
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)

    trade_account_value, _ = DRLAgent.DRL_prediction(
        model=model, environment=e_trade_gym
    )

    train_account_value, _ = DRLAgent.DRL_prediction(
        model=model, environment=e_train_gym
    )

    trade_perf = backtest_stats(account_value=trade_account_value)
    train_perf = backtest_stats(account_value=train_account_value)

    ret = {
        "train_ret": train_perf.loc["Cumulative returns"],
        "trade_ret": trade_perf.loc["Cumulative returns"],
    }

    return ret


if __name__ == "__main__":
    cuda = sys.argv[1]
    result_list = []

    for timesteps in [i * 50000 for i in range(1, 11)]:
        # 50000 ~ 500000
        for iter in range(5):
            print("--------------------")
            print(f"timesteps: {timesteps},iteration: {iter}")
            ret = train_fund_sz50("ddpg", cuda=cuda, timesteps=timesteps)
            ret["timesteps"] = timesteps
            result_list.append(ret)

    result = pd.DataFrame(result_list, columns=["timesteps", "train_ret", "trade_ret"])
    result.to_csv("./results/ret_on_time.csv")
