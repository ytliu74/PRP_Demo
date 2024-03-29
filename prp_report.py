import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import itertools
import os
import datetime

from finrl import config
from finrl import config_tickers
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.finrl_meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

from data_processor.indicator import (
    add_all_indicators,
    add_tech_indicator,
    tech_indicator_only,
)

import data_processor.config


def backtest(df, env_kwargs, model):
    e_trade_gym = StockTradingEnv(df, **env_kwargs)
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model, environment=e_trade_gym
    )
    final_value = df_account_value.iloc[-1].account_value

    return final_value


def main(algo="SAC", run_iter=5, log_points=10):
    print(f"Running {algo}, run_iter={run_iter}, log_points={log_points}")

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

    train = data_split(df, "2009-01-01", "2020-01-01")
    trade = data_split(df, "2020-01-01", "2022-01-01")

    indicators = data_processor.config.INDICATORS
    n_indicators = len(indicators)
    stock_dimension = len(df.tic.unique())
    state_space = 1 + (2 + n_indicators) * stock_dimension

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

    agent = DRLAgent(env=env_train)

    single_timestep = 11000

    results = np.zeros((run_iter, log_points))

    for i in range(run_iter):
        print(f"run_iter: {i}")

        if algo == "SAC":

            SAC_PARAMS = {
                "batch_size": 256,
                "buffer_size": 100000,
                "learning_rate": 0.0001,
                "learning_starts": 100,
                "ent_coef": "auto_0.1",
            }

            model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

            trained_sac = agent.train_model(
                model=model_sac, tb_log_name="sac", total_timesteps=0
            )

            results[i, 0] = backtest(trade, env_kwargs, model=trained_sac)
            print(f"initial result: {results[i, 0]}")

            for log_i in range(1, log_points):
                trained_sac = agent.train_model(
                    model=trained_sac,
                    tb_log_name="sac",
                    total_timesteps=single_timestep,
                )

                results[i, log_i] = backtest(trade, env_kwargs, model=trained_sac)
                print(f"log point {log_i}, result: {results[i, log_i]}")

        elif algo == "DDPG":
            DDPG_PARAMS = {
                "batch_size": 128,
                "buffer_size": 100000,
                "learning_rate": 0.0001,
            }

            agent = DRLAgent(env=env_train)
            # model_ddpg = agent.get_model(
            #     "ddpg", model_kwargs=DDPG_PARAMS, tensorboard_log=config.TENSORBOARD_LOG_DIR
            # )
            model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)
            trained_ddpg = agent.train_model(
                model=model_ddpg, tb_log_name="ddpg", total_timesteps=0
            )

            results[i, 0] = backtest(trade, env_kwargs, model=trained_ddpg)
            print(f"initial result: {results[i, 0]}")

            for log_i in range(1, log_points):
                trained_ddpg = agent.train_model(
                    model=trained_ddpg,
                    tb_log_name="ddpg",
                    total_timesteps=single_timestep,
                )

                results[i, log_i] = backtest(trade, env_kwargs, model=trained_ddpg)
                print(f"log point {log_i}, result: {results[i, log_i]}")

    np.save(f'logs/{algo}_fund', results)

    print(results)

if __name__ == "__main__":
    main("DDPG", run_iter=5, log_points=10)
