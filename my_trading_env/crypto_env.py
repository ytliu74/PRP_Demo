import math
import numpy as np
import pandas as pd


class CryptoEnv:
    """Multi crypto trading environment"""

    def __init__(
        self,
        data_df: pd.DataFrame,
        tech_indicators_list: list,
        min_buy_value=10,
        lookback=1,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        ma_ratio=0.99,
        reward_scaler=1e-3,
    ):
        """Multi crypto trading environment

        Args:
            data_df (pd.DataFrame): A df with price and tech indicators
            tech_indicators_list (list)
            min_buy_value (int, optional): The minimum value to trade. Defaults to 10.
            lookback (int, optional): _description_. Defaults to 1.
            initial_capital (int, optional): _description_. Defaults to 1e6.
            buy_cost_pct (float, optional): _description_. Defaults to 1e-3.
            sell_cost_pct (float, optional): _description_. Defaults to 1e-3.
            ma_ratio (float, optional): _description_. Defaults to 0.99.
            reward_scaler (float, optional): _description_. Defaults to 1e-4.

        """
        self.tech_indicators_list = tech_indicators_list
        self.min_buy_value = min_buy_value
        self.lookback = lookback
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.max_stock = 1
        self.ma_ratio = ma_ratio
        self.reward_scaler = reward_scaler
        self.crypto_num = len(data_df["tic"].unique())
        self.trading_date = data_df["date"].unique()
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)

        self.price_array, self.tech_array = self._df_to_time_array(data_df)
        self._generate_normalizers()

        self.max_step = len(self.price_array) - lookback - 1

        """env information"""
        self.env_name = "MY_MulticryptoEnv"
        # cash + holdings + (price + tech) * lookback
        self.state_dim = (
            1
            + self.crypto_num
            + (self.crypto_num * 1 + self.tech_array.shape[1]) * lookback
        )
        self.action_dim = self.crypto_num
        self.if_discrete = False
        self.target_return = 10

    def reset(self) -> np.ndarray:
        self.time = self.lookback - 1
        self.cash = self.initial_cash  # reset()
        self.total_asset = self.initial_cash
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.episode_return = 0.0
        self.return_ma = 0.0

        state = self.get_state()
        return state

    def step(self, actions):
        """A typical step function

        Args:
            actions (array)

        Returns:
            state, reward, done, None
        """
        self.time += 1

        price = self.price_array[self.time]
        for i in range(self.action_dim):
            norm_vector_i = self.action_norm_vector[i]
            actions[i] = actions[i] * norm_vector_i

        for index in np.where(actions < 0)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

        for index in np.where(actions > 0)[0]:  # buy_index:
            if (
                price[index] > 0
            ):  # Buy only if the price is > 0 (no missing data in this particular date)
                min_share = 10 / price[index]
                buy_num_shares = min(
                    self.cash // (price[index] * (1 + self.buy_cost_pct)),
                    actions[index],
                )
                # less than 10$
                if buy_num_shares < min_share:
                    buy_num_shares = 0
                self.stocks[index] += buy_num_shares
                self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

        """update time"""
        done = self.time == self.max_step
        state = self.get_state()
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        reward = (next_total_asset - self.total_asset) * self.reward_scaler
        self.total_asset = next_total_asset
        self.return_ma = self.return_ma * self.ma_ratio + reward
        self.cumu_return = self.total_asset / self.initial_cash
        if done:
            reward = self.return_ma
            self.episode_return = self.total_asset / self.initial_cash
        return state, reward, done, None

    def get_state(self):
        state = np.hstack(
            (
                self.cash * self.cash_norm,
                self.stocks,
            )
        )
        for i in range(self.lookback):
            price_i = self.price_array[self.time - i]
            tech_i = self.tech_array[self.time - i]
            normalized_price_i = price_i * self.price_norm_vector
            normalized_tech_i = tech_i * self.tech_norm_vector

            state = np.hstack((state, normalized_price_i, normalized_tech_i)).astype(
                np.float32
            )

        return state

    def close(self):
        pass

    def _generate_normalizers(self):
        action_norm_vector = []
        price_norm_vector = []
        tech_norm_vector = []  # 8 tech indicators => 8 times size of price_norm_vector

        price_0 = self.price_array[0]  # price at time 0

        for price in price_0:
            x = len(str(int(price))) - (1 + int(math.log10(self.initial_cash)))
            action_norm = 10 ** (-x)
            action_norm_vector.append(action_norm)

            y = len(str(int(price))) - 1
            price_norm = 10 ** (-y)
            price_norm_vector.append(price_norm)

            tech_norm_vector.extend([price_norm] * 3)  # SMA * 3
            tech_norm_vector.append(price_norm)  # MACDHist
            tech_norm_vector.append(10**-2)  # CCI
            tech_norm_vector.append(10**-2)  # RSI
            tech_norm_vector.append(1)  # NATR
            tech_norm_vector.append(10 ** (y - 6))  # ADOSC

        self.action_norm_vector = np.array(action_norm_vector)
        self.price_norm_vector = np.array(price_norm_vector)
        self.tech_norm_vector = np.array(tech_norm_vector)
        self.cash_norm = 10 ** -int(math.log10(self.initial_cash))

    def _df_to_time_array(self, df):
        tic_list = df["tic"].unique()
        price_list = []
        total_tech_list = []

        for tic in tic_list:
            tic_df = df[df["tic"] == tic]
            price = tic_df["close"].to_numpy().reshape(-1, 1)
            price_list.append(price)

            tech_list = []
            for tech in self.tech_indicators_list:
                tech_data = tic_df[tech].to_numpy().reshape(-1, 1)
                tech_list.append(tech_data)

            tech_array = np.concatenate(tech_list, axis=1)
            total_tech_list.append(tech_array)

        price_array = np.concatenate(price_list, axis=1)
        total_tech_array = np.concatenate(total_tech_list, axis=1)

        return price_array, total_tech_array
