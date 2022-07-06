import pandas as pd
import numpy as np

import torch
from elegantrl.agents import AgentDDPG
from elegantrl.agents import AgentPPO
from elegantrl.agents import AgentSAC
from elegantrl.agents import AgentTD3
from elegantrl.agents import AgentSAC_H
from elegantrl.agents import AgentPPO_H
from elegantrl.train.config import Arguments

# from elegantrl.agents import AgentA2C
from elegantrl.train.run import train_and_evaluate, init_agent

MODELS = {
    "ddpg": AgentDDPG,
    "td3": AgentTD3,
    "sac": AgentSAC,
    "ppo": AgentPPO,
    "sac_h": AgentSAC_H,
    "ppo_h": AgentPPO_H,
}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac", "sac_h"]
ON_POLICY_MODELS = ["ppo", "ppo_h"]
"""MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}"""


class CryptoDRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(self, model_name, model_kwargs):
        self.env.env_num = 1
        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        model = Arguments(agent=agent, env=self.env)
        model.if_off_policy = model_name in OFF_POLICY_MODELS
        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.seed = model_kwargs["seed"]
                model.net_dim = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_gap"]
                model.eval_times = model_kwargs["eval_times"]
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_and_evaluate(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, env):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        agent = MODELS[model_name]
        env.env_num = 1
        args = Arguments(agent=agent, env=env)
        args.cwd = cwd
        args.net_dim = net_dimension
        # load agent
        try:
            agent = init_agent(args, gpu_id=0)
            act = agent.act
            device = agent.device
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state = env.reset()
        episode_returns = []  # the cumulative_return / initial_account
        actions = []
        episode_total_assets = [env.initial_cash]
        with _torch.no_grad():
            for i in range(env.max_step):
                s_tensor = _torch.as_tensor(np.array((state,)), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = a_tensor.cpu().numpy()[0]
                actions.append(action)

                state, reward, done, _ = env.step(action)

                total_asset = env.cash + (env.price_array[env.time] * env.stocks).sum()
                episode_total_assets.append(total_asset)
                episode_return = total_asset / env.initial_cash
                episode_returns.append(episode_return)
                if done:
                    break
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return", episode_return)
        result_df = pd.DataFrame()
        result_df["date"] = env.trading_date[env.lookback + 1:]
        result_df["account_value"] = episode_total_assets
        return result_df, actions
