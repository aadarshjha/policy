import gym
import numpy as np

import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

# call back function to save best model
from SaveOnBestTrainingRewardCallback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common import results_plotter

# import library to read from YAML file
import yaml
import argparse

import os


class ExecuteTraining:
    def __init__(
        self,
        policy,
        env_name,
        learning_policy,
        learning_rate,
        initial_epsilon,
        seed,
        verbose,
        experiment_name,
        timesteps,
    ):
        self.env = gym.make(env_name)

        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.log_dir = "./" + self.experiment_name + "_best_model"
        self.env = Monitor(self.env, self.log_dir)

        # make the self.log_dir directory
        os.makedirs(self.log_dir, exist_ok=True)

        if policy == "DQN":
            self.model = DQN(
                policy=learning_policy,
                env=self.env,
                learning_rate=learning_rate,
                exploration_initial_eps=initial_epsilon,
                seed=seed,
                verbose=verbose,
                tensorboard_log="./dqn_cartpole_tensorboard/" + self.experiment_name
            )

    def run(self):
        callback = SaveOnBestTrainingRewardCallback(
            check_freq=1000, log_dir=self.log_dir
        )
        self.model.learn(total_timesteps=self.timesteps, callback=callback)
        self.env.close()

    def evaluate(self):
        # evaluates the model over 10 episodes, collects mean and standard deviation.
        eval_env = gym.make(self.env_name)
        mean_reward, std_reward = evaluate_policy(
            self.model, eval_env, n_eval_episodes=10
        )
        print("Mean Reward:", mean_reward, "Standard Deviation Of Reward:", std_reward)

    def load_model(self, model_path):
        self.model = DQN.load(model_path)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(
        description="Train an RL agent on the CartPole-v1 environment (https://gym.openai.com/envs/CartPole-v1/)."
    )
    parser.add_argument("--config", type=str, help="Config YAML File path")
    args = parser.parse_args()

    # read the config file
    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    env = ExecuteTraining(
        config["MODEL"],
        config["ENV"],
        config["LEARNING_POLICY"],
        config["LEARNING_RATE"],
        config["INTIAL_EPSILON"],
        config["SEED"],
        config["VERBOSE"],
        config["EXPERIMENT_NAME"],
        config["TIMESTEPS"],
    )

    env.run()
