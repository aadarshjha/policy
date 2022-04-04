import gym
import numpy as np

import os

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.results_plotter import load_results, ts2xy
from PlotAverageAndStdDev import PlotAverageAndStdDev

import matplotlib.pyplot as plt

import yaml
import argparse

# global log directory
log_dir = "./logs"


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

        self.log_dir = log_dir + "/" + self.experiment_name

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.env = Monitor(self.env, self.log_dir)

        if policy == "DQN":
            self.model = DQN(
                policy=learning_policy,
                env=self.env,
                learning_rate=learning_rate,
                exploration_initial_eps=initial_epsilon,
                seed=seed,
                verbose=0,
            )

    def run(self):
        # centralized call back function.
        callback = PlotAverageAndStdDev(check_freq=1, log_dir=self.log_dir, verbose=1)
        # need to collect information when each episode ends.
        self.model.learn(total_timesteps=self.timesteps, callback=callback)
        self.env.close()

    def evaluate(self):
        # evaluates the model over 10 episodes, collects mean and standard deviation.
        eval_env = gym.make('CartPole-v1')
        mean_reward, std_reward = evaluate_policy(
            DQN.load("./logs/dqn_experiment_1/best_model"), eval_env, n_eval_episodes=10, render=True, deterministic=True
        )
        print("Mean Reward:", mean_reward, "Standard Deviation Of Reward:", std_reward)

    def load_model(self, model_path):
        self.model = DQN.load(model_path)

    def plot_figures(self):
        self.plot_episodic_reward()
        self.plot_average_reward()

    def plot_episodic_reward(self):
        # plot the results
        _, y = ts2xy(load_results(self.log_dir), "timesteps")

        print(len(y), " Episodes")

        plt.plot(np.arange(len(y)), y)
        plt.xlabel("Episodes")
        plt.savefig(self.log_dir + "average.png")
        plt.close()

    def plot_average_reward(self):
        # plot the results
        _, y = ts2xy(load_results(self.log_dir), "timesteps")

        y_moving_average = []

        # computes average of y
        for i in range(len(y)):
            # get average moving up until ith element
            average = np.sum(y[: i + 1]) / (i + 1)
            y_moving_average.append(average)

        print(len(y), " Episodes")

        plt.plot(np.arange(len(y)), y_moving_average)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.savefig(self.log_dir + "moving_average.png")
        plt.close()

    def plot_episodic_deviation(self):
        _, y = ts2xy(load_results(self.log_dir), "timesteps")

        y_std = []

        # computes average of y
        for i in range(len(y)):
            # get average moving up until ith element
            std = np.std(y[: i + 1])
            y_std.append(std)

        plt.plot(np.arange(len(y)), y_std)
        plt.xlabel("Episodes")
        plt.savefig(self.log_dir + "std.png")
        plt.close()


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

    # env.run()
    # env.plot_figures()
    # env.plot_episodic_deviation()

    env.evaluate()