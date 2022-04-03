import gym

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy

# import library to read from YAML file
import yaml
import argparse


class ExecuteTraining:
    def __init__(
        self,
        policy,
        env_name,
        learning_policy,
        learning_rate,
        seed,
        initial_epsilon,
        double_q,
    ):
        self.env = gym.make(env_name)
        if policy == "DQN":
            self.model = DQN(
                policy=learning_policy,
                learning_Rate=learning_rate,
                exploration_initial_eps=initial_epsilon,
                double_q=double_q,
                seed=seed,
            )

    def run(self):
        self.model.learn(total_timesteps=10000)
        self.model.save("cartpole_model")
        self.env.close()

    def load_model(self, model_path):
        self.model = DQN.load(model_path)


if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser(
        description="Train an RL agent on the CartPole-v0 environment."
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
        config["SEED"],
        config["INTIAL_EPSILON"],
        config["DOUBLE_Q"],
    )

    env.run()
