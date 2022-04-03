import gym

import stable_baselines3
from stable_baselines3 import DQN
print(stable_baselines3.__version__)
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
        initial_epsilon,
        seed,
        verbose,
    ):
        self.env = gym.make(env_name)
        if policy == "DQN":
            self.model = DQN(
                policy=learning_policy,
                env=self.env,
                learning_rate=learning_rate,
                exploration_initial_eps=initial_epsilon,
                seed=seed,
                verbose=verbose,
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
        config["INTIAL_EPSILON"],
        config["SEED"],
        config["VERBOSE"],
    )

    env.run()
