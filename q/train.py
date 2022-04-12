import argparse
import random
import pandas as pd
import numpy as np
import argparse
import os
import yaml
import gym

# parse the command line for a file argument
parser = argparse.ArgumentParser()
parser.add_argument("--file", help="the file to train on")
args = parser.parse_args()

# read yaml file
with open("./experiments/" + args.file, "r") as ymlfile:
    cfg = yaml.load(ymlfile)

seed = cfg["SEED"]

np.random.seed(seed)
random.seed(seed)

if cfg["DRIVE"]:
    PREFIX = "../../drive/MyDrive/policy/"
else:
    PREFIX = ""

PATH = PREFIX + "logs/" + cfg["EXP_NAME"] + "/"


class Q(object):
    def __init__(
        self,
        N_EPISODES,
        GAMMA,
        EPSILON,
        ALPHA,
        ENV_NAME,
    ):
        self.n_episodes = N_EPISODES
        self.env = gym.make(ENV_NAME)
        self.alpha = ALPHA  # Learning factor
        self.epsilon = EPSILON
        self.gamma = GAMMA  # Discount factor
        self.Q_table = dict()
        # Following variables for statistics
        self.training_trials = 0
        self.testing_trials = 0

    def build_state(self, features):
        """Build state by concatenating features (bins) into 4 digit int."""
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def create_state(self, obs):
        """Create state variable from observation.
        Args:
            obs: Observation list with format [horizontal position, velocity,
                 angle of pole, angular velocity].
        Returns:
            state: State tuple
        """
        cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
        pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
        cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
        angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]
        state = self.build_state(
            [
                np.digitize(x=[obs[0]], bins=cart_position_bins)[0],
                np.digitize(x=[obs[1]], bins=pole_angle_bins)[0],
                np.digitize(x=[obs[2]], bins=cart_velocity_bins)[0],
                np.digitize(x=[obs[3]], bins=angle_rate_bins)[0],
            ]
        )
        return state

    def choose_action(self, state):
        """Given a state, choose an action.
        Args:
            state: State of the agent.
        Returns:
            action: Action that agent will take.
        """
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            # Find max Q value
            max_Q = self.get_maxQ(state)
            actions = []
            for key, value in self.Q_table[state].items():
                if value == max_Q:
                    actions.append(key)
            if len(actions) != 0:
                action = random.choice(actions)
        return action

    def create_Q(self, state, valid_actions):
        """Update the Q table given a new state/action pair.
        Args:
            state: List of state booleans.
            valid_actions: List of valid actions for environment.
        """
        if state not in self.Q_table:
            self.Q_table[state] = dict()
            for action in valid_actions:
                self.Q_table[state][action] = 0.0
        return

    def get_maxQ(self, state):
        """Find the maximum Q value in a given Q table.
        Args:
            Q_table: Q table dictionary.
            state: List of state booleans.
        Returns:
            maxQ: Maximum Q value for a given state.
        """
        maxQ = max(self.Q_table[state].values())
        return maxQ

    def learn(self, state, action, prev_reward, prev_state, prev_action):
        """Update the Q-values
        Args:
            state: State at current time step.
            action: Action at current time step.
            prev_reward: Reward at previous time step.
            prev_state: State at previous time step.
            prev_action: Action at previous time step.
        """
        # Updating previous state/action pair so I can use the 'future state'
        self.Q_table[prev_state][prev_action] = (1 - self.alpha) * self.Q_table[
            prev_state
        ][prev_action] + self.alpha * (
            prev_reward + (self.gamma * self.get_maxQ(state))
        )
        return

    def run(self):
        valid_actions = [0, 1]
        training_totals = []
        testing_totals = []
        history = {"epsilon": [], "alpha": []}

        for _ in range(self.n_episodes):  # 688 testing trials
            episode_rewards = 0
            obs = self.env.reset()
            agent.epsilon = agent.epsilon * 0.99  # 99% of epsilon value
            for step in range(200):  # 200 steps max
                state = self.create_state(obs)  # Get state
                agent.create_Q(state, valid_actions)  # Create state in Q_table
                action = self.choose_action(state)  # Choose action
                obs, reward, done, info = self.env.step(action)  # Do action
                print(reward)
                episode_rewards += reward

                if step != 0:
                    self.learn(state, action, prev_reward, prev_state, prev_action)

                prev_state = state
                prev_action = action
                prev_reward = reward

                if done:
                    break

            training_totals.append(episode_rewards)
            agent.training_trials += 1
            history["epsilon"].append(agent.epsilon)
            history["alpha"].append(agent.alpha)

        return training_totals, testing_totals, history


if __name__ == "__main__":
    # always create a logs folder
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    agent = Q(
        cfg["N_EPISODES"],
        cfg["GAMMA"],
        cfg["EPSILON"],
        cfg["ALPHA"],
        cfg["ENV_NAME"],
    )

    agent.run()
