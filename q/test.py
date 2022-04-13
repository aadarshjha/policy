# test the agent
import gym
import pickle
import os
import argparse
from agent import AgentLearning
import random
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="default")

args = parser.parse_args()
experience_name = args.exp

DRIVE = False

if DRIVE:
    PREFIX = "../../drive/MyDrive/policy/"
else:
    PREFIX = ""

PATH = PREFIX + "logs/" + experience_name + "/"

if not os.path.exists(PATH):
    os.makedirs(PATH)

Q_table = None

# if solved_q_table.pkl exists, load it
if os.path.exists(PATH + "solved_q_table.pkl"):
    with open(PATH + "solved_q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)
else:
    # load the max_q_table
    with open(PATH + "max_q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)

# use the Q_table to test the env
env = gym.make("CartPole-v0")
agent = AgentLearning(Q_table, env)


def choose_action(state):
    # Find max Q value
    # check if state does not exist Q_table
    if state not in Q_table:
        print("HELLO")
    max_Q = max(Q_table[state].values())
    actions = []
    for key, value in Q_table[state].items():
        if value == max_Q:
            actions.append(key)
        if len(actions) != 0:
            action = random.choice(actions)
    return action


def create_Q(state, valid_actions):
    if state not in Q_table:
        Q_table[state] = dict()
        for action in valid_actions:
            Q_table[state][action] = 0.0
    return


env = gym.make("CartPole-v0")
observation = env.reset()


reward_history = []

# iterate through 100 episodes
for episode in range(100):
    observation = env.reset()
    episode_reward = 0
    # iterate through 200 steps
    done = False
    while done == False:
        # env.render()
        action = choose_action(agent.create_state(observation))
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        create_Q(agent.create_state(observation), [0, 1])
        reward_history.append(episode_reward)
        if done:
            break
    # print the episode reward
    print(episode_reward)
    # save the episode reward
    with open(PATH + "episode_rewards.json", "w") as f:
        JSON_object = {"episode_rewards": reward_history, "average_reward": np.mean(reward_history)}
        json.dump(JSON_object, f)