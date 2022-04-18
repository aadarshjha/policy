# test the agent
import gym
import pickle
import os
import argparse
import random
import json
import numpy as np
import keras
import tensorflow as tf

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

env = gym.make("CartPole-v0")
observation = env.reset()

reward_history = []

# load model with tf
# config model network
json_file = open(PATH + "final_regular_model_architecture.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights(PATH + "final_regular_network_weights.h5")


model = loaded_model

# iterate through 100 episodes
for episode in range(100):
    observation = env.reset()
    episode_reward = 0
    # iterate through 200 steps
    done = False
    while done == False:
        action = np.argmax(model.predict(observation.reshape(1, 4)))
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            break
    print(episode_reward)
    reward_history.append(episode_reward)

with open(PATH + "test_episode_rewards.json", "w") as f:
    JSON_object = {
        "episode_rewards": reward_history,
        "average_reward": np.mean(reward_history),
    }
    json.dump(JSON_object, f)
