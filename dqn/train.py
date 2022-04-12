import gym
import math
import keras
import random
import numpy as np
import tensorflow
import os
import yaml
import argparse
from collections import deque
import json
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# parse the command line for a file argument
parser = argparse.ArgumentParser()
parser.add_argument("--file", help="the file to train on")
args = parser.parse_args()

# read yaml file
with open("./experiments/" + args.file, "r") as ymlfile:
    cfg = yaml.load(ymlfile)

seed = cfg["SEED"]

tensorflow.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

if cfg["DRIVE"]:
    PREFIX = "../../drive/MyDrive/policy/"
else:
    PREFIX = ""

PATH = PREFIX + "logs/" + cfg["EXP_NAME"] + "/"


class DQN:
    def __init__(
        self,
        N_EPISODES,
        GAMMA,
        EPSILON,
        EPSILON_MIN,
        EPSILON_DECAY,
        ALPHA,
        ALPHA_DECAY,
        BATCH_SIZE,
        ENV_NAME,
        seed,
    ):

        self.max_score = 0
        self.n_episodes = N_EPISODES
        self.n_win_tick = 195
        self.max_env_steps = 1000

        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.alpha = ALPHA
        self.alpha_decay = ALPHA_DECAY
        self.alpha_test_factor = 1.0

        self.batch_size = BATCH_SIZE
        self.monitor = False
        self.quiet = False

        # environment Parameters
        self.memory = deque(maxlen=100000)
        self.env = gym.make(ENV_NAME)

        # set seed of gym env
        self.env.seed(seed)

        if self.max_env_steps is not None:
            self.env._max_episode_steps = self.max_env_steps

        self.model = Sequential(
            [
                Dense(24, input_shape=(4,), activation="relu"),
                Dense(48, activation="relu"),
                Dense(96, activation="relu"),
                Dense(48, activation="relu"),
                Dense(24, activation="relu"),
                Dense(2, activation="relu"),
            ]
        )

        self.model.compile(
            loss="mse", optimizer=Adam(lr=self.alpha, decay=self.alpha_decay)
        )

    # Define functions
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        return max(
            self.epsilon_min,
            min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)),
        )

    def preprocess(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size, epsilon):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = (
                reward
                if done
                else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            )
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        self.model.fit(
            np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0
        )
        if epsilon > self.epsilon_min:
            epsilon *= self.epsilon_decay

    # run function
    def run(self):
        scores = []
        mean_scores = [] 
        std_scores = [] 

        for e in range(self.n_episodes):
            print("Episode: ", e)
            if e > self.n_episodes - 2:
                global epsilon
                epsilon = 0.0
            state = self.preprocess(self.env.reset())
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
            if i > self.max_score:

                self.max_score = i

                self.model.save_weights(PATH + "weights.h5")
                with open(PATH + "model_architecture.json", "w") as f:
                    f.write(self.model.to_json())

                # save a JSON file with the number of episodes so far and the max score
                with open(PATH + "info.json", "w") as f:
                    f.write(
                        '{"episodes":'
                        + str(e)
                        + ',"max_score":'
                        + str(self.max_score)
                        + "}"
                    )

            scores.append(i)
            mean_score = np.mean(scores)
            mean_scores.append(mean_score)
            std_scores.append(np.std(scores))

            if mean_score >= self.n_win_tick and e >= 100:
                if not self.quiet:
                    print(
                        "Ran "
                        + str(e)
                        + " episodes. Solved after "
                        + str(e)
                        + "trials"
                    )
                # Save the weights
                self.model.save_weights(PATH +  "final_model_weights.h5")

                # Save the model architecture
                with open(PATH + "final_model_architecture.json", "w") as f:
                    f.write(self.model.to_json())
                
                # save scores, mean_scores, std_scores dump to JSON
                with open(PATH + "scores.json", "w") as f:
                    # dump to JSON
                    JSON_object = {
                        "scores": scores,
                        "mean_scores": mean_scores,
                        "std_scores": std_scores,
                        "episodes": e,
                    }
                    json.dump(JSON_object, f)
                return e

            if e % 100 == 0 and not self.quiet:
                print(
                    "episode "
                    + str(e)
                    + " mean survival time over last 100 episodes was "
                    + str(mean_score)
                    + " ticks"
                )
            self.replay(self.batch_size, self.get_epsilon(e))
        if not self.quiet:
            print("did not solve after " + str(e) + " episodes")

        # save the scores 
        with open(PATH + "_failed_scores.json", "w") as f:
            # dump to JSON
            JSON_object = {
                "scores": scores,
                "mean_scores": mean_scores,
                "std_scores": std_scores,
                "episodes": e
            }
            json.dump(JSON_object, f)
        
        # save the weights and architecturer 
        self.model.save_weights(PATH + "failed_model_weights.h5")
        with open(PATH + "failed_model_architecture.json", "w") as f:
            f.write(self.model.to_json())
        return e


# Training the network
if __name__ == "__main__":

    # always create a logs folder
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    agent = DQN(
        cfg["N_EPISODES"],
        cfg["GAMMA"],
        cfg["EPSILON"],
        cfg["EPSILON_MIN"],
        cfg["EPSILON_DECAY"],
        cfg["ALPHA"],
        cfg["ALPHA_DECAY"],
        cfg["BATCH_SIZE"],
        cfg["ENV_NAME"],
        seed,
    )
    agent.run()