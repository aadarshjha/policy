# Imports and gym creation
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow
from tensorflow import keras
import random
import gym
import argparse
import yaml
import tensorflow
import os

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epsilon", type=float, default=0.1)
parser.add_argument("--exp", type=str, default="default")

args = parser.parse_args()
seed = args.seed
epsilon = args.epsilon
experience_name = args.exp

tensorflow.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


DRIVE = True

if DRIVE:
    PREFIX = "../../drive/MyDrive/policy/"
else:
    PREFIX = ""


PATH = PREFIX + "logs/" + experience_name + "/"
print(PATH)

if not os.path.exists(PATH):
    os.makedirs(PATH)


envCartPole = gym.make("CartPole-v0")

EPISODES = 1000
TRAIN_END = 0


def discount_rate():  # Gamma
    return 0.95


def learning_rate():  # Alpha
    return 0.001


def batch_size():
    return 24


class DoubleDeepQNetwork:
    def __init__(
        self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay
    ):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.model_target = self.build_model()  # Second (target) neural network
        self.update_target_from_model()  # Update weights
        self.loss = []

    def build_model(self):
        model = keras.Sequential(
            [
                keras.layers.Dense(24, input_dim=self.nS, activation="relu"),
                keras.layers.Dense(48, activation="relu"),
                keras.layers.Dense(96, activation="relu"),
                keras.layers.Dense(48, activation="relu"),
                keras.layers.Dense(24, activation="relu"),
                keras.layers.Dense(self.nA, activation="relu"),
            ]
        )

        model.compile(
            loss="mean_squared_error",  # Loss function: Mean Squared Error
            optimizer=keras.optimizers.Adam(lr=self.alpha),
        )  # Optimaizer: Adam (Feel free to check other options)
        return model

    def update_target_from_model(self):
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)  # Explore
        action_vals = self.model.predict(
            state
        )  # Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def test_action(self, state):  # Exploit
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, nstate, done))

    def experience_replay(self, batch_size):
        # Execute the experience replay
        minibatch = random.sample(
            self.memory, batch_size
        )  # Randomly sample from memory

        # Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch)
        st = np.zeros((0, self.nS))  # States
        nst = np.zeros((0, self.nS))  # Next States
        for i in range(len(np_array)):  # Creating the state and next state np arrays
            st = np.append(st, np_array[i, 0], axis=0)
            nst = np.append(nst, np_array[i, 3], axis=0)
        st_predict = self.model.predict(
            st
        )  # Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.model_target.predict(nst)  # Predict from the TARGET
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            # Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            if (
                done == True
            ):  # Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:  # Non terminal
                target = (
                    reward
                    + self.gamma
                    * nst_action_predict_target[np.argmax(nst_action_predict_model)]
                )  # Using Q to get T is Double DQN
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        # Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size, self.nS)
        y_reshape = np.array(y)
        epoch_count = 1
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        # Graph Losses
        for i in range(epoch_count):
            self.loss.append(hist.history["loss"][i])
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Create the agents
nS = envCartPole.observation_space.shape[0]  # This is only 4
nA = envCartPole.action_space.n  # Actions
dqn = DoubleDeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.001, 0.995)

batch_size = batch_size()

max_reward = 0
scores = []
mean_scores = []
std_scores = []

TEST_Episodes = 0
for e in range(EPISODES):
    state = envCartPole.reset()
    state = np.reshape(state, [1, nS])  # Resize to store in memory to pass to .predict
    tot_rewards = 0
    for time in range(
        200
    ):  # 200 is when you "solve" the game. This can continue forever as far as I know
        action = dqn.action(state)
        nstate, reward, done, _ = envCartPole.step(action)
        nstate = np.reshape(nstate, [1, nS])
        tot_rewards += reward
        dqn.store(
            state, action, reward, nstate, done
        )  # Resize to store in memory to pass to .predict
        state = nstate

        if done:
            break
        # Experience Replays
        if len(dqn.memory) > batch_size:
            dqn.experience_replay(batch_size)

    # store all the values
    scores.append(tot_rewards)
    mean_scores.append(np.mean(scores))
    std_scores.append(np.std(scores))

    mean_score = np.mean(scores[-100:])

    # we save if its a max
    if tot_rewards > max_reward:

        max_reward = tot_rewards

        print("Maximum Reward Achieved, " + str(e) + "Reward, " + str(tot_rewards) + "\n")

        dqn.model.save_weights(PATH + "max_regular_network_weights.h5")
        dqn.model_target.save_weights(PATH + "max_target_network_weights.h5")

        with open(PATH + "max_regular_model_architecture.json", "w") as f:
            f.write(dqn.model.to_json())

        with open(PATH + "max_target_model_architecture.json", "w") as f:
            f.write(dqn.model_target.to_json())

        # save a JSON file with the number of episodes so far and the max score
        with open(PATH + "max_info.json", "w") as f:
            f.write('{"episodes":' + str(e) + ',"max_score":' + str(max_reward) + "}")

    if len(scores) > 100 and np.average(scores[-100:]) > 195:
        dqn.model.save_weights(PATH + "solved_regular_network_weights.h5")
        dqn.model_target.save_weights(PATH + "solved_target_network_weights.h5")

        with open(PATH + "solved_regular_model_architecture.json", "w") as f:
            f.write(dqn.model.to_json())

        with open(PATH + "solved_target_model_architecture.json", "w") as f:
            f.write(dqn.model_target.to_json())

        # save a JSON file with the number of episodes so far and the max score
        with open(PATH + "solved_info.json", "w") as f:
            f.write(
                '{"episodes":' + str(e) + ',"solved_score":' + str(max_reward) + "}"
            )
        break

    print(
        "Episode: {}/{}".format(e, EPISODES),
        "Total Reward: {}".format(tot_rewards),
        "Mean Reward (Last 100): {}".format(mean_score),
    )
    dqn.update_target_from_model()

if e > EPISODES:
    print("Failed to solved")

    dqn.model.save_weights(PATH + "failed_regular_network_weights.h5")
    dqn.model_target.save_weights(PATH + "failed_target_network_weights.h5")

    with open(PATH + "failed_regular_model_architecture.json", "w") as f:
        f.write(dqn.model.to_json())

    with open(PATH + "failed_target_model_architecture.json", "w") as f:
        f.write(dqn.model_target.to_json())
