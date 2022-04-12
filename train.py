import gym
import math
import keras
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tf.keras.optimizers import Adam


class DQN:
    def __init__(self):
        self.max_score = 0
        self.n_episodes = 5000
        self.n_win_tick = 195
        self.max_env_steps = 1000

        self.gamma = 1.0
        self.epsilon = 1.0  # exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        self.alpha = 0.01  # learning rate
        self.alpha_decay = 0.01
        self.alpha_test_factor = 1.0

        self.batch_size = 256
        self.monitor = False
        self.quiet = False

        # environment Parameters
        self.memory = deque(maxlen=100000)
        self.env = gym.make("CartPole-v0")

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
            min(epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)),
        )


    def preprocess(self, state):
        return np.reshape(state, [1, 4])


    def replay(self, batch_size, epsilon):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = (
                reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            )
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if epsilon > self.epsilon_min:
            epsilon *= self.epsilon_decay


    # run function
    def run(self):
        global max_score
        scores = deque(maxlen=100)
        for e in range(self.n_episodes):
            print(e)
            if e > self.n_episodes - 2:
                global epsilon
                epsilon = 0.0
            state = self.preprocess(self.env.reset())
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                # env.render()
                next_state = self.preprocess(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
            if i > max_score:
                max_score = i
                # Save the weights
                self.model.save_weights(str(max_score) + "model_weights.h5")

                # Save the model architecture
                with open(str(max_score) + "model_architecture.json", "w") as f:
                    f.write(self.model.to_json())

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_tick and e >= 100:
                if not self.quiet:
                    print(
                        "Ran "
                        + str(e)
                        + " episodes. Solved after "
                        + str(e - 100)
                        + "trials"
                    )
                # Save the weights
                self.model.save_weights(str(max_score) + "final_model_weights.h5")

                # Save the model architecture
                with open(str(max_score) + "final_model_architecture.json", "w") as f:
                    f.write(self.model.to_json())

                return e - 100
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
        return e

# Training the network
if __name__ == "__main__":
    agent = DQN()
    agent.run()
