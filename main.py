import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from argparse import ArgumentParser
import yaml
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

class DQN:
    def __init__(
        self,
        ENV_NAME,
        GAMMA,
        LEARNING_RATE,
        MEMORY_SIZE,
        BATCH_SIZE,
        EXPLORATION_MAX,
        EXPLORATION_MIN,
        EXPLORATION_DECAY,
    ):
        self.exploration_rate = EXPLORATION_MAX
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.exploration_min = EXPLORATION_MIN
        self.exploration_decay = EXPLORATION_DECAY
        self.env_name = ENV_NAME
        self.env = gym.make(ENV_NAME)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

        # seed the env
        self.env.seed(seed)

        self.model = Sequential(
            [
                Dense(24, input_shape=(self.observation_space,), activation="relu"),
                Dense(24, activation="relu"),
                Dense(self.action_space, activation="linear"),
            ]
        )
        self.model.compile(
            loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=["accuracy"]
        )

        # print a summary of the model
        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = reward + self.gamma * np.amax(
                    self.model.predict(state_next)[0]
                )
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def train(self):
        env = gym.make(self.env_name)
        run = 0

        episodic_rewards = []
        average_episodic_rewards = []
        standard_dev_rewards = [] 

        while True:

            # if the average of the episodic_rewards is greater than or equal to 195, over the past 100 episodes, stop training
            if len(episodic_rewards) >= 100:
                if sum(episodic_rewards[-100:]) / 100 >= 195:
                    # save the model
                    self.model.save("best_model.h5")
                    break

            run += 1
            state = env.reset()
            state = np.reshape(state, [1, self.observation_space])
            step = 0
            while True:
                step += 1
                # env.render()
                action = self.act(state)
                state_next, reward, terminal, _ = env.step(action)
                reward = reward if not terminal else -reward
                state_next = np.reshape(state_next, [1, self.observation_space])
                self.remember(state, action, reward, state_next, terminal)
                state = state_next
                if terminal:
                    # save episodic reward 
                    episodic_rewards.append(step)

                    # save average episodic reward
                    average_episodic_rewards.append(np.mean(episodic_rewards))

                    print(
                        "Episode: "
                        + str(run)
                        + ", exploration: "
                        + str(self.exploration_rate)
                        + ", score: "
                        + str(step)
                    )
                    break
                self.experience_replay()

        print("Training terminated.")


if __name__ == "__main__":


    # argparse to fetch the yaml config file
    parser = ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # seeding
    seed = config["SEED"]
    np.random.seed(seed)
    random.seed(seed)
    # set seed of tf
    tf.random.set_seed(seed)

    DQNAgent = DQN(
        ENV_NAME=config["ENV_NAME"],
        GAMMA=config["GAMMA"],
        LEARNING_RATE=config["LEARNING_RATE"],
        MEMORY_SIZE=config["MEMORY_SIZE"],
        BATCH_SIZE=config["BATCH_SIZE"],
        EXPLORATION_MAX=config["EXPLORATION_MAX"],
        EXPLORATION_MIN=config["EXPLORATION_MIN"],
        EXPLORATION_DECAY=config["EXPLORATION_DECAY"],
    )

    DQNAgent.train()
