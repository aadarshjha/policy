import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from matplotlib import pyplot as plt
import os
import argparse
import json

#  collect seed argument and epsilon
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epsilon", type=float, default=0.1)
parser.add_argument("--exp", type=str, default="default")

args = parser.parse_args()
seed = args.seed
epsilon = args.epsilon
experience_name = args.exp

# seed tensorflow
tf.random.set_seed(seed)
# seed random
random.seed(seed)
# seed numpy
np.random.seed(seed)

DRIVE = False

if DRIVE:
    PREFIX = "../../drive/MyDrive/policy/"
else:
    PREFIX = ""

PATH = PREFIX + "logs/" + experience_name + "/"

if not os.path.exists(PATH):
    os.makedirs(PATH)

# CARTPOLE GAME SETTINGS
OBSERVATION_SPACE_DIMS = 4
ACTION_SPACE = [0, 1]

# AGENT/NETWORK HYPERPARAMETERS
EPSILON_INITIAL = epsilon
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
ALPHA = 0.001  # learning rate
GAMMA = 0.99  # discount factor
TAU = 0.1  # target network soft update hyperparameter
EXPERIENCE_REPLAY_BATCH_SIZE = 32
AGENT_MEMORY_LIMIT = 2000
MIN_MEMORY_FOR_EXPERIENCE_REPLAY = 500


def create_dqn():

    model = Sequential(
        [
            Dense(24, input_shape=(4,), activation="relu"),
            Dense(48, activation="relu"),
            Dense(96, activation="relu"),
            Dense(48, activation="relu"),
            Dense(24, activation="relu"),
            Dense(2, activation="relu"),
        ]
    )

    model.compile(loss="mse", optimizer=Adam(lr=ALPHA))

    return model


class DoubleDQNAgent(object):
    def __init__(self):
        self.memory = []
        self.online_network = create_dqn()
        self.target_network = create_dqn()
        self.epsilon = EPSILON_INITIAL
        self.has_talked = False

    def act(self, state):
        if self.epsilon > np.random.rand():
            # explore
            return np.random.choice(ACTION_SPACE)
        else:
            # exploit
            state = self._reshape_state_for_net(state)
            q_values = self.online_network.predict(state)[0]
            return np.argmax(q_values)

    def experience_replay(self):

        minibatch = random.sample(self.memory, EXPERIENCE_REPLAY_BATCH_SIZE)
        minibatch_new_q_values = []

        for experience in minibatch:
            state, action, reward, next_state, done = experience
            state = self._reshape_state_for_net(state)
            experience_new_q_values = self.online_network.predict(state)[0]
            if done:
                q_update = reward
            else:
                next_state = self._reshape_state_for_net(next_state)
                # using online network to SELECT action
                online_net_selected_action = np.argmax(
                    self.online_network.predict(next_state)
                )
                # using target network to EVALUATE action
                target_net_evaluated_q_value = self.target_network.predict(next_state)[
                    0
                ][online_net_selected_action]
                q_update = reward + GAMMA * target_net_evaluated_q_value
            experience_new_q_values[action] = q_update
            minibatch_new_q_values.append(experience_new_q_values)
        minibatch_states = np.array([e[0] for e in minibatch])
        minibatch_new_q_values = np.array(minibatch_new_q_values)
        self.online_network.fit(
            minibatch_states, minibatch_new_q_values, verbose=False, epochs=1
        )

    def update_target_network(self):
        q_network_theta = self.online_network.get_weights()
        target_network_theta = self.target_network.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta, target_network_theta):
            target_weight = target_weight * (1 - TAU) + q_weight * TAU
            target_network_theta[counter] = target_weight
            counter += 1
        self.target_network.set_weights(target_network_theta)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) <= AGENT_MEMORY_LIMIT:
            experience = (state, action, reward, next_state, done)
            self.memory.append(experience)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)

    def _reshape_state_for_net(self, state):
        return np.reshape(state, (1, OBSERVATION_SPACE_DIMS))


def train():
    env = gym.make("CartPole-v0")
    env.seed(seed)
    MAX_TRAINING_EPISODES = 2000
    MAX_STEPS_PER_EPISODE = 200

    max_score = 0
    scores = []
    mean_scores = []
    std_scores = []

    agent = DoubleDQNAgent()

    for episode_index in range(0, MAX_TRAINING_EPISODES):
        state = env.reset()
        episode_score = 0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            episode_score += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > MIN_MEMORY_FOR_EXPERIENCE_REPLAY:
                agent.experience_replay()
                agent.update_target_network()
            if done:
                break

        scores.append(episode_score)
        mean_scores.append(np.mean(scores))
        std_scores.append(np.std(scores))

        mean_score = np.mean(scores[-100:])
        agent.update_epsilon()

        # if the score is maximum
        if episode_score > max_score:
            max_score = episode_score

            # save the maximum score
            print("Max found at {} trials, Reward {}".format(episode_index, max_score))

            # Save the weights
            agent.online_network.save_weights(
                PATH + "max_online_final_model_weights.h5"
            )

            # Save the model architecture
            with open(PATH + "max_online_model_architecture.json", "w") as f:
                f.write(agent.target_network.to_json())

            # Save the weights
            agent.online_network.save_weights(
                PATH + "max_target_final_model_weights.h5"
            )

            # Save the model architecture
            with open(PATH + "max_target_model_architecture.json", "w") as f:
                f.write(agent.target_network.to_json())

            # save scores, mean_scores, std_scores dump to JSON
            with open(PATH + "max_scores.json", "w") as f:
                # dump to JSON
                JSON_object = {
                    "scores": scores,
                    "mean_scores": mean_scores,
                    "std_scores": std_scores,
                    "episodes": episode_index,
                }
                json.dump(JSON_object, f)

        print(
            "Episode %d scored %d, avg %.2f"
            % (episode_index, episode_score, mean_score)
        )

        # if we reach convregence
        if mean_score >= 195 and episode_index > 100:
            print("Env solved in {} trials".format(episode_index))

            # Save the weights
            agent.online_network.save_weights(PATH + "online_solved_model_weights.h5")

            # Save the model architecture
            with open(PATH + "solved_online_model_architecture.json", "w") as f:
                f.write(agent.target_network.to_json())

            # Save the weights
            agent.online_network.save_weights(PATH + "target_solved_model_weights.h5")

            # Save the model architecture
            with open(PATH + "solved_target_model_architecture.json", "w") as f:
                f.write(agent.target_network.to_json())

            # save scores, mean_scores, std_scores dump to JSON
            with open(PATH + "solved_scores.json", "w") as f:
                # dump to JSON
                JSON_object = {
                    "scores": scores,
                    "mean_scores": mean_scores,
                    "std_scores": std_scores,
                    "episodes": episode_index,
                }
                json.dump(JSON_object, f)
            return

    print("Failed to solve in {} trials".format(episode_index))

    # Save the weights
    agent.online_network.save_weights(PATH + "online_failed_model_weights.h5")

    # Save the model architecture
    with open(PATH + "failed_online_model_architecture.json", "w") as f:
        f.write(agent.target_network.to_json())

    # Save the weights
    agent.online_network.save_weights(PATH + "target_failed_model_weights.h5")

    # Save the model architecture
    with open(PATH + "failed_target_model_architecture.json", "w") as f:
        f.write(agent.target_network.to_json())

    # save scores, mean_scores, std_scores dump to JSON
    with open(PATH + "failed_scores.json", "w") as f:
        # dump to JSON
        JSON_object = {
            "scores": scores,
            "mean_scores": mean_scores,
            "std_scores": std_scores,
            "episodes": episode_index,
        }
        json.dump(JSON_object, f)

    return


if __name__ == "__main__":
    trials = train()
