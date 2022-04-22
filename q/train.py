import gym
import numpy as np
import argparse
from agent import AgentLearning
import json
import pickle
import os

#  collect seed argument and epsilon
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epsilon", type=float, default=0.1)
parser.add_argument("--exp", type=str, default="default")

args = parser.parse_args()
seed = args.seed
epsilon = args.epsilon
experience_name = args.exp


DRIVE = False

if DRIVE:
    PREFIX = "../../drive/MyDrive/policy/"
else:
    PREFIX = ""

PATH = PREFIX + "logs/" + experience_name + "/"

if not os.path.exists(PATH):
    os.makedirs(PATH)


def q_learning(env, agent):

    valid_actions = [0, 1]

    max_reward = 0
    scores = []
    mean_scores = []
    std_scores = []

    for episode in range(2000):  # 688 testing trials

        episode_rewards = 0
        obs = env.reset()
        agent.epsilon = agent.epsilon * 0.99

        for step in range(200):
            state = agent.create_state(obs)
            agent.create_Q(state, valid_actions)
            action = agent.choose_action(state)
            obs, reward, done, _ = env.step(action)
            print(reward)
            episode_rewards += reward

            if step != 0:
                agent.learn(state, action, prev_reward, prev_state, prev_action)
            prev_state = state
            prev_action = action
            prev_reward = reward
            if done:
                # Terminal state reached, reset environment
                break

        scores.append(episode_rewards)
        mean_scores.append(np.mean(scores))
        std_scores.append(np.std(scores))
        mean_score = np.mean(scores[-100:])

        if episode_rewards > max_reward:

            print("New max reward recorded: " + str(episode))

            max_reward = episode_rewards

            # save the q-table with pickle, create if does not exist
            with open(PATH + "max_q_table.pkl", "wb") as f:
                pickle.dump(agent.Q_table, f)

            # save a JSON file with the number of episodes so far and the max score
            with open(PATH + "max_info.json", "w") as f:
                JSON_object = {
                    "scores": scores,
                    "mean_scores": mean_scores,
                    "std_scores": std_scores,
                    "episodes": episode,
                }
                json.dump(JSON_object, f)

        # if we've reached the goal, reset the environment
        if mean_score >= 195:

            print("Solved in %d steps" % step)

            # make a pickle q table if it does not exist
            with open(PATH + "solved_q_table.pkl", "wb") as f:
                pickle.dump(agent.Q_table, f)

                # save scores, mean_scores, std_scores dump to JSON
            with open(PATH + "solved_scores.json", "w") as f:
                # dump to JSON
                JSON_object = {
                    "scores": scores,
                    "mean_scores": mean_scores,
                    "std_scores": std_scores,
                    "episodes": episode,
                }
                json.dump(JSON_object, f)

            return

        print(
            "Episode: {}, Average Reward: {}, Max Reward: {}".format(
                episode, np.mean(scores[-100:]), max_reward
            )
        )

    print("Unable to solve environment in %d episodes" % episode)

    # save the q-table with pickle
    with open(PATH + "final_q_table.pkl", "wb") as f:
        pickle.dump(agent.Q_table, f)

    # save scores, mean_scores, std_scores dump to JSON
    with open(PATH + "final_scores.json", "w") as f:
        # dump to JSON
        JSON_object = {
            "scores": scores,
            "mean_scores": mean_scores,
            "std_scores": std_scores,
            "episodes": episode,
        }
        json.dump(JSON_object, f)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env.seed(seed)
    # set numpy seed
    np.random.seed(seed)
    agent = AgentLearning(env, 0.1, epsilon=epsilon, gamma=0.9)
    q_learning(env, agent)
