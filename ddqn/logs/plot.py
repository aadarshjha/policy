# plot the experiments
import json
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_episodes(file_name, episodes, scores, mean_scores, std_scores):
    plt.plot(np.arange(episodes), scores)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title("Episodic Scores")
    plt.savefig(file_name + "_episodic_scores.png")
    plt.close()

    plt.plot(np.arange(episodes), mean_scores)
    plt.xlabel("Episodes")
    plt.ylabel("Mean Score")
    plt.title("Mean Episodic Scores")
    plt.savefig(file_name + "_mean_scores.png")
    plt.close()

    plt.plot(np.arange(episodes), std_scores)
    plt.xlabel("Episodes")
    plt.ylabel("Standard Deviation")
    plt.title("Standard Deviation Of Scores")
    plt.savefig(file_name + "_std_scores.png")
    plt.close()



if __name__ == "__main__":
    file_names = [
        ["./SEED_0_DDQN/max_info.json", "seed_0_DDQN"],
        ["./SEED_0_DDQN_04/max_info.json", "SEED_0_DDQN_04"],
        ["./SEED_0_DDQN_06/max_info.json", "SEED_0_DDQN_06"],
        ["./SEED_123_DDQN/final_info.json", "seed_123_DDQN"],
    ]

    for element in file_names:
        file_name = element[0]

        # load the file test_seed_0/max_info.json
        with open(file_name, "r") as f:
            max_info = json.load(f)

        scores = max_info["scores"]
        mean_scores = max_info["mean_scores"]
        std_scores = max_info["std_scores"]

        episodes = len(scores)

        plot_episodes(element[1], episodes, scores, mean_scores, std_scores)
