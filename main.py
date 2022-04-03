# main.py
# Centralizes tesing experimentation
# Aadarsh Jha

import gym


class ExecuteEnviornment:
    def __init__(self):
        self.env = gym.make("CartPole-v0")

    def run(self):
        self.env.reset()
        for _ in range(1000):
            self.env.render()
            # take a random action
            self.env.step(self.env.action_space.sample())
        self.env.close()


if __name__ == "__main__":
    env = ExecuteEnviornment()
    env.run()
