import os
import numpy as np

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class PlotAverageAndStdDev(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(PlotAverageAndStdDev, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.n_episodes = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # increment the self.ep_counter
        self.n_episodes += np.sum(self.locals["dones"]).item()

        # by default is at 1, so we save everything.
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")

            if len(x) > 0:
                mean_reward = np.mean(y[-100:])

                if mean_reward > 195.0:
                    # terminate the training, save the model.
                    print(
                        f"Termination critereion reached, saving... {self.save_path}.zip"
                    )
                    self.model.save(self.save_path)
                    # stops the training.
                    return False

                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Num episodes: {self.n_episodes}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} | Mean over the last 100 episodes: {mean_reward:.2f}"
                    )

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True
