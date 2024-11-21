from stable_baselines3 import HybridDQN, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import numpy as np

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, label=""):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]
    plt.plot(x, y, label=label)
    
    

params_cartpole_zoo = {
    "learning_rate": 2.3e-3,
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "gamma": 0.99,
    "target_update_interval": 10,
    "train_freq": 256,
    "gradient_steps": 128,
    "exploration_fraction": 0.16,
    "exploration_final_eps": 0.04,
    "policy_kwargs": dict(net_arch=[256, 256]),
}


# Create log dir
log_dir_dqn = "continual_learning_exps/DQN/"
log_dir_hdqn = "continual_learning_exps/HDQN/"

os.makedirs(log_dir_dqn, exist_ok=True)
os.makedirs(log_dir_hdqn, exist_ok=True)


# Create and wrap the environment
# Logs will be saved in log_dir/monitor.csv
env = gym.make("CartPole-v1")
env_dqn = Monitor(env, log_dir_dqn)
env_hdqn = Monitor(env, log_dir_hdqn)

model1 = DQN("MlpPolicy", env_dqn, seed=42, **params_cartpole_zoo)
model1.learn(1e5)

params_cartpole_zoo["policy_kwargs"].update(dict(new_estimators=1, lr_gb=2.3e-3, depth=3))
model2 = HybridDQN("HybridPolicy", env_hdqn, gb_freq=1, seed=42, **params_cartpole_zoo)
model2.learn(1e5)



# Helper from the library
plot_results(log_dir_dqn, label="DQN")
plot_results(log_dir_hdqn, label="HDQN")


plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.title("DQN vs HybridDQN" + " Smoothed")
plt.legend()
plt.savefig("comparison_gb_freq")