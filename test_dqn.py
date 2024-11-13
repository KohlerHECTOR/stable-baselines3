from stable_baselines3 import HybridDQN
import gymnasium as gym

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

env = gym.make("CartPole-v1")
model = HybridDQN(env, gb_freq=5, verbose=1, **params_cartpole_zoo)
model.learn(5e4)