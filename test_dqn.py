from stable_baselines3 import HybridDQN
import gymnasium as gym

env = gym.make("CartPole-v1")
model = HybridDQN(env)
model.learn(1e4)