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
    "new_estimators":1,
    "lr_gb":2.3e-3,
    "depth":3,
    "gb_freq":1,
}


for seed in range(10):
    for new_estimator in [1, 10, 100]:
        params_cartpole_zoo["new_estimators"] = new_estimator
        env = gym.make("CartPole-v1")
        model = HybridDQN("HybridPolicy", env, seed=seed, **params_cartpole_zoo, tensorboard_log="./hybrid_dqn/")
        model.learn(1e5, tb_log_name=f"new_estimators_{new_estimator}_seed_{seed}")