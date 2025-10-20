# 好好学习 天天向上
# {2024/7/19} {15:16}
# import gymnasium as gym
# from gymnasium import envs
# env_names=[spec for spec in envs.registry.keys()]
# print(env_names)
# env = gym.make("CartPole-v1")
#
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()
import gymnasium as gym
import torch
from torch import nn
env = gym.make("Pendulum-v1", render_mode="human")
print(env.observation_space.shape)
print(env.action_space.shape)
observation, info = env.reset()
print(observation)
observation=torch.tensor(observation,dtype=torch.float)
for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward, terminated, truncated, info)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
NetConstruction=nn.Sequential(
            nn.Linear(3, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
        )
print(NetConstruction(observation))