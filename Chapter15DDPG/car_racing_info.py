import gymnasium as gym

# 创建环境
env = gym.make("CarRacing-v2", render_mode="human")  # render_mode 可选 "human" 或 "rgb_array"

print("环境名:", env.spec.id)
print("观测空间:", env.observation_space)
print("动作空间:", env.action_space)
print("奖励范围:", env.reward_range)
print("最大 episode 步数:", env.spec.max_episode_steps)

# 重置环境，获取初始观测
obs, info = env.reset()
print("\n初始观测 shape:", obs.shape)
print("初始观测 dtype:", obs.dtype)
print("reset info:", info)

# 随机采样一个动作并执行
action = env.action_space.sample()
print("\n随机动作:", action)

next_obs, reward, terminated, truncated, info = env.step(action)
print("下一观测 shape:", next_obs.shape)
print("奖励:", reward)
print("是否终止:", terminated)
print("是否截断:", truncated)
print("step info:", info)

env.close()
