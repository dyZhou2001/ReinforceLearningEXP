import gymnasium as gym
import numpy as np

def run_random_episode(env_id="Pendulum-v1"):
    env = gym.make(env_id)
    obs, _ = env.reset()
    total_reward = 0.0
    step = 0
    rewards = []
    done = False
    print(f"环境: {env_id}")
    while not done:
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, is_tr, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        step += 1
        print(f"第{step}步，动作: {action}, 奖励: {reward:.3f}")
        if is_tr:
            break
    avg_reward = total_reward / step if step > 0 else 0.0
    print(f"总步数: {step}")
    print(f"总奖励: {total_reward:.3f}")
    print(f"平均每步奖励: {avg_reward:.3f}")

if __name__ == "__main__":
    run_random_episode()
