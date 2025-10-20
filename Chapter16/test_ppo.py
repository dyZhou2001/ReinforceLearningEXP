#!/usr/bin/env python3
"""
PPO测试和演示脚本
用于加载训练好的模型并进行测试
"""

import torch
import gymnasium as gym
import numpy as np
from PPOALLINONE import Actor, PPOAgent
import argparse


def load_and_test_model(model_path, env_name="Pendulum-v1", num_episodes=10, render=False):
    """加载并测试训练好的模型"""
    
    # 创建环境
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)
    
    # 获取环境参数
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 创建智能体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(state_dim, action_dim, device)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    
    print(f"已加载模型: {model_path}")
    print(f"训练episode: {checkpoint.get('episode', 'Unknown')}")
    print(f"最佳奖励: {checkpoint.get('best_reward', 'Unknown')}")
    print("=" * 50)
    
    # 测试模型
    total_rewards = []
    total_steps = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # 获取动作（确定性策略，用于测试）
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.actor(state_tensor).cpu().numpy()[0]
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        
        print(f"Episode {episode + 1:2d}: Reward = {episode_reward:8.2f}, Steps = {episode_steps:3d}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_steps = np.mean(total_steps)
    
    print("=" * 50)
    print(f"测试完成！")
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"平均步数: {avg_steps:.1f}")
    print(f"最大奖励: {max(total_rewards):.2f}")
    print(f"最小奖励: {min(total_rewards):.2f}")
    
    env.close()
    return avg_reward, std_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./ppo_saves/best_model.pth", 
                       help="Path to the trained model")
    parser.add_argument("--env", default="Pendulum-v1", help="Environment name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()
    
    # 测试模型
    try:
        load_and_test_model(args.model_path, args.env, args.episodes, args.render)
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {args.model_path}")
        print("请先运行训练脚本生成模型文件。")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")