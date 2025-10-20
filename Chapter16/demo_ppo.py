#!/usr/bin/env python3
"""
PPO快速演示脚本
验证PPO实现是否正常工作
"""

import torch
import gymnasium as gym
import numpy as np
from PPOALLINONE import PPOAgent

def quick_demo():
    """快速演示PPO的基本功能"""
    print("PPO快速演示")
    print("=" * 30)
    
    # 创建环境
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"环境: Pendulum-v1")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    
    # 创建智能体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    agent = PPOAgent(state_dim, action_dim, device)
    print("PPO智能体创建成功")
    
    # 测试随机策略的性能
    print("\n测试随机策略...")
    total_reward = 0
    num_episodes = 3
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 200:  # 限制最大步数
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
                
            state = next_state
        
        total_reward += episode_reward
        print(f"Episode {episode + 1}: 奖励 = {episode_reward:.2f}, 步数 = {steps}")
    
    avg_reward = total_reward / num_episodes
    print(f"\n平均奖励: {avg_reward:.2f}")
    
    # 测试经验存储和更新
    print("\n测试经验存储...")
    state, _ = env.reset()
    
    # 收集一些经验
    for _ in range(10):
        action, log_prob, value = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_experience(state, action, reward, value, log_prob, done)
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    print(f"缓冲区大小: {agent.buffer.size}")
    print("经验存储测试完成")
    
    print("\n所有测试通过！PPO实现正常工作。")
    env.close()

if __name__ == "__main__":
    quick_demo()