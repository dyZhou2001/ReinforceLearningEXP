#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) 完整实现
所有相关代码整合在一个文件中，无需ptan库，使用倒立摆环境
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from collections import deque
import argparse

# ================================
# 超参数配置
# ================================
GAMMA = 0.99              # 折扣因子
GAE_LAMBDA = 0.95         # GAE参数
TRAJECTORY_SIZE = 2048    # 轨迹长度
LEARNING_RATE_ACTOR = 3e-4    # Actor学习率
LEARNING_RATE_CRITIC = 1e-3   # Critic学习率
PPO_EPS = 0.2            # PPO截断参数
PPO_EPOCHS = 10          # PPO更新轮数
PPO_BATCH_SIZE = 64      # 批量大小
TEST_ITERS = 10000       # 测试间隔
ENTROPY_COEF = 0.01      # 熵系数，鼓励探索
VALUE_LOSS_COEF = 0.5    # 价值损失系数

# ================================
# 神经网络模型定义
# ================================

class Actor(nn.Module):
    """Actor网络：输出动作的均值和标准差"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        # 构建前向传播网络
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出tanh激活，动作范围在[-1, 1]
        )
        # 学习标准差参数（对数形式，保证为正）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        """前向传播，输出动作均值"""
        mu = self.fc(state)
        return mu
    
    def get_action_and_logprob(self, state):
        """获取动作和对数概率"""
        mu = self.forward(state)
        std = torch.exp(self.log_std.clamp(-20, 2))  # 限制标准差范围
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        # 将动作限制在[-1, 1]范围内
        action = torch.clamp(action, -1.0, 1.0)
        return action, log_prob
    
    def get_logprob_and_entropy(self, state, action):
        """给定状态和动作，计算对数概率和熵"""
        mu = self.forward(state)
        std = torch.exp(self.log_std.clamp(-20, 2))
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Critic网络：估计状态价值函数V(s)"""
    
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        # 构建价值网络
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """前向传播，输出状态价值"""
        return self.fc(state)


# ================================
# 经验收集和存储
# ================================

class PPOBuffer:
    """PPO经验缓冲区，用于存储轨迹数据"""
    
    def __init__(self, max_size, state_dim, action_dim, device):
        self.max_size = max_size
        self.device = device
        
        # 初始化缓冲区
        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(max_size, dtype=torch.float32, device=device)
        
        self.ptr = 0
        self.size = 0
    
    def store(self, state, action, reward, value, log_prob, done):
        """存储一步经验"""
        self.states[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.actions[self.ptr] = torch.FloatTensor(action).to(self.device)
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def get(self):
        """获取所有数据并计算优势和回报"""
        assert self.size == self.max_size
        
        # 计算GAE优势
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        gae = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = 0  # 轨迹结束，下一个价值为0
                next_done = 1
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]
            
            # 计算TD误差
            delta = self.rewards[t] + GAMMA * next_value * (1 - next_done) - self.values[t]
            # 计算GAE
            gae = delta + GAMMA * GAE_LAMBDA * (1 - next_done) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 重置缓冲区
        self.ptr = 0
        self.size = 0
        
        return (self.states, self.actions, self.log_probs, 
                returns, advantages, self.values)


# ================================
# PPO智能体
# ================================

class PPOAgent:
    """PPO智能体类"""
    
    def __init__(self, state_dim, action_dim, device='cuda'):
        self.device = device
        
        # 初始化网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE_CRITIC)
        
        # 初始化经验缓冲区
        self.buffer = PPOBuffer(TRAJECTORY_SIZE, state_dim, action_dim, device)
        
    def get_action(self, state):
        """根据当前策略选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取动作和对数概率
            action, log_prob = self.actor.get_action_and_logprob(state)
            # 获取状态价值
            value = self.critic(state)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def get_deterministic_action(self, state):
        """在评估时根据当前策略选择确定性动作（均值）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu = self.actor(state)
        return mu.cpu().numpy()[0]

    def store_experience(self, state, action, reward, value, log_prob, done):
        """存储经验到缓冲区"""
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def update(self):
        """使用PPO算法更新策略"""
        if self.buffer.size < TRAJECTORY_SIZE:
            return {}
        
        # 获取缓冲区数据
        states, actions, old_log_probs, returns, advantages, old_values = self.buffer.get()
        
        stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'kl_div': 0,
            'clip_fraction': 0
        }
        
        # PPO更新循环
        for epoch in range(PPO_EPOCHS):
            # 随机打乱数据
            indices = torch.randperm(TRAJECTORY_SIZE)
            
            # 分批更新
            for start in range(0, TRAJECTORY_SIZE, PPO_BATCH_SIZE):
                end = start + PPO_BATCH_SIZE
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # 计算新的对数概率和熵
                log_probs, entropy = self.actor.get_logprob_and_entropy(batch_states, batch_actions)
                
                # 计算概率比率
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # 计算策略损失（PPO损失）
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - PPO_EPS, 1 + PPO_EPS) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                values = self.critic(batch_states).squeeze()
                # 使用截断的价值损失
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -PPO_EPS, PPO_EPS)
                value_loss1 = F.mse_loss(values, batch_returns)
                value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                value_loss = VALUE_LOSS_COEF * torch.max(value_loss1, value_loss2)
                
                # 熵损失（鼓励探索）
                entropy_loss = -ENTROPY_COEF * entropy.mean()
                
                # 总损失
                total_loss = policy_loss + value_loss + entropy_loss
                
                # 更新actor
                self.actor_optimizer.zero_grad()
                (policy_loss + entropy_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # 更新critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                # 记录统计信息
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > PPO_EPS).float().mean()
                
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.mean().item()
                stats['kl_div'] += kl_div.item()
                stats['clip_fraction'] += clip_fraction.item()
        
        # 平均统计信息
        num_updates = PPO_EPOCHS * (TRAJECTORY_SIZE // PPO_BATCH_SIZE)
        for key in stats:
            stats[key] /= num_updates
            
        return stats


# ================================
# 测试函数
# ================================

def test_agent(agent, env, num_episodes=10):
    """测试智能体性能"""
    total_rewards = []
    total_steps = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # 获取动作（测试时不需要log_prob和value）
            action = agent.get_deterministic_action(state)
            
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
    
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    
    return avg_reward, avg_steps


# ================================
# 主训练函数
# ================================

def main():
    """主训练循环"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1", help="Environment name")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Maximum training episodes")
    parser.add_argument("--save_dir", default="./ppo_saves", help="Save directory")
    parser.add_argument("--log_dir", default="./ppo_logs", help="Log directory")
    args = parser.parse_args()
    
    # 检查设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建环境
    env = gym.make(args.env)
    test_env = gym.make(args.env)
    
    # 获取环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # 创建智能体
    agent = PPOAgent(state_dim, action_dim, device)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 训练统计
    episode_rewards = deque(maxlen=100)
    best_reward = float('-inf')
    step_count = 0
    
    print("开始PPO训练...")
    print("=" * 50)
    
    # 主训练循环
    for episode in range(args.max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # 收集一个轨迹的经验
        while True:
            # 获取动作
            action, log_prob, value = agent.get_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.store_experience(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_steps += 1
            step_count += 1
            state = next_state
            
            # 当缓冲区满了或episode结束时进行更新
            if agent.buffer.size == TRAJECTORY_SIZE or done:
                if agent.buffer.size == TRAJECTORY_SIZE:
                    # 更新策略
                    stats = agent.update()
                    
                    # 记录训练统计信息
                    if stats:
                        writer.add_scalar("Loss/Policy", stats['policy_loss'], step_count)
                        writer.add_scalar("Loss/Value", stats['value_loss'], step_count)
                        writer.add_scalar("Loss/Entropy", stats['entropy'], step_count)
                        writer.add_scalar("Training/KL_Divergence", stats['kl_div'], step_count)
                        writer.add_scalar("Training/Clip_Fraction", stats['clip_fraction'], step_count)
                
                if done:
                    break
        
        # 记录episode统计信息
        episode_rewards.append(episode_reward)
        writer.add_scalar("Episode/Reward", episode_reward, episode)
        writer.add_scalar("Episode/Steps", episode_steps, episode)
        writer.add_scalar("Episode/Average_Reward_100", np.mean(episode_rewards), episode)
        
        # 定期测试和保存模型
        if episode % 50 == 0 and episode > 0:
            test_reward, test_steps = test_agent(agent, test_env, num_episodes=5)
            writer.add_scalar("Test/Reward", test_reward, episode)
            writer.add_scalar("Test/Steps", test_steps, episode)
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg(100): {np.mean(episode_rewards):8.2f} | "
                  f"Test: {test_reward:8.2f}")
            
            # 保存最佳模型
            if test_reward > best_reward:
                best_reward = test_reward
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'episode': episode,
                    'best_reward': best_reward
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"保存新的最佳模型，奖励: {best_reward:.2f}")
        
        # 每1000个episode保存检查点
        if episode % 1000 == 0 and episode > 0:
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'episode': episode,
                'best_reward': best_reward
            }, os.path.join(args.save_dir, f'checkpoint_ep{episode}.pth'))
    
    # 保存最终模型
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'episode': args.max_episodes,
        'best_reward': best_reward
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    print("\n训练完成！")
    print(f"最佳测试奖励: {best_reward:.2f}")
    
    # 关闭环境和writer
    env.close()
    test_env.close()
    writer.close()


if __name__ == "__main__":
    main()