#!/usr/bin/env python3
"""综合版DDPG训练脚本，移除对PTAN库的依赖。

该脚本包含环境交互、经验回放缓冲区、噪声模型、网络结构以及训练循环，
用于训练在连续动作空间下工作的DDPG智能体。
"""
import argparse
import os
import random
import time
from collections import deque
from typing import Deque, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# -------------------------
# 超参数配置
# -------------------------
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
REPLAY_BUFFER_SIZE = 100_000
REPLAY_INITIAL = 10_000
TAU = 1e-3  # 软更新速率
TEST_INTERVAL = 1_000  # 测试频率（交互步数）
MAX_FRAMES = 1_000_000  # 最多交互步数，可根据需求修改


def set_global_seed(seed: int) -> None:
    """设置随机种子，保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    """固定大小的经验回放缓冲区，负责缓存和采样交互数据。"""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """向缓冲区中追加一个经验样本。"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """随机采样一个批次的经验，直接转换为训练所需的Tensor。"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

    # torch.as_tensor避免不必要的数据拷贝，dones保持0/1形式便于在训练中处理终止
        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        dones_t = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(-1)
        return states_t, actions_t, rewards_t, next_states_t, dones_t


class OrnsteinUhlenbeckNoise:
    """OU噪声过程，用于在连续动作空间中注入平滑的探索噪声。"""

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        x0: np.ndarray | None = None,
    ) -> None:
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x_prev = np.zeros(self.size) if x0 is None else x0

    def reset(self) -> None:
        """在每个回合开始时重置噪声状态。"""
        self.x_prev = np.zeros(self.size)

    def sample(self) -> np.ndarray:
        """生成一次噪声样本。"""
        noise = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        self.x_prev = noise
        return noise


class Actor(nn.Module):
    """DDPG策略网络：输入状态，输出环境动作。"""

    def __init__(self, obs_size: int, act_size: int, action_low: np.ndarray, action_high: np.ndarray) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh(),
        )
        # 注册张量常量，便于在forward中做动作缩放
        self.register_buffer("action_low", torch.FloatTensor(action_low))
        self.register_buffer("action_high", torch.FloatTensor(action_high))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """将网络输出的[-1, 1]区间映射回原动作空间。"""
        x = states.view(states.size(0), -1)
        raw_action = self.net(x)
        action_scale = (self.action_high - self.action_low) / 2.0
        action_bias = (self.action_high + self.action_low) / 2.0
        return raw_action * action_scale + action_bias


class Critic(nn.Module):
    """DDPG价值网络：联合状态与动作估计Q值。"""

    def __init__(self, obs_size: int, act_size: int) -> None:
        super().__init__()
        self.obs_net = nn.Sequential(nn.Linear(obs_size, 400), nn.ReLU())
        self.q_net = nn.Sequential(nn.Linear(400 + act_size, 300), nn.ReLU(), nn.Linear(300, 1))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = states.view(states.size(0), -1)
        obs_out = self.obs_net(x)
        return self.q_net(torch.cat([obs_out, actions], dim=1))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """目标网络软更新：target←τ·source+(1-τ)·target。"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


@torch.no_grad()
def test_agent(
    actor: Actor,
    env: gym.Env,
    device: torch.device,
    episodes: int = 10,
) -> Tuple[float, float]:
    """评估智能体，在测试环境中运行多次并返回平均奖励与步数。"""
    total_reward = 0.0
    total_steps = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        while not (done or truncated):
            state_v = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action_v = actor(state_v)
            action = action_v.squeeze(0).cpu().numpy()
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
        total_steps += steps
    return total_reward / episodes, total_steps / episodes


def train_ddpg(
    env_id: str,
    run_name: str,
    device: torch.device,
    seed: int = 1234,
) -> None:
    """主训练函数，封装DDPG训练流程。"""
    set_global_seed(seed)

    # 创建训练与测试环境
    env = gym.make(env_id)
    test_env = gym.make(env_id)

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    # 初始化网络与优化器
    actor = Actor(obs_size, act_size, action_low, action_high).to(device)
    critic = Critic(obs_size, act_size).to(device)
    target_actor = Actor(obs_size, act_size, action_low, action_high).to(device)
    target_critic = Critic(obs_size, act_size).to(device)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=LEARNING_RATE_ACTOR)
    critic_opt = optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC)

    # 探索噪声与经验回放初始化
    noise = OrnsteinUhlenbeckNoise(size=act_size)
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    # 跟踪指标
    writer = SummaryWriter(comment=f"-ddpg_{run_name}")
    frame_idx = 0
    best_reward = None
    episode_reward = 0.0
    episode_steps = 0
    episode_idx = 0

    obs, _ = env.reset(seed=seed)

    while frame_idx < MAX_FRAMES:
        frame_idx += 1
        episode_steps += 1

        # 策略网络给出确定性动作并注入OU噪声，实现探索
        state_v = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action_v = actor(state_v)
        action = action_v.squeeze(0).cpu().detach().numpy()
        action += noise.sample()
        action = np.clip(action, action_low, action_high)

        next_obs, reward, done, truncated, _ = env.step(action)
        terminal = done or truncated
        buffer.push(obs, action, reward, next_obs, terminal)

        obs = next_obs
        episode_reward += reward

        # 网络更新：需要足够的经验后才启动
        if len(buffer) >= REPLAY_INITIAL:
            states_v, actions_v, rewards_v, next_states_v, dones_v = buffer.sample(BATCH_SIZE, device)

            # 1) 更新Critic
            with torch.no_grad():
                next_actions_v = target_actor(next_states_v)
                next_q_v = target_critic(next_states_v, next_actions_v)
                q_ref_v = rewards_v + GAMMA * (1.0 - dones_v) * next_q_v
            q_v = critic(states_v, actions_v)
            critic_loss = F.mse_loss(q_v, q_ref_v)

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # 2) 更新Actor
            current_actions_v = actor(states_v)
            actor_loss = -critic(states_v, current_actions_v).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # 3) 软更新目标网络
            soft_update(target_actor, actor, TAU)
            soft_update(target_critic, critic, TAU)

            # TensorBoard记录训练指标
            writer.add_scalar("loss_critic", critic_loss.item(), frame_idx)
            writer.add_scalar("loss_actor", actor_loss.item(), frame_idx)

        if terminal:
            writer.add_scalar("episode_reward", episode_reward, frame_idx)
            writer.add_scalar("episode_steps", episode_steps, frame_idx)
            obs, _ = env.reset()
            noise.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_idx += 1

        # 周期性评估智能体表现
        if frame_idx % TEST_INTERVAL == 0:
            start_ts = time.time()
            test_reward, test_steps = test_agent(actor, test_env, device)
            duration = time.time() - start_ts
            writer.add_scalar("test_reward", test_reward, frame_idx)
            writer.add_scalar("test_steps", test_steps, frame_idx)
            print(
                f"Frame {frame_idx}: test_reward={test_reward:.3f}, test_steps={test_steps:.1f}, time={duration:.2f}s"
            )

            # 保存性能最佳的策略
            if best_reward is None or test_reward > best_reward:
                print(f"Best reward updated: {best_reward} -> {test_reward:.3f}")
                best_reward = test_reward
                save_dir = os.path.join("saves", f"ddpg-{run_name}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(actor.state_dict(), os.path.join(save_dir, f"best_{test_reward:+.3f}_{frame_idx}.pth"))

    env.close()
    test_env.close()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="DDPG训练脚本（无PTAN版本）")
    parser.add_argument("--env", default="Pendulum-v1", help="Gymnasium环境ID")
    parser.add_argument("--dev", default="cpu", help="计算设备，例如cpu或cuda:0")
    parser.add_argument("-n", "--name", required=True, help="运行名称，用于区分实验")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    args = parser.parse_args()

    device = torch.device(args.dev)
    train_ddpg(env_id=args.env, run_name=args.name, device=device, seed=args.seed)


if __name__ == "__main__":
    main()
