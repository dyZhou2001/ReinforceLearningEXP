#!/usr/bin/env python3
"""Single-file Soft Actor-Critic implementation without PTAN.

The script trains an SAC agent on the Pendulum-v1 environment from Gymnasium.
Networks, replay buffer, and the full training loop live inside this module so
that it can be run standalone.
"""

# Step 1: Standard library imports
import argparse
import collections
import os
import random
import time

# Step 2: Third-party imports
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Step 3: Hyperparameter defaults (can be overridden via CLI)
DEFAULT_GAMMA = 0.99
DEFAULT_ALPHA = 0.2
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR_ACTOR = 3e-4
DEFAULT_LR_CRITIC = 3e-4
DEFAULT_REPLAY_SIZE = 100_000
DEFAULT_WARMUP_STEPS = 1_000
DEFAULT_MAX_STEPS = 200_000
DEFAULT_EVAL_INTERVAL = 5_000
DEFAULT_UPDATES_PER_STEP = 1
LOG_STD_MIN = -20
LOG_STD_MAX = 2


# Step 4: Simple replay buffer to collect experience tuples
class ReplayBuffer:
    """Fixed-size buffer that stores transition tuples for off-policy updates."""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones


# Step 5: Utility to softly copy parameters between networks
@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-averages the parameters of the target network towards the source."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * source_param.data)


# Step 6: Actor network that outputs a squashed Gaussian policy
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, action_scale: np.ndarray):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, state: torch.Tensor):
        hidden = self.net(state)
        mu = self.mu_layer(hidden)
        log_std = self.log_std_layer(hidden)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        """Samples an action using the reparameterization trick."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        noise = normal.rsample()
        tanh_action = torch.tanh(noise)
        action = tanh_action * self.action_scale + self.action_bias
        log_prob = normal.log_prob(noise)
        log_prob -= torch.log(1.0 - tanh_action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        scale_adjust = torch.log(self.action_scale).sum()
        log_prob -= scale_adjust
        deterministic_action = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action


# Step 7: Critic networks for SAC (Twin Q + Value)
class TwinQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor):
        return self.net(state)


# Step 8: Evaluation helper to monitor policy performance
@torch.no_grad()
def evaluate_policy(env: gym.Env, policy: PolicyNetwork, device: torch.device, episodes: int = 5) -> float:
    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_v = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, action_v = policy.sample(obs_v)
            action = action_v.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            obs = next_obs
            done = terminated or truncated
    return total_reward / episodes


# Step 9: Main training routine

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC agent on Pendulum-v1 without PTAN")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Entropy coefficient")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for updates")
    parser.add_argument("--lr-actor", type=float, default=DEFAULT_LR_ACTOR, help="Learning rate for the policy")
    parser.add_argument("--lr-critic", type=float, default=DEFAULT_LR_CRITIC, help="Learning rate for critics")
    parser.add_argument("--replay-size", type=int, default=DEFAULT_REPLAY_SIZE, help="Replay buffer capacity")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Steps collected before training")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Total interaction steps")
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL, help="Steps between evaluations")
    parser.add_argument("--updates-per-step", type=int, default=DEFAULT_UPDATES_PER_STEP, help="Gradient steps per environment step")
    parser.add_argument("--hidden-size", type=int, default=256, help="Width of hidden layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--logdir", type=str, default="runs_sac_pendulum", help="TensorBoard log directory")
    parser.add_argument("--tau", type=float, default=5e-3, help="Target network smoothing coefficient")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g., cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Step 10: Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Step 11: Prepare training and evaluation environments
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    env.action_space.seed(args.seed)
    eval_env.action_space.seed(args.seed + 1)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = env.action_space.high

    # Step 12: Build networks and optimizers
    policy_net = PolicyNetwork(obs_dim, act_dim, args.hidden_size, action_scale).to(device)
    value_net = ValueNetwork(obs_dim, args.hidden_size).to(device)
    target_value_net = ValueNetwork(obs_dim, args.hidden_size).to(device)
    target_value_net.load_state_dict(value_net.state_dict())
    twin_q_net = TwinQNetwork(obs_dim, act_dim, args.hidden_size).to(device)
    actor_optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr_actor)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr_critic)
    q_optimizer = torch.optim.Adam(twin_q_net.parameters(), lr=args.lr_critic)

    # Step 13: Initialize replay buffer and logging utilities
    replay_buffer = ReplayBuffer(args.replay_size)
    writer = SummaryWriter(log_dir=args.logdir)
    best_eval_reward = None

    # Step 14: Main interaction loop with the environment
    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_steps = 0
    start_time = time.time()

    for step_idx in range(1, args.max_steps + 1):
        obs_v = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if step_idx < args.warmup_steps:
            action = env.action_space.sample()
            log_prob = None
        else:
            with torch.no_grad():
                action_v, log_prob_v, _ = policy_net.sample(obs_v)
            action = action_v.squeeze(0).cpu().numpy()
            log_prob = log_prob_v.item()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(obs, action, reward, next_obs, done)
        episode_reward += reward
        episode_steps += 1

        # Step 15: Trigger parameter updates once warmup is finished
        if step_idx >= args.warmup_steps and len(replay_buffer) >= args.batch_size:
            for _ in range(args.updates_per_step):
                states, actions, rewards, next_states, dones = replay_buffer.sample(args.batch_size)
                states_v = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions_v = torch.as_tensor(actions, dtype=torch.float32, device=device)
                rewards_v = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states_v = torch.as_tensor(next_states, dtype=torch.float32, device=device)
                dones_v = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)

                # Step 15a: Update Q-functions using Bellman residual
                with torch.no_grad():
                    next_value_v = target_value_net(next_states_v)
                    target_q_v = rewards_v + args.gamma * (1.0 - dones_v) * next_value_v
                q1_v, q2_v = twin_q_net(states_v, actions_v)
                q_loss = F.mse_loss(q1_v, target_q_v) + F.mse_loss(q2_v, target_q_v)
                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                # Step 15b: Update the state-value network to match the soft value target
                sampled_actions_v, log_prob_v, _ = policy_net.sample(states_v)
                q1_pi_v, q2_pi_v = twin_q_net(states_v, sampled_actions_v)
                soft_value_target = torch.min(q1_pi_v, q2_pi_v) - args.alpha * log_prob_v
                value_v = value_net(states_v)
                value_loss = F.mse_loss(value_v, soft_value_target.detach())
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

                # Step 15c: Update the policy to maximize expected Q while respecting entropy
                sampled_actions_v, log_prob_v, _ = policy_net.sample(states_v)
                q1_pi_v, q2_pi_v = twin_q_net(states_v, sampled_actions_v)
                policy_loss = (args.alpha * log_prob_v - torch.min(q1_pi_v, q2_pi_v)).mean()
                actor_optimizer.zero_grad()
                policy_loss.backward()
                actor_optimizer.step()

                # Step 15d: Keep the target value network in sync
                soft_update(target_value_net, value_net, args.tau)

            writer.add_scalar("loss/q", q_loss.item(), step_idx)
            writer.add_scalar("loss/value", value_loss.item(), step_idx)
            writer.add_scalar("loss/policy", policy_loss.item(), step_idx)
            if log_prob is not None:
                writer.add_scalar("policy/log_prob", log_prob, step_idx)

        obs = next_obs

        if done:
            writer.add_scalar("train/episode_reward", episode_reward, step_idx)
            writer.add_scalar("train/episode_steps", episode_steps, step_idx)
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        # Step 16: Periodically evaluate the deterministic policy
        if step_idx % args.eval_interval == 0:
            eval_reward = evaluate_policy(eval_env, policy_net, device)
            writer.add_scalar("eval/average_reward", eval_reward, step_idx)
            elapsed = time.time() - start_time
            print(f"Step {step_idx}: eval_reward={eval_reward:.2f}, elapsed={elapsed/60:.1f} min")
            if best_eval_reward is None or eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                os.makedirs("saves", exist_ok=True)
                torch.save(policy_net.state_dict(), os.path.join("saves", "sac_pendulum_best.pt"))

    # Step 17: Finish up with a final log entry
    writer.add_scalar("train/total_time_minutes", (time.time() - start_time) / 60.0, args.max_steps)
    writer.close()


if __name__ == "__main__":
    main()
