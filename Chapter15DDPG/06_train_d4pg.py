#!/usr/bin/env python3
# D4PG算法训练主程序
import os  # 文件操作
from lib.replay_buffer import ReplayBuffer
from lib.ou_noise import OUNoise
import time  # 时间统计
import gymnasium as gym  # 新版gym库
import argparse  # 命令行参数解析
from torch.utils.tensorboard.writer import SummaryWriter  # TensorBoard日志
import numpy as np  # 数值计算

from lib import model, common  # 导入自定义模型和工具

import torch  # PyTorch主库
import torch.optim as optim  # 优化器
import torch.nn.functional as F  # 常用函数

# 超参数设置
GAMMA = 0.99  # 折扣因子
BATCH_SIZE = 64  # 批量大小
LEARNING_RATE = 1e-4  # 学习率
REPLAY_SIZE = 100_000  # 回放缓冲区大小（使用整数，避免浮点取模后 position 成为浮点导致索引报错）
REPLAY_INITIAL = REPLAY_SIZE // 10  # 初始填充 10%
REWARD_STEPS = 5  # 多步奖励

TEST_ITERS = 1000  # 测试间隔

Vmax = 10  # 分布最大值
Vmin = -10  # 分布最小值
N_ATOMS = 51  # 分布原子数
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)  # 原子间隔

# 测试网络性能
def test_net(net: model.DDPGActor, env: gym.Env, count: int = 10,
             device: torch.device = torch.device("cpu")):
    rewards = 0.0
    steps = 0
    maxSteep=100
    for _ in range(count):
        obs, _ = env.reset()
        while maxSteep>0:
            maxSteep-=1
            obs_v = torch.FloatTensor(np.array(obs).reshape(1, -1)).to(device)
            mu_v = net(obs_v)
            action = mu_v[0].data.cpu().numpy()  # 使用索引而不是squeeze
            # 通用裁剪：按环境动作空间范围
            if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, is_tr, _ = env.step(action)
            rewards += reward
            steps += 1
            if done or is_tr:
                break
    return rewards / count, steps / count

# 分布投影（分布式RL核心）
def distr_projection(
        next_distr: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        gamma: float
):
    """
    分布投影算法，参考《A Distributional Perspective on RL》
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)  # 投影分布初始化
    delta_z = (Vmax - Vmin) / (N_ATOMS - 1)
    for atom in range(N_ATOMS):
        v = rewards + (Vmin + atom * delta_z) * gamma  # 计算每个原子的投影位置
        tz_j = np.minimum(Vmax, np.maximum(Vmin, v))  # 限制在区间内
        b_j = (tz_j - Vmin) / delta_z  # 归一化
        l = np.floor(b_j).astype(np.int64)  # 下界索引
        u = np.ceil(b_j).astype(np.int64)  # 上界索引
        eq_mask = u == l  # 是否整数
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0  # 终止状态分布清零
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0  # 终止状态分布赋值
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr


if __name__ == "__main__":
    # 训练最大步数
    MAX_FRAMES = 1000000
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device to use, default=cpu")
    # 之前版本使用 required=True 导致必须输入；改为可选，并在为空时自动生成时间戳名称
    parser.add_argument("-n", "--name", default=None, help="Name of the run (optional)")
    args = parser.parse_args()
    if not args.name:
        # 自动生成名称：d4pg_YYYYmmdd_HHMMSS
        args.name = time.strftime("auto_%Y%m%d_%H%M%S")
    device = torch.device(args.dev)
    # 若请求CUDA但当前Torch无CUDA支持，则回退CPU
    if 'cuda' in str(device) and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU 运行。")
        device = torch.device('cpu')

    # 保存路径
    save_path = os.path.join("saves", "d4pg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    # CarRacing-v2为标准环境，无需注册
    env = gym.make(common.ENV_ID)
    test_env = gym.make(common.ENV_ID)

    obs_size = int(np.prod(env.observation_space.shape))
    act_size = int(np.prod(env.action_space.shape))
    act_net = model.DDPGActor(obs_size, act_size).to(device)
    crt_net = model.D4PGCritic(obs_size, act_size, N_ATOMS, Vmin, Vmax).to(device)
    print(act_net)
    print(crt_net)
    tgt_act_net = model.DDPGActor(obs_size, act_size).to(device)
    tgt_crt_net = model.D4PGCritic(obs_size, act_size, N_ATOMS, Vmin, Vmax).to(device)
    tgt_act_net.load_state_dict(act_net.state_dict())
    tgt_crt_net.load_state_dict(crt_net.state_dict())

    writer = SummaryWriter(comment="-d4pg_" + args.name)
    buffer = ReplayBuffer(REPLAY_SIZE)
    ou_noise = OUNoise(act_size)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    episode_reward = 0.0
    episode_steps = 0
    obs, _ = env.reset()
    ou_noise.reset()

    while True:
        if frame_idx >= MAX_FRAMES:
            print(f"训练终止：达到最大步数 {MAX_FRAMES}")
            break
        frame_idx += 1
        obs_v = torch.FloatTensor(np.array(obs).reshape(1, -1)).to(device)
        action = act_net(obs_v).cpu().detach().numpy()[0]  # 使用索引而不是squeeze
        action += ou_noise.sample()
        # 通用裁剪：按当前环境动作空间范围
        if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, done, trunc, _ = env.step(action)
        episode_reward += reward
        episode_steps += 1
        buffer.push(obs, action, reward, next_obs, done or trunc)
        obs = next_obs

        if done or trunc:
            writer.add_scalar("episode_reward", episode_reward, frame_idx)
            writer.add_scalar("episode_steps", episode_steps, frame_idx)
            obs, _ = env.reset()
            ou_noise.reset()
            episode_reward = 0.0
            episode_steps = 0

        if len(buffer) < REPLAY_INITIAL:
            continue

        # 采样批次
        states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
        states_v = torch.FloatTensor(states).to(device)
        actions_v = torch.FloatTensor(actions).to(device)
        rewards_v = torch.FloatTensor(rewards).to(device)
        next_states_v = torch.FloatTensor(next_states).to(device)
        dones_v = torch.BoolTensor(dones).to(device)

        # 训练critic
        crt_opt.zero_grad()
        crt_distr_v = crt_net(states_v, actions_v)
        with torch.no_grad():
            next_actions_v = tgt_act_net(next_states_v)
            next_distr_v = F.softmax(tgt_crt_net(next_states_v, next_actions_v), dim=1)
            proj_distr = distr_projection(
                next_distr_v.cpu().numpy(), rewards_v.cpu().numpy(),
                dones_v.cpu().numpy(), gamma=GAMMA**REWARD_STEPS)
            proj_distr_v = torch.tensor(proj_distr).to(device)
        prob_dist_v = -F.log_softmax(crt_distr_v, dim=1) * proj_distr_v
        critic_loss_v = prob_dist_v.sum(dim=1).mean()
        critic_loss_v.backward()
        crt_opt.step()
        writer.add_scalar("loss_critic", critic_loss_v.item(), frame_idx)

        # 训练actor
        act_opt.zero_grad()
        cur_actions_v = act_net(states_v)
        crt_distr_v = crt_net(states_v, cur_actions_v)
        actor_loss_v = -crt_net.distr_to_q(crt_distr_v)
        actor_loss_v = actor_loss_v.mean()
        actor_loss_v.backward()
        act_opt.step()
        writer.add_scalar("loss_actor", actor_loss_v.item(), frame_idx)

        # 软更新目标网络
        def soft_update(target, source, tau=1e-3):
            for t_param, s_param in zip(target.parameters(), source.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)
        soft_update(tgt_act_net, act_net)
        soft_update(tgt_crt_net, crt_net)

        # 定期测试
        if frame_idx % TEST_ITERS == 0:
            ts = time.time()
            rewards, steps = test_net(act_net, test_env, device=device)
            print("Test done in %.2f sec, reward %.3f, steps %d" % (
                time.time() - ts, rewards, steps))
            writer.add_scalar("test_reward", rewards, frame_idx)
            writer.add_scalar("test_steps", steps, frame_idx)
            if best_reward is None or best_reward < rewards:
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                    name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                    fname = os.path.join(save_path, name)
                    torch.save(act_net.state_dict(), fname)
                best_reward = rewards


# python .\06_train_d4pg.py --dev cuda:0 --name myrun