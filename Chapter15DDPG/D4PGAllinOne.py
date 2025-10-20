import argparse
import os
import random
import time
import gymnasium as gym  # 新版gym库
import numpy as np  # 数值计算
import torch  # PyTorch主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 日志
import platform

###############################参数配置###############################
# 环境ID
# ENV_ID = "CarRacing-v2"
ENV_ID = "Pendulum-v1"  # 使用更简单的Pendulum环境进行测试
HID_SIZE= 256  # 隐藏层大小
N_ATOMS = 101  # 减少原子数量，提高训练效率
Vmin = -1800  # 根据Pendulum环境调整，原范围约[-16.2, 0]
Vmax = 500    # 留出一些余量
REPLAY_SIZE = 100000
BATCH_SIZE = 128  # 减小批次大小，提高训练稳定性
ACT_LEARNING_RATE = 1e-4
CRT_LEARNING_RATE = 3e-4  # Critic使用更高的学习率
GAMMA = 0.99
TARGET_NET_SYNC = 100  # 目标网络更新频率改为1000步，提高稳定性
EVAL_FREQ = 5000  # 评估频率
WARMUP_SIZE = REPLAY_SIZE//5  # 增加预热步数到20000
UPDATE_FREQ = 1  # 每步都更新，提高学习效率


# 组件定义
# 自定义状态预处理函数，替代ptan.agent.float32_preprocessor
def float32_preprocessor(states):
    # 支持list、ndarray、torch.Tensor输入
    if isinstance(states, list):
        states = np.array(states)
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states).float()
    elif not isinstance(states, torch.Tensor):
        states = torch.tensor(states, dtype=torch.float32)
    # 自动加batch维度
    if states.dim() == 1:
        states = states.unsqueeze(0)
    return states
# DDPG Actor网络
class DDPGActor(nn.Module):
    def __init__(self, obs_size: int, act_size: int ,HID_SIZE: int=256 ,envinfo:dict=None):
        super(DDPGActor, self).__init__()
        self.envinfo=envinfo
        # 提前转换为张量，避免每次forward时重复创建
        self.action_scale = torch.tensor(self.envinfo['action_scale'], dtype=torch.float32)
        self.action_bias = torch.tensor(self.envinfo['action_bias'], dtype=torch.float32)
        # 三层全连接网络
        self.envinfo=envinfo
        self.net = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh()  # 输出动作范围[-1,1]
        )

    def forward(self, x: torch.Tensor):
        # 展平特征以适配图像观测 (B, C,H,W) 或 (B,H,W,C)
        x = x.view(x.size(0), -1)
        raw = self.net(x)
        # 拼接为最终动作  这里执行的是缩放操作，因此使用envinfo中的action_scale和action_bias比较方便
        action_scale = self.action_scale.to(x.device)
        action_bias = self.action_bias.to(x.device)
        return raw * action_scale + action_bias
# D4PG Critic分布式Q网络
class D4PGCritic(nn.Module):
    def __init__(self, obs_size: int, act_size: int,
                 n_atoms: int, v_min: float, v_max: float, HID_SIZE: int=256):
        super(D4PGCritic, self).__init__()
        # 状态特征提取
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )
        # 输出分布（n_atoms个原子）
        self.out_net = nn.Sequential(
            nn.Linear(HID_SIZE + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, n_atoms)
        )
        # 构建支持集（分布的离散点）
        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        # 展平特征
        x = x.view(x.size(0), -1)
        obs = self.obs_net(x)
        # 输出分布
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q(self, distr: torch.Tensor):
        # 将分布转为Q值
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
    
# D4PG智能体，带自适应高斯噪声
class AgentD4PG():
    """
    使用自适应高斯噪声进行探索
    """
    def __init__(self, net: DDPGActor, device: torch.device = torch.device("cpu"),
                 epsilon: float = 0.3, envinfo:dict=None, epsilon_decay: float = 0.9999):
        self.net = net
        self.device = device
        self.epsilon = epsilon  # 初始噪声强度
        self.epsilon_min = 0.05  # 最小噪声强度
        self.epsilon_decay = epsilon_decay  # 噪声衰减率
        self.envinfo = envinfo

    def __call__(self, states, agent_states):
        states_v = float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.detach().cpu().numpy()
        # 添加自适应高斯噪声
        actions += self.epsilon * np.random.normal(size=actions.shape)
        # 噪声衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # 通用动作裁剪：根据envinfo中的动作范围自动裁剪 这里执行的是裁剪操作，因此使用envinfo中的action_low和action_high比较方便
        if 'action_low' in self.envinfo and 'action_high' in self.envinfo:
            actions = np.clip(actions, self.envinfo['action_low'], self.envinfo['action_high'])
        return actions, agent_states

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
# 分布投影（分布式RL核心）
def distr_projection(
        next_distr: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        gamma: float,
        Vmin: float = -150,
        Vmax: float = 150,
        N_ATOMS: int = 51
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

def distr_projection_cuda(
    next_distr: torch.Tensor,  # [batch, N_ATOMS]，必须是float32，device可为cuda
    rewards: torch.Tensor,     # [batch]
    dones: torch.Tensor,       # [batch]，bool类型
    gamma: float,
    Vmin: float = -150,
    Vmax: float = 150,
    N_ATOMS: int = 51
):
    """
    优化的分布投影算法（纯PyTorch版，提高数值稳定性）
    """
    device = next_distr.device
    batch_size = rewards.shape[0]
    delta_z = (Vmax - Vmin) / (N_ATOMS - 1)
    proj_distr = torch.zeros((batch_size, N_ATOMS), dtype=next_distr.dtype, device=device)
    
    # 创建支持集
    support = torch.linspace(Vmin, Vmax, N_ATOMS, device=device, dtype=torch.float32)
    
    # 对于每个样本和原子，计算投影
    for j in range(N_ATOMS):
        # Bellman方程: T_z_j = r + gamma * z_j (如果不是终止状态)
        tz_j = rewards + gamma * support[j] * (~dones).float()
        # 对于终止状态，T_z_j = r
        tz_j = torch.where(dones, rewards, tz_j)
        
        # 将T_z_j裁剪到支持集范围内
        tz_j = torch.clamp(tz_j, Vmin, Vmax)
        
        # 计算在离散支持集中的位置
        b_j = (tz_j - Vmin) / delta_z
        l = torch.floor(b_j).long()
        u = torch.ceil(b_j).long()
        
        # 边界处理 - 确保索引在有效范围内
        l = torch.clamp(l, 0, N_ATOMS - 1)
        u = torch.clamp(u, 0, N_ATOMS - 1)
        
        # 分配概率质量
        # 对于l==u的情况（b_j是整数）
        eq_mask = (l == u)
        if eq_mask.any():
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, j]
        
        # 对于l!=u的情况（需要插值）
        ne_mask = (l != u)
        if ne_mask.any():
            # 确保权重计算的数值稳定性
            weight_u = (b_j[ne_mask] - l[ne_mask].float()).clamp(0, 1)
            weight_l = (1.0 - weight_u).clamp(0, 1)
            
            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, j] * weight_l
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, j] * weight_u
    
    # 确保分布归一化（数值稳定性）
    proj_distr = proj_distr / (proj_distr.sum(dim=1, keepdim=True) + 1e-8)
    
    return proj_distr
# 评估函数
def evaluate(env, net, device="cpu", count=10, maxSteep=200):
    rewards = 0.0
    steps = 0
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
# 评估函数 --- IGNORE ---
if __name__ == "__main__":
    # 训练最大步数
    MAX_FRAMES = 1000000
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda:0", help="Device to use, default=cuda:0")
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
    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)
    envinfo = {
        'action_low': env.action_space.low,
        'action_high': env.action_space.high,
        'action_scale': (env.action_space.high - env.action_space.low) / 2.0,
        'action_bias': (env.action_space.high + env.action_space.low) / 2.0,
        # 其他信息
    }
    obs_size = int(np.prod(env.observation_space.shape))
    act_size = int(np.prod(env.action_space.shape))
    print("观测空间:", env.observation_space)
    print("动作空间:", env.action_space)
    print("动作范围:", env.action_space.low, "~", env.action_space.high)
    print("动作缩放:", envinfo['action_scale'], "偏置:", envinfo['action_bias'])
   
    act_net = DDPGActor(obs_size, act_size, HID_SIZE, envinfo).to(device)
    crt_net = D4PGCritic(obs_size, act_size, N_ATOMS, Vmin, Vmax, HID_SIZE).to(device)
    print(act_net)
    print(crt_net)
    tgt_act_net = DDPGActor(obs_size, act_size, HID_SIZE, envinfo).to(device)
    tgt_crt_net = D4PGCritic(obs_size, act_size, N_ATOMS, Vmin, Vmax, HID_SIZE).to(device)
    tgt_act_net.load_state_dict(act_net.state_dict())   
    tgt_crt_net.load_state_dict(crt_net.state_dict())
    buffer = ReplayBuffer(REPLAY_SIZE)
    agent = AgentD4PG(act_net, device, epsilon=0.2, envinfo=envinfo, epsilon_decay=0.9995)  # 降低初始噪声并增加衰减
    act_opt = torch.optim.Adam(act_net.parameters(), lr=ACT_LEARNING_RATE)
    crt_opt = torch.optim.Adam(crt_net.parameters(), lr=CRT_LEARNING_RATE)

    # 日志：TensorBoard
    node_name = platform.node() or "host"
    log_dir = os.path.join("runs", f"d4pg_{args.name}_{node_name}")
    try:
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard 日志目录: {log_dir}")
    except Exception as e:
        writer = None
        print(f"TensorBoard 写入器创建失败，将仅打印控制台日志: {e}")

    delta_z = (Vmax - Vmin) / (N_ATOMS - 1)
    if writer is not None:
        writer.add_scalar("config/Vmin", Vmin, 0)
        writer.add_scalar("config/Vmax", Vmax, 0)
        writer.add_scalar("config/N_ATOMS", N_ATOMS, 0)
        writer.add_scalar("config/delta_z", delta_z, 0)
    frame_idx = 0
    best_reward = None
    episode_reward = 0.0
    episode_steps = 0   
    obs, _ = env.reset()
    while True:
        if frame_idx >= MAX_FRAMES:
            print(f"训练终止：达到最大步数 {MAX_FRAMES}")
            break
        frame_idx += 1
        obs_v = torch.FloatTensor(np.array(obs).reshape(1, -1)).to(device)
        action, _ = agent(obs, None) 
        action = action[0]  # 提取动作
        next_obs, reward, done, is_tr, _ = env.step(action)
        episode_reward += reward
        episode_steps += 1
        buffer.push(obs, action, reward, next_obs, done or is_tr)
        obs = next_obs

        if done or is_tr:
            print(f"第{frame_idx}步，回合奖励{episode_reward:.2f}，回合长度{episode_steps}")
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        if len(buffer) >= WARMUP_SIZE and frame_idx % UPDATE_FREQ == 0:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
            states_v = torch.FloatTensor(states).to(device)
            actions_v = torch.FloatTensor(actions).to(device)
            rewards_v = torch.FloatTensor(rewards).to(device)
            next_states_v = torch.FloatTensor(next_states).to(device)
            dones_t = torch.BoolTensor(dones).to(device)

            # 计算目标分布
            with torch.no_grad():
                next_actions_v = tgt_act_net(next_states_v)
                next_distr_v = tgt_crt_net(next_states_v, next_actions_v)
                # next_distr_v = F.softmax(next_distr_v, dim=1).cpu().numpy() #为什么在网络外部使用softmax
                # proj_distr = distr_projection(
                #     next_distr_v, rewards, dones, GAMMA, Vmin, Vmax, N_ATOMS
                # )
                # proj_distr_v = torch.FloatTensor(proj_distr).to(device)
                next_distr_v = F.softmax(next_distr_v, dim=1)
                proj_distr_v = distr_projection_cuda(
                    next_distr_v, rewards_v, dones_t, GAMMA, Vmin, Vmax, N_ATOMS
                )
            # 计算当前分布
            distr_v = crt_net(states_v, actions_v)
            log_p = F.log_softmax(distr_v, dim=1)

            # 计算Critic损失（交叉熵）
            crt_loss = -(proj_distr_v * log_p).sum(dim=1).mean()

            crt_opt.zero_grad()
            crt_loss.backward()
            # 添加梯度裁剪提高稳定性
            torch.nn.utils.clip_grad_norm_(crt_net.parameters(), max_norm=1.0)
            crt_opt.step()
            # 每TARGET_NET_SYNC步同步一次目标网络
            if frame_idx % TARGET_NET_SYNC == 0:
                tgt_crt_net.load_state_dict(crt_net.state_dict())
            # 计算Actor损失（最大化Q值）
            act_opt.zero_grad()
            actions_pred = act_net(states_v)
            distr_pred = crt_net(states_v, actions_pred)
            q_pred = crt_net.distr_to_q(distr_pred)
            act_loss = -q_pred.mean()  
            act_loss.backward()
            # 添加梯度裁剪提高稳定性
            torch.nn.utils.clip_grad_norm_(act_net.parameters(), max_norm=1.0)
            act_opt.step()
            if frame_idx % TARGET_NET_SYNC == 0:
                tgt_act_net.load_state_dict(act_net.state_dict())
            # ===== 分布监测与日志 =====
            with torch.no_grad():
                pred_p = F.softmax(distr_v, dim=1)
                # 边界质量（预测/目标）
                left_mass_pred = pred_p[:, 0].mean().item()
                right_mass_pred = pred_p[:, -1].mean().item()
                target_p = proj_distr_v
                target_p = target_p / target_p.sum(dim=1, keepdim=True).clamp(min=1e-8)
                left_mass_tgt = target_p[:, 0].mean().item()
                right_mass_tgt = target_p[:, -1].mean().item()
                # 熵（预测分布）
                entropy_pred = (-(pred_p * (pred_p.clamp(min=1e-8).log())).sum(dim=1)).mean().item()
                # KL(target || pred)
                kl_tgt_pred = (target_p * (target_p.clamp(min=1e-8).log() - pred_p.clamp(min=1e-8).log())).sum(dim=1).mean().item()
                # Q 值统计
                q_mean = q_pred.mean().item()
                q_std = q_pred.std(unbiased=False).item()

            if writer is not None:
                writer.add_scalar("loss/critic", float(crt_loss.item()), frame_idx)
                writer.add_scalar("loss/actor", float(act_loss.item()), frame_idx)
                writer.add_scalar("agent/epsilon", agent.epsilon, frame_idx)  # 添加噪声监控
                writer.add_scalar("dist_pred/left_mass", left_mass_pred, frame_idx)
                writer.add_scalar("dist_pred/right_mass", right_mass_pred, frame_idx)
                writer.add_scalar("dist_pred/entropy", entropy_pred, frame_idx)
                writer.add_scalar("dist_target/left_mass", left_mass_tgt, frame_idx)
                writer.add_scalar("dist_target/right_mass", right_mass_tgt, frame_idx)
                writer.add_scalar("dist/kl_target_pred", kl_tgt_pred, frame_idx)
                writer.add_scalar("q/mean", q_mean, frame_idx)
                writer.add_scalar("q/std", q_std, frame_idx)

            # 间隔性打印，便于快速观察是否贴边
            if frame_idx % (EVAL_FREQ // 2 if EVAL_FREQ >= 2 else 500) == 0:
                print(
                    f"[frame {frame_idx}] Lp={left_mass_pred:.3f} Rp={right_mass_pred:.3f} "
                    f"Lt={left_mass_tgt:.3f} Rt={right_mass_tgt:.3f} H={entropy_pred:.2f} "
                    f"KL(t||p)={kl_tgt_pred:.3f} Qμ={q_mean:.1f} Qσ={q_std:.1f}"
                )
            if frame_idx % 10 == 0:
                print(f"第{frame_idx}步，Critic损失{crt_loss.item():.3f}，Actor损失{act_loss.item():.3f}")
        # 定期评估
        if frame_idx % EVAL_FREQ == 0:
            eval_reward, eval_steps = evaluate(test_env, act_net, device, count=5)
            print(f"第{frame_idx}步，评估奖励{eval_reward:.2f}，评估长度{eval_steps}")
            if best_reward is None or eval_reward > best_reward:
                best_reward = eval_reward
                torch.save(act_net.state_dict(), os.path.join(save_path, "best_actor.dat"))
                torch.save(crt_net.state_dict(), os.path.join(save_path, "best_critic.dat"))
                print(f"新最佳奖励{best_reward:.2f}，模型已保存。")
            torch.save(act_net.state_dict(), os.path.join(save_path, "last_actor.dat"))
            torch.save(crt_net.state_dict(), os.path.join(save_path, "last_critic.dat"))
            print("最新模型已保存。")

    # 关闭 TensorBoard 写入器
    try:
        if writer is not None:
            writer.close()
    except NameError:
        pass
