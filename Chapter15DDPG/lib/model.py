import numpy as np  # 数值计算

import torch  # PyTorch主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数

# 自定义状态预处理函数，替代ptan.agent.float32_preprocessor
def float32_preprocessor(states):
    if isinstance(states, list):
        states = np.array(states)
    return torch.FloatTensor(states)

# 自定义Agent基类，替代ptan.agent.BaseAgent
class BaseAgent:
    def __call__(self, states, agent_states):
        raise NotImplementedError

# 隐藏层大小
HID_SIZE = 128

# DDPG Actor网络
class DDPGActor(nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super(DDPGActor, self).__init__()
        # 三层全连接网络
        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()  # 输出动作范围[-1,1]
        )

    def forward(self, x: torch.Tensor):
        # 展平特征以适配图像观测 (B, C,H,W) 或 (B,H,W,C)
        x = x.view(x.size(0), -1)
        raw = self.net(x)
        # CarRacing-v2: steer [-1,1], gas [0,1], brake [0,1]
        # raw[...,0]: tanh输出，直接作为steer
        # raw[...,1:]: tanh输出[-1,1]，需映射到[0,1]
        steer = raw[..., 0:1]  # [-1,1]
        gas_brake = (raw[..., 1:3] + 1) / 2  # [-1,1] -> [0,1]
        # 拼接为最终动作
        return torch.cat([steer, gas_brake], dim=-1)

# DDPG Critic网络
class DDPGCritic(nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super(DDPGCritic, self).__init__()
        # 状态特征提取
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )
        # 状态+动作联合特征
        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)  # 输出Q值
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        # 展平特征
        x = x.view(x.size(0), -1)
        obs = self.obs_net(x)  # 状态特征
        # 拼接动作，输出Q值
        return self.out_net(torch.cat([obs, a], dim=1))

# D4PG Critic分布式Q网络
class D4PGCritic(nn.Module):
    def __init__(self, obs_size: int, act_size: int,
                 n_atoms: int, v_min: float, v_max: float):
        super(D4PGCritic, self).__init__()
        # 状态特征提取
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )
        # 输出分布（n_atoms个原子）
        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, n_atoms)
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


# DDPG智能体，带OU噪声
class AgentDDPG(BaseAgent):
    """
    使用Orstein-Uhlenbeck过程进行探索
    """
    def __init__(self, net: DDPGActor, device: torch.device = torch.device('cpu'),
                 ou_enabled: bool = True, ou_mu: float = 0.0, ou_teta: float = 0.15,
                 ou_sigma: float = 0.2, ou_epsilon: float = 1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled  # 是否启用OU噪声
        self.ou_mu = ou_mu  # OU均值
        self.ou_teta = ou_teta  # OU参数
        self.ou_sigma = ou_sigma  # OU参数
        self.ou_epsilon = ou_epsilon  # 噪声强度

    def initial_state(self):
        return None  # 初始状态

    def __call__(self, states, agent_states):
        states_v = float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        # 添加OU噪声
        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)
                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        # CarRacing-v2裁剪：steer[-1,1], gas/brake[0,1]
        actions[..., 0] = np.clip(actions[..., 0], -1, 1)
        actions[..., 1] = np.clip(actions[..., 1], 0, 1)
        actions[..., 2] = np.clip(actions[..., 2], 0, 1)
        return actions, new_a_states

# D4PG智能体，带高斯噪声
class AgentD4PG(BaseAgent):
    """
    使用高斯噪声进行探索
    """
    def __init__(self, net: DDPGActor, device: torch.device = torch.device("cpu"),
                 epsilon: float = 0.3):
        self.net = net
        self.device = device
        self.epsilon = epsilon  # 噪声强度

    def __call__(self, states, agent_states):
        states_v = float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()
        # 添加高斯噪声
        actions += self.epsilon * np.random.normal(size=actions.shape)
        # CarRacing-v2裁剪：steer[-1,1], gas/brake[0,1]
        actions[..., 0] = np.clip(actions[..., 0], -1, 1)
        actions[..., 1] = np.clip(actions[..., 1], 0, 1)
        actions[..., 2] = np.clip(actions[..., 2], 0, 1)
        return actions, agent_states
