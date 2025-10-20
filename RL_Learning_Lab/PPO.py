import collections
import random
import gymnasium as gym
import numpy as np
import torch
from torch import nn
import os

#####基本参数lambda_, epochs, eps, gamma):
MAX_EPISODES = 100
MAX_EP_STEPS = 201
LR_A = 1e-4  # learning rate for actor
LR_C = 3e-3  # learning rate for critic
GAMMA = 0.9  # reward discount
LAMBDA = 0.95
EPISILON = 0.2
EPOCH = 5
MEMORY_CAPACITY = MAX_EP_STEPS
# BATCH_SIZE = 72
RENDER = False
ENV_NAME = 'Pendulum-v1'


#####
# ------------------------------------- #
# 经验回放池
# ------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):  # 经验池的最大容量
        # 创建一个队列，先进先出
        self.buffer = []

    # 在队列中添加数据
    def add(self, state, action, reward, next_state):
        # 以list类型保存
        self.buffer.append((state, action, reward, next_state))

    # 在队列中随机取样batch_size组数据
    # def sample(self, batch_size):
    #     transitions = random.sample(self.buffer, batch_size)
    #     # 将数据集拆分开来
    #     state, action, reward, next_state = zip(*transitions)
    #     return np.array(state), action, reward, np.array(next_state)

    def elements(self):
        state=[a for a,_,_,_ in self.buffer]
        action=[b for _,b,_,_ in self.buffer]
        reward=[c for _,_,c,_ in self.buffer]
        next_state=[d for _,_,_,d in self.buffer]
        return np.array(state),action,reward,np.array(next_state)

    # 测量当前时刻的队列长度
    def size(self):
        return len(self.buffer)


class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_scale = 2.0
        self.NetConstruction = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, 1)
        self.fc_std = nn.Linear(256, 1)
        self.active_tan=nn.Tanh()

    def forward(self, x):
        x = self.NetConstruction(x)
        mu = self.fc_mu(x)
        mu = self.active_tan(mu) * self.output_scale
        std = self.fc_std(x)
        std = nn.functional.softplus(std)
        return mu, std


class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.NetConstruction = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.NetConstruction(x)
        return x


class PPO:
    def __init__(self,
                 actor_lr, critic_lr,
                 lambda_, epochs, eps, gamma):
        # 属性分配
        self.lambda_ = lambda_  # GAE优势函数的缩放因子
        self.epochs = epochs  # 一条序列的数据用来训练多少轮
        self.eps = eps  # 截断范围
        self.gamma = gamma  # 折扣系数
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 实例化策略网络
        self.actor = ActorNet().to(self.device)
        # 实例化价值网络
        self.critic = CriticNet().to(self.device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # 动作选择
    def take_action(self, state):  # 输入当前时刻的状态
        # [n_states]-->[1,n_states]-->tensor
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            # 预测当前状态的动作，输出动作概率的高斯分布
            mu, std = self.actor(state)
            # 构造高斯分布
            action_dict = torch.distributions.Normal(mu, std)
            # 随机选择动作
            action = action_dict.sample().item()
        return [action]  # 返回动作值

    def update_policy(self, transition_dict):
        # print(transition_dict)
        state_batch = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        action_batch = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        reward_batch = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-7)  # 进行标准化
        state_next_batch = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        # print("statebatch:",state_batch,"\n actionbatch",action_batch,"\n rewardbatch",reward_batch,"\n nextstatebatch",state_next_batch)

        # 计算td误差
        with torch.no_grad():
            next_state_value_batch = self.critic(state_next_batch)
            td_target_batch = reward_batch + self.gamma * next_state_value_batch
            state_value_batch = self.critic(state_batch)
            td_error_batch = td_target_batch - state_value_batch
            # print(td_error_batch)

        # 计算GAE优势函数
        advantage_batch = torch.zeros_like(reward_batch)
        advantage = 0
        for t in reversed(range(len(reward_batch))):
            advantage = self.lambda_ * self.gamma * advantage + td_error_batch[t, 0]
            advantage_batch[t, 0] = advantage
        advantage_batch=(advantage_batch-advantage_batch.mean())/(advantage_batch.std()+1e-7)
        # print(advantage_batch)
        # advantage_batch = torch.flip(advantage_batch, dims=[0])
        # v_target_batch=advantage_batch+state_value_batch

        # 计算旧策略的log 也就是重要性采样分母部分
        with torch.no_grad():
            # 策略网络--预测，当前状态选择的动作的高斯分布
            mu, std = self.actor(state_batch)  # [b,1]
            # 基于均值和标准差构造正态分布
            # print('mu:',mu,'std:',std)
            action_dists = torch.distributions.Normal(mu, std)
            # 从正态分布中选择动作，并使用log函数
            old_log_prob = action_dists.log_prob(action_batch)  # 计算value在定义的正态分布（mean,std）中对应的概率的对数

        # 开始更新迭代新的策略的log  这里虽然一开始策略和旧策略一样 但是迭代以后参数会更新
        # 一个序列训练epochs次
        for _ in range(self.epochs):
            # 预测当前状态下的动作
            # print('statebatch')
            # print(state_batch)
            mu, std = self.actor(state_batch)
            # 构造正态分布
            # print('mu')
            # print(mu)
            # print('std')
            # print(std)
            action_dists = torch.distributions.Normal(mu, std)
            dist_entropy=action_dists.entropy().mean()
            # 当前策略在 t 时刻智能体处于状态 s 所采取的行为概率
            log_prob = action_dists.log_prob(action_batch)
            # 计算概率的比值来控制新策略更新幅度
            ratio = torch.exp(log_prob - old_log_prob)

            # 公式的左侧项
            surr1 = ratio * advantage_batch
            # 公式的右侧项，截断
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)*advantage_batch

            # 策略网络的损失PPO-clip
            actor_loss = -torch.mean(torch.min(surr1, surr2))-dist_entropy*0.005
            # 价值网络的当前时刻预测值，与目标价值网络当前时刻的state_value之差
            critic_loss = torch.mean(nn.functional.mse_loss(self.critic(state_batch), td_target_batch))

            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            critic_loss.backward()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()
            self.actor_optimizer.step()


# #####
# env = gym.make(ENV_NAME)
# # print(env.observation_space) #-1,1 -1,1 -8,8
# # print(env.action_space) #-2,2
# replayBuffer = ReplayBuffer(MEMORY_CAPACITY)
# agent = PPO(actor_lr=LR_A, critic_lr=LR_C, eps=EPISILON, lambda_=LAMBDA, gamma=GAMMA, epochs=EPOCH)
# for i in range(MAX_EPISODES):
#     state, info = env.reset()
#     # print(state)
#     reward_all = 0
#     for j in range(MAX_EP_STEPS):
#         action = agent.take_action(state)
#         # print(action)
#         next_state, reward, terminated, truncated, info = env.step(action)
#         reward_all += reward
#         replayBuffer.add(state, action, reward, next_state)
#         state = next_state
#     s, a, r, ns = replayBuffer.elements()
#     # print('s',s)
#     # print('a',a)
#     # print('r',r)
#     # print('n',ns)
#     transition_dict = {
#         'states': s,
#         'actions': a,
#         'rewards': r,
#         'next_states': ns,
#     }
#     agent.update_policy(transition_dict)
#     replayBuffer.buffer.clear()
#     if i%10==0:
#         print(f"------第{i}个episode完成。-------\n"
#               f"上一轮训练的总reward为：{reward_all}")
#
# env.close()

###########以下是测试代码##########
# 创建环境和回放池
env = gym.make(ENV_NAME)
replayBuffer = ReplayBuffer(MEMORY_CAPACITY)
agent = PPO(actor_lr=LR_A, critic_lr=LR_C, eps=EPISILON, lambda_=LAMBDA, gamma=GAMMA, epochs=EPOCH)
# 加载模型
agent.actor.load_state_dict(torch.load('actorPPO.pth'))
agent.critic.load_state_dict(torch.load('criticPPO.pth'))
with torch.no_grad():
    env = gym.make(ENV_NAME, render_mode='human')
    for i in range(MAX_EPISODES):
        state, info = env.reset()
        for j in range(MAX_EP_STEPS):
            action = agent.take_action(state)
            action = list(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
