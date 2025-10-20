# 好好学习 天天向上
# {2024/8/3} {上午9:56}
import collections
import random
import gymnasium as gym
import numpy as np
import torch
from torch import nn
import copy
import os

#####基本参数lambda_, epochs, eps, gamma):
MAX_EPISODES = 60
MAX_EP_STEPS = 300
LR_A = 5e-5  # learning rate for actor
LR_C = 5e-4  # learning rate for critic
LR_alpha = 5e-4
GAMMA = 0.9  # reward discount

ENTROPY = 0.01
EPOCH = 5
TAU=0.05
MEMORY_CAPACITY = 10240
BATCH_SIZE = 72
RENDER = False
ENV_NAME = 'Pendulum-v1'

class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_scale=2.0
        self.net_structure=nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.meanfuc=nn.Linear(256,1)
        self.stdfunc=nn.Linear(256,1)

    def forward(self,x):
        x=self.net_structure(x)
        mean_=self.meanfuc(x)
        std=nn.functional.softplus(self.stdfunc(x))
        dist=torch.distributions.Normal(mean_,std)
        sample=dist.rsample()
        actions=nn.functional.tanh(sample)
        log_prob=dist.log_prob(sample)
        log_prob=log_prob - torch.log(1 - torch.tanh(actions).pow(2) + 1e-7)

        return actions*self.output_scale,log_prob


class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_construction=nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self,s,a):
        x = torch.cat([s, a], dim=1)
        x = self.net_construction(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):  # 经验池的最大容量
        # 创建一个队列，先进先出
        self.buffer = collections.deque(maxlen=capacity)

    # 在队列中添加数据
    def add(self, state, action, reward, next_state):
        # 以list类型保存
        self.buffer.append((state, action, reward, next_state))

    # 在队列中随机取样batch_size组数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将数据集拆分开来
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)

    # 测量当前时刻的队列长度
    def size(self):
        return len(self.buffer)


class SAC:
    def __init__(self,actor_lr: float,critic_lr: float,alpha_lr: float,gamma: float,tau:float,target_entropy):
        #基础参数
        self.actor_lr=actor_lr
        self.critic_lr=critic_lr
        self.alpha_lr=alpha_lr
        self.gamma=gamma
        self.tau=tau
        self.target_entropy=target_entropy
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        #定义了5个网络
        self.actor=ActorNet().to(self.device)
        self.critic_1=CriticNet().to(self.device)
        self.critic_2=CriticNet().to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
        #三个优化器
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=self.actor_lr)
        self.critic_1_optimizer=torch.optim.Adam(self.critic_1.parameters(),lr=self.critic_lr)
        self.critic_2_optimizer=torch.optim.Adam(self.critic_2.parameters(),lr=self.critic_lr)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1 - self.tau) + param.data * self.tau
            )

    def take_action(self,state):
        with torch.no_grad():
            state=torch.tensor(state,dtype=torch.float).to(self.device)
            action,_=self.actor(state)
            action = list(action.cpu())
        return action

    def calcu_td_target_batch(self,reward_batch,state_next_batch):
        with torch.no_grad():
            next_action_batch,next_logprob_batch=self.actor(state_next_batch)
            Qvalue1_next=self.target_critic_1(state_next_batch,next_action_batch)
            Qvalue2_next=self.target_critic_2(state_next_batch,next_action_batch)
            Qvalue_next=torch.min(Qvalue1_next,Qvalue2_next)
            td_error_batch=Qvalue_next-self.log_alpha.exp()*next_logprob_batch
        return  reward_batch+self.gamma*td_error_batch


    def update(self,transition_dict):
        #导出所需batch
        state_batch = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        action_batch = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        reward_batch = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        state_next_batch = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)

        td_target_batch=self.calcu_td_target_batch(reward_batch,state_next_batch)
        critic_loss1=torch.mean(nn.functional.mse_loss(self.critic_1(state_batch,action_batch),td_target_batch))
        critic_loss2=torch.mean(nn.functional.mse_loss(self.critic_2(state_batch,action_batch),td_target_batch))

        self.critic_1_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic_2_optimizer.step()

        current_action_batch_new,cabw_logprob=self.actor(state_batch)
        Qvalue1=self.critic_1(state_batch,current_action_batch_new)
        Qvalue2=self.critic_2(state_batch,current_action_batch_new)
        actor_loss=torch.mean(self.log_alpha.exp()*cabw_logprob-torch.min(Qvalue1,Qvalue2))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((-cabw_logprob - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_opt.zero_grad()
        alpha_loss.backward()
        self.log_alpha_opt.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)









if __name__ == '__main__':
    # #####
    # env = gym.make(ENV_NAME)
    # # print(env.observation_space) #-1,1 -1,1 -8,8
    # # print(env.action_space) #-2,2
    # replayBuffer = ReplayBuffer(MEMORY_CAPACITY)
    # agent = SAC(actor_lr=LR_A, critic_lr=LR_C,alpha_lr=LR_alpha, gamma=GAMMA,tau=TAU,target_entropy=ENTROPY)
    # for i in range(MAX_EPISODES):
    #     state, info = env.reset()
    #     reward_all = 0
    #     for j in range(MAX_EP_STEPS):
    #         action = agent.take_action(state)
    #         next_state, reward, terminated, truncated, info = env.step(action)
    #         reward_all += reward
    #         replayBuffer.add(state, action, reward, next_state)
    #         state = next_state
    #         if replayBuffer.size() > BATCH_SIZE + 1:
    #             # 经验池随机采样batch_size组
    #             s, a, r, ns = replayBuffer.sample(BATCH_SIZE)
    #             # 构造数据集
    #             transition_dict = {
    #                 'states': s,
    #                 'actions': a,
    #                 'rewards': r,
    #                 'next_states': ns,
    #             }
    #             # 模型训练
    #             agent.update(transition_dict)
    #
    #     print(f"------第{i}个episode完成。-------\n"
    #           f"上一轮训练的总reward为：{reward_all}")
    #
    # env.close()

    ###########以下是测试代码##########
    # 创建环境和回放池
    env = gym.make(ENV_NAME)
    replayBuffer = ReplayBuffer(MEMORY_CAPACITY)
    agent = SAC(actor_lr=LR_A, critic_lr=LR_C, alpha_lr=LR_alpha, gamma=GAMMA, tau=TAU, target_entropy=ENTROPY)
    # 加载模型
    agent.actor.load_state_dict(torch.load( 'actor.pth'))
    agent.critic_1.load_state_dict(torch.load( 'critic1.pth'))
    agent.critic_2.load_state_dict(torch.load('critic2.pth'))

    # 开始测试
    with torch.no_grad():
        env = gym.make(ENV_NAME, render_mode='human')
        for i in range(5):
            state, info = env.reset()
            for j in range(MAX_EP_STEPS):
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                if terminated or truncated:
                    break
    env.close()