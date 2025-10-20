# 好好学习 天天向上
# {2024/7/19} {16:51}
import collections
import random
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
#####基本参数
MAX_EPISODES = 100
MAX_EP_STEPS = 201
LR_A = 3e-4    # learning rate for actor
LR_C = 3e-3    # learning rate for critic
GAMMA = 0.9     # reward discount
MEMORY_CAPACITY = 1000
BATCH_SIZE = 72
RENDER = False
ENV_NAME = 'Pendulum-v1'


#####
# ------------------------------------- #
# 经验回放池
# ------------------------------------- #

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

##### define net
class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_scale=2.0
        self.NetConstruction=nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    def forward(self,x):
        x=self.NetConstruction(x)
        x=torch.tanh(x)
        x=x*self.output_scale
        return x
class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.NetConstruction=nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,s,a):
        x=torch.cat([s,a],dim=1)
        x=self.NetConstruction(x)
        return x
#####
##### main class of DDPG
class DDPG:
    def __init__(self,sigma, actor_lr, critic_lr, tau, gamma):
        self.sigma=sigma
        self.actor_lr=actor_lr
        self.critic_lr=critic_lr
        self.tau=tau
        self.gamma=gamma
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.main_actor=ActorNet().to(self.device)
        self.target_actor = ActorNet().to(self.device)
        self.main_critic=CriticNet().to(self.device)
        self.target_critic = CriticNet().to(self.device)
        # # 初始化价值网络的参数，两个价值网络的参数相同
        # self.target_critic.load_state_dict(self.main_critic.state_dict())
        # # 初始化策略网络的参数，两个策略网络的参数相同
        # self.target_actor.load_state_dict(self.main_actor.state_dict())

        self.actor_optimizer=torch.optim.Adam(params=self.main_actor.parameters(),lr=self.actor_lr)
        self.critic_optimizer=torch.optim.Adam(params=self.main_critic.parameters(),lr=self.critic_lr)
        self.writer = SummaryWriter(comment='_DDPG_Pendulum')

    def take_action(self,state):
        with torch.no_grad():
            state=torch.tensor(state,dtype=torch.float).to(self.device)
            action=self.main_actor(state)
        return action

    def add_noise(self,action):
        action = action + self.sigma * np.random.randn()
        action=list(action.cpu())
        return action

    def update_policy(self,transition_dict,step_num):
        # print(transition_dict)
        state_batch=torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        action_batch =torch.tensor(transition_dict['actions'],dtype=torch.float).view(-1,1).to(self.device)
        reward_batch =torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        state_next_batch =torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        #print("statebatch:",state_batch,"\n actionbatch",action_batch,"\n rewardbatch",reward_batch,"\n nextstatebatch",state_next_batch)
        action_next_batch=self.target_actor(state_next_batch)
        y_batch=reward_batch+self.gamma*self.target_critic(state_next_batch,action_next_batch)
        q_value_batch = self.main_critic(state_batch, action_batch)

        critic_loss_func = nn.MSELoss().to(self.device)
        critic_loss=torch.mean(critic_loss_func(q_value_batch,y_batch))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if step_num%100==0:
            print("第",step_num,'步训练完成，本次训练损失为',critic_loss)

        actor_loss=-torch.mean(self.main_critic(state_batch,self.main_actor(state_batch)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if step_num%5==0:
            self.soft_update(self.main_actor, self.target_actor)
            self.soft_update(self.main_critic, self.target_critic)
        self.writer.add_scalar('Loss/Critic', critic_loss.item(), step_num)  # <source id="1">
        self.writer.add_scalar('Loss/Actor', actor_loss.item(), step_num)  # <source id="1">



    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)
        # target_net.load_state_dict(net.state_dict())

    def save_checkpoint(self, path='ddpg_best_model.pth'):
        torch.save({
            'main_actor': self.main_actor.state_dict(),
            'main_critic': self.main_critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),

        }, path)

    def load_checkpoint(self, path='ddpg_best_model.pth'):
        checkpoint = torch.load(path)
        self.main_actor.load_state_dict(checkpoint['main_actor'])
        self.main_critic.load_state_dict(checkpoint['main_critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])


##### 训练步骤代码
env = gym.make(ENV_NAME)
# print(env.observation_space) #-1,1 -1,1 -8,8
# print(env.action_space) #-2,2
replayBuffer=ReplayBuffer(MEMORY_CAPACITY)
agent=DDPG(actor_lr=LR_A,critic_lr=LR_C,sigma=0.01,tau=0.005,gamma=GAMMA)
best_reward = -float('inf')  # 初始化最佳奖励值
for i in range(MAX_EPISODES):
    state, info = env.reset()
    reward_all=0
    for j in range(MAX_EP_STEPS):
        action=agent.take_action(state)
        action=agent.add_noise(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        reward_all+=reward
        replayBuffer.add(state, action, reward, next_state)
        state = next_state
        if replayBuffer.size() > BATCH_SIZE+1:
            # 经验池随机采样batch_size组
            s, a, r, ns= replayBuffer.sample(BATCH_SIZE)
            # 构造数据集
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
            }
            # 模型训练
            agent.update_policy(transition_dict,j)
    if reward_all > best_reward:
        best_reward = reward_all
        agent.save_checkpoint()  # <source id="1">
        print(f"模型已保存，当前最佳奖励：{best_reward:.2f}")
    else:
        agent.save_checkpoint()
        print(f"奖励未提升，保存模型（当前：{reward_all:.2f}，最佳：{best_reward:.2f}）")
    print(f"------第{i}个episode完成。-------\n"
          f"上一轮训练的总reward为:{reward_all}")
    agent.writer.add_scalar('Reward/Train', reward_all, i)  # <source id="1">
agent.writer.close()
env.close()

# ###########以下是测试代码##########
# 修改测试代码部分（在上下文已有的测试代码中）
########### 以下是修改后的测试代码 ##########
test_agent = DDPG(sigma=0, actor_lr=LR_A, critic_lr=LR_C, tau=0.005, gamma=GAMMA)  # 新建测试用agent
test_agent.load_checkpoint()  # 加载保存的最佳模型  # <source id="1">

with torch.no_grad():
    env = gym.make(ENV_NAME, render_mode='human')
    total_test_reward = 0

    for i in range(MAX_EPISODES):
        state, info = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            action = test_agent.take_action(state)  # 使用加载的模型
            action = list(action.cpu())
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            ep_reward += reward

        total_test_reward += ep_reward

    print(f"测试阶段平均奖励：{total_test_reward / MAX_EPISODES:.2f}")
    env.close()


