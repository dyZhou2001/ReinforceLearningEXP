# 好好学习 天天向上
# {2024/7/7} {15:13}
import numpy as np
import OrinRobot
import random
import torch
from torch import nn
import time
#网格世界  2代表target 0是可行区域 1是障碍区域 三个区域的reward分别为 1 0 -10
#对于agent来说有 0 1 2 3 4五个action 分别对应 上 右 下 左 原地不动
Environment=np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 0, 0, 0]
],dtype=int)

#神经网路类  按理说对于这样一个网格世界 25个state 5个action  表格形式的qpik也只需要125个元素  所以神经网路的参数应该用100个左右就能实现
class QpikEstimation(nn.Module):
    def __init__(self):
        super(QpikEstimation,self).__init__()
        self.NetConstruction=nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            # nn.Linear(64, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self,x):
        x=self.NetConstruction(x)
        return x

#agent机器人类  实现了包括探索环境 计算动作奖励值等功能 同时可以使用DQN更新Q网络
class DQLearningRobot(OrinRobot.OrinRobot):
    def __init__(self,state):
        super().__init__()
        self.initstate=state
        self.behavior_policy = np.zeros((5, 5), dtype=int) #用来生成experiencebuffer的策略 使用off policy 所以希望behavior是均匀的 即尽可能多探索
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.main_net=QpikEstimation().to(self.device)
        self.target_net=QpikEstimation().to(self.device)
        self.lostfunc=nn.MSELoss().to(self.device)
        self.learning_rate=0.02
        self.optimizer=torch.optim.Adam(params=self.main_net.parameters(),lr=self.learning_rate)

    def Strider(self,current_state, policy,epsilon):
        weighsList = [epsilon/len(self.action_set)]*5
        actionList = [0, 1, 2, 3, 4]
        weighsList[policy[current_state[0]][current_state[1]]] = 1 - ((len(self.action_set) - 1) * epsilon / len(self.action_set))
        actionChoice = random.choices(actionList, weighsList)
        # print(actionChoice,weighsList)
        newR, newC = self.updateState(current_state[0], current_state[1], actionChoice[0], Environment)
        # print(actionChoice[0],newR,newC)
        return actionChoice[0], newR, newC
#

    #用于生成buffer的函数 使用了set  无重复元素 如果把环境都探索到 共125个元素
    def experienceBufferGeneration(self,initial_state, step, policy,epsilon,environment):
            strideArray = set()
            currentState = initial_state
            # action,newR, newC = Strider(currentState, policy)
            # strideArray.append((initial_state,action))
            while (step):
                action, newR, newC = self.Strider(currentState, policy,epsilon)
                newState=(newR, newC)
                r = self.calculateReward(currentState, action, environment)
                # if environment[newR][newC]==2:
                #     done=True
                # else: done=False
                strideArray.add((currentState, action,r,newState))
                currentState = (newR, newC)
                step -= 1
            print(len(strideArray))
            strideArray=list(strideArray)
            return strideArray
#在程序结束后遍历所有state 生成最优的action
    def updatePolicy(self,environment):
        with torch.no_grad():
            for i in range(environment.shape[0]):
                for j in range(environment.shape[1]):
                    state=torch.tensor([i,j],dtype=torch.float,device=self.device)
                    a=self.main_net(state)
                    self.policy[i][j]=torch.argmax(a)
                    print(a)

    def DQLearning(self,episode,environment):
        # alpha_t = 0.5  # update rate , needed design
        epsilon = 1  # epsilon-greedy,since off policy, we want more explore
        Step = 2000
        gamma = 0.9
        initialState=(0,0)
        BATCH_SIZE=125
        expBuffer=self.experienceBufferGeneration(initialState,Step,self.behavior_policy,epsilon,environment)
        self.main_net.train()
        for i in range(episode):
            startTime=time.perf_counter()
            minibatch=random.sample(expBuffer,BATCH_SIZE) #minibatch的格式为{s,a,r,s',done} ((行，列)，动作值，（新行，新列），完成标志)
            # print(minibatch)
            state_batch=[item[0] for item in minibatch]
            # print("statebatch",state_batch)
            action_batch=torch.tensor([item[1] for item in minibatch],dtype=torch.int64).view(-1,1).to(self.device)
            # print('action batch is: ',action_batch)
            reward_batch=torch.tensor([item[2] for item in minibatch],dtype=torch.float).view(-1,1).to(self.device)
            statenext_batch=[item[3] for item in minibatch]
            # dones = torch.tensor([item[4] for item in minibatch],dtype=torch.float).view(-1, 1).to(self.device)
            mainnetInputs=torch.tensor(state_batch,dtype=torch.float).to(self.device)
            targetInputs=torch.tensor(statenext_batch,dtype=torch.float).to(self.device)
            mainoutputs=self.main_net(mainnetInputs)
            # print(mainnetInputs)
            targetoutputs=self.target_net(targetInputs)

            #DQN核心代码
            max_next_q_values=targetoutputs.max(1)[0].view(-1,1)
            y_batch_tensor=reward_batch + gamma * max_next_q_values
            # print(reward_batch)
            # print(gamma*max_next_q_values)
            q_batch_tensor=mainoutputs.gather(1,action_batch)
            #核心代码结束
            batchLoss=self.lostfunc(q_batch_tensor,y_batch_tensor)
            self.optimizer.zero_grad()
            batchLoss.backward()
            self.optimizer.step()
            # print('preepisode: ',i)
            if i%10==1:
                # print('1 ',q_batch_tensor,"\n 2 ", y_batch_tensor)
                print(batchLoss)
                endTime=time.perf_counter()
                print('第',i,'轮训练完成，花费的时间是：',endTime-startTime)
                self.target_net.load_state_dict(self.main_net.state_dict())
                print('神经网络参数交换完成')

Agent=DQLearningRobot((0,0))
Agent.DQLearning(1500,Environment) #进行10000次循环 每次取batchsize=25个样本进行训练
Agent.updatePolicy(Environment)
print(Agent.policy)









