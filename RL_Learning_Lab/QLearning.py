# 好好学习 天天向上
# {2024/6/23} {19:42}
import numpy as np
import OrinRobot
import random
Environment=np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 0, 0, 0]
],dtype=int)

class QLearningRotot(OrinRobot.OrinRobot):
    def __init__(self,state):
        super().__init__()
        self.initstate=state
        # self.state_n0=(0,0)
        # self.state_n1=(0,0)
        # self.r_t1=0
        # self.action0=0
        # self.action1=0
        self.behavier_policy=np.zeros((5,5),dtype=int)

    # def episodeGeneration(self,initial_state, step, policy,epsilon):
    #     strideArray = []
    #     currentState = initial_state
    #     # action,newR, newC = Strider(currentState, policy)
    #     # strideArray.append((initial_state,action))
    #     while (step):
    #         action, newR, newC = self.Strider(currentState, policy,epsilon)
    #         strideArray.append((currentState, action))
    #         currentState = (newR, newC)
    #         step -= 1
    #     return strideArray

    def Strider(self,current_state, policy,epsilon):
        weighsList = [epsilon/len(self.action_set)]*5
        actionList = [0, 1, 2, 3, 4]
        weighsList[policy[current_state[0]][current_state[1]]] = 1 - ((len(self.action_set) - 1) * epsilon / len(self.action_set))
        actionChoice = random.choices(actionList, weighsList)
        # print(actionChoice,weighsList)
        newR, newC = self.updateState(current_state[0], current_state[1], actionChoice[0], Environment)
        # print(actionChoice[0],newR,newC)
        return actionChoice[0], newR, newC

    def QLearning(self,episode,environment):
        alpha_t=0.5 #update rate , needed design
        epsilon=1 #epsilon-greedy,since off policy, we want more explore
        Step=1000
        gamma=0.9
    #Goal: Learn an optimal path that can lead the agent to the target state from an initial state s0.
        for episode_i in range(episode):
            #stateArray = self.episodeGeneration(self.initstate, Step, self.behavier_policy,epsilon)
            currentState = self.initstate
            for i in range(Step):
                action, newR, newC = self.Strider(currentState, self.behavier_policy, epsilon)
                r=self.calculateReward(currentState,action,environment)
                max_action=np.argmax(self.q_pik, axis=0)[newR,newC]
                max_action_qpik=self.q_pik[max_action,newR,newC]
                self.q_pik[action,currentState[0],currentState[1]]=self.q_pik[action,currentState[0],currentState[1]]\
                                                                   -alpha_t*(self.q_pik[action,currentState[0],currentState[1]]-(r+gamma*max_action_qpik))
                currentState = (newR, newC)
                self.policy=np.argmax(self.q_pik, axis=0)
                self.v_pik=np.max(self.q_pik,axis=0)
                #print(self.q_pik)
            #print(self.policy)

Agent=QLearningRotot((0,0))
Agent.QLearning(10,Environment)
print(Agent.policy)
print(Agent.q_pik)
# print(Agent.v_pik)
#
#
#



            


