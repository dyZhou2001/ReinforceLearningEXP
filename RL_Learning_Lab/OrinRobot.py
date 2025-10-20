# 好好学习 天天向上
# {2024/6/23} {19:47}
import numpy as np
class OrinRobot:
    '''agent的基类 包含了探索环境和计算当前动作reward的基础功能 后续在此基类上添加不同算法'''
    def __init__(self):
        self.policy=np.zeros((5,5),dtype=int)
        self.q_pik=np.zeros((5,5,5),dtype=float)
        self.v_pik=np.zeros((5,5),dtype=float)
        self.action_set=[0,1,2,3,4]

    def toWall(self,state_r, state_c, action, environment):
        if (action == 0) and (state_r == 0):
            return True
        elif (action == 1) and (state_c == (environment.shape[1] - 1)):
            return True
        elif (action == 2) and (state_r == (environment.shape[0] - 1)):
            return True
        elif (action == 3) and (state_c == 0):
            return True
        else:
            return False

    def toSafePlace(self,state_r, state_c, action, environment):
        if action == 0 and environment[state_r - 1, state_c] == 0:
            return True
        elif action == 1 and environment[state_r, state_c + 1] == 0:
            return True
        elif action == 2 and environment[state_r + 1, state_c] == 0:
            return True
        elif action == 3 and environment[state_r, state_c - 1] == 0:
            return True
        elif action == 4 and environment[state_r, state_c] == 0:
            return True
        else:
            return False

    def toTarget(self,state_r, state_c, action, environment):
        if action == 0 and environment[state_r - 1, state_c] == 2:
            return True
        elif action == 1 and environment[state_r, state_c + 1] == 2:
            return True
        elif action == 2 and environment[state_r + 1, state_c] == 2:
            return True
        elif action == 3 and environment[state_r, state_c - 1] == 2:
            return True
        elif action == 4 and environment[state_r, state_c] == 2:
            return True
        else:
            return False

    def updateState(self,stater,statec,action,environment):
        r=stater
        c=statec
        if self.toWall(stater,statec,action,environment):
            return r,c
        elif action==0:
            r-=1
        elif action==1:
            c+=1
        elif action==2:
            r+=1
        elif action==3:
            c-=1
        elif action==4:
            return r,c
        return r,c

    def calculateReward(self,state, action, environment):
        state_r = state[0]
        state_c = state[1]
        reward = 0
        if self.toWall(state_r, state_c, action, environment):
            reward = -1
        elif self.toSafePlace(state_r, state_c, action, environment):
            reward = 0
        elif self.toTarget(state_r, state_c, action, environment):
            reward = 1
        else:
            reward = -10
        return reward