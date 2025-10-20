# 好好学习 天天向上
# {2024/6/8} {10:27}
import numpy as np
import random
class SAMap:
    def __init__(self):
        self.returnMap={}
        self.numMap={}

def updateState(stater,statec,action,environment):
    r=stater
    c=statec
    if toWall(stater,statec,action,environment):
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

def calculateReward(state,action,environment):
    state_r=state[0]
    state_c=state[1]
    reward=0
    if toWall(state_r,state_c,action,environment):
        reward=-10
    elif toSafePlace(state_r,state_c,action,environment):
        reward=0
    elif toTarget(state_r,state_c,action,environment):
        reward=1
    else:reward=-10
    return reward

def toWall(state_r,state_c,action,environment):
    if (action==0) and (state_r==0):
        return True
    elif (action==1) and (state_c==(environment.shape[1]-1)):
        return True
    elif (action==2) and (state_r==(environment.shape[0]-1)):
        return True
    elif (action==3) and (state_c==0):
        return True
    else:return False

def toSafePlace(state_r,state_c,action,environment):
    if action==0 and environment[state_r-1,state_c]==0:
        return True
    elif action==1 and environment[state_r,state_c+1]==0:
        return True
    elif action==2 and environment[state_r+1,state_c]==0:
        return True
    elif action==3 and environment[state_r,state_c-1]==0:
        return True
    elif action==4 and environment[state_r,state_c]==0:
        return True
    else:return False

def toTarget(state_r,state_c,action,environment):
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

# def iterateV_Pik(v_pik,q_pik):
#     rowMax=np.max(q_pik,axis=1)
#     print(rowMax)
#     for i in range(v_pik.shape[0]):
#         for j in range(v_pik.shape[1]):
#             v_pik[i][j]=rowMax[i*5+j]
#     print(v_pik)

def episodeGeneration(initial_state,step,policy):
    strideArray=[]
    currentState=initial_state
    # action,newR, newC = Strider(currentState, policy)
    # strideArray.append((initial_state,action))
    while(step):
        action,newR,newC=Strider(currentState,policy)
        strideArray.append((currentState,action))
        currentState=(newR,newC)
        step-=1
    return strideArray

def Strider(current_state,policy):
    Aset=5
    weighsList=[]
    for i in range(5):
        weighsList.append(Epsilon/Aset)
    actionList=[0,1,2,3,4]
    weighsList[policy[current_state[0]][current_state[1]]]=1-((Aset-1)*Epsilon/Aset)
    actionChoice=random.choices(actionList,weighsList)
    # print(actionChoice,weighsList)
    newR,newC=updateState(current_state[0], current_state[1], actionChoice[0], Environment)
    # print(actionChoice[0],newR,newC)
    return actionChoice[0],newR,newC

def policyEvaluation(q_pik,sa_map):
    for key in sa_map.returnMap.keys():
        i=key[0][0]
        j=key[0][1]
        k=key[1]
        q_pik[k][i][j]=sa_map.returnMap[key]/sa_map.numMap[key]

    # for i in range(q_pik.shape[0]):
    #     for j in range(q_pik.shape[1]):
    #         for k in range(q_pik.shape[2]):
    #             if ((i,j),k) in sa_map.returnMap:
    #                 q_pik[k][i][j]=sa_map.returnMap[((i,j),k)]/sa_map.numMap[((i,j),k)]
    #             else:q_pik[i][j][k]=0





Environment=np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 0, 0, 0]
],dtype=float)
Policy = np.random.randint(0, 5, size=(5, 5))  # 随机初始化策略
Q_Pik=np.zeros((5,5,5),dtype=float)
V_Pik=np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
],dtype=float)


saMap = SAMap()
for Episode in range(10):
    # Epsilon=0.001
    Epsilon=1/(1.2*Episode+1)
    if Epsilon<0.001: Epsilon=0.001
    gamma=0.9
    Step=1e5

    initialState=(0,0)
    stateArray=episodeGeneration(initialState,Step,Policy)
    #print(stateArray)
    g=0.0
    for s,a in reversed(stateArray):
        r=calculateReward(s,a,Environment)
        g=gamma*g+r
        key=(s,a)
        if key in saMap.returnMap:
            saMap.returnMap[key]=+g
            saMap.numMap[key]+=1
        else:
            saMap.returnMap[key] = g
            saMap.numMap[key] = 1
        policyEvaluation(Q_Pik,saMap)
        # print(Q_Pik)
    Policy = np.argmax(Q_Pik, axis=0)# updatePolicy(Policy, Q_Pik)
    V_Pik=np.max(Q_Pik,axis=1)# iterateV_Pik(V_Pik, Q_Pik)
print(Policy)
