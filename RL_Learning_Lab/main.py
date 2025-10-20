# 好好学习 天天向上
# {2024/5/30} {16:39}
import numpy as np

def cauculateQ_Pik(state_r,state_c,action,v_pik,environment):
    Gamma=0.9
    r=cauculateReward(state_r,state_c,action,environment)
    row,cal=updateState(state_r,state_c,action,environment)
    return r+Gamma*v_pik[row,cal]

def updatePolicy(policy,q_pik):
    rowMaxIndices=np.argmax(q_pik,axis=1)
    print(rowMaxIndices)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            policy[i,j]=rowMaxIndices[i*5+j]
    print(policy)

def iterateV_Pik(v_pik,q_pik):
    rowMax=np.max(q_pik,axis=1)
    print(rowMax)
    for i in range(v_pik.shape[0]):
        for j in range(v_pik.shape[1]):
            v_pik[i][j]=rowMax[i*5+j]
    print(v_pik)

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



def cauculateReward(state_r,state_c,action,environment):
    reward=0
    if toWall(state_r,state_c,action,environment):
        reward=-1
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

Environment=np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 0, 0, 0]
],dtype=np.float)
Policy=np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
Q_Pik=np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
],dtype=np.float)
V_Pik=np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
],dtype=np.float)
for Episode in range(20):
    for i in range(Environment.shape[0]):
        for j in range(Environment.shape[1]):
            for action in range(5):
                Q_Pik[i*5+j][action] = cauculateQ_Pik(i, j, action, V_Pik, Environment)

    updatePolicy(Policy, Q_Pik)
    iterateV_Pik(V_Pik, Q_Pik)

print(Q_Pik)
print(Policy)
print(V_Pik)
# cauculateQ_Pik(state_r=2,state_c=4,action=1,v_pik=V_Pik,environment=Environment)

