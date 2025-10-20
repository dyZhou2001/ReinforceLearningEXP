import gymnasium as gym  # 新版gym库
import numpy as np  # 数值计算
import torch  # PyTorch主库

# 自定义状态预处理函数，替代ptan.agent.float32_preprocessor
def float32_preprocessor(states):
    if isinstance(states, list):
        states = np.array(states)
    return torch.FloatTensor(states)
# 环境ID
# ENV_ID = "CarRacing-v2"
ENV_ID = "Pendulum-v1"  # 使用更简单的Pendulum环境进行测试
# CarRacing-v2为标准环境，无需自定义注册

def register_env():
    pass  # 保留接口，实际不做任何操作



# DDQN批量数据解包
def unpack_batch_ddqn(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        # 判断是否终止
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    # 转为张量
    states_v = float32_preprocessor(states).to(device)
    actions_v = float32_preprocessor(actions).to(device)
    rewards_v = float32_preprocessor(rewards).to(device)
    last_states_v = float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v
