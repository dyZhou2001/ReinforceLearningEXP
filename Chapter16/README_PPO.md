# PPO强化学习完整实现

这是一个完整的PPO（Proximal Policy Optimization）强化学习智能体实现，专门为倒立摆（Pendulum-v1）环境设计，但也可以轻松适配其他连续控制环境。

## 特点

- **完全自包含**: 所有代码在一个文件中，无需ptan库
- **详细注释**: 每个组件都有清晰的中文注释
- **模块化设计**: Actor、Critic、Buffer等组件分离
- **完整实现**: 包含GAE、PPO截断、熵正则化等关键技术
- **易于使用**: 提供简单的训练和测试脚本

## 文件结构

```
├── PPOALLINONE.py      # 主要的PPO实现文件
├── test_ppo.py         # 模型测试脚本
├── run_ppo.py          # 训练启动脚本
├── requirements_ppo.txt # 依赖包列表
└── README_PPO.md       # 说明文档（本文件）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_ppo.txt
```

### 2. 开始训练

**方法1: 使用启动脚本（推荐）**
```bash
python run_ppo.py
```

**方法2: 直接运行**
```bash
python PPOALLINONE.py --max_episodes 1000
```

### 3. 测试模型

```bash
python test_ppo.py --model_path ./ppo_saves/best_model.pth --render
```

## 算法原理

### PPO核心思想

PPO通过限制策略更新的幅度来保证训练稳定性：

1. **策略比率**: `r(θ) = π_θ(a|s) / π_θ_old(a|s)`
2. **截断目标**: `L^CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)`
3. **总损失**: `L = L^CLIP - c1*L^VF + c2*H[π_θ]`

### 关键组件

#### 1. Actor网络
- 输出动作的均值μ(s)
- 学习可训练的对数标准差log_σ
- 使用Normal分布采样动作

#### 2. Critic网络
- 估计状态价值函数V(s)
- 用于计算优势函数A(s,a)

#### 3. GAE优势估计
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)
A_t = δ_t + γλA_{t+1}
```

#### 4. 经验缓冲区
- 存储完整轨迹数据
- 计算GAE和回报
- 支持批量更新

## 超参数说明

| 参数 | 值 | 说明 |
|------|----|----|
| GAMMA | 0.99 | 折扣因子 |
| GAE_LAMBDA | 0.95 | GAE参数λ |
| TRAJECTORY_SIZE | 2048 | 轨迹长度 |
| PPO_EPS | 0.2 | PPO截断参数ε |
| PPO_EPOCHS | 10 | 每次更新的轮数 |
| PPO_BATCH_SIZE | 64 | 批量大小 |
| LEARNING_RATE_ACTOR | 3e-4 | Actor学习率 |
| LEARNING_RATE_CRITIC | 1e-3 | Critic学习率 |
| ENTROPY_COEF | 0.01 | 熵系数 |
| VALUE_LOSS_COEF | 0.5 | 价值损失系数 |

## 训练监控

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir ./ppo_logs
```

可以观察的指标：
- Episode/Reward: 每个episode的奖励
- Loss/Policy: 策略损失
- Loss/Value: 价值损失  
- Loss/Entropy: 熵损失
- Training/KL_Divergence: KL散度
- Training/Clip_Fraction: 截断比例

## 环境适配

要适配其他环境，只需修改以下参数：

1. 改变环境名称：
```python
parser.add_argument("--env", default="CartPole-v1", help="Environment name")
```

2. 对于离散动作空间，需要修改Actor网络输出层。

## 性能调优建议

### 提升训练效果：
1. **调整学习率**: 降低学习率可能提高稳定性
2. **增加网络深度**: 对复杂任务使用更深的网络
3. **调整GAE_LAMBDA**: 影响偏差-方差权衡
4. **修改熵系数**: 平衡探索和利用

### 加速训练：
1. **增加批量大小**: 提高GPU利用率
2. **减少PPO epochs**: 降低每次更新时间
3. **并行环境**: 使用多个环境并行收集数据

## 常见问题

### Q: 训练不收敛怎么办？
A: 
1. 降低学习率
2. 增加轨迹长度
3. 调整PPO截断参数
4. 检查奖励函数设计

### Q: 如何保存中间结果？
A: 程序会自动每1000个episode保存检查点，每50个episode测试并可能保存最佳模型。

### Q: 内存不足怎么办？
A: 
1. 减少TRAJECTORY_SIZE
2. 减少PPO_BATCH_SIZE  
3. 使用CPU训练

### Q: 如何可视化结果？
A: 使用`--render`参数运行test_ppo.py，或在训练时定期查看TensorBoard。

## 技术细节

### 网络架构
- Actor: 3层全连接网络，隐藏层64个神经元，Tanh激活
- Critic: 3层全连接网络，隐藏层64个神经元，ReLU激活

### 动作空间处理
- 连续动作使用正态分布采样
- 动作被截断到[-1, 1]范围
- 支持多维动作空间

### 数值稳定性
- 梯度截断（0.5）
- 标准差截断（1e-20 到 2）
- 优势标准化

## 扩展建议

1. **添加RNN支持**: 处理部分可观察环境
2. **实现并行训练**: 使用多进程收集经验
3. **添加好奇心机制**: 提升探索能力
4. **实现分层强化学习**: 处理复杂任务

## 参考文献

1. Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
2. Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).

## 许可证

MIT License