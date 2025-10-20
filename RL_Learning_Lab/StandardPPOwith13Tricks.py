# 好好学习 天天向上
# {2024/9/10} {下午2:52}
import argparse
import os
import random
from distutils.util import strtobool
import time
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np



def make_env(gym_id,index,video):
    def thunk():
        env=gym.make(gym_id,render_mode="rgb_array")
        env=gym.wrappers.RecordEpisodeStatistics(env)
        if video:
            if index==0:
                env=gym.wrappers.RecordVideo(env,video_folder="videos/",episode_trigger=lambda t:t%40==0)
        return env
    return thunk

def layer_init(layer,std=np.sqrt(2),bias_const=0.0):
    nn.init.orthogonal_(layer.weight,std)
    nn.init.constant_(layer.bias,bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # 输出维度是 1
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),  # 输出维度是动作空间的维度
        )

    def get_critic_value(self, x):
        # print("Critic input shape:", x.shape)  # 调试输出
        value = self.critic(x)
        # print("Critic output shape:", value.shape)  # 调试输出
        return value

    def get_action_and_value(self, x, action=None):
        # print("Actor input shape:", x.shape)  # 调试输出
        logits = self.actor(x)
        # print("Actor logits shape:", logits.shape)  # 调试输出
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(x)
        # print("Actor output shapes - action:", action.shape, "log_prob:", log_prob.shape, "entropy:", entropy.shape, "value:", value.shape)  # 调试输出
        return action, log_prob, entropy, value




def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exp-name',type=str,default='carpole',
                        help='The name of the experiment')
    parser.add_argument('--gym-id', type=str, default='CartPole-v1',
                        help='The id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='The learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='The seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000,
                        help='The total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?',const=True,help='If toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True,
                        nargs='?', const=True, help='If toggled, cuda will not be enabled by default')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='The number of the parallel game environment')
    parser.add_argument('--num-steps', type=int, default=128,
                        help='The number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True,
                        help='The number of steps to run in each environment per policy rollout')
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True,
                        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor gamma')
    parser.add_argument('--gaelambda', type=float, default=0.95,
                        help='lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4,
                        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4,
                        help='the K epochs to update the policy')
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True,
                        help='Toggle advantages normalization')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help='The surrogate clipping coefficient')
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True,
                        help='Toggle whether or not to use a clipped loss for the value function')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='The entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='The value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='The maximum norm of gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
                        help='The target kl divergence threshold')
    args=parser.parse_args()
    args.batch_size= int(args.num_envs*args.num_steps)
    args.minibatch_size=int(args.batch_size//args.num_minibatches)
    return args

if __name__=='__main__':
    args=parse_args()
    # print(args)
    runName=f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer=SummaryWriter(f"runs/{runName}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key,value in vars(args).items() ]))
    )
    # for i in range(100):
    #     Writer.add_scalar('test_loss',i*2,global_step=i)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=args.torch_deterministic

    device='cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

    # envs=gym.make_vec(args.gym_id,num_envs=3)
    # envs=gym.wrappers.vector.RecordEpisodeStatistics(envs)
    # envs=gym.wrappers.RecordVideo(envs,"videos/")
    envs=gym.vector.SyncVectorEnv([make_env(args.gym_id,i,True) for i in range(args.num_envs)])
    agent=Agent(envs).to(device)
    optimizer=optim.Adam(agent.parameters(),lr=args.learning_rate,eps=1e-5)  #epsilon设置为1e-5

    obs=torch.zeros((args.num_steps,args.num_envs)+envs.single_observation_space.shape).to(device) #通过 + 操作符，将这三部分组合成一个元组，表示最终张量的形状
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step=0
    start_time=time.time()
    next_obs, _ = envs.reset()  # 提取观测值，忽略 info
    next_obs = torch.Tensor(next_obs).to(device)
    next_done=torch.zeros(args.num_envs).to(device)
    num_updates=args.total_timesteps // args.batch_size
    # print(num_updates)
    # print(next_obs)
    # print(next_dones)
    # print(agent.get_critic_value(next_obs))
    # print(agent.get_critic_value(next_obs).shape)
    # print(agent.get_action_and_value(next_obs))
    for update in range(1,num_updates+1):
        if args.anneal_lr:
            frac=1.0-(update-1.0)/num_updates
            lrnow=frac*args.learning_rate
            optimizer.param_groups[0]["lr"]=lrnow

        for step in range(0,args.num_steps):
            global_step+=1*args.num_envs
            obs[step]=next_obs
            dones[step]=next_done

            with torch.no_grad():
                action,logprob,_,value=agent.get_action_and_value(next_obs)
                values[step]=value.flatten()
            logprobs[step]=logprob
            actions[step]=action

            next_obs,reward,done,terminated,info=envs.step(action.cpu().numpy())
            rewards[step]=torch.tensor(reward).to(device).view(-1)
            next_obs,next_done=torch.Tensor(next_obs).to(device),torch.Tensor(done).to(device) #必须用Tensor而不是tensor  否则bool无法和float运算
            # if 'episode' in info:
            #     print(global_step,info['episode']['r'])

        with torch.no_grad():
            next_value=agent.get_critic_value(next_obs).reshape(1,-1)
            if args.gae:
                advantages=torch.zeros_like(rewards).to(device)
                last_gae_lam=0
                for t in reversed(range(args.num_steps)):
                    if t==(args.num_steps-1):
                        next_non_terminal=1.0-next_done
                        next_values=next_value
                    else:
                        next_non_terminal=1.0-dones[t+1]
                        next_values=values[t+1]
                    delta=rewards[t]+args.gamma*next_non_terminal*next_values-values[t]
                    advantages[t]=last_gae_lam=delta+args.gamma*args.gaelambda*next_non_terminal*last_gae_lam
                returns=advantages+values

            else:
                returns=torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == (args.num_steps-1):
                        next_non_terminal=1.0-next_done
                        next_return=next_value
                    else:
                        next_non_terminal=1.0-dones[t+1]
                        next_return=returns[t+1]
                    returns[t]=rewards[t]+args.gamma*next_non_terminal*next_return
                advantages=returns-values

        b_obs=obs.reshape((-1,)+envs.single_observation_space.shape)
        b_logprobs=logprobs.reshape(-1)
        b_advantages=advantages.reshape(-1)
        b_returns=returns.reshape(-1)
        b_actions=actions.reshape((-1,)+envs.single_action_space.shape)
        b_values=values.reshape(-1)

        b_inds=np.arange(args.batch_size)
        clipfracs=[]
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0,args.batch_size,args.minibatch_size):
                end=start+args.minibatch_size
                mb_inds=b_inds[start:end]
                # print('start and end index',start,end)
                _,newlogprob,entropy,newvalue=agent.get_action_and_value(b_obs[mb_inds],b_actions.long()[mb_inds])
                logratio=newlogprob-b_logprobs[mb_inds]
                ratio=logratio.exp()
                # print(ratio)

                with torch.no_grad():
                    old_approx_kl=(-logratio).mean()
                    approx_kl=((ratio-1)-logratio).mean()
                    clipfracs+=[((ratio-1.0).abs()>args.clip_coef).float().mean().item()]


                mb_advantages=b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages=(mb_advantages-mb_advantages.mean())/(mb_advantages.std()+1e-8)
                pgloss1=-mb_advantages*ratio
                pgloss2=-mb_advantages*torch.clip(ratio,1-args.clip_coef,1+args.clip_coef)
                pgloss=torch.max(pgloss1,pgloss2).mean()

                newvalue=newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped=(newvalue-b_returns[mb_inds])**2
                    v_clipped=b_values[mb_inds]+torch.clip(newvalue-b_returns[mb_inds],-args.clip_coef,args.clip_coef)
                    v_loss_clipped=(v_clipped-b_returns[mb_inds])**2
                    vloss=torch.max(v_loss_unclipped,v_loss_clipped).mean()*0.5
                else:
                    vloss=0.5*((newvalue-b_returns[mb_inds])**2).mean()

                entropy_loss=entropy.mean()
                loss=pgloss-args.entropy_coef*entropy_loss+args.vf_coef*vloss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(agent.parameters(),args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl>args.target_kl:
                    break

        y_pred,y_true=b_values.cpu().numpy(),b_returns.cpu().numpy()
        var_y=np.var(y_true)
        explained_var=np.nan if var_y==0 else 1-np.var(y_true-y_pred)/var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", vloss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pgloss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)



    envs.close()

#tensorboard --logdir=runs