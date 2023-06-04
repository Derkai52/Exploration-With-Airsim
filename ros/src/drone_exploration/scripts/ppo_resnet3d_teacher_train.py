# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from gym_env.envs.drone_explore_env import DroneExploreEnv
from gym_env.envs.env_parallel import SyncVectorEnv,DroneExploreEnvParallel
from models.ppo_network import Agent_Resnet,Agent_Resnet_Imitate

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="explore_airsim",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="DroneExploreEnv",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=2,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=200,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.05,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    parser.add_argument("--savedmodelpath", type=str, default="/media/mrmmm/Data/Graduation_Data/model/Graduation_Data/ppo_resnet3d_imitate_learning/",
        help="the id of the environment")
    parser.add_argument("--checkpointpath", type=str, default="/media/mrmmm/Data/Graduation_Data/model/Graduation_Data/ppo_resnet3d_imitate_learning/best.pth",
        help="the id of the environment")
    parser.add_argument("--trainingdata", type=str, default="/media/mrmmm/Data/Graduation_Data/Teacher_Data/",
        help="the id of the environment")
        

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args



if __name__ == "__main__":
    #STEP1 接受参数输入，记录log
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    best_pg_loss = 0

    #STEP2 用随机种子初始化
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    #STEP3 环境设置，模型定义，优化器初始化
    agent = Agent_Resnet_Imitate(8).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if os.path.exists(args.checkpointpath):
        checkpoint = torch.load(args.checkpointpath)
        agent.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("模型加载：", args.checkpointpath)


    start_time = time.time()
    global_epoch = checkpoint['global_step']

    ##STEP5 数据加载并训练
    for epoch in range(args.update_epochs):
        for data_file in os.listdir(args.trainingdata + "actions/"):
            if data_file.endswith(".npy"):

                #STEP6.1 学习率调整
                # Annealing the rate if instructed to do so.
                # if args.anneal_lr:
                #     frac = 1.0 - (update - 1.0) / num_updates
                #     lrnow = frac * args.learning_rate
                #     optimizer.param_groups[0]["lr"] = lrnow

                #STEP6.2 数据加载
                actions_file_path = args.trainingdata + "actions/" + data_file
                observations_file_path = args.trainingdata + "observations/" + data_file
                rewards_file_path = args.trainingdata + "rewards/" + data_file

                observations = np.load(observations_file_path)
                observations = np.reshape(observations,(observations.shape[0],1,observations.shape[1],observations.shape[2],observations.shape[3]))
                actions = torch.tensor(np.load(actions_file_path)).to(device)
                rewards = np.load(rewards_file_path)
                print("数据大小:",observations.shape,actions.shape,rewards.shape)
                returns = np.zeros(rewards.shape,dtype=float)
                for t in range(returns.shape[0]):
                    for i in range(t,returns.shape[0]):
                        returns[t] += args.gamma**(i-t)*rewards[t]
                returns = torch.tensor(returns).to(device)


                #STEP6.5 优化策略函数和状态价值函数
                # Optimizing the policy and value network
                b_inds = np.arange(returns.shape[0])
                clipfracs = []
            #for epoch in range(args.update_epochs):
                #STEP6.5.1 随机取样
                correct_num = 0
                np.random.shuffle(b_inds)
                for start in range(0, returns.shape[0], args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    #STEP6.5.2 重新计算网络输出
                    _, action_logits, newvalue = agent.get_action_and_value(torch.tensor(observations[mb_inds]).float().to(device))


                    #STEP6.5.3 总损失及反向传播
                    MSE_loss = nn.MSELoss()
                    cross_entropy_loss = nn.CrossEntropyLoss()
                    
                    actor_loss = cross_entropy_loss(nn.Softmax(dim=0)(action_logits),actions[mb_inds])
                    value_loss = MSE_loss(newvalue.squeeze().float(),returns[mb_inds].squeeze().float())
                    loss = actor_loss + value_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    pred_action = torch.argmax(action_logits,dim=1)
                    correct_num = torch.eq(actions[mb_inds],pred_action).sum().float().item()
                correct_ratio = correct_num/args.minibatch_size
                if global_epoch==1:
                    best_pg_loss = actor_loss.item()

                if(epoch%20 == 0):
                    state = {'net':agent.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'global_step':global_epoch,
                        'pg_loss':actor_loss.item(),
                        'v_loss':value_loss.item()}
                    torch.save(state, args.savedmodelpath+str(epoch)+".pth")
                    print("save best model!!!",epoch)

                if(actor_loss.item() < best_pg_loss):
                    best_pg_loss = actor_loss.item()
                    state = {'net':agent.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'global_step':global_epoch,
                        'pg_loss':actor_loss.item(),
                        'v_loss':value_loss.item()}
                    torch.save(state, args.savedmodelpath+"best.pth")
                    print("save best model!!!")


                print("-------------------- epoch ",global_epoch," --------------------")
                print("correct_ratio: ",correct_ratio, "  value_loss: ", value_loss.item(),"   policy_loss: ", actor_loss.item())
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar("charts/correct_ratio", correct_ratio, global_epoch)
                writer.add_scalar("losses/value_loss", value_loss.item(), global_epoch)
                writer.add_scalar("losses/policy_loss", actor_loss.item(), global_epoch)
                writer.add_scalar("losses/loss", loss.item(), global_epoch)
                global_epoch += 1


    writer.close()
