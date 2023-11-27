import os
import random

import gym
from tensorboardX import SummaryWriter

from gnn_schedule.models.graph_util import *
from gnn_schedule.official.agent_utils import eval_actions
from gnn_schedule.models.actor_critic import ActorCritic
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np


class PPO:
    def __init__(
        self,
        lr,
        gamma,
        k_epochs,
        eps_clip,
        n_j,
        n_m,
        num_layers,
        input_dim,
        hidden_dim,
        num_mlp_layers_feature_extract,
        num_mlp_layers_actor,
        hidden_dim_actor,
        num_mlp_layers_critic,
        hidden_dim_critic,
        configs,
    ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.configs = configs

        self.policy = ActorCritic(
            n_j=n_j,
            n_m=n_m,
            num_layers=num_layers,
            learn_eps=False,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
            num_mlp_layers_actor=num_mlp_layers_actor,
            hidden_dim_actor=hidden_dim_actor,
            num_mlp_layers_critic=num_mlp_layers_critic,
            hidden_dim_critic=hidden_dim_critic,
            device=configs.device,
        )
        self.policy_old = deepcopy(self.policy)

        """self.policy.load_state_dict(
            torch.load(path='./{}.pth'.format(str(n_j) + '_' + str(n_m) + '_' + str(1) + '_' + str(99))))"""

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio
        )

        self.V_loss_2 = nn.MSELoss()
        self.writer = SummaryWriter()

    def update(self, memories, n_tasks):
        vloss_coef = self.configs.vloss_coef
        ploss_coef = self.configs.ploss_coef
        entloss_coef = self.configs.entloss_coef

        rewards_all_env = []
        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        # store data for all env
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(self.configs.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)
            # process each env data
            adj_mb_t_all_env.append(aggr_adjs(memories[i].adj_mb, n_tasks, self.configs.device))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(self.configs.device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(self.configs.device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(self.configs.device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(self.configs.device).squeeze())
            old_logprobs_mb_t_all_env.append(
                torch.stack(memories[i].logprobs).to(self.configs.device).squeeze().detach()
            )

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(memories)):
                pis, vals = self.policy(
                    (adj_mb_t_all_env[i], fea_mb_t_all_env[i], candidate_mb_t_all_env[i], mask_mb_t_all_env[i])
                )
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.view(-1).detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                p_loss = -torch.min(surr1, surr2).mean()

                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                ent_loss = -ent_loss.clone()

                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss_sum += loss
                vloss_sum += v_loss
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        if self.configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()


def set_random_seed(random_seed: int):
    """
    Setup all possible random seeds so results can be reproduced
    """
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    # tf.set_random_seed(random_seed) # if you use tensorflow
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)
    if hasattr(gym.spaces, "prng"):
        gym.spaces.prng.seed(random_seed)
