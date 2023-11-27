import os
import pickle
import random

import torch
import numpy as np
import global_util
from gnn_schedule.official.my_memory import MyMemory
from jssp_tool.util import logger
from tensorboardX import SummaryWriter

from gnn_schedule.util.result_generator import to_dataframe


class Runner:
    def __init__(self, configs, env, vali_data):
        self.configs = configs
        self.output = os.path.join(global_util.get_project_root(), self.configs.output)
        self.model_dir = os.path.join(self.output, configs.model_dir)
        os.makedirs(self.model_dir, exist_ok=True)

        self.env = env
        self.vali_data = vali_data
        self.device = configs.device
        self.writer = SummaryWriter()
        self.gamma = configs.gamma

    def to_tensor(self, adj, fea, candidate, mask):
        adj_tensor = torch.from_numpy(np.copy(adj)).to(self.device).to_sparse()
        fea_tensor = torch.from_numpy(np.copy(fea)).to(self.device)
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(self.device).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(self.device).unsqueeze(0)
        return adj_tensor, fea_tensor, candidate_tensor, mask_tensor

    def collect_data(self, ppo, memories, ep_rewards, ep_makespan):
        n_j = random.randint(6, 10)
        n_m = random.randint(6, 10)
        with torch.no_grad():
            for i in range(self.configs.num_envs):
                obs, _ = self.env.reset(n_j=n_j, n_m=n_m)
                done = False
                while not done:
                    obs = self.to_tensor(*obs)
                    pi, value = ppo.policy_old(obs)
                    action, a_idx, logprob = ppo.sample_action(pi, obs[2])
                    next_obs, reward, done, _, _ = self.env.step(action.item())
                    memories[i].append(obs, a_idx, reward, done, logprob, value)
                    obs = next_obs

                    ep_rewards[i] += reward
                memories[i].compute_monte_carlo_returns(self.gamma)
                ep_makespan[i] = self.env.cur_make_span

    def train(self, agent):
        memories = [MyMemory() for _ in range(self.configs.num_envs)]
        best_result = float("inf")
        for i_update in range(self.configs.max_updates):
            ep_rewards = [0 for _ in range(self.configs.num_envs)]
            ep_makespan = [0 for _ in range(self.configs.num_envs)]

            # 收集数据
            self.collect_data(agent, memories, ep_rewards, ep_makespan)
            # 训练模型
            v_loss, a_loss, e_loss = agent.update(memories)
            loss_sum = v_loss + a_loss + e_loss
            for memory in memories:
                memory.clear()

            # 以下均为记录或者验证
            mean_ep_reward = sum(ep_rewards) / len(ep_rewards)

            # log results
            logger.info(
                "Episode {}\t Last reward: {:.2f}\t loss: {:.8f}\t make span:{}".format(
                    i_update + 1, mean_ep_reward, loss_sum, sum(ep_makespan) / len(ep_makespan)
                )
            )
            self.writer.add_scalar("train/reward", mean_ep_reward, i_update)
            self.writer.add_scalar("train/loss", loss_sum, i_update)
            self.writer.add_scalar("train/make_span", sum(ep_makespan) / len(ep_makespan), i_update)

            if (i_update + 1) % 100 == 0:
                best_result = self.test(self.vali_data, agent, best_result, i_update)

    def test(self, vali_data, ppo, best_result, i_update, phase="val"):
        make_spans = []
        schedule_list = []
        for data in vali_data:
            obs, _ = self.env.reset(data=data)
            rewards = 0
            while True:
                obs = self.to_tensor(*obs)
                with torch.no_grad():
                    pi, _ = ppo.policy(obs)
                action = ppo.greedy_select_action(pi, obs[2])
                obs, reward, done, _, _ = self.env.step(action.item())
                rewards += reward
                if done:
                    break
            make_spans.append(self.env.cur_make_span)
            if phase == "test":
                df_schedule = to_dataframe(self.env.operations)
                schedule_list.append(df_schedule)

        avg_makespan = np.mean(make_spans)
        if avg_makespan < best_result:
            torch.save(ppo.policy.state_dict(), os.path.join(self.output, "best.pth"))
            best_result = avg_makespan
        logger.info("i_update: {}, 测试平均制造周期：{}".format(i_update, avg_makespan))
        self.writer.add_scalar("val/平均制造周期", avg_makespan, i_update)

        if phase == "test":
            with open(os.path.join(self.output, "obj{}.pickle".format(self.env.n_j)), "wb") as f:
                pickle.dump(make_spans, f)
            with open(os.path.join(self.output, "sol{}.pickle".format(self.env.n_j)), "wb") as f:
                pickle.dump(schedule_list, f)

        return best_result
