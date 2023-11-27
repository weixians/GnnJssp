import copy
import random
from typing import Tuple

import numpy as np
from gym.core import ObsType

from jssp_tool.env.data.data_generator import gen_instance_uniformly
from gnn_dispatch.env.base_env import BaseEnv


class JsspEnv(BaseEnv):
    def __init__(self, n_j, n_m, dur_low, dur_high, device):
        """
        机器作业调度环境
        Args:
            n_j:  job数量
            n_m: machine数量
            dur_low: task最短处理时间
            dur_high: task最长处理时间
        """
        super().__init__(n_j, n_m, device)
        self.dur_low = dur_low
        self.dur_high = dur_high

        # 给每个task起个id
        self.task_ids = None

        # 标记哪些task已调度
        self.scheduled_marks = None
        # 邻接矩阵
        self.adj_matrix = None
        self.estimated_max_end_time = 0
        self.low_bounds = None

        # 当前可供选择的task id，最多n_j个
        self.candidates = None
        # 标志哪些job是否结束
        self.mask = None

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        self.episode_reward = 0
        self.cur_make_span = 0
        self.step_count = 0
        if "data" in kwargs:
            self.task_durations, self.task_machines = kwargs.get("data")
            self.n_j, self.n_m = self.task_durations.shape
        else:
            self.n_j = kwargs.get("n_j")
            self.n_m = kwargs.get("n_m")
            # 生成一个新的调度案例
            self.task_durations, self.task_machines = gen_instance_uniformly(
                self.n_j, self.n_m, self.dur_low, self.dur_high
            )
        self.task_size = self.n_j * self.n_m
        self.machine_occupied_times = [[] for _ in range(self.n_m)]
        self.task_ids = np.array([i for i in range(self.n_j * self.n_m)], dtype=np.int64).reshape((self.n_j, self.n_m))
        self.scheduled_task_ids = []
        self.scheduled_marks = np.zeros_like(self.task_machines, dtype=np.int32)
        self.low_bounds = np.cumsum(self.task_durations, axis=1, dtype=np.single)
        self.estimated_max_end_time = np.max(self.low_bounds)
        self.adj_matrix = self.build_adjacency_matrix()
        self.task_finish_times = np.zeros_like(self.task_durations)

        feature = np.concatenate(
            [self.low_bounds.reshape(-1, 1) / 1000, self.scheduled_marks.reshape(-1, 1)], axis=1, dtype=np.float32
        )
        self.candidates = copy.deepcopy(self.task_ids[:, 0])
        self.mask = np.array([False for _ in range(self.n_j)])
        obs = copy.deepcopy((self.adj_matrix, feature, self.candidates, self.mask))
        obs = self._to_tensor(obs)
        info = {}

        return obs, info

    def step(self, task_id: int) -> Tuple[ObsType, float, bool, bool, dict]:
        # task_id = self.candidates[action]
        # task_id=action

        if task_id not in self.scheduled_task_ids:
            # self.scheduled_task_ids.append(task_id)
            self.step_count += 1

            row = task_id // self.n_m
            col = task_id % self.n_m
            self.scheduled_marks[row, col] = 1
            # 计算当前task结束时间
            self.compute_task_schedule_time(task_id, row, col)
            self.update_low_bounds(row, col)
            self.adj_matrix = self.build_adjacency_matrix()

        # 如果不是某个job的最后一个task，更新下一次的candidate
        if task_id not in self.task_ids[:, -1]:
            self.candidates[task_id // self.n_m] += 1
        else:
            # 标志某个job调度结束
            self.mask[task_id // self.n_m] = True
        # 寻找
        feature = np.concatenate(
            [self.low_bounds.reshape(-1, 1) / 1000, self.scheduled_marks.reshape(-1, 1)],
            axis=1,
            dtype=np.float32,
        )
        obs = copy.deepcopy((self.adj_matrix, feature, self.candidates, self.mask))
        obs = self._to_tensor(obs)
        reward = self.estimated_max_end_time - np.max(self.low_bounds)
        self.estimated_max_end_time = np.max(self.low_bounds)
        terminated = len(self.scheduled_task_ids) == self.task_size
        info = {}
        self.episode_reward += reward

        return obs, reward, terminated, False, info

    def render(self):
        pass

    def update_low_bounds(self, row, col):
        # 更新当前task的结束时间
        self.low_bounds[row, col] = self.task_finish_times[row, col]
        # 对于其他未调度的task，采用加上前面完成时间作为low bounds
        for i in range(self.n_j):
            for j in range(self.n_m):
                if self.task_finish_times[i, j] == 0:
                    if j == 0:
                        self.low_bounds[i, j] = self.task_durations[i, j]
                    else:
                        self.low_bounds[i, j] = self.task_durations[i, j] + self.low_bounds[i, j - 1]

    def build_adjacency_matrix(self):
        adj_matrix = np.eye(self.task_size, dtype=np.single)
        for i in range(1, 1 + self.task_size):
            if i == 0 or i % self.n_m != 0:
                for j in range(1, self.task_size):
                    if i == j:
                        adj_matrix[i - 1, j] = 1

        # 根据scheduled_task_ids更新邻接矩阵
        for tasks in self.machine_occupied_times:
            for i in range(0, len(tasks) - 1):
                adj_matrix[tasks[i][0], tasks[i + 1][0]] = 1

        return adj_matrix

    def _to_tensor(self, obs):
        adj, feature, candidate, finish_mark = obs
        # adj = torch.tensor(adj.T, dtype=torch.float32).to(self.device).to_sparse()
        # feature = torch.tensor(feature, dtype=torch.float32).to(self.device)
        # candidate = torch.tensor(candidate, dtype=torch.int64).to(self.device).unsqueeze(0)
        # finish_mark = torch.tensor(finish_mark, dtype=torch.bool).to(self.device).unsqueeze(0)
        return adj.T, feature, candidate, finish_mark
