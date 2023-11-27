import gym
from abc import ABC


class BaseEnv(gym.Env, ABC):
    def __init__(self, configs):
        self.n_j = configs.n_j
        self.n_m = configs.n_m
        self.device = configs.device
        self.task_size = self.n_j * self.n_m

        # 每个task处理时长
        self.task_durations = None
        # 每个task对应的机器编号
        self.task_machines = None

        # ------- 用于辅助计算task的结束时间 ---------

        # 记录机器被task占用的时间段, list of tuples, [(task,起始时间,结束时间),...]
        self.machine_occupied_times = None

        # ------- 用于记录 ---------
        self.episode_reward = 0
        # 当前所有已调度的任务的结束时间，即t时刻的C_max
        self.cur_make_span = 0
        self.t = 0
        self.last_valid_t = 0
        # ------- 用于绘图 ---------
        self.history_make_span = []

    def get_row_col(self, task_id):
        return task_id // self.n_m, task_id % self.n_m

    def render(self):
        pass
