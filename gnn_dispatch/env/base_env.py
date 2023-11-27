import copy
from abc import ABC

import gym
import numpy as np


class BaseEnv(gym.Env, ABC):
    def __init__(self, n_j, n_m, device):
        self.n_j = n_j
        self.n_m = n_m
        self.device = device
        self.task_size = n_j * n_m

        # 每个task处理时长
        self.task_durations = None
        # 每个task对应的机器编号
        self.task_machines = None
        # 已经调度了的task的id列表
        self.scheduled_task_ids = None

        # ------- 用于辅助计算task的结束时间 ---------
        # 已调度任务的结束时间
        self.task_finish_times = None

        # self.machine_start_times = None
        # self.machine_scheduled_tasks = None
        # 记录机器被task占用的时间段, list of tuples, [(task,起始时间,结束时间),...]
        self.machine_occupied_times = None

        # ------- 用于记录 ---------
        self.episode_reward = 0
        # 当前所有已调度的任务的结束时间，即t时刻的C_max
        self.cur_make_span = 0
        self.step_count = 0
        # ------- 用于绘图 ---------
        self.history_make_span = []

    def compute_task_schedule_time(self, task_id, row, col):
        """
        对于新来的task，检测是否可以插入当前已调度机器时间中的空闲片段:
            如果插入会导致覆盖已有调度时间片段，则放到最后;
            否则可以插入空闲片段
        Args:
            task_id:
            row:
            col:

        Returns:

        """
        job_pre_task_col = col - 1 if col > 0 else 0
        job_task_ready_time = self.task_finish_times[row, job_pre_task_col]
        task_duration = self.task_durations[row, col]

        machine_id = self.task_machines[row, col]
        # 寻找可用的插入空隙
        machine_occupied_times = copy.deepcopy(self.machine_occupied_times[machine_id])

        ind = -1
        for i, (oid, ostart, oend) in enumerate(machine_occupied_times):
            if ostart > job_task_ready_time:
                ind = i
                break
        # 机器中开始时间没有晚于当前task开始时间的，直接放到最后
        if ind == -1:
            self.put_end(task_id, machine_id, job_task_ready_time, task_duration)
        else:
            # 添加虚拟的时间片段,方便间隔计算
            machine_occupied_times.insert(ind, (task_id, job_task_ready_time, job_task_ready_time + task_duration))
            inserted = False
            # 计算可用的空闲间隔
            for i in range(ind, len(machine_occupied_times) - 1):
                # 对于后续位置，计算每个位置之间的时间间隔
                if i == ind:
                    start_time = max(machine_occupied_times[i][1], machine_occupied_times[i - 1][2] if i > 0 else 0)
                else:
                    start_time = machine_occupied_times[i][2]
                gap = machine_occupied_times[i + 1][1] - start_time
                # 判断空闲间隔是否足够当前task执行
                if gap >= task_duration:
                    self.put_between(task_id, machine_id, start_time, i, task_duration)
                    inserted = True
                    break
            # 找不到合适的空闲间隔，放到最后
            if not inserted:
                self.put_end(task_id, machine_id, job_task_ready_time, task_duration)

        # 更新机器完成周期
        self.cur_make_span = np.max(self.task_finish_times)

    def put_between(self, task_id, machine_id, start_time, insert_pos, task_duration):
        item = self.machine_occupied_times[machine_id][insert_pos]
        # 插入task_scheduled_id
        ind = np.argwhere(np.array(self.scheduled_task_ids) == item[0])
        self.scheduled_task_ids.insert(ind[0][0], task_id)

        row, col = task_id // self.n_m, task_id % self.n_m
        self.task_finish_times[row, col] = start_time + task_duration
        self.machine_occupied_times[machine_id].insert(insert_pos, (task_id, start_time, start_time + task_duration))

    def put_end(self, task_id, machine_id, job_task_ready_time, task_duration):
        self.scheduled_task_ids.append(task_id)

        # 计算start time
        machine_ready_time = 0
        if len(self.machine_occupied_times[machine_id]) > 0:
            machine_ready_time = self.machine_occupied_times[machine_id][-1][2]
        start_time = max(machine_ready_time, job_task_ready_time)

        row, col = task_id // self.n_m, task_id % self.n_m
        self.task_finish_times[row, col] = start_time + task_duration
        self.machine_occupied_times[machine_id].append((task_id, start_time, start_time + task_duration))
