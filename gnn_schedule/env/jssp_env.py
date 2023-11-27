import copy
import itertools
from typing import Tuple

import numpy as np
from gym.core import ObsType

from gnn_schedule.env.node import Operation, Machine, Job
from gnn_schedule.env.node import OP_STATUS_DONE, OP_STATUS_PROCESSING, OP_STATUS_READY, OP_STATUS_NOT_SCHEDULED
from gnn_schedule.util.data_generator import gen_instance_uniformly
from gnn_schedule.env.base_env import BaseEnv


class JsspEnv(BaseEnv):
    def __init__(self, configs):
        """
        机器作业调度环境
        non-trivial state: t时刻至少有一台机器空闲(processing_ops为空)，并且该机器上的ready_ops不为空
        """
        super().__init__(configs)
        self.dur_low = configs.low
        self.dur_high = configs.high
        self.gamma = configs.gamma

        self.op_ids = None
        self.jobs = []
        self.operations = None
        self.machines = None
        self.available_machines = None
        self.feature_dim = 0
        self.adj_matrix = None
        self.disjunctive_matrix = None
        self.precedent_matrix = None
        self.succedent_matrix = None
        self.all_item_matrix = None
        self.finish_marks = None
        self.schedule_actions = None

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        self.episode_reward = 0
        self.cur_make_span = 0
        self.t = 0
        self.last_valid_t = 0
        self.machine_occupied_times = [[] for _ in range(self.n_m)]
        self.schedule_actions = []

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
        self.all_item_matrix = np.ones((self.n_j * self.n_m, self.n_j * self.n_m), dtype=np.float32)
        self.op_ids = np.array([i for i in range(self.n_j * self.n_m)], dtype=np.int64).reshape((self.n_j, self.n_m))
        self.finish_marks = np.array([False for _ in range(self.n_j * self.n_m)], dtype=np.bool_).reshape(
            (self.n_j, self.n_m)
        )
        self.init_operations(self.op_ids)
        self.init_matrices(self.op_ids)
        self.update()
        obs = self.get_observation()
        info = {}

        return obs, info

    def step(self, action: int) -> Tuple[ObsType, float, bool, bool, dict]:
        self.schedule_actions.append(action)
        reward = self.transit(self.operations[action])
        obs = self.get_observation()
        terminated = self.done()
        info = {}
        self.episode_reward += reward

        if terminated:
            self.compute_makespan()

        return obs, reward, terminated, False, info

    def get_observation(self):
        # 每一个node的precedent,succedent,disjunctive neighbor的matrix
        adjacency_matrices = (
            self.precedent_matrix,
            self.succedent_matrix,
            self.disjunctive_matrix,
            self.all_item_matrix,
        )
        features = np.zeros((self.task_size, *self.feature_dim), dtype=np.float32)
        candidates = np.ones(self.n_j, dtype=np.int64) * -1
        # 标志哪些op是不可用的，为了凑数，保证每次action size相等
        undoable_masks = np.array([True for _ in range(self.n_j)])

        trivial_feature_rows = [i for i in range(self.n_j)]
        for op in self.operations:
            if op.node_status == OP_STATUS_READY:
                # 赋值有意义的op（ready_op)
                row, col = self.get_row_col(op.id)
                candidates[row] = op.id
                if self.machines[op.machine_id].available():
                    undoable_masks[row] = False
                trivial_feature_rows.remove(row)
            features[op.id] = op.to_array(self.n_m)

        # 对于正在运行的job，取其正在运行/完工的op特征
        for row in trivial_feature_rows:
            ops = self.jobs[row].ops
            for i in range(len(ops) - 1, -1, -1):
                if ops[i].node_status == OP_STATUS_DONE or ops[i].node_status == OP_STATUS_PROCESSING:
                    candidates[row] = ops[i].id
                    break

        return adjacency_matrices, features, candidates, undoable_masks

    def transit(self, cur_op: Operation):
        cumulative_reward = 0
        for machine in self.machines:
            # 从ready_ops中删除op，processing_ops加入op
            if cur_op in machine.ready_ops:
                job_pre_op_end_time = cur_op.pre_op.end_time if cur_op.pre_op is not None else 0
                machine_ready_time = machine.done_ops[-1].end_time if len(machine.done_ops) > 0 else 0
                assert max(job_pre_op_end_time, machine_ready_time) == self.t
                if max(job_pre_op_end_time, machine_ready_time) != self.t:
                    print()

                cur_op.set_start_end_time(self.t, self.t + cur_op.processing_time)
                cur_op.node_status = OP_STATUS_PROCESSING

                machine.ready_ops.remove(cur_op)
                machine.processing_ops.append(cur_op)

                # 将该机器的其他备选项放回unscheduled
                for op in reversed(machine.ready_ops):
                    machine.unscheduled_ops.append(op)
                    op.node_status = OP_STATUS_NOT_SCHEDULED
                    machine.ready_ops.remove(op)
                break

        # 转到下一个有意义的状态
        count = 0
        in_trivial_state = True
        while in_trivial_state:
            in_trivial_state, reward = self.update()
            cumulative_reward += pow(self.gamma, count) * reward
            if self.done():
                break
            if in_trivial_state:
                self.t += 1
                for machine in self.machines:
                    for op in machine.ready_ops:
                        op.waiting_time += 1

        return cumulative_reward

    def compute_makespan(self):
        for op in self.operations:
            if self.cur_make_span < op.end_time:
                self.cur_make_span = op.end_time

    def done(self):
        for machine in self.machines:
            if len(machine.done_ops) != self.n_j:
                return False
        return True

    def update(self):
        in_trivial_state = True
        reward = 0

        for op in self.operations:
            if op.node_status == OP_STATUS_PROCESSING:
                op.remaining_time = op.end_time - self.t
                if self.t >= op.end_time:
                    op.node_status = OP_STATUS_DONE
                    op.remaining_time = -1

        # 更新节点状态并判断是否为non-trivial states
        for machine in self.machines:
            if len(machine.done_ops) == self.n_j:
                continue
            # 根据当前时间更新op是否结束
            for op in reversed(machine.processing_ops):
                if op.node_status == OP_STATUS_DONE:
                    machine.processing_ops.remove(op)
                    machine.done_ops.append(op)
                    self.finish_marks[self.get_row_col(op.id)] = True
            for op in reversed(machine.unscheduled_ops):
                # JOB维度上该op无pre_op或者pre_op已经处理完成，且机器空闲，则该op可以进入ready_ops，准备被选
                if op.pre_op is None or op.pre_op.node_status == OP_STATUS_DONE and len(machine.processing_ops) == 0:
                    op.node_status = OP_STATUS_READY
                    machine.unscheduled_ops.remove(op)
                    machine.ready_ops.append(op)

            # 机器空闲，且机器中有可调度op，为有意义的状态
            if len(machine.ready_ops) > 0 and len(machine.processing_ops) == 0:
                in_trivial_state = False
            # reward为负的等待op数量
            reward -= len(machine.ready_ops)

        return in_trivial_state, reward

    def init_operations(self, op_ids):
        """
        初始化operation数据，加入机器中
        :return:
        """
        self.jobs = []
        self.operations = []
        machine_operations = {i: [] for i in range(self.n_m)}

        cumsum_durations = np.cumsum(self.task_durations, axis=1)
        complete_ratios = cumsum_durations / cumsum_durations[:, -1].reshape(-1, 1)
        for i in range(self.n_j):
            pre_op = None
            job_ops = []
            for j in range(self.n_m):
                op = Operation(
                    i,
                    op_ids[i, j],
                    pre_op,
                    self.task_durations[i, j],
                    complete_ratios[i, j],
                    self.n_m - j,
                    cumsum_durations[i, -1],
                    self.task_machines[i, j],
                )
                self.operations.append(op)
                machine_operations[self.task_machines[i, j]].append(op)
                pre_op = op
                job_ops.append(op)
            self.jobs.append(Job(job_ops))
        self.feature_dim = self.operations[0].to_array(self.n_m).shape
        self.machines = []
        self.available_machines = []
        for i in range(self.n_m):
            self.machines.append(Machine(i, machine_operations[i]))
            self.available_machines.append(self.machines[-1])

    def init_matrices(self, op_ids):
        # adjacency,precedent,succedent,disjunctive neighbor的matrix

        # conjunctive
        self.adj_matrix = np.eye(self.task_size, dtype=np.single)
        for i in range(1, 1 + self.task_size):
            if i == 0 or i % self.n_m != 0:
                for j in range(1, self.task_size):
                    if i == j:
                        self.adj_matrix[i - 1, j] = 1
                        self.adj_matrix[j, i - 1] = 1

        self.disjunctive_matrix = np.eye(self.task_size, dtype=np.single)
        # disjunctive
        machine_tasks = [[] for _ in range(self.n_m)]
        for i in range(self.task_size):
            machine = np.take(self.task_machines, i)
            machine_tasks[machine].append(np.take(op_ids, i))
        for task_ids in machine_tasks:
            for c in itertools.combinations(task_ids, 2):
                self.adj_matrix[c[0], c[1]] = 1
                self.adj_matrix[c[1], c[0]] = 1
                self.disjunctive_matrix[c[0], c[1]] = 1
                self.disjunctive_matrix[c[1], c[0]] = 1

        self.precedent_matrix = copy.deepcopy(self.adj_matrix)
        self.succedent_matrix = copy.deepcopy(self.adj_matrix)
