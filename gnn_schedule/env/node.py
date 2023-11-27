import numpy as np

# STATUS_NOT_SCHEDULED = [1, 0, 0]
# STATUS_PROCESSING = [0, 1, 0]
# STATUS_DONE = [0, 0, 1]

OP_STATUS_NOT_SCHEDULED = "not_scheduled"
OP_STATUS_READY = "ready"
OP_STATUS_PROCESSING = "processing"
OP_STATUS_DONE = "done"


class Operation:
    def __init__(
        self, job_id, op_id, pre_op, processing_time, complete_ratio, remaining_op_num, job_processing_time, machine_id
    ):
        self.job_id = job_id
        self.id = op_id
        self.pre_op = pre_op

        # node feature
        self.node_status = OP_STATUS_NOT_SCHEDULED
        self.processing_time = processing_time  # 定值
        self.complete_ratio = complete_ratio  # 定值
        self.remaining_op_num = remaining_op_num  # 定值
        self.waiting_time = 0
        self.remaining_time = -1

        # 用于输入模型时，归一化op的processing_time
        self.job_processing_time = job_processing_time  # 定值

        # 记录用
        # 开始加工时间
        self.start_time = None
        # 结束加工时间
        self.end_time = None
        self.machine_id = machine_id

    def set_start_end_time(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time

    def to_array(self, n_m) -> np.ndarray:
        if self.node_status == OP_STATUS_NOT_SCHEDULED:
            node_status = [1, 0, 0]
        elif self.node_status == OP_STATUS_PROCESSING:
            node_status = [0, 1, 0]
        else:
            node_status = [0, 0, 1]

        return np.array(
            [
                *node_status,
                self.processing_time / self.job_processing_time,
                self.complete_ratio,
                self.remaining_op_num / n_m,
                self.waiting_time / self.job_processing_time,
                self.remaining_time / self.job_processing_time,
            ],
            dtype=np.float32,
        )


class Machine:
    def __init__(self, machine_id, ops):
        self.id = machine_id
        # 该机器上未被机器调度的op
        self.unscheduled_ops = ops
        # 即action space，可以作为调度选项的ops：Job维度上pre_op为空，或者pre_op已加工完成或者正在加工 （即从unscheduled_ops去掉pre_op也未调度的op）
        self.ready_ops = []
        # 已进入该机器调度的op（正在加工或者已被选中等待加工）。
        self.processing_ops = []
        self.done_ops = []
        # 记录机器是否处理完所有的op
        self.done = False

    def available(self):
        return len(self.processing_ops) == 0


class Job:
    def __init__(self, ops):
        self.ops = ops
        self.done = False
