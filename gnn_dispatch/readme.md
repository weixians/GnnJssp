# 场景描述
JSSP问题，每个工单包含的工序数量相等，且每个工序有**唯一对应**的机器来处理，工序一旦分配到机器，便不可被中断

# 优化目标
最小化所有工单的makespan

# 数据描述
numpy数据格式，shape=(100,2,6,6)，即：100条数据，每条数据包含2个6*6的数组，
- 第一个数组：每行表示一个job(工单)，行中的每一个元素表示一道工序所需的处理时间(时间范围1-99)，例：
```
27,17,69,43,56,77
80,90,15,92,58,90
12,7,43,57,2,8
52,28,74,16,2,24
86,25,8,23,55,78
17,71,36,32,33,46
```
- 第二个数组：每个元素对应着第一个数组对应元素表示的工序所对应的机器序号（机器序号从0开始编号，不是从1开始），例：
```
5,3,0,2,4,1
2,1,0,5,4,3
1,4,0,5,2,3
1,0,2,4,3,5
0,3,2,4,5,1
1,2,5,4,3,0
```

```python
import os
import numpy as np

n_j = 6 
n_m = 6
data = np.load(os.path.join("generatedData{}_{}_Seed200.npy").format(n_j, n_m))
for i, item in enumerate(data):
    print("第{}条数据：".format(i + 1))
    task_durations, task_machines = item[0], item[1]
    print(task_durations)
    print(task_machines)
```