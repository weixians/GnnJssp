import pandas as pd


def to_dataframe(task_durations, task_machines, task_finish_times):
    df_gantt = pd.DataFrame(columns=["Job", "Operation", "Machine", "start_time", "end_time"])

    for i in range(len(task_durations)):
        for j in range(len(task_durations[i])):
            df = pd.DataFrame(
                {
                    "Job": [i],
                    "Operation": [j],
                    "Machine": [task_machines[i][j]],
                    "start_time": [task_finish_times[i][j] - task_durations[i][j]],
                    "end_time": [task_finish_times[i][j]],
                }
            )
            df_gantt = pd.concat([df_gantt, df])
    df_gantt = df_gantt.reset_index(drop=True)
    return df_gantt
