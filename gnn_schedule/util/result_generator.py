import pandas as pd


def to_dataframe(operations):
    df_gantt = pd.DataFrame(columns=["Job", "Operation", "Machine", "start_time", "end_time"])

    for op in operations:
        df = pd.DataFrame(
            {
                "Job": [op.job_id],
                "Operation": [op.id],
                "Machine": [op.machine_id],
                "start_time": [op.start_time],
                "end_time": [op.end_time],
            }
        )
        df_gantt = pd.concat([df_gantt, df])
    df_gantt = df_gantt.reset_index(drop=True)
    return df_gantt
