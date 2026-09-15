"""Load the eight-column Alibaba Cluster Trace v2017 task table.

Source: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/schema.csv
Raw values and missing resource requests are preserved; this is not ML cleaning.
"""

import csv
from pathlib import Path

import pandas as pd


BATCH_TASK_COLUMNS = (
    "create_timestamp", "end_timestamp", "job_id", "task_id",
    "instance_num", "status", "plan_cpu", "plan_mem",
)
DEFAULT_BATCH_TASK_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "batch_task.csv"


def load_batch_tasks(path: str | Path = DEFAULT_BATCH_TASK_PATH) -> pd.DataFrame:
    """Load v2017 tasks, rejecting shifted columns and invalid numeric fields.

    The second field is named end time in schema.csv and latest state
    modification time in trace_201708.md; do not assume execution duration.
    Memory is normalized, not GB. No resource-unit conversion is applied.
    """
    path = Path(path)
    # Pandas can pad short records or infer an index for excess fields.
    # Check every record first so neither case can silently shift the schema.
    row_count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream, strict=True):
            row_count += 1
            if len(row) != len(BATCH_TASK_COLUMNS):
                raise ValueError(
                    f"{path}: record {row_count}: expected 8 columns for "
                    f"Alibaba v2017 batch_task, got {len(row)}"
                )
    if not row_count:
        raise ValueError(f"{path}: empty Alibaba batch_task file")

    dtypes = {name: "Int64" for name in BATCH_TASK_COLUMNS}
    dtypes.update(status="string", plan_mem="Float64")
    frame = pd.read_csv(
        path, header=None, names=list(BATCH_TASK_COLUMNS), dtype=dtypes,
        encoding="utf-8-sig", keep_default_na=False, na_values=[""],
    )
    required = list(BATCH_TASK_COLUMNS[:6])
    missing = frame[required].isna().sum()
    if missing.any():
        raise ValueError(f"{path}: missing required fields: {missing[missing > 0].to_dict()}")
    allowed_statuses = {"Ready", "Waiting", "Running", "Terminated", "Failed", "Cancelled"}
    unknown = set(frame["status"].unique()) - allowed_statuses
    if unknown:
        raise ValueError(f"{path}: unknown Alibaba task statuses: {sorted(unknown)}")
    return frame
