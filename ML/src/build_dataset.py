"""Prepare task-level targets from Alibaba 2017 without loading all instances.

Run from the project root: python -m ML.src.build_dataset
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess import load_batch_tasks


INSTANCE_COLUMNS = (
    "start_timestamp", "end_timestamp", "job_id", "task_id", "machine_id",
    "status", "seq_no", "total_seq_no", "real_cpu_max", "real_cpu_avg",
    "real_mem_max", "real_mem_avg",
)
KEYS = ["job_id", "task_id"]
FEATURES = ["instance_num", "plan_cpu", "plan_mem"]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = RAW_DIR.parent / "processed"
SCHEMA_URL = "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/schema.csv"
TRACE_URL = "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/trace_201708.md"


def iter_instances(path: str | Path, chunksize: int = 100_000):
    """Validate exact record width before typed, bounded-memory parsing."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    path = Path(path)
    records = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for records, row in enumerate(csv.reader(stream, strict=True), 1):
            if len(row) != len(INSTANCE_COLUMNS):
                raise ValueError(f"{path}: record {records}: expected 12 columns, got {len(row)}")
    if not records:
        raise ValueError(f"{path}: empty instance file")
    types = {name: "Int64" for name in INSTANCE_COLUMNS[:8] if name != "status"}
    types.update({name: "float64" for name in INSTANCE_COLUMNS[8:]})
    types["status"] = "string"
    allowed = {"Ready", "Waiting", "Running", "Terminated", "Failed", "Cancelled", "Interrupted", "Interupted"}
    with pd.read_csv(
        path, names=list(INSTANCE_COLUMNS), header=None, dtype=types,
        keep_default_na=False, na_values=[""], encoding="utf-8-sig",
        chunksize=chunksize,
    ) as reader:
        for chunk in reader:
            if chunk.status.isna().any() or not set(chunk.status.unique()) <= allowed:
                raise ValueError("Missing or unknown instance status")
            yield chunk


def _merge_aggregates(existing, part, operations):
    if existing is None:
        return part
    return pd.concat([existing, part]).groupby(level=KEYS).agg(operations)


def prepare_dataset(task_path, instance_path, chunksize=100_000, progress=None):
    """Return dataset and audit report. No imputation, clipping or unit conversion.

    The row grain is one task. Targets summarize measured completed attempts,
    NOT concurrent task usage or distinct instances. Missing RAM does not remove
    an otherwise valid CPU observation. Counters named diagnostics may overlap.
    """
    tasks = load_batch_tasks(task_path)
    if tasks.duplicated(KEYS).any():
        raise ValueError("Duplicate (job_id, task_id) in task table: ambiguous join")
    known_keys = pd.MultiIndex.from_frame(tasks[KEYS])
    funnel = Counter({name: 0 for name in (
        "rows", "not_terminated", "missing_or_invalid_task_key", "unmatched_task_key",
        "invalid_completed_timestamps", "accepted_completed_attempts",
    )})
    statuses, missing, diagnostics = Counter(), Counter(), Counter()
    totals = None
    resource_totals = {"cpu": None, "mem": None}
    for i, chunk in enumerate(iter_instances(instance_path, chunksize)):
        funnel["rows"] += len(chunk)
        statuses.update(chunk.status.value_counts().to_dict())
        missing.update(chunk.isna().sum().to_dict())
        completed = chunk.status.eq("Terminated")
        funnel["not_terminated"] += int((~completed).sum())
        chunk = chunk.loc[completed].copy()
        valid_key = chunk[KEYS].notna().all(axis=1) & chunk[KEYS].gt(0).all(axis=1)
        funnel["missing_or_invalid_task_key"] += int((~valid_key).sum())
        chunk = chunk.loc[valid_key].copy()
        matched = pd.MultiIndex.from_frame(chunk[KEYS]).isin(known_keys)
        funnel["unmatched_task_key"] += int((~matched).sum())
        chunk = chunk.loc[matched].copy()
        valid_time = (
            chunk.start_timestamp.notna() & chunk.end_timestamp.notna()
            & chunk.end_timestamp.gt(0) & chunk.end_timestamp.ge(chunk.start_timestamp)
        ).fillna(False)
        funnel["invalid_completed_timestamps"] += int((~valid_time).sum())
        chunk = chunk.loc[valid_time].copy()
        funnel["accepted_completed_attempts"] += len(chunk)
        if chunk.empty:
            continue
        part = chunk.groupby(KEYS).size().to_frame("completed_attempt_count")
        totals = _merge_aggregates(totals, part, {"completed_attempt_count": "sum"})
        for resource in ("cpu", "mem"):
            avg, peak = f"real_{resource}_avg", f"real_{resource}_max"
            finite = pd.Series(np.isfinite(chunk[[avg, peak]]).all(axis=1), index=chunk.index)
            nonnegative = chunk[[avg, peak]].ge(0).all(axis=1)
            ordered = chunk[avg].le(chunk[peak])
            valid = finite & nonnegative & ordered
            diagnostics[f"{resource}_missing_or_nonfinite"] += int((~finite).sum())
            diagnostics[f"{resource}_negative"] += int((chunk[[avg, peak]] < 0).any(axis=1).sum())
            diagnostics[f"{resource}_avg_above_max"] += int(chunk[avg].gt(chunk[peak]).sum())
            diagnostics[f"{resource}_valid_attempts"] += int(valid.sum())
            selected = chunk.loc[valid]
            if selected.empty:
                continue
            part = selected.groupby(KEYS).agg(
                **{f"{resource}_measured_attempt_count": (avg, "count"),
                   f"{resource}_average_sum": (avg, "sum"),
                   f"{resource}_observed_peak": (peak, "max")}
            )
            operations = {column: ("max" if column.endswith("peak") else "sum") for column in part}
            resource_totals[resource] = _merge_aggregates(resource_totals[resource], part, operations)
        if progress is not None and i % 20 == 0:
            progress(f"Processed {funnel['rows']:,} instance records")

    if totals is None:
        totals = pd.DataFrame(columns=KEYS + ["completed_attempt_count"]).set_index(KEYS)
    result = tasks.merge(totals.reset_index(), on=KEYS, how="left", validate="one_to_one")
    result["completed_attempt_count"] = result.completed_attempt_count.fillna(0).astype("int64")
    task_valid = (
        result.status.eq("Terminated") & result[FEATURES].notna().all(axis=1)
        & result[FEATURES].gt(0).all(axis=1)
        & pd.Series(np.isfinite(result[FEATURES].astype(float)).all(axis=1), index=result.index)
    )
    for resource, aggregate in resource_totals.items():
        count = f"{resource}_measured_attempt_count"
        mean = f"{resource}_mean_of_attempt_averages"
        peak = f"{resource}_observed_peak"
        if aggregate is None:
            result[count], result[mean], result[peak] = 0, np.nan, np.nan
        else:
            aggregate[mean] = aggregate[f"{resource}_average_sum"] / aggregate[count]
            result = result.merge(aggregate[[count, mean, peak]].reset_index(), on=KEYS, how="left", validate="one_to_one")
            result[count] = result[count].fillna(0).astype("int64")
        result[f"{resource}_measurement_coverage"] = result[count].div(result.completed_attempt_count.replace(0, np.nan))
        result[f"eligible_{resource}"] = task_valid & result[count].gt(0)

    dataset = result.loc[result.eligible_cpu | result.eligible_mem].copy()
    dataset = dataset.sort_values(KEYS).reset_index(drop=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [SCHEMA_URL, TRACE_URL],
        "inputs": {"tasks": str(Path(task_path).resolve()), "instances": str(Path(instance_path).resolve())},
        "instance_funnel_exclusive": dict(funnel),
        "instance_statuses": dict(statuses), "instance_missing_fields": dict(missing),
        "resource_diagnostics_on_accepted_attempts_overlapping": dict(diagnostics),
        "tasks": {"input": len(tasks), "with_accepted_attempts": int(result.completed_attempt_count.gt(0).sum()),
                  "terminated_with_valid_requests": int(task_valid.sum()), "output": len(dataset),
                  "eligible_cpu": int(dataset.eligible_cpu.sum()), "eligible_mem": int(dataset.eligible_mem.sum()),
                  "eligible_both": int((dataset.eligible_cpu & dataset.eligible_mem).sum())},
        "model_contract": {
            "features": FEATURES, "split_group": "job_id",
            "targets": {r: [f"{r}_mean_of_attempt_averages", f"{r}_observed_peak"] for r in ("cpu", "mem")},
            "selection": "Filter eligible_cpu or eligible_mem for the chosen target before training.",
            "excluded_from_features": "IDs, status, timestamps, observation counts, coverage and all measured targets.",
        },
        "limitations": [
            "Targets describe observed completed attempts, not simultaneous total task consumption.",
            "Averages weight each measured attempt equally, not by duration; peaks are maxima, not sums.",
            "No unique instance identifier is provided: attempts are not deduplicated by task/machine/sequence.",
            "Coverage is relative to accepted completed attempts, not the requested instance count.",
            "Memory stays normalized; CPU requests and observations stay in original source units. No vCPU/GB conversion or request/usage ratio is applied.",
            "CPU and RAM may describe different subsets of attempts; missing measurements are never filled with zero.",
            "Completed-only selection and missing keys/measurements can bias the dataset.",
            "Split by job_id to prevent related tasks leaking across training and test sets.",
            "These features do not map directly to the API inputs expected_users/workload_type; API integration needs a separate design.",
        ],
    }
    assert funnel["rows"] == sum(value for key, value in funnel.items() if key != "rows")
    return dataset, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    dataset, report = prepare_dataset(
        args.raw_dir / "batch_task.csv", args.raw_dir / "batch_instance.csv",
        chunksize=args.chunksize, progress=lambda message: print(message, flush=True),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_dir / "task_resource_dataset.csv", index=False)
    (args.output_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["tasks"], indent=2))


if __name__ == "__main__":
    main()
