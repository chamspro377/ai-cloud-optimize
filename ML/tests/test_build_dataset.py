from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import pandas as pd

from ML.src.build_dataset import iter_instances, prepare_dataset


@contextmanager
def csv_file(text):
    path = Path(tempfile.gettempdir()) / f"alibaba-{uuid4().hex}.csv"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
        yield path
    finally:
        path.unlink(missing_ok=True)


class DatasetTests(unittest.TestCase):
    task = "1,10,1,1,3,Terminated,50,0.02\n"

    def prepare(self, instances, tasks=None, chunksize=2):
        with csv_file(self.task if tasks is None else tasks) as task_path, csv_file(instances) as instance_path:
            return prepare_dataset(task_path, instance_path, chunksize)

    def test_aggregation_is_independent_of_chunk_boundaries(self):
        rows = (
            "1,10,1,1,1,Terminated,1,1,2,1,0.2,0.1\n"
            "1,10,1,1,2,Terminated,1,1,4,3,,\n"
            "1,10,1,1,3,Terminated,1,1,10,8,0.8,0.4\n"
        )
        small, report = self.prepare(rows, chunksize=2)
        large, _ = self.prepare(rows, chunksize=100)
        pd.testing.assert_frame_equal(small, large)
        row = small.iloc[0]
        self.assertEqual(row.cpu_mean_of_attempt_averages, 4)
        self.assertEqual(row.cpu_observed_peak, 10)
        self.assertEqual(row.mem_mean_of_attempt_averages, 0.25)
        self.assertAlmostEqual(row.mem_measurement_coverage, 2 / 3)
        self.assertEqual(report["tasks"]["eligible_both"], 1)

    def test_missing_memory_does_not_discard_cpu_or_become_zero(self):
        frame, _ = self.prepare("-1,10,1,1,1,Terminated,1,1,2,1,,\n")
        self.assertTrue(frame.iloc[0].eligible_cpu)
        self.assertFalse(frame.iloc[0].eligible_mem)
        self.assertTrue(pd.isna(frame.iloc[0].mem_observed_peak))

    def test_funnel_join_and_resource_validity(self):
        rows = (
            "1,10,1,1,1,Running,1,1,2,1,,\n"
            "1,10,,,1,Terminated,1,1,2,1,,\n"
            "1,10,99,99,1,Terminated,1,1,2,1,,\n"
            "10,1,1,1,1,Terminated,1,1,2,1,,\n"
            "1,10,1,1,1,Terminated,1,1,2,3,0.1,0.2\n"
            "1,10,1,1,1,Terminated,1,1,2,0,0.2,0.1\n"
        )
        frame, report = self.prepare(rows)
        self.assertEqual(list(report["instance_funnel_exclusive"].values()), [6, 1, 1, 1, 1, 2])
        self.assertEqual(frame.iloc[0].cpu_measured_attempt_count, 1)
        self.assertEqual(frame.iloc[0].cpu_mean_of_attempt_averages, 0)
        self.assertEqual(frame.iloc[0].cpu_measurement_coverage, 0.5)

    def test_invalid_requests_and_unfinished_tasks_excluded(self):
        rows = "1,10,1,1,1,Terminated,1,1,2,1,,\n"
        for task in ("1,10,1,1,3,Terminated,,\n", "1,10,1,1,3,Running,50,0.02\n"):
            with self.subTest(task=task):
                frame, _ = self.prepare(rows, tasks=task)
                self.assertTrue(frame.empty)

    def test_ambiguous_task_join_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare("1,10,1,1,1,Terminated,1,1,2,1,,\n", tasks=self.task * 2)

    def test_empty_accepted_set_produces_empty_dataset(self):
        frame, report = self.prepare("1,10,1,1,1,Running,1,1,2,1,,\n")
        self.assertTrue(frame.empty)
        self.assertEqual(report["tasks"]["output"], 0)

    def test_bad_width_and_unknown_status_rejected(self):
        for row in ("1,2,3\n", "1,10,1,1,1,Unknown,1,1,2,1,,\n"):
            with self.subTest(row=row), csv_file(row) as path, self.assertRaises(ValueError):
                list(iter_instances(path, 2))

    def test_nonfinite_and_negative_cpu_are_excluded(self):
        for value in ("inf", "-1"):
            frame, _ = self.prepare(f"1,10,1,1,1,Terminated,1,1,2,{value},0.2,0.1\n")
            self.assertFalse(frame.iloc[0].eligible_cpu)
            self.assertTrue(frame.iloc[0].eligible_mem)


if __name__ == "__main__":
    unittest.main()
