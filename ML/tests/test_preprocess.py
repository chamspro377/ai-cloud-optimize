"""Regression checks for the Alibaba v2017 task schema (stdlib unittest)."""

from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import pandas as pd

from ML.src.preprocess import load_batch_tasks


class BatchTaskLoadingTests(unittest.TestCase):
    def load_text(self, text):
        path = Path(tempfile.gettempdir()) / f"alibaba-test-{uuid4().hex}.csv"
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text)
            return load_batch_tasks(path)
        finally:
            path.unlink(missing_ok=True)

    def test_actual_v2017_mapping_and_missing_requests(self):
        frame = self.load_text(
            "6459,6524,3,4,15740,Terminated,50,0.007956928014909534\n"
            "-10,0,4,7,393,Waiting,,\n"
        )
        self.assertEqual(frame.shape, (2, 8))
        self.assertEqual(frame.loc[0, "create_timestamp"], 6459)
        self.assertEqual(frame.loc[0, "task_id"], 4)
        self.assertEqual(frame.loc[0, "instance_num"], 15740)
        self.assertEqual(frame.loc[0, "status"], "Terminated")
        self.assertEqual(frame.loc[0, "plan_cpu"], 50)
        self.assertAlmostEqual(frame.loc[0, "plan_mem"], 0.007956928014909534)
        self.assertTrue(pd.isna(frame.loc[1, "plan_cpu"]))
        self.assertTrue(pd.isna(frame.loc[1, "plan_mem"]))
        self.assertEqual(frame.loc[1, "create_timestamp"], -10)

    def test_rejects_short_and_long_records_including_later_rows(self):
        valid = "1,2,3,4,5,Terminated,50,0.01\n"
        for bad in ("1,2,3,4,5,Waiting,\n", "1,2,3,4,5,Waiting,50,0.01,9\n"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "record 2"):
                self.load_text(valid + bad)

    def test_rejects_empty_file(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.load_text("")

    def test_rejects_invalid_numeric_field(self):
        with self.assertRaises(ValueError):
            self.load_text("abc,2,3,4,5,Terminated,50,0.01\n")

    def test_rejects_missing_required_identifier(self):
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.load_text("1,2,,4,5,Waiting,,\n")

    def test_rejects_invalid_status(self):
        with self.assertRaisesRegex(ValueError, "statuses"):
            self.load_text("1,2,3,4,5,123,50,0.01\n")


if __name__ == "__main__":
    unittest.main()
