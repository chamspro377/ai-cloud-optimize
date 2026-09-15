from pathlib import Path
import shutil
import tempfile
import unittest
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from ML.src.predict import predict_resources
from ML.src.train import FEATURES, TARGETS, regression_metrics, split_by_job, train_reference_models, validate_dataset


def synthetic_dataset():
    rows = []
    for job in range(1, 31):
        for task in (1, 2):
            row = dict(job_id=job, task_id=task, instance_num=job + task,
                       plan_cpu=40 + job, plan_mem=0.01 * job)
            for resource, scale in (("cpu", 1), ("mem", 0.001)):
                eligible = resource == "cpu" or job % 3 != 0
                row[f"eligible_{resource}"] = eligible
                row[f"{resource}_measurement_coverage"] = 1.0 if eligible else 0.0
                row[TARGETS[resource][0]] = scale * job if eligible else np.nan
                row[TARGETS[resource][1]] = scale * (job + 5) if eligible else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp_parent = Path(tempfile.gettempdir()).resolve()
        self.directory = self.temp_parent / f"alibaba-model-test-{uuid4().hex}"
        self.directory.mkdir()
        self.frame = synthetic_dataset()

    def tearDown(self):
        if self.directory.resolve().parent != self.temp_parent:
            raise RuntimeError("Unexpected test cleanup target")
        shutil.rmtree(self.directory)

    def train(self, frame=None, directory=None):
        return train_reference_models(
            self.frame if frame is None else frame, self.directory if directory is None else directory,
            forest_params=dict(n_estimators=3, max_depth=3, min_samples_leaf=1), n_jobs=1,
        )

    def test_shared_split_has_no_job_leakage_and_is_reproducible(self):
        split = split_by_job(self.frame)
        pd.testing.assert_series_equal(split, split_by_job(self.frame))
        self.assertFalse(set(self.frame.loc[split.eq("train"), "job_id"]) & set(self.frame.loc[split.eq("test"), "job_id"]))
        self.assertEqual(self.frame.assign(split=split).groupby("job_id").split.nunique().max(), 1)

    def test_baseline_uses_only_training_rows_and_predictions_reload(self):
        report = self.train()
        split = split_by_job(self.frame)
        for resource, targets in TARGETS.items():
            bundle = joblib.load(self.directory / f"{resource}_reference.joblib")
            train = self.frame.loc[split.eq("train") & self.frame[f"eligible_{resource}"]]
            np.testing.assert_allclose(bundle["baseline"].constant_[0], train[targets].mean().to_numpy())
            predictions = pd.read_csv(self.directory / f"{resource}_test_predictions.csv")
            self.assertEqual(len(predictions), report["resources"][resource]["test_rows"])
        result = predict_resources(10, 50, 0.1, self.directory)
        self.assertEqual(result["predictions"]["cpu"]["outside_training_range"], [])
        self.assertTrue(all(np.isfinite(list(result["predictions"][r]["values"].values())).all() for r in ("cpu", "mem")))

    def test_changing_test_targets_does_not_change_fitted_forest(self):
        self.train()
        changed = self.frame.copy()
        mask = split_by_job(changed).eq("test")
        changed.loc[mask, TARGETS["cpu"]] *= 100
        self.train(changed, self.directory / "second")
        original = joblib.load(self.directory / "cpu_reference.joblib")["model"]
        second = joblib.load(self.directory / "second/cpu_reference.joblib")["model"]
        np.testing.assert_array_equal(original.predict(self.frame[FEATURES]), second.predict(self.frame[FEATURES]))

    def test_rejects_invalid_or_ambiguous_training_data(self):
        for column, value in (("eligible_cpu", "False"), ("plan_cpu", 0), ("job_id", 1.2), ("cpu_observed_peak", np.nan)):
            frame = self.frame.copy()
            frame[column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_dataset(frame)
        with self.assertRaises(ValueError):
            validate_dataset(pd.concat([self.frame, self.frame.iloc[[0]]]))

    def test_insufficient_resource_jobs_fail_before_model_writes(self):
        self.frame["eligible_mem"] = False
        with self.assertRaisesRegex(ValueError, "two jobs"):
            self.train()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_prediction_input_validation_and_range_diagnostic(self):
        for values in ((0, 50, 0.1), (2.5, 50, 0.1), (True, 50, 0.1), (1, np.inf, 0.1), (1, 50, None)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                predict_resources(*values, model_dir=self.directory)
        self.train()
        result = predict_resources(1000, 50, 0.1, self.directory)
        self.assertIn("instance_num", result["predictions"]["cpu"]["outside_training_range"])

    def test_metrics_handle_empty_and_constant_targets(self):
        self.assertIsNone(regression_metrics([], [])["mae"])
        self.assertIsNone(regression_metrics([1, 1], [1, 2])["r2"])
        metrics = regression_metrics([1, 3], [2, 2])
        self.assertEqual(metrics["mae"], 1)
        self.assertEqual(metrics["rmse"], 1)
        self.assertEqual(metrics["underprediction_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
