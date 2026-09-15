"""Predict Alibaba observations from trusted, locally trained model bundles."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .build_dataset import FEATURES


MODEL_DIR = Path(__file__).resolve().parents[1] / "models/reference_v1"


def predict_resources(instance_num, plan_cpu, plan_mem, model_dir=MODEL_DIR):
    values = [instance_num, plan_cpu, plan_mem]
    if any(isinstance(value, (bool, str)) for value in values):
        raise ValueError("Features must be numeric, not booleans or strings")
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be numeric") from exc
    if not np.isfinite(numeric).all() or not (numeric > 0).all():
        raise ValueError("Features must be finite and strictly positive")
    if numeric[0] != np.floor(numeric[0]):
        raise ValueError("instance_num must be an integer")
    features = pd.DataFrame([numeric], columns=FEATURES)
    result = {"inputs": dict(zip(FEATURES, numeric.tolist())), "predictions": {},
              "interpretation": "Experimental Alibaba observations; not a VM sizing recommendation."}
    fingerprints = set()
    for resource in ("cpu", "mem"):
        bundle = joblib.load(Path(model_dir) / f"{resource}_reference.joblib")
        if bundle.get("schema_version") != 1 or bundle.get("features") != list(FEATURES) or bundle.get("resource") != resource:
            raise ValueError(f"Incompatible {resource} model bundle")
        fingerprints.add((bundle.get("dataset_sha256"), bundle.get("seed")))
        predicted = bundle["model"].predict(features)[0]
        outside = [name for name, value in zip(FEATURES, numeric)
                   if not bundle["training_feature_ranges"][name]["min"] <= value <= bundle["training_feature_ranges"][name]["max"]]
        result["predictions"][resource] = {
            "values": dict(zip(bundle["targets"], predicted.tolist())), "units": bundle["units"],
            "outside_training_range": outside,
        }
    if len(fingerprints) != 1:
        raise ValueError("CPU and memory bundles come from different datasets or splits")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-num", type=int, required=True)
    parser.add_argument("--plan-cpu", type=float, required=True)
    parser.add_argument("--plan-mem", type=float, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    print(json.dumps(predict_resources(args.instance_num, args.plan_cpu, args.plan_mem, args.model_dir), indent=2))


if __name__ == "__main__":
    main()
