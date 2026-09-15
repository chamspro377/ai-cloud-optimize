"""Train fixed Alibaba reference models and evaluate on held-out jobs.

No tuning or model selection uses the test set. Saved forests remain research
models regardless of whether they beat the constant reference.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from .build_dataset import FEATURES


ML_DIR = Path(__file__).resolve().parents[1]
TARGETS = {r: [f"{r}_mean_of_attempt_averages", f"{r}_observed_peak"] for r in ("cpu", "mem")}
FOREST_PARAMS = {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 5, "max_features": 1.0}


def validate_dataset(frame):
    required = ["job_id", "task_id", *FEATURES]
    for resource, targets in TARGETS.items():
        required += [f"eligible_{resource}", f"{resource}_measurement_coverage", *targets]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    if frame.empty or frame.duplicated(["job_id", "task_id"]).any():
        raise ValueError("Empty dataset or duplicate task keys")
    for columns in (["job_id", "task_id"], FEATURES):
        values = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError(f"Expected finite positive values: {columns}")
    for column in ("job_id", "task_id", "instance_num"):
        if not frame[column].eq(np.floor(frame[column])).all():
            raise ValueError(f"Expected integer values: {column}")
    for resource, targets in TARGETS.items():
        eligible = frame[f"eligible_{resource}"]
        if eligible.dtype != bool:
            raise ValueError(f"eligible_{resource} must contain booleans, not strings or missing values")
        subset = frame.loc[eligible]
        values = subset[targets].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not (values >= 0).all():
            raise ValueError(f"Invalid {resource} targets")
        if not (values[:, 0] <= values[:, 1] + 1e-12).all():
            raise ValueError(f"{resource} mean exceeds peak")
        coverage = subset[f"{resource}_measurement_coverage"]
        if not (coverage.gt(0) & coverage.le(1)).all():
            raise ValueError(f"Invalid {resource} measurement coverage")


def split_by_job(frame, seed=42, test_size=0.2):
    """One shared deterministic partition before filtering CPU/RAM eligibility."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if frame.job_id.nunique() < 3:
        raise ValueError("At least three distinct jobs are required")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(frame, groups=frame.job_id))
    assignment = pd.Series("train", index=frame.index, name="split")
    assignment.iloc[test_index] = "test"
    assert not set(frame.iloc[train_index].job_id) & set(frame.iloc[test_index].job_id)
    return assignment


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return {"rows": 0, "mae": None, "rmse": None, "r2": None, "underprediction_rate": None}
    return {
        "rows": int(actual.size),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if actual.size >= 2 and np.var(actual) > 0 else None,
        "underprediction_rate": float(np.mean(predicted < actual)),
    }


def write_markdown_report(report, path):
    labels = {
        "cpu_mean_of_attempt_averages": "CPU moyen par tentative",
        "cpu_observed_peak": "Pic CPU observé",
        "mem_mean_of_attempt_averages": "RAM moyenne par tentative",
        "mem_observed_peak": "Pic RAM observé",
    }
    lines = [
        "# Premier modèle Alibaba — évaluation", "",
        f"Exécution UTC : {report['created_at_utc']}", "",
        "Deux forêts aléatoires multi-sorties (CPU et RAM), comparées à une constante",
        "égale à la moyenne des cibles d'apprentissage. Paramètres fixés avant le test.", "",
        f"Séparation commune : {report['split']['train']['jobs']} jobs d'apprentissage et "
        f"{report['split']['test']['jobs']} jobs de test. Aucun job partagé.", "",
        "| Cible | Tâches test | MAE constante | MAE forêt | Réduction MAE | R² forêt |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for resource in report["resources"].values():
        for target, values in resource["targets"].items():
            baseline, forest = values["baseline_test"], values["forest_test"]
            reduction = values["mae_reduction_vs_constant"]
            reduction_text = f"{reduction:.1%}" if reduction is not None else "non défini"
            r2 = f"{forest['r2']:.3f}" if forest['r2'] is not None else "non défini"
            lines.append(f"| {labels[target]} | {forest['rows']} | {baseline['mae']:.6f} | {forest['mae']:.6f} | {reduction_text} | {r2} |")
    lines += [
        "", "MAE : erreur absolue moyenne, dans l'unité de la cible. Une valeur plus basse est meilleure.",
        "La réduction MAE est relative à la constante, pas un pourcentage de précision.",
        "R² : part de variabilité expliquée sur le test ; ce score peut être négatif.", "",
        "## Portée des résultats", "",
        "Le détail des erreurs d'apprentissage, RMSE, sous-estimations et couvertures",
        "complètes/partielles se trouve dans `evaluation.json` et le notebook 03.",
        "Les fichiers de prédictions permettent de recalculer les scores. Les modèles",
        "sauvegardés sont exactement ceux évalués, sans réentraînement sur le test.", "",
        "Les scores indiquent un signal prédictif encore limité. Aucun seuil de qualité",
        "opérationnelle n'est établi ; une amélioration face à une constante ne suffit",
        "pas à valider le dimensionnement automatique des VM.", "",
        *[f"- {item}" for item in report["limitations"]], "",
        "## Reproduire", "", "Depuis la racine du projet :", "", "```powershell",
        ".venv\\Scripts\\python.exe -m ML.src.train",
        ".venv\\Scripts\\python.exe -m ML.src.predict --instance-num 2 --plan-cpu 50 --plan-mem 0.005",
        "```", "",
        "Les trois entrées proviennent du schéma Alibaba. Elles ne correspondent pas",
        "encore aux entrées métier `expected_users` ou `workload_type` de l'API.", "",
        "## Références", "",
        "- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html",
        "- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
        "- https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/schema.csv", "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def train_reference_models(frame, output_dir, *, seed=42, test_size=0.2,
                           n_jobs=1, forest_params=None, dataset_sha256=None, progress=None):
    """Persist models, the exact group split, predictions and evaluation."""
    validate_dataset(frame)
    frame = frame.sort_values(["job_id", "task_id"]).reset_index(drop=True)
    assignments = split_by_job(frame, seed, test_size)
    # Fail before writing files if either resource cannot be evaluated.
    for resource in TARGETS:
        usable = frame[f"eligible_{resource}"]
        for split in ("train", "test"):
            subset = frame.loc[usable & assignments.eq(split)]
            if len(subset) < 2 or subset.job_id.nunique() < 2:
                raise ValueError(f"Need at least two jobs and rows for {resource}/{split}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = dict(FOREST_PARAMS if forest_params is None else forest_params)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_sha256,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "protocol": {
            "seed": seed, "requested_test_job_fraction": test_size,
            "group_column": "job_id", "features": list(FEATURES),
            "forest_parameters": params, "n_jobs": n_jobs,
            "baseline": "DummyRegressor(strategy='mean'), fitted on training targets only",
            "selection": "Fixed parameters before evaluation; no tuning or winner selection on test data.",
            "fit_scope": "Training jobs only; saved models are the exact evaluated estimators, without a full-data refit.",
        },
        "split": {name: {"rows": int(assignments.eq(name).sum()),
                         "jobs": int(frame.loc[assignments.eq(name), "job_id"].nunique())}
                  for name in ("train", "test")},
        "resources": {},
        "limitations": [
            "Random held-out jobs measure generalization within this trace, not future-time or VM performance.",
            "CPU stays in original source units; memory stays normalized. Predictions are not vCPU/GB recommendations.",
            "Targets summarize measured completed attempts; missing measurements and completed-only selection may bias results.",
            "High coverage only means coverage among accepted completed attempts, not all requested instances.",
            "Observed peaks are not simultaneous total task consumption and are not guaranteed upper bounds.",
            "These three Alibaba request features do not map directly to the current API's expected_users/workload_type.",
            "Test results are now visible; further model selection requires training-only validation and a fresh final evaluation protocol.",
        ],
    }
    job_assignments = frame[["job_id"]].assign(split=assignments).drop_duplicates().sort_values("job_id")
    job_assignments.to_csv(output_dir / "job_split.csv", index=False)
    for resource, targets in TARGETS.items():
        if progress:
            progress(f"Training {resource.upper()} reference model")
        usable = frame[f"eligible_{resource}"]
        train = frame.loc[usable & assignments.eq("train")]
        test = frame.loc[usable & assignments.eq("test")]
        x_train, y_train = train[FEATURES], train[targets]
        forest = RandomForestRegressor(**params, random_state=seed, n_jobs=n_jobs)
        baseline = DummyRegressor(strategy="mean")
        forest.fit(x_train, y_train)
        baseline.fit(x_train, y_train)
        forest_train = forest.predict(x_train)
        forest_test = forest.predict(test[FEATURES])
        baseline_test = baseline.predict(test[FEATURES])
        metrics = {}
        for j, target in enumerate(targets):
            forest_metrics = regression_metrics(test[target], forest_test[:, j])
            baseline_metrics = regression_metrics(test[target], baseline_test[:, j])
            baseline_mae = baseline_metrics["mae"]
            metrics[target] = {
                "forest_train": regression_metrics(train[target], forest_train[:, j]),
                "forest_test": forest_metrics, "baseline_test": baseline_metrics,
                "mae_reduction_vs_constant": (1 - forest_metrics["mae"] / baseline_mae) if baseline_mae > 0 else None,
                "test_by_coverage": {},
            }
            coverage = test[f"{resource}_measurement_coverage"]
            for label, mask in {"full": coverage.ge(1 - 1e-12), "partial": coverage.lt(1 - 1e-12)}.items():
                metrics[target]["test_by_coverage"][label] = {
                    "forest": regression_metrics(test.loc[mask, target], forest_test[mask.to_numpy(), j]),
                    "baseline": regression_metrics(test.loc[mask, target], baseline_test[mask.to_numpy(), j]),
                }
        predictions = test[["job_id", "task_id", f"{resource}_measurement_coverage", *targets]].copy()
        for j, target in enumerate(targets):
            predictions[f"forest_{target}"] = forest_test[:, j]
            predictions[f"baseline_{target}"] = baseline_test[:, j]
        predictions.to_csv(output_dir / f"{resource}_test_predictions.csv", index=False)
        feature_ranges = {name: {"min": float(x_train[name].min()), "max": float(x_train[name].max())} for name in FEATURES}
        bundle = {
            "schema_version": 1, "resource": resource, "features": list(FEATURES), "targets": targets,
            "model": forest, "baseline": baseline, "training_feature_ranges": feature_ranges,
            "dataset_sha256": dataset_sha256, "sklearn_version": sklearn.__version__,
            "seed": seed, "units": "source CPU values" if resource == "cpu" else "normalized memory",
        }
        joblib.dump(bundle, output_dir / f"{resource}_reference.joblib", compress=3)
        reloaded = joblib.load(output_dir / f"{resource}_reference.joblib")
        np.testing.assert_allclose(reloaded["model"].predict(test[FEATURES].head(10)), forest_test[:10])
        report["resources"][resource] = {
            "train_rows": len(train), "test_rows": len(test),
            "train_jobs": int(train.job_id.nunique()), "test_jobs": int(test.job_id.nunique()),
            "training_feature_ranges": feature_ranges, "targets": metrics,
        }
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown_report(report, output_dir / "evaluation.md")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ML_DIR / "data/processed/task_resource_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ML_DIR / "models/reference_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    frame = pd.read_csv(args.dataset)
    report = train_reference_models(frame, args.output_dir, seed=args.seed, n_jobs=args.n_jobs,
                                    dataset_sha256=digest, progress=lambda message: print(message, flush=True))
    for resource in report["resources"].values():
        for target, metrics in resource["targets"].items():
            print(target, json.dumps(metrics["forest_test"]), flush=True)


if __name__ == "__main__":
    main()
