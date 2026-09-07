"""Run the task-disjoint, picture-level closed-set identification baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_contract import load_sessions, manifest, select_cohort
from src.protocol import feature_matrix, partition_rows
from src.revised_features import FeatureSettings, build_feature_rows
from src.revised_metrics import (
    participant_bootstrap,
    ranking_metrics,
    subject_label_permutation_pvalue,
)
from src.revised_models import candidates, class_scores


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _implementation_sha256() -> str:
    """Hash the code paths that define this experiment's behavior."""
    source_files = [
        PROJECT_ROOT / "scripts" / "run_experiment.py",
        PROJECT_ROOT / "src" / "data_contract.py",
        PROJECT_ROOT / "src" / "protocol.py",
        PROJECT_ROOT / "src" / "revised_features.py",
        PROJECT_ROOT / "src" / "revised_metrics.py",
        PROJECT_ROOT / "src" / "revised_models.py",
    ]
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT.parent / "asreog_filter_order3_all_data"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.json")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "task_disjoint_baseline"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    protocol, experiment = config["protocol"], config["experiment"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sessions, load_exclusions = load_sessions(args.data_dir)
    cohort, cohort_exclusions = select_cohort(sessions, **config["cohort"])
    settings = FeatureSettings(**config["features"])
    rows, feature_exclusions = build_feature_rows(
        cohort, settings, **{k: v for k, v in config["cohort"].items() if k != "session_policy"}
    )
    partitions = partition_rows(rows, protocol)
    subjects = sorted({row["subject_id"] for row in partitions["train"]})
    label_map = {subject: index for index, subject in enumerate(subjects)}
    feature_names = sorted(
        set(rows[0])
        - {
            "row_id",
            "subject_id",
            "session_id",
            "task_id",
            "picture_id",
            "n_samples",
            "missing_fraction",
        }
    )
    matrices = {
        name: feature_matrix(data, feature_names, label_map) for name, data in partitions.items()
    }
    train_variance = np.var(matrices["train"][0], axis=0)
    retained_feature_mask = train_variance > 1e-12
    if not retained_feature_mask.any():
        raise ValueError("all features have zero variance in the training partition")
    feature_names = [feature for feature, keep in zip(feature_names, retained_feature_mask) if keep]
    matrices = {
        name: (matrix[:, retained_feature_mask], labels)
        for name, (matrix, labels) in matrices.items()
    }

    validation_results = []
    x_train, y_train = matrices["train"]
    x_validation, y_validation = matrices["validation"]
    for name, model in candidates(experiment["seed"]).items():
        model.fit(x_train, y_train)
        scores, classes = class_scores(model, x_validation)
        validation_results.append(
            {"model": name, **ranking_metrics(y_validation, scores, classes, experiment["top_k"])}
        )
    selected = max(validation_results, key=lambda result: result["top1_accuracy"])["model"]

    x_final = np.concatenate([matrices["train"][0], matrices["validation"][0]])
    y_final = np.concatenate([matrices["train"][1], matrices["validation"][1]])
    fitted = candidates(experiment["seed"])[selected].fit(x_final, y_final)
    x_test, y_test = matrices["test"]
    scores, classes = class_scores(fitted, x_test)
    y_pred = classes[np.argmax(scores, axis=1)]
    metrics = ranking_metrics(y_test, scores, classes, experiment["top_k"])
    metrics["top1_bootstrap_95_ci"] = participant_bootstrap(
        y_test,
        y_pred,
        np.array([row["subject_id"] for row in partitions["test"]]),
        experiment["bootstrap_resamples"],
        experiment["seed"],
    )
    metrics["label_permutation_p_value"] = subject_label_permutation_pvalue(
        y_test,
        y_pred,
        np.array([row["subject_id"] for row in partitions["test"]]),
        experiment["permutation_resamples"],
        experiment["seed"],
    )
    metrics["closed_set_chance_accuracy"] = 1.0 / len(subjects)

    predictions = [
        {
            "row_id": row["row_id"],
            "subject_id": row["subject_id"],
            "task_id": row["task_id"],
            "true_label": int(truth),
            "predicted_label": int(prediction),
        }
        for row, truth, prediction in zip(partitions["test"], y_test, y_pred)
    ]
    with (args.output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    _write_json(
        args.output_dir / "cohort_manifest.json",
        manifest(cohort, [*load_exclusions, *cohort_exclusions], config),
    )
    _write_json(args.output_dir / "segment_exclusions.json", feature_exclusions)
    _write_json(
        args.output_dir / "results.json",
        {
            "protocol": protocol,
            "n_subjects": len(subjects),
            "n_feature_rows": {name: len(data) for name, data in partitions.items()},
            "n_features": len(feature_names),
            "n_zero_variance_features_removed": int((~retained_feature_mask).sum()),
            "feature_names": feature_names,
            "validation": validation_results,
            "selected_model": selected,
            "test_metrics": metrics,
            "implementation_sha256": _implementation_sha256(),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )
    _write_json(
        args.output_dir / "run_metadata.json",
        {"created_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    print(f"Selected model: {selected}")
    print(
        f"Test top-1 accuracy: {metrics['top1_accuracy']:.4f}; chance: {metrics['closed_set_chance_accuracy']:.4f}"
    )
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
