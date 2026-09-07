"""Main pipeline script for cross-task gaze identification.

Usage:
    python scripts/run_pipeline.py [--data-dir DIR] [--split SPLIT] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset, dataset_summary, get_subject_task_matrix
from src.features import build_feature_matrix
from src.models import build_models, train_and_evaluate, encode_labels
from src.evaluation import (
    define_task_splits,
    prepare_cross_task_data,
    compute_metrics,
    save_results,
    plot_confusion_matrix,
    plot_feature_importance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_single_split(
    dataset: dict,
    split: dict,
    output_dir: Path,
    random_state: int = 42,
) -> list:
    """Run a single train/test split across all models.

    Returns:
        list of result dicts.
    """
    split_name = split["name"]
    logger.info("=" * 70)
    logger.info("Split: %s", split_name)
    logger.info("  Train tasks: %s", split["train_tasks"])
    logger.info("  Test tasks:  %s", split["test_tasks"])
    logger.info("  Rationale: %s", split["rationale"])

    try:
        X_train, y_train, X_test, y_test, train_subjects, test_subjects = (
            prepare_cross_task_data(
                feature_matrix=None,
                feature_names=None,
                subject_ids=None,
                train_tasks=split["train_tasks"],
                test_tasks=split["test_tasks"],
                dataset=dataset,
                use_pictures=True,
            )
        )
    except ValueError as e:
        logger.error("Failed to prepare data for split %s: %s", split_name, e)
        return []

    n_subjects = len(set(y_train))
    logger.info("  Subjects: %d", n_subjects)
    logger.info("  Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)

    # Handle NaN/Inf in features
    import numpy as np
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove zero-variance features
    var_train = np.var(X_train, axis=0)
    non_zero_var = var_train > 1e-10
    if non_zero_var.sum() < X_train.shape[1]:
        logger.info(
            "  Removing %d zero-variance features (%d remaining)",
            X_train.shape[1] - non_zero_var.sum(),
            non_zero_var.sum(),
        )
        X_train = X_train[:, non_zero_var]
        X_test = X_test[:, non_zero_var]

    # Build and evaluate models
    models = build_models(random_state=random_state)
    model_results = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        models=models,
        random_state=random_state,
    )

    # Package results
    results = []
    for model_name, res in model_results.items():
        metrics = compute_metrics(y_test, res["y_pred"], n_subjects)
        # Add confusion matrix from raw metrics
        metrics["confusion_matrix"] = res.get("y_pred") is not None and compute_metrics(
            y_test, res["y_pred"], n_subjects
        ).get("confusion_matrix")

        entry = {
            "split_name": split_name,
            "split_info": {
                "train_tasks": split["train_tasks"],
                "test_tasks": split["test_tasks"],
                "rationale": split["rationale"],
            },
            "model_name": model_name,
            "metrics": metrics,
            "report": res["report"],
        }
        results.append(entry)

        logger.info(
            "  %s: acc=%.4f, bal_acc=%.4f, f1=%.4f (random=%.4f)",
            model_name,
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["f1_macro"],
            metrics["random_baseline"],
        )

    # Plot confusion matrix for best model
    best_model_name = max(
        model_results.keys(),
        key=lambda k: model_results[k]["accuracy"],
    )
    best_pred = model_results[best_model_name]["y_pred"]
    if best_pred is not None:
        best_metrics = compute_metrics(y_test, best_pred, n_subjects)
        plot_confusion_matrix(
            best_metrics["confusion_matrix"],
            title=f"Best Model: {best_model_name} | Split: {split_name}",
            save_path=str(output_dir / f"cm_{split_name}_{best_model_name}.png"),
        )

        # Feature importance for tree-based models
        best_model = model_results[best_model_name]["model"]
        if best_model is not None and hasattr(best_model, "named_steps"):
            clf = best_model.named_steps.get("clf")
            if clf is not None and hasattr(clf, "feature_importances_"):
                # Get feature names after variance filtering
                from src.features import build_feature_matrix as bfm
                _, feat_names_train, _ = bfm(
                    dataset, split["train_tasks"], use_pictures=True,
                )
                # Apply same variance filter
                filtered_names = [
                    n for n, v in zip(feat_names_train, non_zero_var) if v
                ]
                if len(filtered_names) == len(clf.feature_importances_):
                    plot_feature_importance(
                        clf.feature_importances_,
                        filtered_names,
                        title=f"Feature Importance: {best_model_name} | {split_name}",
                        save_path=str(output_dir / f"feat_imp_{split_name}_{best_model_name}.png"),
                    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-task gaze-based person identification pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT.parent / "asreog_filter_order3_all_data"),
        help="Path to the dataset directory with .npz files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Run only this split (by name). If None, run all splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset from %s ...", args.data_dir)
    t0 = time.time()
    dataset = load_dataset(args.data_dir, prefer_first_session=True)
    logger.info("Dataset loaded in %.1f seconds", time.time() - t0)

    # Summary
    summary = dataset_summary(dataset)
    logger.info("Dataset summary:")
    logger.info("  Subjects: %d", summary["n_subjects"])
    for tk, count in summary["task_counts"].items():
        shape = summary["task_sample_shapes"][tk]
        logger.info("  %s: %d subjects, shape=%s", tk, count, shape)

    # Define splits
    splits = define_task_splits()
    if args.split:
        splits = [s for s in splits if s["name"] == args.split]
        if not splits:
            logger.error("Split '%s' not found", args.split)
            sys.exit(1)

    # Run each split
    all_results = []
    for split in splits:
        results = run_single_split(dataset, split, output_dir, args.random_state)
        all_results.extend(results)

    # Save all results
    save_results(all_results, str(output_dir))

    # Print final summary
    logger.info("=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    for r in all_results:
        m = r["metrics"]
        logger.info(
            "%-25s | %-20s | acc=%.4f | bal=%.4f | f1=%.4f | rand=%.4f",
            r["split_name"],
            r["model_name"],
            m["accuracy"],
            m["balanced_accuracy"],
            m["f1_macro"],
            m["random_baseline"],
        )

    logger.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()
