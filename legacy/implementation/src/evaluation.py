"""Evaluation utilities for cross-task gaze identification.

Handles cross-task train/test splits, metric computation, visualization,
and result aggregation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)

# Cognitive load mapping per task
TASK_COGNITIVE_LOAD = {
    "task1": "low",
    "task2": "low",
    "task3": "low",
    "task4": "medium",
    "task5": "medium",
    "task6": "medium",
    "task7": "high",
    "task8": "high",
    "task9": "high",
}

TASK_DESCRIPTIONS = {
    "task1": "Scene viewing",
    "task2": "Face viewing",
    "task3": "Scene elements",
    "task4": "Semantic processing",
    "task5": "Visuospatial pursuit",
    "task6": "Memory recognition",
    "task7": "Arithmetic",
    "task8": "Executive function",
    "task9": "Complex executive",
}


def define_task_splits() -> List[Dict[str, Any]]:
    """Define all train/test task splits for evaluation.

    Returns:
        list of dicts with keys:
            'name': str, split name
            'train_tasks': list of task keys
            'test_tasks': list of task keys
            'rationale': str, justification
    """
    splits = [
        {
            "name": "low_to_high_A",
            "train_tasks": ["task1", "task2"],
            "test_tasks": ["task7", "task8"],
            "rationale": (
                "Train on low cognitive load (scene/face viewing), "
                "test on high load (arithmetic/executive). "
                "Tests generalization under maximum cognitive shift."
            ),
        },
        {
            "name": "low_to_high_B",
            "train_tasks": ["task1", "task3"],
            "test_tasks": ["task8", "task9"],
            "rationale": (
                "Alternative low-to-high split to verify robustness."
            ),
        },
        {
            "name": "high_to_low",
            "train_tasks": ["task7", "task8"],
            "test_tasks": ["task1", "task2"],
            "rationale": (
                "Reverse direction: train on high load, test on low. "
                "Tests if high-load gaze patterns transfer to low-load."
            ),
        },
        {
            "name": "medium_to_extremes",
            "train_tasks": ["task4", "task5"],
            "test_tasks": ["task1", "task9"],
            "rationale": (
                "Train on medium load, test on both extremes (low + high). "
                "Tests transfer from middle cognitive demands."
            ),
        },
        {
            "name": "same_level_low",
            "train_tasks": ["task1", "task2"],
            "test_tasks": ["task2", "task3"],
            "rationale": (
                "Control: same cognitive level (low). "
                "Shows upper bound for within-level transfer."
            ),
        },
        {
            "name": "same_level_high",
            "train_tasks": ["task7", "task8"],
            "test_tasks": ["task8", "task9"],
            "rationale": (
                "Control: same cognitive level (high). "
                "Shows upper bound for within-level transfer at high load."
            ),
        },
        {
            "name": "all_low_to_all_high",
            "train_tasks": ["task1", "task2", "task3"],
            "test_tasks": ["task7", "task8", "task9"],
            "rationale": (
                "Full low-to-high: use all 3 low tasks for train, "
                "all 3 high tasks for test."
            ),
        },
    ]
    return splits


def prepare_cross_task_data(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    subject_ids: List[str],
    train_tasks: List[str],
    test_tasks: List[str],
    dataset: dict,
    use_pictures: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Prepare train/test data for a cross-task split.

    The function re-extracts features separately for train tasks and test tasks,
    then aligns subjects present in both.

    Args:
        feature_matrix: unused placeholder (features are re-extracted).
        feature_names: unused placeholder.
        subject_ids: unused placeholder.
        train_tasks: list of task keys for training.
        test_tasks: list of task keys for testing.
        dataset: full dataset dict from data_loader.
        use_pictures: whether to use picture-level features.

    Returns:
        (X_train, y_train, X_test, y_test, train_subjects, test_subjects)
    """
    from .features import build_feature_matrix

    n_train = len(train_tasks)
    n_test = len(test_tasks)

    # Build features for train tasks with generic prefixes (t0_, t1_, ...)
    X_train, feat_names_train, train_subjects = build_feature_matrix(
        dataset, train_tasks, use_pictures=use_pictures, generic_prefix=True,
    )
    # Build features for test tasks with same generic prefixes
    X_test, feat_names_test, test_subjects = build_feature_matrix(
        dataset, test_tasks, use_pictures=use_pictures, generic_prefix=True,
    )

    # Find common subjects
    common_subjects = sorted(set(train_subjects) & set(test_subjects))
    if len(common_subjects) < 2:
        raise ValueError(
            f"Need at least 2 common subjects, got {len(common_subjects)}"
        )

    logger.info(
        "Cross-task: %d common subjects (train=%d, test=%d, common=%d)",
        len(train_subjects), len(train_subjects), len(test_subjects), len(common_subjects),
    )

    # Align train
    train_idx = [train_subjects.index(s) for s in common_subjects]
    X_train_aligned = X_train[train_idx]

    # Align test
    test_idx = [test_subjects.index(s) for s in common_subjects]
    X_test_aligned = X_test[test_idx]

    # Handle feature name mismatch: use intersection or pad
    common_feats_train = feat_names_train
    common_feats_test = feat_names_test

    # If feature sets differ (shouldn't normally), intersect
    if set(common_feats_train) != set(common_feats_test):
        common_feats = sorted(set(common_feats_train) & set(common_feats_test))
        if not common_feats:
            raise ValueError("No common features between train and test tasks")
        train_feat_idx = [common_feats_train.index(f) for f in common_feats]
        test_feat_idx = [common_feats_test.index(f) for f in common_feats]
        X_train_aligned = X_train_aligned[:, train_feat_idx]
        X_test_aligned = X_test_aligned[:, test_feat_idx]
        logger.warning("Feature mismatch: using %d common features", len(common_feats))

    # Encode labels (subject IDs -> integers)
    subject_to_label = {s: i for i, s in enumerate(common_subjects)}
    y_train = np.array([subject_to_label[s] for s in common_subjects])
    y_test = np.array([subject_to_label[s] for s in common_subjects])

    return (
        X_train_aligned, y_train,
        X_test_aligned, y_test,
        common_subjects, common_subjects,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_subjects: int,
) -> Dict[str, Any]:
    """Compute classification metrics.

    Returns:
        dict with accuracy, balanced_accuracy, f1_macro, f1_weighted,
        confusion_matrix, per_class_accuracy, random_baseline.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_subjects)))

    # Per-class accuracy
    per_class_acc = {}
    for i in range(n_subjects):
        mask = y_true == i
        if mask.sum() > 0:
            per_class_acc[i] = float(np.mean(y_pred[mask] == i))

    random_baseline = 1.0 / n_subjects if n_subjects > 0 else 0.0

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
        "random_baseline": random_baseline,
        "n_subjects": n_subjects,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    save_path: str,
    max_labels: int = 30,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """Plot and save a confusion matrix heatmap."""
    n = cm.shape[0]
    if n > max_labels:
        # Subsample for readability
        step = max(1, n // max_labels)
        indices = list(range(0, n, step))
        cm_plot = cm[np.ix_(indices, indices)]
        tick_labels = [str(i) for i in indices]
    else:
        cm_plot = cm
        tick_labels = [str(i) for i in range(n)]

    # Normalize by row
    cm_norm = cm_plot.astype(float) / (cm_plot.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted Subject")
    ax.set_ylabel("True Subject")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", save_path)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    title: str,
    save_path: str,
    top_n: int = 30,
) -> None:
    """Plot top-N feature importances bar chart."""
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), values[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance to %s", save_path)


def plot_results_summary(
    all_results: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Plot a summary bar chart of results across splits and models."""
    split_names = [r["split_name"] for r in all_results]
    model_names = sorted(set(r["model_name"] for r in all_results))

    data = {}
    for model in model_names:
        data[model] = []
        for r in all_results:
            if r["model_name"] == model:
                data[model].append(r["metrics"]["accuracy"])
            else:
                data[model].append(0.0)

    x = np.arange(len(split_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(model_names):
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, data[model], width, label=model)

    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Task Person Identification Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(split_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.axhline(y=1.0 / 69, color="r", linestyle="--", alpha=0.5, label="Random baseline")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved results summary to %s", save_path)


def save_results(
    all_results: List[Dict[str, Any]],
    save_dir: str,
) -> None:
    """Save all results to JSON and generate plots.

    Args:
        all_results: list of dicts, each with:
            split_name, model_name, metrics (without confusion_matrix), report
        save_dir: directory to save outputs.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results as JSON
    json_results = []
    for r in all_results:
        entry = {
            "split_name": r["split_name"],
            "split_info": r.get("split_info", {}),
            "model_name": r["model_name"],
            "accuracy": r["metrics"]["accuracy"],
            "balanced_accuracy": r["metrics"]["balanced_accuracy"],
            "f1_macro": r["metrics"]["f1_macro"],
            "f1_weighted": r["metrics"]["f1_weighted"],
            "random_baseline": r["metrics"]["random_baseline"],
            "n_subjects": r["metrics"]["n_subjects"],
        }
        json_results.append(entry)

    with open(save_dir / "results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Save summary table as CSV-like text
    lines = ["split,model,accuracy,bal_acc,f1_macro,f1_weighted,random_baseline,n_subjects"]
    for r in all_results:
        m = r["metrics"]
        lines.append(
            f"{r['split_name']},{r['model_name']},"
            f"{m['accuracy']:.4f},{m['balanced_accuracy']:.4f},"
            f"{m['f1_macro']:.4f},{m['f1_weighted']:.4f},"
            f"{m['random_baseline']:.4f},{m['n_subjects']}"
        )
    with open(save_dir / "summary.csv", "w") as f:
        f.write("\n".join(lines))

    # Generate plots
    plot_results_summary(all_results, str(save_dir / "results_summary.png"))

    # Confusion matrices for best model per split
    seen_splits = set()
    for r in all_results:
        if r["split_name"] not in seen_splits and r["metrics"].get("confusion_matrix") is not None:
            seen_splits.add(r["split_name"])
            cm = r["metrics"]["confusion_matrix"]
            plot_confusion_matrix(
                cm,
                title=f"Confusion Matrix: {r['split_name']} / {r['model_name']}",
                save_path=str(save_dir / f"cm_{r['split_name']}_{r['model_name']}.png"),
            )

    logger.info("All results saved to %s", save_dir)
