"""Metrics and uncertainty estimates for closed-set identification."""
from __future__ import annotations

import numpy as np


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray, top_k: int) -> dict[str, float]:
    ranked_labels = classes[np.argsort(scores, axis=1)[:, ::-1]]
    ranks = np.array([np.flatnonzero(row == truth)[0] + 1 for row, truth in zip(ranked_labels, y_true)])
    return {"top1_accuracy": float((ranks == 1).mean()),
            f"top{top_k}_accuracy": float((ranks <= top_k).mean()),
            "mean_reciprocal_rank": float((1.0 / ranks).mean())}


def participant_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, subject_ids: np.ndarray,
                          resamples: int, seed: int) -> tuple[float, float]:
    """95% percentile interval from resampling subjects, not picture rows."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject_ids)
    by_subject = {subject: np.flatnonzero(subject_ids == subject) for subject in subjects}
    values = []
    for _ in range(resamples):
        sample = rng.choice(subjects, len(subjects), replace=True)
        indexes = np.concatenate([by_subject[subject] for subject in sample])
        values.append(float((y_true[indexes] == y_pred[indexes]).mean()))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def subject_label_permutation_pvalue(y_true: np.ndarray, y_pred: np.ndarray, subject_ids: np.ndarray,
                                     resamples: int, seed: int) -> float:
    """One-sided label-permutation p-value, preserving rows within a subject."""
    observed = float((y_true == y_pred).mean())
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject_ids)
    label_for_subject = {subject: y_true[np.flatnonzero(subject_ids == subject)[0]] for subject in subjects}
    original = np.array([label_for_subject[subject] for subject in subjects])
    null = []
    for _ in range(resamples):
        shuffled = rng.permutation(original)
        mapping = dict(zip(subjects, shuffled))
        permuted = np.array([mapping[subject] for subject in subject_ids])
        null.append(float((permuted == y_pred).mean()))
    return float((1 + np.count_nonzero(np.asarray(null) >= observed)) / (resamples + 1))
