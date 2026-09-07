"""Small, explicit model set for the task-disjoint baseline."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "nearest_centroid": Pipeline([("scale", StandardScaler()), ("model", NearestCentroid())]),
        "logistic_regression_c0.1": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=3000, random_state=seed)),
        ]),
        "logistic_regression_c1": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=3000, random_state=seed)),
        ]),
    }


def class_scores(model: Pipeline, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return larger-is-better class scores and their corresponding labels."""
    classifier: Any = model.named_steps["model"]
    if hasattr(classifier, "decision_function"):
        return classifier.decision_function(model.named_steps["scale"].transform(matrix)), classifier.classes_
    if isinstance(classifier, NearestCentroid):
        scaled = model.named_steps["scale"].transform(matrix)
        distances = ((scaled[:, None, :] - classifier.centroids_[None, :, :]) ** 2).sum(axis=2)
        return -distances, classifier.classes_
    raise TypeError(f"No score adapter for {type(classifier).__name__}")
