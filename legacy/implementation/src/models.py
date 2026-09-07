"""Machine learning models for gaze-based person identification.

Provides multiple classifiers for multi-class subject identification
and pairwise verification.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def build_models(random_state: int = 42) -> Dict[str, Any]:
    """Build a dictionary of named classifiers.

    Returns:
        dict mapping model_name -> sklearn-compatible estimator.
    """
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=random_state,
            )),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                decision_function_shape="ovr",
                random_state=random_state,
                probability=True,
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                eval_metric="mlogloss",
                n_jobs=-1,
            )),
        ]),
    }
    return models


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Train each model and evaluate on test data.

    Args:
        X_train: (n_train, n_features) training features.
        y_train: (n_train,) training labels (integer subject IDs).
        X_test: (n_test, n_features) test features.
        y_test: (n_test,) test labels.
        models: dict of models; if None, builds default set.
        random_state: random seed.

    Returns:
        dict mapping model_name -> {
            'model': fitted estimator,
            'accuracy': float,
            'balanced_accuracy': float,
            'f1_macro': float,
            'y_pred': array,
            'y_prob': array or None,
            'report': str,
        }
    """
    if models is None:
        models = build_models(random_state=random_state)

    results = {}
    for name, model in models.items():
        logger.info("Training %s ...", name)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Get probabilities if available
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                except Exception:
                    pass

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            report = classification_report(y_test, y_pred, zero_division=0)

            results[name] = {
                "model": model,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "f1_macro": f1,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "report": report,
            }
            logger.info(
                "  %s: acc=%.4f, bal_acc=%.4f, f1=%.4f",
                name, acc, bal_acc, f1,
            )
        except Exception as e:
            logger.warning("  %s failed: %s", name, e)
            results[name] = {
                "model": None,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "f1_macro": 0.0,
                "y_pred": None,
                "y_prob": None,
                "report": f"Error: {e}",
            }

    return results


def evaluate_single_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate a single pre-built model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            pass

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": classification_report(y_test, y_pred, zero_division=0),
    }


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    model_builder_fn,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Perform stratified k-fold cross-validation.

    Args:
        X: (n_samples, n_features)
        y: (n_samples,)
        model_builder_fn: callable that returns a fresh model instance.
        n_folds: number of folds.
        random_state: random seed.

    Returns:
        dict with per-fold and mean metrics.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_builder_fn()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        fold_metrics.append({
            "fold": fold_idx,
            "accuracy": accuracy_score(y_val, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
            "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
        })

    # Compute means
    mean_metrics = {
        k: np.mean([fm[k] for fm in fold_metrics])
        for k in ["accuracy", "balanced_accuracy", "f1_macro"]
    }
    std_metrics = {
        f"{k}_std": np.std([fm[k] for fm in fold_metrics])
        for k in ["accuracy", "balanced_accuracy", "f1_macro"]
    }

    return {
        "folds": fold_metrics,
        "mean": mean_metrics,
        "std": std_metrics,
    }


def encode_labels(subject_ids: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
    """Encode string subject IDs to integer labels.

    Returns:
        (label_array, label_to_id_map)
    """
    unique_subjects = sorted(set(subject_ids))
    id_to_label = {sid: i for i, sid in enumerate(unique_subjects)}
    label_to_id = {i: sid for sid, i in id_to_label.items()}
    labels = np.array([id_to_label[sid] for sid in subject_ids])
    return labels, label_to_id
