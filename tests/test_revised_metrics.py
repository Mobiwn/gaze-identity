import numpy as np

from src.revised_metrics import ranking_metrics, subject_label_permutation_pvalue


def test_ranking_metrics_use_class_labels_not_score_column_numbers() -> None:
    metrics = ranking_metrics(
        np.array([10, 20]),
        np.array([[0.1, 0.9], [0.7, 0.3]]),
        np.array([20, 10]),
        top_k=2,
    )
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["top2_accuracy"] == 1.0
    assert metrics["mean_reciprocal_rank"] == 1.0


def test_permutation_pvalue_is_bounded() -> None:
    y_true = np.array([0, 0, 1, 1])
    value = subject_label_permutation_pvalue(
        y_true,
        y_true,
        np.array(["a", "a", "b", "b"]),
        resamples=20,
        seed=42,
    )
    assert 0 < value <= 1
