import numpy as np

from src.revised_features import FeatureSettings, extract_picture_features


def test_features_are_finite_and_deterministic_with_missing_samples() -> None:
    gaze = np.column_stack([np.linspace(0, 100, 250), np.linspace(50, 150, 250)])
    gaze[20:30] = np.nan
    settings = FeatureSettings()
    first = extract_picture_features(gaze, settings)
    second = extract_picture_features(gaze, settings)
    assert first == second
    assert len(first) > 20
    assert all(np.isfinite(value) for value in first.values())


def test_constant_gaze_has_defined_shape_statistics() -> None:
    features = extract_picture_features(np.full((125, 2), 20.0), FeatureSettings())
    assert features["pos_skew_x"] == 0.0
    assert features["pos_kurtosis_y"] == 0.0
    assert features["entropy_2d"] == 0.0
