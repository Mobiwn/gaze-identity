"""Auditable picture-level gaze features for the revised protocol.

The extractor intentionally excludes AOI-grid features until screen bounds and
coordinate semantics are verified. Entropy features are computed from counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from .data_contract import SessionRecord


@dataclass(frozen=True)
class FeatureSettings:
    sampling_rate_hz: float = 125.0
    velocity_threshold_px_s: float = 30.0
    minimum_fixation_duration_s: float = 0.1
    histogram_bins: int = 20


def _interpolate(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(values)
    if valid.sum() < 2:
        raise ValueError("fewer than two finite coordinate values")
    if not valid.all():
        index = np.arange(len(values))
        values[~valid] = np.interp(index[~valid], index[valid], values[valid])
    return values


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _shape_stat(values: np.ndarray, statistic: str) -> float:
    """Return a stable skew/kurtosis value for constant or near-constant data."""
    if np.std(values) < 1e-12:
        return 0.0
    value = stats.skew(values) if statistic == "skew" else stats.kurtosis(values)
    return float(value) if np.isfinite(value) else 0.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(int)
    boundaries = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in boundaries.reshape(-1, 2)]


def extract_picture_features(gaze: np.ndarray, settings: FeatureSettings) -> dict[str, float]:
    """Extract deterministic features from one stored picture segment."""
    gaze = np.asarray(gaze, dtype=float)
    if gaze.ndim != 2 or gaze.shape[1] != 2 or len(gaze) < 2:
        raise ValueError("gaze must have shape (n_samples >= 2, 2)")
    x, y = _interpolate(gaze[:, 0]), _interpolate(gaze[:, 1])
    fs = settings.sampling_rate_hz
    dx, dy = np.diff(x), np.diff(y)
    speed = np.hypot(dx, dy) * fs
    acceleration = np.diff(speed) * fs if len(speed) > 1 else np.array([0.0])
    duration = len(gaze) / fs
    saccade_mask = np.r_[False, speed > settings.velocity_threshold_px_s]
    fixation_runs = [run for run in _runs(~saccade_mask)
                     if (run[1] - run[0]) / fs >= settings.minimum_fixation_duration_s]
    saccade_runs = _runs(saccade_mask)
    fixation_durations = np.array([(end - start) / fs for start, end in fixation_runs], dtype=float)
    fixation_dispersion = np.array([np.ptp(x[start:end]) + np.ptp(y[start:end])
                                    for start, end in fixation_runs], dtype=float)
    saccade_amplitudes = np.array([np.hypot(x[end - 1] - x[start], y[end - 1] - y[start])
                                   for start, end in saccade_runs], dtype=float)
    saccade_peaks = np.array([speed[start:end].max() for start, end in saccade_runs], dtype=float)

    def average(values: np.ndarray) -> float:
        return float(values.mean()) if len(values) else 0.0

    def spread(values: np.ndarray) -> float:
        return float(values.std()) if len(values) else 0.0

    hist_2d, _, _ = np.histogram2d(x, y, bins=settings.histogram_bins)
    histogram_x, _ = np.histogram(x, bins=settings.histogram_bins)
    histogram_y, _ = np.histogram(y, bins=settings.histogram_bins)
    centroid_distance = np.hypot(x - x.mean(), y - y.mean())
    scanpath = np.hypot(dx, dy).sum()
    return {
        "pos_mean_x": float(x.mean()), "pos_mean_y": float(y.mean()),
        "pos_std_x": float(x.std()), "pos_std_y": float(y.std()),
        "pos_median_x": float(np.median(x)), "pos_median_y": float(np.median(y)),
        "pos_iqr_x": float(stats.iqr(x)), "pos_iqr_y": float(stats.iqr(y)),
        "pos_range_x": float(np.ptp(x)), "pos_range_y": float(np.ptp(y)),
        "pos_skew_x": _shape_stat(x, "skew"), "pos_skew_y": _shape_stat(y, "skew"),
        "pos_kurtosis_x": _shape_stat(x, "kurtosis"), "pos_kurtosis_y": _shape_stat(y, "kurtosis"),
        "pos_centroid_distance": float(centroid_distance.mean()),
        "vel_mean": float(speed.mean()), "vel_std": float(speed.std()),
        "vel_median": float(np.median(speed)), "vel_p95": float(np.percentile(speed, 95)),
        "vel_p99": float(np.percentile(speed, 99)), "vel_max": float(speed.max()),
        "accel_abs_mean": float(np.abs(acceleration).mean()), "accel_std": float(acceleration.std()),
        "fixation_count": float(len(fixation_runs)), "fixation_rate": float(len(fixation_runs) / duration),
        "fixation_duration_mean": average(fixation_durations), "fixation_duration_std": spread(fixation_durations),
        "fixation_dispersion_mean": average(fixation_dispersion),
        "fixation_time_ratio": float(fixation_durations.sum() / duration) if len(fixation_durations) else 0.0,
        "saccade_count": float(len(saccade_runs)), "saccade_rate": float(len(saccade_runs) / duration),
        "saccade_amplitude_mean": average(saccade_amplitudes), "saccade_amplitude_std": spread(saccade_amplitudes),
        "saccade_peak_velocity_mean": average(saccade_peaks),
        "scanpath_length": float(scanpath), "scanpath_length_per_second": float(scanpath / duration),
        "dispersion_total": float(x.std() + y.std()), "entropy_2d": _entropy_from_counts(hist_2d),
        "entropy_x": _entropy_from_counts(histogram_x), "entropy_y": _entropy_from_counts(histogram_y),
    }


def build_feature_rows(sessions: list[SessionRecord], settings: FeatureSettings,
                       minimum_picture_samples: int, maximum_missing_fraction: float
                       ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return feature rows and explicit segment-level exclusions."""
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for session in sessions:
        for task_id, pictures in session.tasks.items():
            for picture in pictures:
                row_id = f"{picture.subject_id}/{picture.session_id}/{task_id}/{picture.picture_id}"
                if len(picture.gaze) < minimum_picture_samples:
                    exclusions.append({"row_id": row_id, "reason": "too few samples"})
                    continue
                if picture.missing_fraction > maximum_missing_fraction:
                    exclusions.append({"row_id": row_id, "reason": "missingness exceeds threshold"})
                    continue
                try:
                    features = extract_picture_features(picture.gaze, settings)
                except ValueError as error:
                    exclusions.append({"row_id": row_id, "reason": str(error)})
                    continue
                if not all(np.isfinite(value) for value in features.values()):
                    exclusions.append({"row_id": row_id, "reason": "non-finite feature"})
                    continue
                rows.append({"row_id": row_id, "subject_id": picture.subject_id,
                             "session_id": picture.session_id, "task_id": task_id,
                             "picture_id": picture.picture_id, "n_samples": len(picture.gaze),
                             "missing_fraction": picture.missing_fraction, **features})
    return rows, exclusions
