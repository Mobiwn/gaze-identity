"""Gaze feature extraction for eye-tracking data.

Extracts handcrafted features from (x, y) gaze time series at ~125 Hz.
Features are designed to capture individual-specific gaze behavior patterns
that may generalize across different cognitive tasks.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

SAMPLING_RATE = 125.0  # Hz


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe_divide(a: float, b: float, fill: float = 0.0) -> float:
    """Divide a / b, returning fill if b is zero."""
    return a / b if abs(b) > 1e-12 else fill


def _zero_features(prefix: str) -> Dict[str, float]:
    """Return a dictionary of zero-valued features for degenerate signals.

    This ensures consistent feature dimensions even for all-NaN or constant data.
    """
    # Build a template by running on a small valid signal
    template_gaze = np.array([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0]])
    template = extract_task_features(template_gaze, prefix=prefix)
    return {k: 0.0 for k in template.keys()}


def _interpolate_nans(signal_1d: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1-D signal.

    If all values are NaN, returns zeros. If fewer than 2 good values exist,
    returns the signal with NaNs replaced by 0.
    """
    nans = np.isnan(signal_1d)
    if not np.any(nans):
        return signal_1d
    good = ~nans
    n_good = np.sum(good)
    if n_good < 2:
        # Not enough good points to interpolate; replace with 0
        signal_1d[nans] = 0.0
        return signal_1d
    indices = np.arange(len(signal_1d))
    signal_1d[nans] = np.interp(indices[nans], indices[good], signal_1d[good])
    return signal_1d


# ---------------------------------------------------------------------------
# Fixation / saccade detection (I-VT velocity-threshold)
# ---------------------------------------------------------------------------

def detect_fixations_saccades(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = SAMPLING_RATE,
    vel_threshold: float = 30.0,
    min_fix_duration_s: float = 0.1,
) -> Dict[str, object]:
    """Detect fixations and saccades using the I-VT algorithm.

    Args:
        x, y: gaze coordinates (1-D arrays of same length).
        fs: sampling rate in Hz.
        vel_threshold: velocity threshold in pixels/s (above = saccade).
        min_fix_duration_s: minimum fixation duration in seconds.

    Returns:
        dict with 'fixations' list, 'saccades' list, 'velocity' array.
    """
    n = len(x)
    dx = np.diff(x)
    dy = np.diff(y)
    dt = 1.0 / fs
    speed = np.sqrt(dx**2 + dy**2) / dt  # pixels/s

    # Pad speed to match original length (insert 0 at start)
    speed_padded = np.zeros(n)
    speed_padded[1:] = speed

    is_saccade = speed_padded > vel_threshold

    # Segment into fixations and saccades
    min_fix_samples = int(min_fix_duration_s * fs)
    fixations = []
    saccades = []
    i = 0
    while i < n:
        if not is_saccade[i]:
            # Find end of fixation
            j = i
            while j < n and not is_saccade[j]:
                j += 1
            dur = j - i
            if dur >= min_fix_samples:
                fixations.append({
                    "start": i,
                    "end": j,
                    "duration_s": dur / fs,
                    "mean_x": np.mean(x[i:j]),
                    "mean_y": np.mean(y[i:j]),
                    "dispersion": np.ptp(x[i:j]) + np.ptp(y[i:j]),
                })
            i = j
        else:
            # Find end of saccade
            j = i
            while j < n and is_saccade[j]:
                j += 1
            dur = j - i
            amp = np.sqrt((x[j - 1] - x[i]) ** 2 + (y[j - 1] - y[i]) ** 2)
            peak_vel = np.max(speed_padded[i:j]) if j > i else 0.0
            saccades.append({
                "start": i,
                "end": j,
                "duration_s": dur / fs,
                "amplitude": amp,
                "peak_velocity": peak_vel,
            })
            i = j

    return {
        "fixations": fixations,
        "saccades": saccades,
        "velocity": speed_padded,
    }


# ---------------------------------------------------------------------------
# Feature extraction: single task (full time-series)
# ---------------------------------------------------------------------------

def extract_task_features(
    gaze: np.ndarray,
    fs: float = SAMPLING_RATE,
    prefix: str = "",
) -> Dict[str, float]:
    """Extract features from a single task's gaze data.

    Args:
        gaze: array of shape (n_samples, 2) with (x, y) columns.
        fs: sampling rate in Hz.
        prefix: optional string prefix for feature names (e.g. 'task1_').

    Returns:
        dict mapping feature_name -> value.
    """
    if gaze is None or len(gaze) == 0:
        return {}

    x = _interpolate_nans(gaze[:, 0].copy())
    y = _interpolate_nans(gaze[:, 1].copy())

    # Check if signal is degenerate (all same value after interpolation)
    if np.std(x) < 1e-10 and np.std(y) < 1e-10:
        # Return zero features for degenerate signals
        return _zero_features(prefix)

    n = len(x)

    feats: Dict[str, float] = {}
    p = prefix

    # --- Position features ---
    feats[f"{p}pos_mean_x"] = float(np.mean(x))
    feats[f"{p}pos_mean_y"] = float(np.mean(y))
    feats[f"{p}pos_std_x"] = float(np.std(x))
    feats[f"{p}pos_std_y"] = float(np.std(y))
    feats[f"{p}pos_range_x"] = float(np.ptp(x))
    feats[f"{p}pos_range_y"] = float(np.ptp(y))
    feats[f"{p}pos_median_x"] = float(np.median(x))
    feats[f"{p}pos_median_y"] = float(np.median(y))
    feats[f"{p}pos_iqr_x"] = float(stats.iqr(x))
    feats[f"{p}pos_iqr_y"] = float(stats.iqr(y))
    feats[f"{p}pos_skew_x"] = float(stats.skew(x))
    feats[f"{p}pos_skew_y"] = float(stats.skew(y))
    feats[f"{p}pos_kurtosis_x"] = float(stats.kurtosis(x))
    feats[f"{p}pos_kurtosis_y"] = float(stats.kurtosis(y))

    # Centroid
    cx, cy = np.mean(x), np.mean(y)
    feats[f"{p}pos_centroid_dist"] = float(np.mean(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))
    feats[f"{p}pos_bounding_area"] = float(np.ptp(x) * np.ptp(y))

    # --- Velocity features ---
    dx = np.diff(x)
    dy = np.diff(y)
    dt = 1.0 / fs
    speed = np.sqrt(dx**2 + dy**2) / dt
    vx = dx / dt
    vy = dy / dt

    feats[f"{p}vel_mean"] = float(np.mean(speed))
    feats[f"{p}vel_std"] = float(np.std(speed))
    feats[f"{p}vel_max"] = float(np.max(speed))
    feats[f"{p}vel_median"] = float(np.median(speed))
    feats[f"{p}vel_p95"] = float(np.percentile(speed, 95))
    feats[f"{p}vel_p99"] = float(np.percentile(speed, 99))
    feats[f"{p}vel_mean_x"] = float(np.mean(np.abs(vx)))
    feats[f"{p}vel_mean_y"] = float(np.mean(np.abs(vy)))
    feats[f"{p}vel_ratio_x_y"] = float(
        _safe_divide(np.mean(np.abs(vx)), np.mean(np.abs(vy)))
    )

    # --- Acceleration features ---
    if len(speed) > 1:
        accel = np.diff(speed) * fs
        feats[f"{p}accel_mean"] = float(np.mean(np.abs(accel)))
        feats[f"{p}accel_std"] = float(np.std(accel))
    else:
        feats[f"{p}accel_mean"] = 0.0
        feats[f"{p}accel_std"] = 0.0

    # --- Fixation / saccade features ---
    fsd = detect_fixations_saccades(x, y, fs=fs)
    fixs = fsd["fixations"]
    sacs = fsd["saccades"]

    n_fix = len(fixs)
    n_sac = len(sacs)
    total_dur = n / fs

    feats[f"{p}fix_count"] = float(n_fix)
    feats[f"{p}fix_rate"] = float(_safe_divide(n_fix, total_dur))
    if n_fix > 0:
        fix_durs = [f["duration_s"] for f in fixs]
        feats[f"{p}fix_mean_dur"] = float(np.mean(fix_durs))
        feats[f"{p}fix_std_dur"] = float(np.std(fix_durs))
        feats[f"{p}fix_median_dur"] = float(np.median(fix_durs))
        fix_disps = [f["dispersion"] for f in fixs]
        feats[f"{p}fix_mean_dispersion"] = float(np.mean(fix_disps))
        fix_areas = [f["duration_s"] for f in fixs]
        feats[f"{p}fix_total_time_ratio"] = float(
            _safe_divide(sum(fix_areas), total_dur)
        )
    else:
        feats[f"{p}fix_mean_dur"] = 0.0
        feats[f"{p}fix_std_dur"] = 0.0
        feats[f"{p}fix_median_dur"] = 0.0
        feats[f"{p}fix_mean_dispersion"] = 0.0
        feats[f"{p}fix_total_time_ratio"] = 0.0

    feats[f"{p}sac_count"] = float(n_sac)
    feats[f"{p}sac_rate"] = float(_safe_divide(n_sac, total_dur))
    if n_sac > 0:
        sac_amps = [s["amplitude"] for s in sacs]
        sac_peaks = [s["peak_velocity"] for s in sacs]
        sac_durs = [s["duration_s"] for s in sacs]
        feats[f"{p}sac_mean_amp"] = float(np.mean(sac_amps))
        feats[f"{p}sac_std_amp"] = float(np.std(sac_amps))
        feats[f"{p}sac_mean_peak_vel"] = float(np.mean(sac_peaks))
        feats[f"{p}sac_mean_dur"] = float(np.mean(sac_durs))
    else:
        feats[f"{p}sac_mean_amp"] = 0.0
        feats[f"{p}sac_std_amp"] = 0.0
        feats[f"{p}sac_mean_peak_vel"] = 0.0
        feats[f"{p}sac_mean_dur"] = 0.0

    # --- Scanpath features ---
    scanpath_len = float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)))
    feats[f"{p}scan_total_length"] = scanpath_len
    feats[f"{p}scan_length_per_second"] = float(_safe_divide(scanpath_len, total_dur))
    # Bounding box ratio (width / height)
    bx, by = np.ptp(x), np.ptp(y)
    feats[f"{p}scan_bbox_ratio"] = float(_safe_divide(bx, by))

    # --- Dispersion / entropy features ---
    # Spatial dispersion (std of gaze positions)
    feats[f"{p}disp_total"] = float(np.std(x) + np.std(y))
    # Shannon entropy of binned gaze positions
    try:
        H, _ = np.histogramdd(np.column_stack([x, y]), bins=20)
        p_dist = H / (H.sum() + 1e-12)
        feats[f"{p}entropy_2d"] = float(-np.sum(p_dist[p_dist > 0] * np.log2(p_dist[p_dist > 0])))
    except Exception:
        feats[f"{p}entropy_2d"] = 0.0

    # 1-D entropy for x and y separately
    try:
        Hx, _ = np.histogram(x, bins=20, density=True)
        Hy, _ = np.histogram(y, bins=20, density=True)
        feats[f"{p}entropy_x"] = float(-np.sum(Hx[Hx > 0] * np.log2(Hx[Hx > 0])))
        feats[f"{p}entropy_y"] = float(-np.sum(Hy[Hy > 0] * np.log2(Hy[Hy > 0])))
    except Exception:
        feats[f"{p}entropy_x"] = 0.0
        feats[f"{p}entropy_y"] = 0.0

    # --- Grid-based AOI features (4x4 grid) ---
    x_edges = np.linspace(np.min(x) - 1e-6, np.max(x) + 1e-6, 5)
    y_edges = np.linspace(np.min(y) - 1e-6, np.max(y) + 1e-6, 5)
    grid_counts = np.zeros((4, 4))
    for xi, yi in zip(x, y):
        ix = np.searchsorted(x_edges[1:], xi)
        iy = np.searchsorted(y_edges[1:], yi)
        ix = min(ix, 3)
        iy = min(iy, 3)
        grid_counts[iy, ix] += 1
    grid_props = grid_counts / (n + 1e-12)
    for iy in range(4):
        for ix in range(4):
            feats[f"{p}grid_{iy}{ix}"] = float(grid_props[iy, ix])

    # Grid entropy
    p_grid = grid_props.flatten() + 1e-12
    feats[f"{p}grid_entropy"] = float(-np.sum(p_grid * np.log2(p_grid)))

    return feats


# ---------------------------------------------------------------------------
# Feature extraction: per-picture then aggregate
# ---------------------------------------------------------------------------

def extract_features_from_pictures(
    pictures: List[np.ndarray],
    fs: float = SAMPLING_RATE,
    prefix: str = "",
) -> Dict[str, float]:
    """Extract features per picture and aggregate with statistics.

    Args:
        pictures: list of arrays, each (n_samples_per_pic, 2).
        fs: sampling rate.
        prefix: feature name prefix.

    Returns:
        dict of aggregated features.
    """
    if not pictures:
        return {}

    pic_features_list = []
    for i, pic in enumerate(pictures):
        pic_feats = extract_task_features(pic, fs=fs, prefix="")
        pic_features_list.append(pic_feats)

    if not pic_features_list:
        return {}

    # Get feature names from first picture
    feat_names = sorted(pic_features_list[0].keys())
    aggregated = {}

    for fname in feat_names:
        vals = [pf[fname] for pf in pic_features_list if fname in pf]
        if not vals:
            continue
        vals_arr = np.array(vals)
        aggregated[f"{prefix}{fname}_mean"] = float(np.mean(vals_arr))
        aggregated[f"{prefix}{fname}_std"] = float(np.std(vals_arr))
        aggregated[f"{prefix}{fname}_min"] = float(np.min(vals_arr))
        aggregated[f"{prefix}{fname}_max"] = float(np.max(vals_arr))
        aggregated[f"{prefix}{fname}_range"] = float(np.ptp(vals_arr))

    return aggregated


# ---------------------------------------------------------------------------
# High-level: build feature matrix for a dataset
# ---------------------------------------------------------------------------

def build_feature_matrix(
    dataset: dict,
    task_keys: List[str],
    use_pictures: bool = True,
    fs: float = SAMPLING_RATE,
    generic_prefix: bool = False,
) -> tuple:
    """Build feature matrix for a set of subjects and tasks.

    Args:
        dataset: dict from data_loader (subject_id -> task_et dict).
        task_keys: which tasks to extract features from.
        use_pictures: if True, extract per-pic features and aggregate.
        fs: sampling rate.
        generic_prefix: if True, use generic slot-based prefixes (t0_, t1_, ...)
            instead of task-specific prefixes. This allows matching features
            between different task groups for cross-task evaluation.

    Returns:
        (feature_matrix, feature_names, subject_ids)
        feature_matrix: np.ndarray of shape (n_subjects, n_features)
        feature_names: list of str
        subject_ids: list of str
    """
    all_features = []
    subject_ids = []
    feature_names = None

    for subject_id in sorted(dataset.keys()):
        tasks = dataset[subject_id]
        subject_feats = {}

        for slot_idx, tk in enumerate(task_keys):
            if tk not in tasks or tasks[tk] is None:
                continue

            gaze = tasks[tk]
            if generic_prefix:
                prefix = f"t{slot_idx}_"
            else:
                prefix = f"{tk}_"

            if use_pictures:
                pics = _segment_task_into_pictures(gaze)
                pic_feats = extract_features_from_pictures(pics, fs=fs, prefix=prefix)
                subject_feats.update(pic_feats)
            else:
                task_feats = extract_task_features(gaze, fs=fs, prefix=prefix)
                subject_feats.update(task_feats)

        if subject_feats:
            if feature_names is None:
                feature_names = sorted(subject_feats.keys())
            all_features.append([subject_feats.get(f, 0.0) for f in feature_names])
            subject_ids.append(subject_id)

    if not all_features:
        return np.array([]), [], []

    return np.array(all_features, dtype=np.float64), feature_names, subject_ids


def _segment_task_into_pictures(
    gaze: np.ndarray,
    n_pictures: int = 10,
) -> List[np.ndarray]:
    """Split a task's gaze data into roughly equal picture segments.

    This is a fallback when per-picture data is not directly available.
    """
    n = len(gaze)
    seg_len = n // n_pictures
    pics = []
    for i in range(n_pictures):
        start = i * seg_len
        end = start + seg_len if i < n_pictures - 1 else n
        pics.append(gaze[start:end])
    return pics


def get_feature_names() -> List[str]:
    """Return the list of base feature names (without prefix)."""
    # Extract from a dummy signal to discover feature names
    dummy = np.random.randn(6250, 2)
    feats = extract_task_features(dummy, prefix="")
    return sorted(feats.keys())
