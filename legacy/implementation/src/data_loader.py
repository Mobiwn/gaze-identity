"""Data loader for EasyCog eye-tracking dataset.

Loads .npz files from the unsliced folder and extracts task-level
eye-tracking data organized by subject and task.
"""

from __future__ import annotations

import glob
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Task keys in the task_et dictionary
TASK_KEYS = [f"task{i}" for i in range(1, 10)]


def parse_filename(filepath: str) -> Tuple[str, str, str]:
    """Parse subject ID, date, and data type from filename.

    Expected format: ``{subject}_patient-{date}-{type}.npz``
    e.g. ``002_patient-2024_12_04_17_40_08-video.npz``

    Returns:
        (subject_id, date_str, data_type)
    """
    name = Path(filepath).stem  # remove .npz
    # Match: subject_patient-date-type
    m = re.match(r"^(\d+)_patient-([\d_]+)-(\w+)$", name)
    if m is None:
        raise ValueError(f"Cannot parse filename: {filepath}")
    subject_id = m.group(1) + "_patient"
    date_str = m.group(2)
    data_type = m.group(3)
    return subject_id, date_str, data_type


def load_subject_video(
    filepath: str,
) -> Dict[str, np.ndarray]:
    """Load eye-tracking data from a single video .npz file.

    Returns:
        dict with keys:
            'subject': str
            'date': str
            'task_et': dict mapping task_key -> np.ndarray (n_samples, 2)
            'task_et_pic': dict mapping task_key -> list of np.ndarray per picture
    """
    data = np.load(filepath, allow_pickle=True)
    subject_id = str(data["subject"].item())
    date_str = str(data["date"].item())
    data_type = str(data["type"].item())

    if data_type != "video":
        raise ValueError(f"Expected video data, got {data_type} for {filepath}")

    task_et_raw = data["task_et"].item()

    result = {
        "subject": subject_id,
        "date": date_str,
        "filepath": filepath,
        "task_et": {},
        "task_et_pic": {},
    }

    for key in TASK_KEYS:
        if key in task_et_raw:
            arr = task_et_raw[key]
            if arr is not None and isinstance(arr, np.ndarray):
                result["task_et"][key] = arr.astype(np.float64)
            else:
                result["task_et"][key] = None
        else:
            result["task_et"][key] = None

        pic_key = f"{key}_pic"
        if pic_key in task_et_raw:
            pic_data = task_et_raw[pic_key]
            if pic_data is not None and isinstance(pic_data, np.ndarray):
                pics = []
                for i in range(len(pic_data)):
                    p = pic_data[i]
                    if p is not None and isinstance(p, np.ndarray):
                        pics.append(p.astype(np.float64))
                result["task_et_pic"][key] = pics
            else:
                result["task_et_pic"][key] = []
        else:
            result["task_et_pic"][key] = []

    return result


def load_dataset(
    data_dir: str,
    subjects: Optional[List[str]] = None,
    prefer_first_session: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load all subjects' eye-tracking data from the dataset directory.

    Args:
        data_dir: Path to folder containing .npz files.
        subjects: If provided, only load these subject IDs (e.g. ['002_patient']).
        prefer_first_session: If True and a subject has multiple sessions,
            keep only the first session (sorted by date).

    Returns:
        dict mapping subject_id -> {task_key -> np.ndarray(n_samples, 2)}
    """
    data_dir = Path(data_dir)
    npz_files = sorted(glob.glob(str(data_dir / "*_patient-*-video.npz")))

    if not npz_files:
        raise FileNotFoundError(f"No video .npz files found in {data_dir}")

    logger.info("Found %d video files in %s", len(npz_files), data_dir)

    # Group by subject
    subject_files: Dict[str, List[str]] = {}
    for f in npz_files:
        try:
            subject_id, date_str, data_type = parse_filename(f)
        except ValueError as e:
            logger.warning("Skipping file: %s", e)
            continue
        if subjects is not None and subject_id not in subjects:
            continue
        subject_files.setdefault(subject_id, []).append((date_str, f))

    # Sort sessions by date and optionally keep only the first
    dataset: Dict[str, Dict[str, np.ndarray]] = {}
    for subject_id, file_list in sorted(subject_files.items()):
        file_list.sort(key=lambda x: x[0])  # sort by date
        if prefer_first_session:
            file_list = file_list[:1]

        for date_str, filepath in file_list:
            try:
                subject_data = load_subject_video(filepath)
                # Store under subject_id; if multiple sessions, merge or overwrite
                if subject_id not in dataset:
                    dataset[subject_id] = subject_data["task_et"]
                else:
                    # Already have data for this subject (multiple sessions kept)
                    # Just keep the first one
                    pass
            except Exception as e:
                logger.warning("Error loading %s: %s", filepath, e)
                continue

    logger.info("Loaded data for %d subjects", len(dataset))
    return dataset


def get_task_samples_per_subject(
    dataset: Dict[str, Dict[str, np.ndarray]],
    task_key: str,
) -> Tuple[List[str], List[np.ndarray]]:
    """Get all subjects' data for a specific task.

    Returns:
        (subject_ids, gaze_arrays) where each gaze_array has shape (n_samples, 2)
    """
    subject_ids = []
    gaze_arrays = []
    for subject_id in sorted(dataset.keys()):
        if task_key in dataset[subject_id] and dataset[subject_id][task_key] is not None:
            subject_ids.append(subject_id)
            gaze_arrays.append(dataset[subject_id][task_key])
    return subject_ids, gaze_arrays


def get_subject_task_matrix(
    dataset: Dict[str, Dict[str, np.ndarray]],
    task_keys: List[str],
) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]]]:
    """Get subjects that have data for ALL specified tasks.

    Returns:
        (valid_subjects, filtered_dataset)
    """
    valid_subjects = []
    for subject_id in sorted(dataset.keys()):
        has_all = all(
            tk in dataset[subject_id] and dataset[subject_id][tk] is not None
            for tk in task_keys
        )
        if has_all:
            valid_subjects.append(subject_id)

    filtered = {s: dataset[s] for s in valid_subjects}
    return valid_subjects, filtered


def dataset_summary(dataset: Dict[str, Dict[str, np.ndarray]]) -> dict:
    """Compute summary statistics for the loaded dataset."""
    n_subjects = len(dataset)
    task_counts = {tk: 0 for tk in TASK_KEYS}
    task_shapes = {tk: [] for tk in TASK_KEYS}

    for subject_id, tasks in dataset.items():
        for tk in TASK_KEYS:
            if tk in tasks and tasks[tk] is not None:
                task_counts[tk] += 1
                task_shapes[tk].append(tasks[tk].shape)

    summary = {
        "n_subjects": n_subjects,
        "task_counts": task_counts,
        "task_sample_shapes": {
            tk: task_shapes[tk][0] if task_shapes[tk] else None for tk in TASK_KEYS
        },
    }
    return summary
