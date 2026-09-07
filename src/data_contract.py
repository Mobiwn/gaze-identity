"""Validated access to the local EasyCog eye-tracking NPZ files.

This module is intentionally independent of the legacy loader. It preserves
picture boundaries and session provenance so experiments can be audited.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TASK_IDS = tuple(f"task{i}" for i in range(1, 10))
REQUIRED_KEYS = {"subject", "date", "type", "task_et"}
_FILENAME = re.compile(r"^(?P<subject>\d+)_patient-(?P<date>[\d_]+)-video\.npz$")


@dataclass(frozen=True)
class PictureRecord:
    subject_id: str
    session_id: str
    task_id: str
    picture_id: int
    gaze: np.ndarray
    missing_fraction: float


@dataclass(frozen=True)
class SessionRecord:
    subject_id: str
    session_id: str
    path: Path
    tasks: dict[str, tuple[PictureRecord, ...]]


@dataclass(frozen=True)
class Exclusion:
    path: str
    reason: str


def _parse_filename(path: Path) -> tuple[str, str]:
    match = _FILENAME.match(path.name)
    if not match:
        raise ValueError("filename does not match '<id>_patient-<date>-video.npz'")
    return f"{match.group('subject')}_patient", match.group("date")


def _as_gaze(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{label} must have shape (n_samples, 2), got {array.shape}")
    if len(array) == 0:
        raise ValueError(f"{label} is empty")
    return array


def load_video_session(path: str | Path) -> SessionRecord:
    """Load one video session and retain its stored picture-level gaze arrays."""
    path = Path(path)
    filename_subject, filename_date = _parse_filename(path)
    with np.load(path, allow_pickle=True) as loaded:
        missing = REQUIRED_KEYS.difference(loaded.files)
        if missing:
            raise ValueError(f"missing required keys: {sorted(missing)}")
        subject_id = str(loaded["subject"].item())
        session_id = str(loaded["date"].item())
        if str(loaded["type"].item()) != "video":
            raise ValueError("NPZ type is not 'video'")
        if subject_id != filename_subject or session_id != filename_date:
            raise ValueError("filename and NPZ subject/date disagree")
        task_et = loaded["task_et"].item()
    if not isinstance(task_et, dict):
        raise ValueError("task_et is not a dictionary")

    tasks: dict[str, tuple[PictureRecord, ...]] = {}
    for task_id in TASK_IDS:
        full_gaze = task_et.get(task_id)
        pictures = task_et.get(f"{task_id}_pic")
        if full_gaze is None or pictures is None:
            raise ValueError(f"{task_id} or {task_id}_pic is missing")
        _as_gaze(full_gaze, task_id)
        if not isinstance(pictures, (list, tuple, np.ndarray)) or len(pictures) == 0:
            raise ValueError(f"{task_id}_pic is empty or not a sequence")
        records = []
        for picture_id, picture in enumerate(pictures):
            gaze = _as_gaze(picture, f"{task_id}_pic[{picture_id}]")
            records.append(PictureRecord(subject_id, session_id, task_id, picture_id, gaze,
                                         float(np.isnan(gaze).any(axis=1).mean())))
        tasks[task_id] = tuple(records)
    return SessionRecord(subject_id, session_id, path, tasks)


def load_sessions(data_dir: str | Path) -> tuple[list[SessionRecord], list[Exclusion]]:
    """Load every candidate video file, returning explicit exclusions."""
    paths = sorted(Path(data_dir).glob("*_patient-*-video.npz"))
    if not paths:
        raise FileNotFoundError(f"no video NPZ files found in {data_dir}")
    sessions, exclusions = [], []
    for path in paths:
        try:
            sessions.append(load_video_session(path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            exclusions.append(Exclusion(str(path), str(error)))
    return sessions, exclusions


def select_cohort(sessions: Iterable[SessionRecord], *, session_policy: str = "earliest_valid",
                  minimum_picture_samples: int = 100, maximum_missing_fraction: float = 0.2
                  ) -> tuple[list[SessionRecord], list[Exclusion]]:
    """Select exactly one quality-qualified session per subject deterministically."""
    if session_policy not in {"earliest_valid", "latest_valid"}:
        raise ValueError("session_policy must be 'earliest_valid' or 'latest_valid'")
    grouped: dict[str, list[SessionRecord]] = {}
    exclusions: list[Exclusion] = []
    for session in sessions:
        valid = all(len(pic.gaze) >= minimum_picture_samples and pic.missing_fraction <= maximum_missing_fraction
                    for pictures in session.tasks.values() for pic in pictures)
        if valid:
            grouped.setdefault(session.subject_id, []).append(session)
        else:
            exclusions.append(Exclusion(str(session.path), "failed picture length or missingness quality rule"))
    reverse = session_policy == "latest_valid"
    selected = [sorted(candidates, key=lambda item: item.session_id, reverse=reverse)[0]
                for _, candidates in sorted(grouped.items())]
    selected_paths = {session.path for session in selected}
    for candidates in grouped.values():
        for session in candidates:
            if session.path not in selected_paths:
                exclusions.append(Exclusion(str(session.path), f"not selected by {session_policy} policy"))
    return selected, exclusions


def manifest(sessions: Iterable[SessionRecord], exclusions: Iterable[Exclusion], config: dict[str, Any]) -> dict[str, Any]:
    """Create JSON-safe provenance for an experiment cohort."""
    sessions, exclusions = list(sessions), list(exclusions)
    files = []
    for session in sessions:
        files.append({"subject_id": session.subject_id, "session_id": session.session_id,
                      "path": str(session.path), "sha256": hashlib.sha256(session.path.read_bytes()).hexdigest(),
                      "picture_counts": {task: len(pictures) for task, pictures in session.tasks.items()}})
    return {"schema": "gaze-identity-cohort-manifest/v1", "n_subjects": len({s.subject_id for s in sessions}),
            "n_sessions": len(sessions), "task_ids": list(TASK_IDS), "files": files,
            "exclusions": [asdict(item) for item in exclusions], "config": config}
