from pathlib import Path

import numpy as np

from src.data_contract import TASK_IDS, load_video_session, select_cohort


def _write_session(path: Path, subject: str, date: str) -> None:
    task_et = {}
    for task in TASK_IDS:
        gaze = np.arange(20, dtype=float).reshape(10, 2)
        task_et[task] = gaze
        task_et[f"{task}_pic"] = [gaze[:5], gaze[5:]]
    np.savez(path, subject=np.array(subject), date=np.array(date), type=np.array("video"),
             task_et=np.array(task_et, dtype=object))


def test_loader_preserves_stored_picture_boundaries_and_session_choice(tmp_path: Path) -> None:
    old = tmp_path / "002_patient-2024_01_01_00_00_00-video.npz"
    new = tmp_path / "002_patient-2024_01_02_00_00_00-video.npz"
    _write_session(old, "002_patient", "2024_01_01_00_00_00")
    _write_session(new, "002_patient", "2024_01_02_00_00_00")
    first = load_video_session(old)
    second = load_video_session(new)
    selected, exclusions = select_cohort([second, first], minimum_picture_samples=1)
    assert selected == [first]
    assert len(selected[0].tasks["task1"]) == 2
    assert any("not selected" in exclusion.reason for exclusion in exclusions)
