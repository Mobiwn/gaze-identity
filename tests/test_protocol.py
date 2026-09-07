import pytest

from src.protocol import partition_rows, validate_task_disjoint


def test_protocol_rejects_shared_task() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_task_disjoint({
            "train_tasks": ["task1", "task2"],
            "validation_tasks": ["task2"],
            "test_tasks": ["task3"],
        })


def test_partition_keeps_only_subjects_present_everywhere() -> None:
    rows = [
        {"subject_id": "a", "task_id": "task1"}, {"subject_id": "a", "task_id": "task2"},
        {"subject_id": "a", "task_id": "task3"}, {"subject_id": "b", "task_id": "task1"},
        {"subject_id": "b", "task_id": "task2"}, {"subject_id": "b", "task_id": "task3"},
        {"subject_id": "c", "task_id": "task1"},
    ]
    partitions = partition_rows(rows, {
        "train_tasks": ["task1"], "validation_tasks": ["task2"], "test_tasks": ["task3"],
    })
    assert {row["subject_id"] for row in partitions["train"]} == {"a", "b"}
