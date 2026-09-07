"""Task-disjoint protocol checks and deterministic feature-table partitioning."""
from __future__ import annotations

from typing import Any

import numpy as np


def validate_task_disjoint(protocol: dict[str, Any]) -> None:
    partitions = {name: set(protocol[f"{name}_tasks"]) for name in ("train", "validation", "test")}
    if any(not tasks for tasks in partitions.values()):
        raise ValueError("train, validation, and test task lists must be non-empty")
    names = list(partitions)
    overlaps = {f"{left}/{right}": sorted(partitions[left] & partitions[right])
                for index, left in enumerate(names) for right in names[index + 1:]
                if partitions[left] & partitions[right]}
    if overlaps:
        raise ValueError(f"task partitions overlap: {overlaps}")


def partition_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    validate_task_disjoint(protocol)
    partitions = {name: [row for row in rows if row["task_id"] in protocol[f"{name}_tasks"]]
                  for name in ("train", "validation", "test")}
    subject_sets = {name: {row["subject_id"] for row in data} for name, data in partitions.items()}
    common = set.intersection(*subject_sets.values())
    if len(common) < 2:
        raise ValueError("fewer than two common subjects across task partitions")
    return {name: [row for row in data if row["subject_id"] in common] for name, data in partitions.items()}


def feature_matrix(rows: list[dict[str, Any]], feature_names: list[str], label_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([[row[name] for name in feature_names] for row in rows], dtype=float),
            np.array([label_map[row["subject_id"]] for row in rows], dtype=int))
