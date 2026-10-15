from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NvSpatialData:
    features: np.ndarray
    positions_um: np.ndarray
    replicate_counts: np.ndarray
    coordinate_ids: np.ndarray
    block_ids: np.ndarray


def _inventory_member(root: Path, suffix: str) -> Path:
    inventory = root / "results" / "study_l_l1_selective_inventory.csv"
    with inventory.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if "Fig.3 and ML model and data/Fig. 3d and Fig. 3e/" in row["member"]
        and row["member"].endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one L1 member ending in {suffix}, found {len(matches)}")
    path = root / matches[0]["local_file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def author_features(parameters: np.ndarray, parameter_std: np.ndarray) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=float)
    parameter_std = np.asarray(parameter_std, dtype=float)
    if parameters.shape != parameter_std.shape or parameters.ndim != 3:
        raise ValueError("parameter arrays must have identical [sample,sensor,fit] shape")
    if parameters.shape[1:] != (8, 6):
        raise ValueError("expected eight sensors and six fit values")
    split = np.abs(parameters[:, :, 1] - parameters[:, :, 4])
    stacked = np.stack(
        [split, parameter_std[:, :, 1], parameter_std[:, :, 4]], axis=2
    ).reshape(parameters.shape[0], 24)
    if np.any(stacked <= 0.0) or not np.isfinite(stacked).all():
        raise ValueError("log features must be finite and strictly positive")
    return np.log(stacked)


def spatial_block_ids(positions_um: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_um, dtype=float)
    x_edges = np.array([1395.0, 1495.0])
    y_edges = np.array([1450.0, 1710.0, 1970.0])
    x_band = np.searchsorted(x_edges, positions[:, 0], side="right")
    y_band = np.searchsorted(y_edges, positions[:, 1], side="right")
    return (x_band * 4 + y_band).astype(int)


def load_l1_spatial_data(root: Path) -> NvSpatialData:
    parameters = np.load(_inventory_member(root, "para_list.npy"), allow_pickle=False)
    parameter_std = np.load(
        _inventory_member(root, "para_std_list.npy"), allow_pickle=False
    )
    x = np.load(_inventory_member(root, "x_pos.npy"), allow_pickle=False).astype(float)
    y = np.load(_inventory_member(root, "y_pos.npy"), allow_pickle=False).astype(float)
    raw_features = author_features(parameters, parameter_std)
    raw_positions = np.column_stack([x, y])

    coordinates, inverse, counts = np.unique(
        raw_positions, axis=0, return_inverse=True, return_counts=True
    )
    features = np.vstack(
        [raw_features[inverse == index].mean(axis=0) for index in range(len(coordinates))]
    )
    coordinate_ids = np.array(
        [f"x{int(x_value)}_y{int(y_value)}" for x_value, y_value in coordinates]
    )
    return NvSpatialData(
        features=features,
        positions_um=coordinates,
        replicate_counts=counts,
        coordinate_ids=coordinate_ids,
        block_ids=spatial_block_ids(coordinates),
    )


def outer_spatial_indices(
    positions_um: np.ndarray, block_ids: np.ndarray, test_block: int, guard_um: float
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions_um, dtype=float)
    test = np.flatnonzero(block_ids == test_block)
    if test.size == 0:
        raise ValueError(f"empty test block {test_block}")
    low = positions[test].min(axis=0) - guard_um
    high = positions[test].max(axis=0) + guard_um
    in_guard_box = np.all((positions >= low) & (positions <= high), axis=1)
    train = np.flatnonzero(~in_guard_box)
    return train, test


def farthest_point_indices(positions_um: np.ndarray, budget: int) -> np.ndarray:
    positions = np.asarray(positions_um, dtype=float)
    if budget <= 0 or budget > len(positions):
        raise ValueError("budget must be within available training coordinates")
    center = positions.mean(axis=0)
    selected = [int(np.argmin(np.sum((positions - center) ** 2, axis=1)))]
    min_distance = np.sum((positions - positions[selected[0]]) ** 2, axis=1)
    while len(selected) < budget:
        min_distance[selected] = -1.0
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        distance = np.sum((positions - positions[next_index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return np.array(selected, dtype=int)
