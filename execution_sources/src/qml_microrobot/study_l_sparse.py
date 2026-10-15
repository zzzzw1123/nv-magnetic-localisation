from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def select_feature_indices(
    features: np.ndarray,
    targets: np.ndarray,
    family: str,
    count: int = 12,
) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    positions = np.asarray(targets, dtype=float)
    if count <= 0 or count > values.shape[1]:
        raise ValueError("invalid feature count")
    standardized = StandardScaler().fit_transform(values)
    balanced = family.startswith("sensor_balanced_")
    scoring_family = family.removeprefix("sensor_balanced_") if balanced else family
    if scoring_family in {"variance_top12", "variance_rank"}:
        score = np.var(values, axis=0)
    elif scoring_family in {"target_correlation_top12", "target_correlation_rank"}:
        centered_x = standardized - standardized.mean(axis=0)
        centered_y = positions - positions.mean(axis=0)
        denominator = np.sqrt(np.sum(centered_x**2, axis=0)[:, None] * np.sum(centered_y**2, axis=0)[None, :])
        correlations = np.abs((centered_x.T @ centered_y) / np.maximum(denominator, 1e-12))
        score = np.linalg.norm(correlations, axis=1)
    elif scoring_family in {"ridge_weight_top12", "ridge_weight_rank"}:
        target_scaler = StandardScaler().fit(positions)
        model = Ridge(alpha=1.0).fit(standardized, target_scaler.transform(positions))
        score = np.linalg.norm(model.coef_, axis=0)
    else:
        raise ValueError(f"unknown selector family: {family}")
    if balanced:
        if count != 16:
            raise ValueError("sensor-balanced selection is fixed to two of three features per sensor")
        selected = []
        for sensor in range(8):
            group = np.arange(3 * sensor, 3 * sensor + 3)
            order = np.lexsort((group, -score[group]))
            selected.extend(group[order[:2]])
        return np.sort(np.asarray(selected, dtype=int))
    order = np.lexsort((np.arange(values.shape[1]), -score))
    return np.sort(order[:count])
