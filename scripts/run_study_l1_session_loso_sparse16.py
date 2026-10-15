"""Evaluate Full24 and Sparse16 Ridge readouts across eight NV sessions.

The script reads 819 observations with 24 spectral features and two coordinates
in micrometres. It writes LOAO and chronological-repeat errors to CSV plus a
JSON summary. Feature rankings and Ridge penalties are selected within each
outer training split.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from qml_microrobot.study_l_session import load_l1_session_data
from qml_microrobot.study_l_sparse import select_feature_indices


ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.01, 1.0, 100.0)
FAMILIES = (
    "ridge_weight_rank",
    "target_correlation_rank",
    "sensor_balanced_ridge_weight_rank",
    "sensor_balanced_target_correlation_rank",
)


def error_um(truth, prediction):
    """Return mean Euclidean error for paired two-coordinate arrays in micrometres."""

    return float(np.linalg.norm(truth - prediction, axis=1).mean())


def fit_predict(features, targets, train, test, alpha, indices=None):
    """Fit a training-standardised Ridge model and predict the requested test rows."""

    use = np.arange(features.shape[1]) if indices is None else np.asarray(indices)
    scaler = StandardScaler().fit(features[train][:, use])
    model = Ridge(alpha=alpha).fit(scaler.transform(features[train][:, use]), targets[train])
    return model.predict(scaler.transform(features[test][:, use]))


def select_full(features, targets, indices, groups):
    """Select the dense 24-feature Ridge penalty by grouped inner cross-validation."""

    splits = GroupKFold(min(4, len(np.unique(groups[indices]))))
    candidates = []
    for alpha in ALPHAS:
        errors = []
        for local_train, local_valid in splits.split(
            features[indices], targets[indices], groups=groups[indices]
        ):
            train = indices[local_train]
            valid = indices[local_valid]
            errors.append(error_um(targets[valid], fit_predict(features, targets, train, valid, alpha)))
        candidates.append((float(np.mean(errors)), alpha))
    return min(candidates)[1]


def select_sparse(features, targets, indices, groups):
    """Select a 16-feature ranking family and Ridge penalty using inner groups.

    Feature ranking is recomputed inside every inner training fold and once more
    on the complete outer-training partition before the returned refit.
    """

    splits = GroupKFold(min(4, len(np.unique(groups[indices]))))
    candidates = []
    for family in FAMILIES:
        for alpha in ALPHAS:
            errors = []
            for local_train, local_valid in splits.split(
                features[indices], targets[indices], groups=groups[indices]
            ):
                train = indices[local_train]
                valid = indices[local_valid]
                selected = select_feature_indices(features[train], targets[train], family, 16)
                prediction = fit_predict(features, targets, train, valid, alpha, selected)
                errors.append(error_um(targets[valid], prediction))
            candidates.append((float(np.mean(errors)), family, alpha))
    _, family, alpha = min(candidates)
    selected = select_feature_indices(features[indices], targets[indices], family, 16)
    return family, alpha, selected


def evaluate_split(data, train, test):
    """Evaluate dense and nested Sparse16 routes on one disjoint outer split."""

    alpha = select_full(data.features, data.positions_um, train, data.session_ids)
    full_prediction = fit_predict(
        data.features, data.positions_um, train, test, alpha
    )
    family, sparse_alpha, selected = select_sparse(
        data.features, data.positions_um, train, data.session_ids
    )
    sparse_prediction = fit_predict(
        data.features,
        data.positions_um,
        train,
        test,
        sparse_alpha,
        selected,
    )
    return {
        "full_error_um": error_um(data.positions_um[test], full_prediction),
        "sparse_error_um": error_um(data.positions_um[test], sparse_prediction),
        "full_alpha": alpha,
        "sparse_alpha": sparse_alpha,
        "selector_family": family,
        "selected_indices": "-".join(map(str, selected)),
    }


def main() -> None:
    """Run eight session holdouts plus eligible chronological-repeat evaluations."""

    data = load_l1_session_data(ROOT)
    sessions = list(dict.fromkeys(data.session_ids.tolist()))
    rows = []
    for session in sessions:
        test = np.flatnonzero(data.session_ids == session)
        train = np.flatnonzero(data.session_ids != session)
        result = evaluate_split(data, train, test)
        rows.append(
            {
                "evaluation": "session_loso",
                "test_session": session,
                "train_sessions": len(np.unique(data.session_ids[train])),
                "test_rows": len(test),
                **result,
            }
        )

    chronological = []
    for session_index in range(2, len(sessions)):
        session = sessions[session_index]
        earlier_sessions = set(sessions[:session_index])
        earlier_coordinates = set(
            data.coordinate_ids[np.isin(data.session_ids, list(earlier_sessions))]
        )
        test = np.flatnonzero(
            (data.session_ids == session)
            & np.isin(data.coordinate_ids, list(earlier_coordinates))
        )
        if len(test) == 0:
            continue
        train = np.flatnonzero(np.isin(data.session_ids, list(earlier_sessions)))
        result = evaluate_split(data, train, test)
        record = {
            "evaluation": "chronological_repeat",
            "test_session": session,
            "train_sessions": len(earlier_sessions),
            "test_rows": len(test),
            **result,
        }
        rows.append(record)
        chronological.append(record)

    output_csv = ROOT / "results" / "study_l1_session_loso_sparse16_rows.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loso = [row for row in rows if row["evaluation"] == "session_loso"]
    full_mean = float(np.mean([row["full_error_um"] for row in loso]))
    sparse_mean = float(np.mean([row["sparse_error_um"] for row in loso]))
    wins = sum(row["sparse_error_um"] <= 1.05 * row["full_error_um"] for row in loso)
    worst_relative = max(
        (row["sparse_error_um"] - row["full_error_um"]) / row["full_error_um"]
        for row in loso
    )
    repeat_full = float(np.mean([row["full_error_um"] for row in chronological]))
    repeat_sparse = float(np.mean([row["sparse_error_um"] for row in chronological]))
    reduction = 1.0 - 34.0 / 50.0
    conditions = {
        "equal_session_mean": sparse_mean <= 1.05 * full_mean,
        "session_stability": wins >= 6,
        "worst_session": worst_relative <= 0.15,
        "parameter_reduction": reduction >= 0.30,
        "chronological_repeat_mean": repeat_sparse <= 1.05 * repeat_full,
        "chronological_repeat_sessions": len(chronological) >= 3,
    }
    passed = all(conditions.values())
    result = {
        "protocol_id": "STUDY_L1_NV_SESSION_LOSO_SPARSE16_V1",
        "status": (
            "L1_SPARSE16_INTERNAL_CROSS_SESSION_HIGH_BAR_PASSED_EXTERNAL_ARRAY_CONFIRMATION_REQUIRED"
            if passed
            else "L1_SPARSE16_CROSS_SESSION_HIGH_BAR_FAILED_INTERNAL_CANDIDATE_CLOSED"
        ),
        "session_loso": {
            "sessions": len(loso),
            "full24_equal_session_mean_error_um": full_mean,
            "sparse16_equal_session_mean_error_um": sparse_mean,
            "relative_change": (sparse_mean - full_mean) / full_mean,
            "noninferiority_sessions": wins,
            "worst_session_relative_increase": worst_relative,
        },
        "chronological_same_coordinate_repeats": {
            "sessions": len(chronological),
            "rows": sum(row["test_rows"] for row in chronological),
            "full24_equal_session_mean_error_um": repeat_full,
            "sparse16_equal_session_mean_error_um": repeat_sparse,
            "relative_change": (repeat_sparse - repeat_full) / repeat_full,
        },
        "trainable_parameters": {"full24": 50, "sparse16": 34},
        "parameter_reduction": reduction,
        "conditions": conditions,
        "high_bar_passed": passed,
        "independent_sensor_arrays": 1,
        "external_confirmation": "REQUIRED_UNAVAILABLE",
        "quantum_calls": 0,
        "speed_claim_allowed": False,
    }
    (ROOT / "results" / "study_l1_session_loso_sparse16_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
