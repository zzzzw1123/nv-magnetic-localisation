from __future__ import annotations

import csv
import hashlib
import io
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qml_microrobot.study_l30_author_cnn import (  # noqa: E402
    AuthorLocalizationCNN,
    fit_author_cnn,
    load_author_data,
    mean_euclidean_error_um,
    predict_author_cnn,
    trainable_parameters,
)
from qml_microrobot.study_l31_compact import (  # noqa: E402
    classical_state_bytes,
    fit_classical_candidate,
    fit_compact_mlp,
    fitted_parameter_count,
    predict_compact_mlp,
)
from scripts.run_study_l1_session_loso_sparse16 import select_full, select_sparse  # noqa: E402


PROTOCOL_PATH = ROOT / "study_l/l31_compact_nonlinear_bridge_protocol.json"
CV_PATH = ROOT / "results/study_l31_compact_candidate_cv.csv"
REPLAY_PATH = ROOT / "results/study_l31_compact_replay_rows.csv"
RESOURCE_PATH = ROOT / "results/study_l31_compact_resource.csv"
RESULT_PATH = ROOT / "results/study_l31_compact_nonlinear_bridge_result.json"
MANIFEST_PATH = ROOT / "results/study_l31_evidence_manifest.json"
CNN_STATE_PATH = ROOT / "checkpoints/study_l31_author_first5.pt"
CNN_META_PATH = ROOT / "checkpoints/study_l31_author_first5.json"
COMPACT_STATE_PATH = ROOT / "checkpoints/study_l31_selected_compact.pkl"
DOCUMENT_PATH = ROOT / "docs/study_l31_compact_nonlinear_bridge.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error(truth, prediction) -> float:
    return mean_euclidean_error_um(truth, prediction)


def candidate_ids(protocol: dict) -> list[str]:
    values = []
    for alpha in protocol["candidates"]["polynomial2_ridge"]["alphas"]:
        values.append(f"poly2_a{alpha}")
    for dimension in protocol["candidates"]["rff_ridge"]["dimensions"]:
        for alpha in protocol["candidates"]["rff_ridge"]["alphas"]:
            values.append(f"rff{dimension}_a{alpha}")
    for width in protocol["candidates"]["compact_mlp"]["hidden_widths"]:
        values.append(f"mlp{width}_fixed")
    return values


def fit_candidate(candidate_id: str, features, targets, protocol):
    if candidate_id.startswith("mlp"):
        width = int(candidate_id.split("_", 1)[0][3:])
        config = protocol["candidates"]["compact_mlp"]
        return fit_compact_mlp(
            features,
            targets,
            width,
            seed=int(config["seed"]),
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
        )
    return fit_classical_candidate(candidate_id, features, targets)


def predict_candidate(candidate_id: str, model, features):
    if candidate_id.startswith("mlp"):
        return predict_compact_mlp(model, features)
    return model.predict(features)


def candidate_bytes(candidate_id: str, model) -> int:
    if candidate_id.startswith("mlp"):
        return model.serialized_bytes()
    return classical_state_bytes(model)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def development_cv(data, development_sessions, protocol):
    rows = []
    development_mask = np.isin(data.session_ids, development_sessions)
    for candidate in candidate_ids(protocol):
        for fold, validation_session in enumerate(development_sessions):
            validation = np.flatnonzero(data.session_ids == validation_session)
            train = np.flatnonzero(development_mask & (data.session_ids != validation_session))
            started = time.perf_counter()
            model = fit_candidate(candidate, data.features[train], data.positions_um[train], protocol)
            fit_sec = time.perf_counter() - started
            prediction = predict_candidate(candidate, model, data.features[validation])
            rows.append(
                {
                    "candidate": candidate,
                    "fitted_parameters": fitted_parameter_count(candidate),
                    "validation_fold": fold,
                    "validation_session": validation_session,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "error_um": error(data.positions_um[validation], prediction),
                    "fit_wall_sec": fit_sec,
                }
            )
            print(
                f"L31 development {candidate} fold {fold + 1}/5: "
                f"{rows[-1]['error_um']:.3f} um",
                flush=True,
            )
    summaries = []
    for candidate in candidate_ids(protocol):
        selected = [row for row in rows if row["candidate"] == candidate]
        summaries.append(
            {
                "candidate": candidate,
                "mean_error_um": float(np.mean([row["error_um"] for row in selected])),
                "worst_error_um": float(np.max([row["error_um"] for row in selected])),
                "fitted_parameters": fitted_parameter_count(candidate),
            }
        )
    minimum = min(row["mean_error_um"] for row in summaries)
    eligible = [row for row in summaries if row["mean_error_um"] <= minimum + 1.0]
    selected = min(
        eligible,
        key=lambda row: (row["fitted_parameters"], row["mean_error_um"], row["candidate"]),
    )
    return rows, summaries, selected


def train_or_load_author_cnn(data, train, protocol):
    l30 = json.loads((ROOT / "study_l/l30_author_cnn_fair_comparison_protocol.json").read_text(encoding="utf-8"))
    expected = {
        "train_indices_sha256": hashlib.sha256(np.asarray(train, dtype=np.int64).tobytes()).hexdigest(),
        "seed": l30["source_model"]["seed"],
        "epochs": l30["source_model"]["epochs"],
    }
    if CNN_STATE_PATH.is_file() and CNN_META_PATH.is_file():
        metadata = json.loads(CNN_META_PATH.read_text(encoding="utf-8"))
        if (
            all(metadata.get(key) == value for key, value in expected.items())
            and metadata.get("state_sha256") == sha256(CNN_STATE_PATH)
        ):
            model = AuthorLocalizationCNN()
            model.load_state_dict(torch.load(CNN_STATE_PATH, map_location="cpu", weights_only=True))
            return model, float(metadata["fit_wall_sec"]), True
    started = time.perf_counter()
    model, losses = fit_author_cnn(
        data.features,
        data.positions_um,
        train,
        seed=int(l30["source_model"]["seed"]),
        epochs=int(l30["source_model"]["epochs"]),
        batch_size=int(l30["source_model"]["batch_size"]),
        learning_rate=float(l30["source_model"]["learning_rate"]),
    )
    fit_sec = time.perf_counter() - started
    CNN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CNN_STATE_PATH)
    CNN_META_PATH.write_text(
        json.dumps(
            {
                **expected,
                "fit_wall_sec": fit_sec,
                "loss_checkpoints": losses,
                "state_sha256": sha256(CNN_STATE_PATH),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return model, fit_sec, False


def fit_linear_routes(data, development, protocol):
    alpha = select_full(data.features, data.positions_um, development, data.session_ids)
    full_scaler = StandardScaler().fit(data.features[development])
    full = Ridge(alpha=alpha).fit(
        full_scaler.transform(data.features[development]), data.positions_um[development]
    )
    family, sparse_alpha, selected = select_sparse(
        data.features, data.positions_um, development, data.session_ids
    )
    sparse_scaler = StandardScaler().fit(data.features[development][:, selected])
    sparse = Ridge(alpha=sparse_alpha).fit(
        sparse_scaler.transform(data.features[development][:, selected]),
        data.positions_um[development],
    )
    return {
        "full": full,
        "full_scaler": full_scaler,
        "full_alpha": alpha,
        "sparse": sparse,
        "sparse_scaler": sparse_scaler,
        "sparse_alpha": sparse_alpha,
        "sparse_family": family,
        "sparse_selected": selected,
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_L31_REPLAY_RESULTS":
        raise RuntimeError("L31 protocol is not frozen")
    torch.set_num_threads(1)
    data = load_author_data(ROOT)
    sessions = list(dict.fromkeys(data.session_ids.tolist()))
    development_sessions, replay_sessions = sessions[:5], sessions[5:]
    cv_rows, summaries, selected = development_cv(data, development_sessions, protocol)
    write_csv(CV_PATH, cv_rows)
    development = np.flatnonzero(np.isin(data.session_ids, development_sessions))
    started = time.perf_counter()
    compact = fit_candidate(selected["candidate"], data.features[development], data.positions_um[development], protocol)
    compact_fit_sec = time.perf_counter() - started
    COMPACT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPACT_STATE_PATH.write_bytes(pickle.dumps(compact, protocol=5))
    cnn, cnn_fit_sec, cnn_resumed = train_or_load_author_cnn(data, development, protocol)
    linear = fit_linear_routes(data, development, protocol)
    rows = []
    for session in replay_sessions:
        test = np.flatnonzero(data.session_ids == session)
        predictions = {
            "author_cnn": predict_author_cnn(cnn, data.features[test]),
            "full24": linear["full"].predict(linear["full_scaler"].transform(data.features[test])),
            "sparse16": linear["sparse"].predict(
                linear["sparse_scaler"].transform(data.features[test][:, linear["sparse_selected"]])
            ),
            "compact": predict_candidate(selected["candidate"], compact, data.features[test]),
        }
        errors = {name: error(data.positions_um[test], prediction) for name, prediction in predictions.items()}
        rows.append(
            {
                "test_session": session,
                "test_rows": len(test),
                "author_cnn_error_um": errors["author_cnn"],
                "full24_error_um": errors["full24"],
                "sparse16_error_um": errors["sparse16"],
                "selected_compact": selected["candidate"],
                "compact_error_um": errors["compact"],
                "compact_minus_cnn_um": errors["compact"] - errors["author_cnn"],
                "compact_win_or_tie": errors["compact"] <= errors["author_cnn"],
            }
        )
    write_csv(REPLAY_PATH, rows)
    means = {
        name: float(np.mean([row[f"{name}_error_um"] for row in rows]))
        for name in ("author_cnn", "full24", "sparse16", "compact")
    }
    reduction = 1.0 - selected["fitted_parameters"] / 721858.0
    checks = {
        "mean_error_ratio": means["compact"] <= protocol["promotion"]["maximum_compact_to_author_cnn_mean_error_ratio"] * means["author_cnn"],
        "session_wins_or_ties": sum(row["compact_win_or_tie"] for row in rows) >= protocol["promotion"]["minimum_compact_session_wins_or_ties_vs_author_cnn"],
        "parameter_reduction": reduction >= protocol["promotion"]["minimum_parameter_reduction_vs_author_cnn"],
    }
    promoted = all(checks.values())
    resources = [
        {
            "model": "author_cnn_first5",
            "fitted_parameters": trainable_parameters(cnn),
            "fit_wall_sec": cnn_fit_sec,
            "serialized_state_bytes": CNN_STATE_PATH.stat().st_size,
        },
        {
            "model": selected["candidate"],
            "fitted_parameters": selected["fitted_parameters"],
            "fit_wall_sec": compact_fit_sec,
            "serialized_state_bytes": candidate_bytes(selected["candidate"], compact),
        },
        {
            "model": "full24_ridge",
            "fitted_parameters": 50,
            "fit_wall_sec": "not_benchmarked_in_l31",
            "serialized_state_bytes": len(pickle.dumps((linear["full_scaler"], linear["full"]), protocol=5)),
        },
        {
            "model": "sparse16_ridge",
            "fitted_parameters": 34,
            "fit_wall_sec": "not_benchmarked_in_l31",
            "serialized_state_bytes": len(pickle.dumps((linear["sparse_scaler"], linear["sparse"]), protocol=5)),
        },
    ]
    write_csv(RESOURCE_PATH, resources)
    result = {
        "protocol_id": protocol["protocol_id"],
        "status": (
            "L31_COMPACT_NONLINEAR_BRIDGE_PROMOTED_INTERNAL_REPLAY"
            if promoted
            else "L31_COMPACT_NONLINEAR_BRIDGE_NOT_PROMOTED"
        ),
        "development_candidate_summaries": summaries,
        "selected_candidate": selected,
        "replay_sessions": len(rows),
        "equal_session_mean_error_um": means,
        "compact_parameter_reduction_vs_cnn": reduction,
        "compact_wins_or_ties_vs_cnn": sum(row["compact_win_or_tie"] for row in rows),
        "promotion_checks": checks,
        "promoted": promoted,
        "author_cnn_checkpoint_resumed": cnn_resumed,
        "replay_used_for_candidate_selection": False,
        "future_session_opened": False,
        "quantum_calls": 0,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    DOCUMENT_PATH.write_text(
        "# Study L31 Compact Nonlinear Bridge\n\n"
        f"Development-only selection chose `{selected['candidate']}` with "
        f"{selected['fitted_parameters']} fitted parameters. On the three historical "
        f"replay sessions, equal-session mean errors were CNN {means['author_cnn']:.4f} um, "
        f"compact {means['compact']:.4f} um, full24 {means['full24']:.4f} um and "
        f"Sparse16 {means['sparse16']:.4f} um. The compact route reduced parameters "
        f"by {100 * reduction:.4f}% versus CNN and passed {sum(checks.values())}/3 "
        f"predeclared promotion checks. Status: `{result['status']}`.\n\n"
        "Candidate selection used only the first five acquisition sessions. The last "
        "three historical replay sessions were not used for architecture, alpha or "
        "feature selection. This remains one-array internal replay evidence, with "
        "`quantum_calls=0` and no physical deployment claim.\n",
        encoding="utf-8",
    )
    relatives = [
        "study_l/l31_compact_nonlinear_bridge_protocol.json",
        "src/qml_microrobot/study_l31_compact.py",
        "scripts/run_study_l31_compact_nonlinear_bridge.py",
        "tests/test_study_l31_compact.py",
        "docs/study_l31_compact_nonlinear_bridge.md",
        "results/study_l31_compact_candidate_cv.csv",
        "results/study_l31_compact_replay_rows.csv",
        "results/study_l31_compact_resource.csv",
        "results/study_l31_compact_nonlinear_bridge_result.json",
        "checkpoints/study_l31_author_first5.pt",
        "checkpoints/study_l31_author_first5.json",
        "checkpoints/study_l31_selected_compact.pkl",
    ]
    entries = [
        {"path": relative, "bytes": (ROOT / relative).stat().st_size, "sha256": sha256(ROOT / relative)}
        for relative in relatives
    ]
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "status": "L31_COMPACT_NONLINEAR_BRIDGE_EVIDENCE_FROZEN",
                "entries": entries,
                "future_session_opened": False,
                "quantum_calls": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
