"""Evaluate input-group sensitivity using saved LOAO CNN models.

Replace feature groups with same-coordinate training references while keeping
weights, target coordinates and batch order fixed. Average errors over donors,
seeds and acquisitions. Partial replacements form synthetic vectors; full
replacements reuse training inputs. These are model-sensitivity diagnostics.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch


WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qml_microrobot.study_l30_author_cnn import AuthorLocalizationCNN, predict_author_cnn
from qml_microrobot.study_l_session import load_l1_session_data
from qml_microrobot.study_l_nv import _inventory_member

SEEDS = [20260804, 20260805, 20260806]
GROUPS = {"peak_separation": np.arange(0, 24, 3), "u1": np.arange(1, 24, 3),
          "u2": np.arange(2, 24, 3), "u1_u2": np.sort(np.r_[np.arange(1, 24, 3), np.arange(2, 24, 3)]),
          "all24": np.arange(24)}
TOL_UM = 1e-6
SOURCE = ROOT / "results/study_l_supplementary/loso_sample_predictions.csv"


def sha(path):
    """Return the SHA-256 digest of a checkpoint, metadata file, or output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    """Hash a contiguous array together with its current byte representation."""
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8).tobytes()).hexdigest()


def checkpoint_paths(fold, seed):
    """Resolve the saved state and metadata for one acquisition/seed pair."""
    if seed == SEEDS[0]:
        matches = sorted((ROOT / "checkpoints/study_l30_author_cnn").glob(f"session_{fold}_*.pt"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one saved L30 checkpoint: fold={fold}, found={len(matches)}")
        state = matches[0]
    else:
        state = ROOT / f"checkpoints/study_l34_multiseed/cnn_seed{seed}_loso_{fold}.pt"
    metadata = state.with_suffix(".json")
    if not state.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"Saved-only inference requires both {state} and {metadata}")
    return state, metadata


def records(frame):
    """Convert a DataFrame to high-precision JSON row dictionaries."""
    return json.loads(frame.to_json(orient="records", double_precision=15))


def load_saved(path):
    """Load one weights-only CNN checkpoint into the fixed architecture."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = AuthorLocalizationCNN()
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert sum(parameter.numel() for parameter in model.parameters()) == 721858
    return model


def main():
    """Reproduce 2,457 baselines, evaluate fixed donor groups, and export summaries."""
    WORK.mkdir(parents=True, exist_ok=True)
    script_sha = sha(Path(__file__))
    source_files = [SOURCE, ROOT / "src/qml_microrobot/study_l30_author_cnn.py",
                    ROOT / "src/qml_microrobot/study_l_session.py", ROOT / "src/qml_microrobot/study_l_nv.py",
                    ROOT / "results/study_l_l1_selective_inventory.csv", ROOT / "results/study_l1_archive_member_structure.csv"]
    source_files += [_inventory_member(ROOT, suffix) for suffix in ["para_list.npy", "para_std_list.npy", "x_pos.npy", "y_pos.npy"]]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_files}
    data = load_l1_session_data(ROOT)
    raw = pd.read_csv(SOURCE)
    assert raw.shape == (4914, 12) and not raw.isna().any().any()
    assert sorted(raw.seed.unique().tolist()) == SEEDS
    assert set(raw.model) == {"cnn", "mlp8"}
    assert not raw.duplicated(["row_index", "seed", "model"]).any()
    meta = raw.drop_duplicates("row_index").set_index("row_index").sort_index()
    assert np.array_equal(meta.index, np.arange(819))
    assert np.array_equal(meta[["true_x_um", "true_y_um"]].to_numpy(), data.positions_um)
    assert np.array_equal(meta.session.to_numpy(), data.session_ids)
    assert features_are_valid(data.features)
    paired = {}
    for row, info in meta.iterrows():
        train = meta.index[meta.fold.ne(info.fold)].to_numpy()
        donors = train[np.all(data.positions_um[train] == data.positions_um[row], axis=1)]
        if len(donors):
            paired[int(row)] = donors
    assert len(paired) == 209 and sum(map(len, paired.values())) == 212

    checkpoint_info = []
    resolved = {}
    for fold in range(8):
        train = np.flatnonzero(meta.fold.to_numpy() != fold).astype(np.int64)
        train_hash = array_sha(train)
        for seed in SEEDS:
            state, metadata_path = checkpoint_paths(fold, seed)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            assert metadata["seed"] == seed
            assert metadata["train_indices_sha256"] == train_hash
            state_hash = sha(state)
            assert metadata.get("model_state_sha256", metadata.get("state_sha256")) == state_hash
            if "feature_sha256" in metadata:
                assert metadata["feature_sha256"] == array_sha(data.features)
            for path in [state, metadata_path]:
                hashes[str(path.relative_to(ROOT))] = sha(path)
            entry = {"fold": fold, "seed": seed, "acquisition": f"A{fold+1}",
                     "checkpoint": str(state.relative_to(ROOT)), "state_sha256": state_hash,
                     "metadata": str(metadata_path.relative_to(ROOT)), "train_indices_sha256": train_hash}
            checkpoint_info.append(entry)
            resolved[(fold, seed)] = state
    assert len(checkpoint_info) == 24
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    original = raw.loc[raw.model.eq("cnn")].set_index(["seed", "row_index"])
    baselines = []
    baseline_lookup = {}
    reproduction = []
    # Reproduce every saved model before constructing any intervention vector.
    for fold in range(8):
        test = meta.index[meta.fold.eq(fold)].to_numpy()
        for seed in SEEDS:
            model = load_saved(resolved[(fold, seed)])
            prediction = predict_author_cnn(model, data.features[test])
            errors = np.linalg.norm(prediction - data.positions_um[test], axis=1)
            archived = original.loc[[(seed, int(row)) for row in test]]
            coordinate_diff = float(np.max(np.abs(prediction - archived[["pred_x_um", "pred_y_um"]].to_numpy())))
            error_diff = float(np.max(np.abs(errors - archived.euclidean_error_um.to_numpy())))
            reproduction.append({"fold": fold, "seed": seed, "n_test": len(test),
                                 "max_coordinate_difference_um": coordinate_diff, "max_error_difference_um": error_diff})
            if error_diff > TOL_UM:
                failure = {"status": "STOPPED_BASELINE_REPRODUCTION_MISMATCH", "fixed_tolerance_um": TOL_UM,
                           "interventions_performed": False, "models_checked": reproduction, "checkpoint_info": checkpoint_info}
                (WORK / "intervention_cnn_check.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
                raise RuntimeError(f"Baseline mismatch fold={fold}, seed={seed}, max error difference={error_diff}; no interventions performed")
            for row, predicted, error in zip(test, prediction, errors):
                item = {"seed": seed, "fold": fold, "acquisition": f"A{fold+1}", "row_index": int(row),
                        "true_x_um": float(data.positions_um[row, 0]), "true_y_um": float(data.positions_um[row, 1]),
                        "pred_x_um": float(predicted[0]), "pred_y_um": float(predicted[1]),
                        "error_um": float(error), "archived_error_um": float(original.loc[(seed, row), "euclidean_error_um"])}
                baselines.append(item)
                baseline_lookup[(seed, int(row))] = item
            del model
    print("All 24 saved CNN baselines reproduced; starting fixed input interventions.", flush=True)

    outcomes = []
    for fold in range(8):
        test = meta.index[meta.fold.eq(fold)].to_numpy()
        selected = [int(row) for row in test if int(row) in paired]
        if not selected:
            continue
        batch = data.features[test].copy()
        row_positions = {int(row): index for index, row in enumerate(test)}
        for seed in SEEDS:
            model = load_saved(resolved[(fold, seed)])
            state_before = {key: value.clone() for key, value in model.state_dict().items()}
            for row in selected:
                local = row_positions[row]
                baseline = baseline_lookup[(seed, row)]
                for donor in paired[row]:
                    assert meta.loc[donor, "fold"] != fold
                    assert np.array_equal(data.positions_um[row], data.positions_um[donor])
                    for group_name, indices in GROUPS.items():
                        hybrid = batch.copy()
                        hybrid[local, indices] = data.features[donor, indices]
                        kept = np.setdiff1d(np.arange(24), indices)
                        assert np.array_equal(hybrid[local, kept], batch[local, kept])
                        assert np.array_equal(np.delete(hybrid, local, axis=0), np.delete(batch, local, axis=0))
                        full_prediction = predict_author_cnn(model, hybrid)
                        prediction = full_prediction[local]
                        error = float(np.linalg.norm(prediction - data.positions_um[row]))
                        shift = prediction - np.array([baseline["pred_x_um"], baseline["pred_y_um"]])
                        outcomes.append({"seed": seed, "fold": fold, "acquisition": f"A{fold+1}",
                                         "row_index": row, "reference_row_index": int(donor),
                                         "reference_acquisition": "A" + str(int(meta.loc[donor, "fold"]) + 1),
                                         "true_x_um": baseline["true_x_um"], "true_y_um": baseline["true_y_um"],
                                         "replacement_group": group_name,
                                         "replaced_feature_indices": ";".join(map(str, indices)),
                                         "donated_log_feature_values": json.dumps(data.features[donor, indices].tolist()),
                                         "baseline_pred_x_um": baseline["pred_x_um"], "baseline_pred_y_um": baseline["pred_y_um"],
                                         "baseline_error_um": baseline["error_um"],
                                         "intervened_pred_x_um": float(prediction[0]), "intervened_pred_y_um": float(prediction[1]),
                                         "intervened_error_um": error, "error_change_um": error - baseline["error_um"],
                                         "prediction_change_x_um": float(shift[0]), "prediction_change_y_um": float(shift[1]),
                                         "prediction_displacement_um": float(np.linalg.norm(shift))})
            assert all(torch.equal(value, state_before[key]) for key, value in model.state_dict().items())
            del model
        print(f"Completed saved-CNN interventions for A{fold+1}: {len(selected)} observations.", flush=True)
    outcomes = pd.DataFrame(outcomes)
    assert len(outcomes) == 212 * 3 * 5 == 3180
    assert not outcomes.duplicated(["seed", "row_index", "reference_row_index", "replacement_group"]).any()
    assert not outcomes.isna().any().any()
    assert np.isfinite(outcomes.select_dtypes(include=[np.number]).to_numpy()).all()
    metrics = ["baseline_error_um", "intervened_error_um", "error_change_um", "prediction_displacement_um"]
    observation_seed = outcomes.groupby(["acquisition", "row_index", "seed", "replacement_group"])[metrics].mean().reset_index()
    observation = observation_seed.groupby(["acquisition", "row_index", "replacement_group"])[metrics].mean().reset_index()
    assert len(observation_seed) == 209 * 3 * 5 and len(observation) == 209 * 5
    summary_rows = []
    for acquisition in [f"A{i}" for i in range(1, 9)]:
        for group_name in GROUPS:
            values = observation.loc[observation.acquisition.eq(acquisition) & observation.replacement_group.eq(group_name)]
            row = {"scope": "per_acquisition", "acquisition": acquisition, "replacement_group": group_name,
                   "n_observations": len(values), "n_acquisitions": int(bool(len(values))),
                   "weighting": "donors_then_seeds_then_observations", "n_lower_error": int(values.error_change_um.lt(0).sum()),
                   "n_higher_error": int(values.error_change_um.gt(0).sum()), "n_equal_error": int(values.error_change_um.eq(0).sum())}
            row.update({key: float(values[key].mean()) if len(values) else None for key in metrics})
            summary_rows.append(row)
    per_acquisition = pd.DataFrame(summary_rows)
    for group_name in GROUPS:
        observed = observation.loc[observation.replacement_group.eq(group_name)]
        acquisitions = per_acquisition.loc[per_acquisition.replacement_group.eq(group_name) & per_acquisition.n_observations.gt(0)]
        for weighting, frame in [("equal_available_acquisition", acquisitions), ("observation_weighted", observed)]:
            row = {"scope": "pooled", "acquisition": "all_eligible", "replacement_group": group_name,
                   "weighting": weighting, "n_observations": len(observed), "n_acquisitions": len(acquisitions),
                   "n_lower_error": int(observed.error_change_um.lt(0).sum()),
                   "n_higher_error": int(observed.error_change_um.gt(0).sum()), "n_equal_error": int(observed.error_change_um.eq(0).sum())}
            row.update({key: float(frame[key].mean()) for key in metrics})
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    case = outcomes.loc[outcomes.row_index.eq(563)].copy()
    assert len(case) == 2 * 3 * 5 == 30
    frames = {"intervention_cnn_baseline.csv": pd.DataFrame(baselines),
              "intervention_cnn_pairs.csv": outcomes, "intervention_cnn_observation_seed.csv": observation_seed,
              "intervention_cnn_observations.csv": observation, "intervention_cnn_summary.csv": summary,
              "intervention_cnn_case563.csv": case}
    for name, frame in frames.items():
        frame.to_csv(WORK / name, index=False, float_format="%.15g")
    for path, digest in hashes.items():
        assert sha(ROOT / path) == digest
    assert sha(Path(__file__)) == script_sha
    report = {"status": "SAVED_CNN_INTERVENTIONS_COMPLETED", "fixed_plan": __doc__, "script_sha256": script_sha,
              "input_sha256": hashes, "checkpoint_info": checkpoint_info, "all_required_checkpoints_present": True,
              "baseline": {"rows": len(baselines), "tolerance_um": TOL_UM, "fold_seed_checks": reproduction,
                           "max_error_difference_um": max(row["max_error_difference_um"] for row in reproduction),
                           "batch_definition": "Complete held-acquisition rows in original row order, same shape for baseline and each intervention"},
              "counts": {"eligible_observations": 209, "same_coordinate_pairs": 212, "seeds": SEEDS,
                         "groups": list(GROUPS), "intervention_rows": len(outcomes), "case563_rows": len(case),
                         "represented_acquisitions": 5, "all_acquisitions_in_summary": 8},
              "group_indices": {key: values.tolist() for key, values in GROUPS.items()},
              "aggregation_order": ["average all actual same-coordinate training donors within observation and seed",
                                    "average three seeds within observation", "average observations within acquisition",
                                    "equal weight for five represented acquisitions; separate observation-weighted alternative"],
              "summary": records(summary), "case563_results": records(case),
              "boundaries": ["Model-input intervention, not physical causal attribution", "Partially mixed vectors may not correspond to a real spectrum",
                             "All24 uses an actual training input, so its error is not a new held-out generalisation endpoint",
                             "No best-donor selection; worsening interventions retained", "This 209-observation subset does not replace the all-819 primary results",
                             "Original MLP fold states were not all retained; no substitute MLP was trained"],
              "training_performed": False, "optimiser_constructed": False, "weights_unchanged_after_inference": True,
              "all_source_hashes_unchanged": True, "p_values_calculated": False,
              "runtime": {"torch": torch.__version__, "device": "cpu", "intraop_threads": torch.get_num_threads(),
                          "interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
              "outputs": {name: {"rows": len(frame), "sha256": sha(WORK / name)} for name, frame in frames.items()}}
    (WORK / "intervention_cnn_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"baseline_max_error_difference_um": report["baseline"]["max_error_difference_um"],
                      "overall": records(summary.loc[summary.scope.eq("pooled")]),
                      "case563_donor_then_seed_means": records(observation.loc[observation.row_index.eq(563)])}, indent=2))


def features_are_valid(features):
    """Check the expected feature shape and finite values."""
    return features.shape == (819, 24) and np.isfinite(features).all()


if __name__ == "__main__":
    main()
