"""Fixed post-hoc diagnostic plan for all repeated-coordinate observations.

This plan is written before inspecting the new feature-difference results.
No localisation model, scaler object or spectrum is fitted. Only existing loaders,
released fitted parameters and locally cached original spectra are read.

For every one of the 209 LOAO observations with an exact training-coordinate
match, obtain the log-transformed 24D feature vector from the existing loader.
For that observation's fold, compute mean and population SD (ddof=0) using all
observations in the other seven acquisitions, then form z=(feature-mean)/SD.
Constant training features are reported as an error rather than silently scaled.
For every same-coordinate training observation j, define d_ij=||z_i-z_j||_2.
Report min_j(d_ij), mean_j(d_ij), sqrt(mean_j(d_ij^2)) (RMS L2), and
sqrt(mean_{j,k}((z_ik-z_jk)^2)) (per-feature RMS). Report the same quantities
within each of the three interleaved eight-channel feature groups: peak
separation, centre standard error u1, and centre standard error u2.

Every repeated-coordinate observation and every corresponding training pair is
retained. Report the count of features outside fold-training min/max ranges and
the maximum absolute standardised feature value. Error associations are
descriptive pooled/within-acquisition rank correlations without p values.

Row 563 is a previously identified large-error case, not a newly selected threshold.
Trace its member path, source row order, original para/para_std values and all
same-coordinate observations. In its fixed fold-training scale, report all 24
feature values, training min/max, standardised values and empirical percentiles.
If its original spectrum is already local, verify its inventory hash and compare
the measured spectrum with the archived Gaussian coefficients without refitting.
Use the source pipeline's APD ordering and three-bin smoothing. Report every
channel, fixed-fit residual RMSE and range-normalised RMSE; neither delete nor
repair any record. Same-coordinate spectra absent locally remain unavailable.
No remote archive download is performed. A numerical anomaly is not causal proof
or proof of an incorrect label; member-order checks have their stated limits.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qml_microrobot.study_l_nv import _inventory_member, author_features
from qml_microrobot.study_l_session import load_l1_session_data, RAW_PATTERN

PREDICTIONS = ROOT / "results/study_l_supplementary/loso_sample_predictions.csv"
RAW_INVENTORY = ROOT / "results/study_l1_sparse16_raw_efficiency_inventory.csv"
CASE_ROW = 563
GROUPS = {"peak_separation": np.arange(0, 24, 3), "u1": np.arange(1, 24, 3), "u2": np.arange(2, 24, 3)}
PARAMETER_NAMES = ["amplitude1", "mu1_GHz", "shared_sigma_GHz", "amplitude2", "mu2_GHz", "constant"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def distance_summary(diff):
    l2 = np.sqrt(np.sum(diff * diff, axis=1))
    return {"min_l2": float(l2.min()), "mean_l2": float(l2.mean()),
            "rms_l2": float(np.sqrt(np.mean(l2 * l2))),
            "per_feature_rms": float(np.sqrt(np.mean(diff * diff)))}


def rank_correlation(frame, x, y):
    if frame[x].nunique() < 2 or frame[y].nunique() < 2:
        return None
    return float(frame[x].rank().corr(frame[y].rank()))


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    files = [PREDICTIONS, RAW_INVENTORY, ROOT / "results/study_l_l1_selective_inventory.csv",
             ROOT / "results/study_l1_archive_member_structure.csv",
             ROOT / "src/qml_microrobot/study_l_nv.py", ROOT / "src/qml_microrobot/study_l_session.py",
             ROOT / "scripts/run_study_l1_sparse16_raw_end_to_end_efficiency.py"]
    parameter_path = _inventory_member(ROOT, "para_list.npy")
    std_path = _inventory_member(ROOT, "para_std_list.npy")
    files += [parameter_path, std_path, _inventory_member(ROOT, "x_pos.npy"), _inventory_member(ROOT, "y_pos.npy")]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    script_sha = sha(Path(__file__))
    data = load_l1_session_data(ROOT)
    parameters = np.load(parameter_path, allow_pickle=False)
    standard = np.load(std_path, allow_pickle=False)
    features = np.asarray(data.features, dtype=float)
    assert features.shape == (819, 24) and parameters.shape == standard.shape == (819, 8, 6)
    assert np.isfinite(features).all()
    assert np.array_equal(features, author_features(parameters, standard))
    raw_features = np.stack([np.abs(parameters[:, :, 1] - parameters[:, :, 4]), standard[:, :, 1], standard[:, :, 4]], axis=2).reshape(819, 24)
    assert np.array_equal(features, np.log(raw_features))
    predictions = pd.read_csv(PREDICTIONS)
    assert predictions.shape == (4914, 12) and not predictions.isna().any().any()
    assert not predictions.duplicated(["row_index", "seed", "model"]).any()
    assert (predictions.groupby(["row_index", "model"]).seed.nunique() == 3).all()
    meta = predictions.drop_duplicates("row_index").set_index("row_index").sort_index()
    assert np.array_equal(meta.index, np.arange(819))
    assert np.array_equal(meta[["true_x_um", "true_y_um"]].to_numpy(), data.positions_um)
    assert np.array_equal(meta.session.to_numpy(), data.session_ids)
    errors = predictions.groupby(["row_index", "model"]).euclidean_error_um.mean().unstack("model")
    meta["acquisition"] = meta.fold.map(lambda v: "A" + str(int(v) + 1))
    member_checks = []
    for row, member in enumerate(data.raw_members):
        match = RAW_PATTERN.search(str(member))
        assert match
        matched = (float(match["x"]) == data.positions_um[row, 0] and
                   float(match["y"]) == data.positions_um[row, 1] and match["session"] == data.session_ids[row])
        assert matched
        member_checks.append(matched)
    fold_stats = {}
    for acquisition, group in meta.groupby("acquisition", sort=True):
        train = meta.index[meta.acquisition.ne(acquisition)].to_numpy()
        values = features[train]
        mu, sd = values.mean(axis=0), values.std(axis=0, ddof=0)
        assert np.all(sd > 0), (acquisition, np.flatnonzero(sd <= 0).tolist())
        fold_stats[acquisition] = {"train_rows": train, "mean": mu, "sd": sd,
                                   "min": values.min(axis=0), "max": values.max(axis=0)}

    summaries, pairs, matching_by_row = [], [], {}
    for row, info in meta.iterrows():
        stats = fold_stats[info.acquisition]
        train = stats["train_rows"]
        matching = train[np.all(data.positions_um[train] == data.positions_um[row], axis=1)]
        if not len(matching):
            continue
        matching_by_row[int(row)] = matching.tolist()
        diff = (features[row] - features[matching]) / stats["sd"]
        standardised = (features[row] - stats["mean"]) / stats["sd"]
        out_of_range = (features[row] < stats["min"]) | (features[row] > stats["max"])
        entry = {"row_index": int(row), "acquisition": info.acquisition, "session": info.session,
                 "true_x_um": float(info.true_x_um), "true_y_um": float(info.true_y_um),
                 "n_same_coordinate_training_observations": len(matching),
                 "training_match_row_indices": ";".join(map(str, matching)),
                 "raw_member": str(data.raw_members[row]),
                 "cnn_mean_error_um": float(errors.loc[row, "cnn"]), "mlp8_mean_error_um": float(errors.loc[row, "mlp8"]),
                 "mlp8_minus_cnn_error_um": float(errors.loc[row, "mlp8"] - errors.loc[row, "cnn"]),
                 "n_features_outside_training_range": int(out_of_range.sum()),
                 "max_abs_feature_z": float(np.abs(standardised).max())}
        entry.update({"all24_" + key: value for key, value in distance_summary(diff).items()})
        for name, indices in GROUPS.items():
            entry.update({name + "_" + key: value for key, value in distance_summary(diff[:, indices]).items()})
            entry[name + "_n_features_outside_training_range"] = int(out_of_range[indices].sum())
        summaries.append(entry)
        for index, train_row in enumerate(matching):
            pair = {"test_row": int(row), "test_acquisition": info.acquisition,
                    "training_row": int(train_row), "training_acquisition": meta.loc[train_row, "acquisition"],
                    "true_x_um": float(info.true_x_um), "true_y_um": float(info.true_y_um),
                    "all24_l2": float(np.linalg.norm(diff[index])),
                    "all24_per_feature_rms": float(np.sqrt(np.mean(diff[index] ** 2)))}
            for name, indices in GROUPS.items():
                pair[name + "_l2"] = float(np.linalg.norm(diff[index, indices]))
                pair[name + "_per_feature_rms"] = float(np.sqrt(np.mean(diff[index, indices] ** 2)))
            pairs.append(pair)
    summary = pd.DataFrame(summaries)
    assert len(summary) == 209 and summary.row_index.nunique() == 209
    assert not summary.isna().any().any()
    assert len(pairs) == 212
    assert np.allclose(summary.all24_rms_l2 / np.sqrt(24), summary.all24_per_feature_rms, atol=1e-12)
    assert np.allclose(summary.all24_rms_l2 ** 2,
                       summary.peak_separation_rms_l2 ** 2 + summary.u1_rms_l2 ** 2 + summary.u2_rms_l2 ** 2, atol=1e-10)
    case_stats = fold_stats[meta.loc[CASE_ROW, "acquisition"]]
    case_members = [CASE_ROW] + matching_by_row[CASE_ROW]
    case_features, case_parameters, case_provenance = [], [], []
    raw_inventory = pd.read_csv(RAW_INVENTORY)
    raw_by_member = {row.raw_member: row for row in raw_inventory.itertuples()}
    raw_spectrum_rows, raw_fit_diagnostics = [], []
    source_freq = np.linspace(2.77, 2.97, 51)
    fit_freq = np.array([np.mean(source_freq[i:i+3]) for i in range(48)])
    source_channels = np.array([15, 6, 11, 4, 13, 7, 3, 1]) - 1
    for row in case_members:
        provenance = {"row_index": int(row), "acquisition": meta.loc[row, "acquisition"],
                      "session": str(data.session_ids[row]), "raw_member": str(data.raw_members[row]),
                      "xy_um": data.positions_um[row].tolist(), "member_coordinates_match_labels": True,
                      "cnn_mean_error_um": float(errors.loc[row, "cnn"]), "mlp8_mean_error_um": float(errors.loc[row, "mlp8"])}
        for k in range(24):
            train_values = features[case_stats["train_rows"], k]
            case_features.append({"row_index": int(row), "acquisition": meta.loc[row, "acquisition"],
                                  "feature_index": k, "channel": k // 3 + 1, "feature_group": list(GROUPS)[k % 3],
                                  "positive_prelog_value": float(raw_features[row, k]), "log_feature_value": float(features[row, k]),
                                  "reference_fold": "A6", "train_mean_log": float(case_stats["mean"][k]),
                                  "train_sd_log_ddof0": float(case_stats["sd"][k]),
                                  "train_min_log": float(case_stats["min"][k]), "train_max_log": float(case_stats["max"][k]),
                                  "z_in_A6_training_scale": float((features[row, k] - case_stats["mean"][k]) / case_stats["sd"][k]),
                                  "train_empirical_percentile_leq": float(100 * np.mean(train_values <= features[row, k])),
                                  "outside_A6_training_range": bool(features[row, k] < train_values.min() or features[row, k] > train_values.max())})
        for channel in range(8):
            record = {"row_index": int(row), "acquisition": meta.loc[row, "acquisition"], "channel": channel + 1}
            for index, name in enumerate(PARAMETER_NAMES):
                record[name] = float(parameters[row, channel, index])
                record[name + "_estimated_std"] = float(standard[row, channel, index])
            record["peak_separation_GHz"] = float(raw_features[row, 3 * channel])
            record["u1_over_separation"] = float(raw_features[row, 3 * channel + 1] / raw_features[row, 3 * channel])
            record["u2_over_separation"] = float(raw_features[row, 3 * channel + 2] / raw_features[row, 3 * channel])
            case_parameters.append(record)
        inventory_row = raw_by_member.get(str(data.raw_members[row]))
        if inventory_row is None or not (ROOT / inventory_row.local_file).is_file():
            provenance["local_raw_spectrum"] = "unavailable in the retained local raw inventory; fitted features only"
        else:
            raw_path = ROOT / inventory_row.local_file
            assert int(inventory_row.row_index) == row
            assert float(inventory_row.x_um) == data.positions_um[row, 0] and float(inventory_row.y_um) == data.positions_um[row, 1]
            raw_hash = sha(raw_path)
            assert raw_hash == inventory_row.sha256 and raw_path.stat().st_size == int(inventory_row.bytes)
            hashes[str(raw_path.relative_to(ROOT))] = raw_hash
            counts = np.load(raw_path, allow_pickle=False, mmap_mode="r")
            assert counts.ndim == 3 and counts.shape[1:] == (16, 51)
            assert np.isfinite(counts).all()
            averaged = np.mean(counts, axis=0)
            smoothed = np.array([averaged[:, i:i+3].mean(axis=1) for i in range(48)]).T
            provenance["local_raw_spectrum"] = {"path": str(raw_path.relative_to(ROOT)), "sha256": raw_hash,
                                                "shape": list(counts.shape), "dtype": str(counts.dtype),
                                                "inventory_hash_and_member_match": True,
                                                "new_spectrum_fit_performed": False}
            for channel, physical_channel in enumerate(source_channels):
                amplitude1, mu1, sigma, amplitude2, mu2, constant = parameters[row, channel]
                predicted = (-amplitude1 * np.exp(-((fit_freq - mu1) ** 2) / (2 * sigma ** 2))
                             - amplitude2 * np.exp(-((fit_freq - mu2) ** 2) / (2 * sigma ** 2)) + constant)
                observed = smoothed[physical_channel]
                residual = observed - predicted
                span = float(np.ptp(observed))
                raw_fit_diagnostics.append({"row_index": int(row), "channel": channel + 1, "physical_APD_channel": int(physical_channel) + 1,
                                            "observed_spectrum_range_counts": span, "fixed_fit_rmse_counts": float(np.sqrt(np.mean(residual ** 2))),
                                            "rmse_divided_by_observed_range": float(np.sqrt(np.mean(residual ** 2)) / span) if span > 0 else None})
                for frequency, observed_value, fixed_value, residual_value in zip(fit_freq, observed, predicted, residual):
                    raw_spectrum_rows.append({"row_index": int(row), "channel": channel + 1,
                                              "physical_APD_channel": int(physical_channel) + 1,
                                              "frequency_GHz": float(frequency), "smoothed_mean_counts": float(observed_value),
                                              "archived_gaussian_counts": float(fixed_value), "residual_counts": float(residual_value)})
        case_provenance.append(provenance)

    associations = []
    for acquisition, group in [("pooled", summary)] + list(summary.groupby("acquisition", sort=True)):
        record = {"acquisition": acquisition, "n_observations": len(group)}
        for x in ["all24_min_l2", "all24_per_feature_rms", "peak_separation_per_feature_rms", "u1_per_feature_rms", "u2_per_feature_rms"]:
            for y in ["cnn_mean_error_um", "mlp8_mean_error_um", "mlp8_minus_cnn_error_um"]:
                record[x + "_spearman_" + y] = rank_correlation(group, x, y)
        associations.append(record)
    case_row = summary.loc[summary.row_index.eq(CASE_ROW)].iloc[0]
    case_ranks = {column: {"value": float(case_row[column]), "n_larger_among_209": int((summary[column] > case_row[column]).sum()),
                           "n_equal_among_209": int((summary[column] == case_row[column]).sum())}
                  for column in ["all24_min_l2", "all24_per_feature_rms", "peak_separation_per_feature_rms", "u1_per_feature_rms", "u2_per_feature_rms", "n_features_outside_training_range", "max_abs_feature_z"]}
    output_frames = {"coverage_repeat_features_observations.csv": summary,
                     "coverage_repeat_features_pairs.csv": pd.DataFrame(pairs),
                     "coverage_repeat_features_case_features.csv": pd.DataFrame(case_features),
                     "coverage_repeat_features_case_parameters.csv": pd.DataFrame(case_parameters),
                     "coverage_repeat_features_case_spectrum.csv": pd.DataFrame(raw_spectrum_rows)}
    for name, frame in output_frames.items():
        frame.to_csv(WORK / name, index=False, float_format="%.15g")
    report = {"diagnostic_role": "Post-hoc repeated-coordinate feature comparison; no model or spectral fitting, no independent-sample p values.",
              "fixed_plan": __doc__, "script_sha256": script_sha, "input_sha256": hashes,
              "counts": {"all_observations_checked": 819, "repeated_coordinate_observations": len(summary),
                         "directed_same_coordinate_training_pairs": len(pairs), "source_member_matches": sum(member_checks),
                         "case_compared_rows": case_members},
              "feature_definition": "Natural logs of [absolute resonance separation, centre SE u1, centre SE u2], interleaved across eight channels.",
              "distance_definitions": {"pair_l2": "sqrt(sum_k((feature_i[k]-feature_j[k])/training_sd[k])^2)",
                                       "min_l2": "minimum pair L2 among same-coordinate training observations",
                                       "mean_l2": "arithmetic mean of pair L2 values",
                                       "rms_l2": "sqrt(mean_j(pair_L2_ij^2))",
                                       "per_feature_rms": "sqrt(mean_(j,k)(standardised_feature_difference_ijk^2))"},
              "fold_training_feature_statistics": {key: {name: value for name, value in values.items() if name != "train_rows"} for key, values in fold_stats.items()},
              "case": {"row_index": CASE_ROW, "fold": "A6", "ranking_among_all_repeated_observations": case_ranks,
                       "provenance": case_provenance, "raw_fit_diagnostics": raw_fit_diagnostics,
                       "index_alignment_result": "All 819 archive-member coordinates and acquisition IDs agree with source positions and prediction metadata. This does not independently validate the physical stage reading.",
                       "uncertainty_interpretation": "u1 and u2 are estimated resonance-centre standard errors from the original covariance diagonal, not measured sensor noise or resonance widths."},
              "descriptive_feature_error_rank_associations": associations,
              "training_performed": False, "spectrum_refitting_performed": False, "remote_downloads": 0,
              "outputs": {name: {"rows": len(frame), "sha256": sha(WORK / name)} for name, frame in output_frames.items()}}
    assert sha(Path(__file__)) == script_sha
    for path, digest in hashes.items():
        assert sha(ROOT / path) == digest
    report["all_input_hashes_unchanged"] = True
    (WORK / "coverage_repeat_features.json").write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(clean({"counts": report["counts"], "case_ranks": case_ranks, "case_provenance": case_provenance,
                            "raw_fit_diagnostics": raw_fit_diagnostics,
                            "case_observation_summary": case_row.to_dict()}), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
