"""Check Meneses position labels and evaluation-window targets.

Use the specified source commit. Identify windows whose next-sample targets
match released floor levels. The post-hoc boundary analysis uses saved
predictions to recalculate errors after excluding two targets.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMIT = "7e8a78a33078a1479fe66b9c73bf3c6d7e14ca02"
REPO = "Fertmeneses/ML_QDM_Meneses_et_al"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
LEVELS = np.array([0., 4.1, 7.8, 11.5, 15.2, 18.9, 22.6, 26.3])
OUTPUT = HERE / "meneses_label_audit.csv"
SENSITIVITY_OUTPUT = HERE / "meneses_boundary_sensitivity.csv"
BOUNDARY_IDS = ((2, 1873, 74960), (3, 2215, 88640))
# Git blob identities from the complete inspected tree at the pinned commit.
EXPECTED_BLOBS = {
    "MLQDM/src/MLQDM/timewindows.py": "535ee9a374ee2fb95c4a8156be7129b88bf1e925",
    "MLQDM/src/MLQDM/MLanalysis.py": "5eec3c2de0e2e6e0c92c4cc6af9731dcfe56161d",
    "Codes/ML_General.ipynb": "74ffb1574e0060503e023042c91e6c4884743100",
    "Data/lin_approx_data_zAut_segm1.csv": "3b5d08b1290681584add45d344fc84def6efa349",
    "Data/lin_approx_data_zAut_segm2.csv": "503bcc1b55b160e4ad49481fc70fceed74c44b9e",
    "Data/lin_approx_data_zAut_segm3.csv": "9b94278060356c628120249806c996bbe7b46d93",
    "Data/phys_model_data_zAut_segm1.csv": "74dbb5c12f6e6f728af0e2520db1980ae00b7e2c",
    "Data/phys_model_data_zAut_segm2.csv": "bb08ef130aa5663944abe4f31c81b69d44f70681",
    "Data/phys_model_data_zAut_segm3.csv": "188338af47c0775c5226d75b11b25040a7546b73",
    "Data/phys_model_data_zAut_segm4.csv": "988e27d6daf4970b5a1e18cc21480ba71ed22e43",
    "Data/phys_model_data_zAut_segm5.csv": "75c549efa3ab82f64439f72c3bbb67f719b95aef",
}


def read_url(url):
    """Read one pinned public artifact as bytes with a bounded timeout."""
    request = urllib.request.Request(url, headers={"User-Agent": "NV-label-provenance-audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def blob_sha(content):
    """Compute the Git blob identity for an in-memory byte sequence."""
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()


def sha_file(path):
    """Return the streaming SHA-256 digest of a protected local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_sources():
    """Verify selected public blobs and return source identities and linear targets."""
    paths = {path: {"sha": digest} for path, digest in EXPECTED_BLOBS.items()}
    wanted = ["MLQDM/src/MLQDM/timewindows.py", "MLQDM/src/MLQDM/MLanalysis.py",
              "Codes/ML_General.ipynb"]
    wanted += [f"Data/lin_approx_data_zAut_segm{i}.csv" for i in (1, 2, 3)]
    def fetch(path):
        """Fetch one pinned path and reject any Git-blob mismatch."""
        content = read_url(RAW + path)
        if blob_sha(content) != paths[path]["sha"]:
            raise ValueError(f"Public blob does not match the pinned tree: {path}")
        return path, content
    with ThreadPoolExecutor(max_workers=6) as pool:
        source = dict(pool.map(fetch, wanted))
    tw = source[wanted[0]].decode("utf-8")
    analysis = source[wanted[1]].decode("utf-8")
    if "z = df['zTrue_m'] if self.gr_tr else df['zAut_m']" not in tw:
        raise ValueError("Published target-column selector differs from the inspected source")
    if 'i_park = preds_track[RF]["Truth_m"].apply(lambda x: x in park_lvls)' not in analysis:
        raise ValueError("Published floor-value parking selector differs from the inspected source")
    for path in wanted[:3]:
        print("PUBLIC_CODE", path, "git_blob", paths[path]["sha"])
    print("PUBLIC_TREE", COMMIT, "selected_blob_identities", len(paths),
          "independent_camera_interval_manifest", "not located in the inspected complete tree")
    print("SOURCE_BOUNDARY: code reproduces a floor-value selector; no independent camera parking-interval table was located in the pinned tree.")
    linear = {i: pd.read_csv(io.BytesIO(source[f"Data/lin_approx_data_zAut_segm{i}.csv"]),
                            float_precision="round_trip")
              for i in (1, 2, 3)}
    return paths, linear


def settings_check():
    """Confirm that the executed Meneses design matches the audited constants."""
    tree = ast.parse((HERE / "meneses_external.py").read_text(encoding="utf-8"))
    constants = {}
    for item in tree.body:
        if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name):
            try:
                constants[item.targets[0].id] = ast.literal_eval(item.value)
            except (ValueError, TypeError):
                pass
    expected = {"SOURCE_COMMIT": COMMIT, "TRAIN_SEGMENTS": (1, 4, 5),
                "TEST_SEGMENTS": (2, 3), "WINDOW": 40, "TRAIN_WINDOWS": 819,
                "SEEDS": (20260804, 20260805, 20260806), "SNAPSHOTS": (200, 400)}
    if any(constants.get(key) != value for key, value in expected.items()):
        raise ValueError("Executed Meneses settings differ from the audited configuration")


def main():
    """Reconstruct window targets, compare released label definitions, and export the audit."""
    settings_check()
    protected = [HERE / "meneses_external.py", HERE / "meneses_full.csv",
                 HERE / "meneses_pilot.csv"]
    protected += [HERE / "meneses_data" / f"phys_model_data_zAut_segm{i}.csv" for i in range(1, 6)]
    before = {path: sha_file(path) for path in protected}
    paths, linear = public_sources()
    frames, rows, data_digest = {}, [], hashlib.sha256()
    for segment in range(1, 6):
        name = f"phys_model_data_zAut_segm{segment}.csv"
        content = (HERE / "meneses_data" / name).read_bytes()
        public_path = "Data/" + name
        if blob_sha(content) != paths[public_path]["sha"]:
            raise ValueError(f"Local data differs from pinned public bytes: {name}")
        data_digest.update(name.encode("utf-8")); data_digest.update(content)
        frame = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
        frames[segment] = frame
        starts = np.arange(0, len(frame) - 40, 40, dtype=np.int64)
        target_rows = starts + 40
        col = "zTrue_m" if "zTrue_m" in frame else "zAut_m"
        selected = (np.linspace(0, len(starts) - 1, 273, dtype=np.int64)
                    if segment in (1, 4, 5) else np.arange(len(starts)))
        if segment in linear:
            alternate = linear[segment]
            if len(alternate) != len(frame) or not np.array_equal(alternate.Time_s, frame.Time_s):
                raise ValueError("Alternative interpolation uses a different time grid")
            differences = frame.zTrue_m.to_numpy() != alternate.zTrue_m.to_numpy()
            print("LABEL_COLUMN", segment, "target_column", col,
                  "zTrue_physical_vs_linear_different_rows", int(differences.sum()))
        for window_index in selected:
            target_row = int(target_rows[window_index]); physical = float(frame.iloc[target_row][col])
            alt = float(linear[segment].iloc[target_row][col]) if segment in linear else np.nan
            current_mask = bool(physical in LEVELS) if segment in (2, 3) else None
            alt_mask = bool(alt in LEVELS) if segment in (2, 3) else None
            rows.append({
                "role": "train_selected" if segment in (1, 4, 5) else "test_all",
                "segment": segment, "window_index": int(window_index),
                "input_start_row": int(starts[window_index]),
                "input_end_row": target_row - 1, "target_row": target_row,
                "target_time_s": float(frame.iloc[target_row].Time_s),
                "target_column": col, "target_value_m": physical,
                "retained_floor_value_mask": current_mask,
                "public_linear_target_value_m": alt,
                "public_linear_floor_value_mask": alt_mask,
                "interpolation_masks_agree": current_mask == alt_mask if current_mask is not None else None,
                "independent_parking_interval_verified": None,
                "interval_provenance_status": "independent_camera_interval_not_located",
                "source_commit": COMMIT, "source_csv_git_blob": paths[public_path]["sha"],
            })
        print("WINDOW_PLAN", segment, "rows", len(frame), "all_windows", len(starts),
              "selected_training_windows", len(selected) if segment in (1, 4, 5) else 0,
              "target_column", col)
    audit = pd.DataFrame(rows)
    cols = ["protocol", "mode", "source_commit", "source_sha256", "data_sha256",
            "job_id", "model", "seed", "input_standardised", "target_standardised",
            "epoch", "train_windows", "optimizer_updates", "batch_order_sha256",
            "test_segment", "window_index", "target_time_s", "true_z_m", "parking_ground_truth"]
    saved = pd.read_csv(HERE / "meneses_full.csv", usecols=cols, float_precision="round_trip")
    if len(saved) != 257664 or saved.job_id.nunique() != 24 or set(saved.epoch) != {200, 400}:
        raise ValueError("Saved full experiment fit/snapshot coverage differs")
    if not saved.source_commit.eq(COMMIT).all() or not saved.train_windows.eq(819).all():
        raise ValueError("Saved commit or training-window count differs")
    if not saved.data_sha256.eq(data_digest.hexdigest()).all():
        raise ValueError("Saved data fingerprint does not match the local public bytes")
    source_digest = hashlib.sha256()
    source_digest.update((HERE / "meneses_external.py").read_bytes())
    source_digest.update((ROOT / "src/qml_microrobot/study_l30_author_cnn.py").read_bytes())
    if not saved.source_sha256.eq(source_digest.hexdigest()).all():
        raise ValueError("Saved execution source fingerprint differs")
    if not (saved.optimizer_updates == saved.epoch * 4).all():
        raise ValueError("Recorded optimiser-update count differs from the 819-window batch256 plan")
    if saved.duplicated(["job_id", "epoch", "test_segment", "window_index"]).any():
        raise ValueError("Duplicate saved prediction endpoint")
    if not saved.groupby(["seed", "epoch"]).batch_order_sha256.nunique().eq(1).all():
        raise ValueError("Saved minibatch orders differ across matched conditions")
    test = audit[audit.role == "test_all"]
    reference = test.set_index(["segment", "window_index"])
    checked = saved.join(reference[["target_time_s", "target_value_m", "retained_floor_value_mask"]],
                         on=["test_segment", "window_index"], rsuffix="_source")
    if checked.target_time_s_source.isna().any():
        raise ValueError("Saved window does not belong to the reconstructed test list")
    if not np.array_equal(checked.target_time_s, checked.target_time_s_source):
        raise ValueError("Saved target-time alignment differs")
    if not np.array_equal(checked.true_z_m, checked.target_value_m):
        raise ValueError("Saved target values differ from source next-sample targets")
    if not np.array_equal(checked.parking_ground_truth.astype(bool), checked.retained_floor_value_mask.astype(bool)):
        raise ValueError("Saved masks differ from the original physical-CSV floor-value mask")
    copies = saved.groupby(["test_segment", "window_index"]).size()
    if not copies.eq(48).all() or len(copies) != 5368:
        raise ValueError("Each test endpoint must occur in 24 fits and two snapshots")
    audit["saved_prediction_copies"] = [int(copies.loc[(s, w)]) if role == "test_all" else 0
                                         for s, w, role in zip(audit.segment, audit.window_index, audit.role)]
    audit["saved_target_and_mask_agree"] = [True if role == "test_all" else None for role in audit.role]
    if any(sha_file(path) != value for path, value in before.items()):
        raise RuntimeError("Protected source data or results changed during the audit")
    audit.to_csv(OUTPUT, index=False, float_format="%.17g")
    print("SAVED_RECORDS_PASS", "fits", 24, "snapshots", [200, 400], "training_windows", 819,
          "prediction_rows", len(saved), "audit_rows", len(audit), "data_sha256", data_digest.hexdigest())
    for segment in (2, 3):
        subset = audit[(audit.role == "test_all") & (audit.segment == segment)]
        print("MASK_AUDIT", segment, "test_windows", len(subset),
              "retained_physical_floor_mask", int(subset.retained_floor_value_mask.astype(bool).sum()),
              "alternative_linear_floor_mask", int(subset.public_linear_floor_value_mask.astype(bool).sum()),
              "mask_disagreements", int((~subset.interpolation_masks_agree.astype(bool)).sum()))
        print(subset[~subset.interpolation_masks_agree.astype(bool)][[
            "window_index", "target_row", "target_time_s", "target_value_m",
            "public_linear_target_value_m", "retained_floor_value_mask",
            "public_linear_floor_value_mask"]].to_string(index=False))
    print("SOURCE_INTERVAL_VERIFICATION_UNRESOLVED; masks and experimental results were preserved byte-for-byte.")
    print("AUDIT_CSV", OUTPUT)


def boundary_sensitivity():
    """Aggregate saved predictions after excluding two fixed boundary targets.

    The diagnostic returns segment-level and equal-segment CNN/MLP8 MAE and
    MLP8-minus-CNN gaps in metres for four preprocessing conditions. Seeds are
    averaged before segments. Protected data, masks, and prediction files remain
    byte-identical.
    """
    settings_check()
    prediction_path = HERE / "meneses_full.csv"
    audit_path = OUTPUT
    raw_paths = [HERE / "meneses_data" / f"phys_model_data_zAut_segm{i}.csv"
                 for i in range(1, 6)]
    execution_paths = [HERE / "meneses_external.py",
                       ROOT / "src/qml_microrobot/study_l30_author_cnn.py"]
    protected = [prediction_path, audit_path, HERE / "meneses_pilot.csv"] + raw_paths + execution_paths
    before = {path: sha_file(path) for path in protected}
    data_digest = hashlib.sha256()
    for path in raw_paths:
        content = path.read_bytes()
        if blob_sha(content) != EXPECTED_BLOBS["Data/" + path.name]:
            raise ValueError("A protected source CSV differs from its pinned public identity")
        data_digest.update(path.name.encode("utf-8")); data_digest.update(content)
    execution_digest = hashlib.sha256()
    for path in execution_paths:
        execution_digest.update(path.read_bytes())
    audit = pd.read_csv(audit_path, float_precision="round_trip")
    reference = audit[audit.role == "test_all"].set_index(["segment", "window_index"])
    if len(reference) != 5368 or not reference.index.is_unique:
        raise ValueError("The preserved audit test-window inventory differs")
    for segment, window, target_row in BOUNDARY_IDS:
        point = reference.loc[(segment, window)]
        if (int(point.target_row) != target_row
                or not bool(point.retained_floor_value_mask)
                or bool(point.public_linear_floor_value_mask)):
            raise ValueError("The prespecified boundary exclusion does not match the audit")
    columns = ["protocol", "source_commit", "source_sha256", "data_sha256", "model", "seed",
               "input_standardised", "target_standardised", "epoch", "test_segment", "window_index",
               "target_time_s", "parking_ground_truth", "abs_error_m", "true_z_m", "pred_z_m"]
    saved = pd.read_csv(prediction_path, usecols=columns, float_precision="round_trip")
    if (len(saved) != 257664 or not saved.source_commit.eq(COMMIT).all()
            or not saved.data_sha256.eq(data_digest.hexdigest()).all()
            or not saved.source_sha256.eq(execution_digest.hexdigest()).all()):
        raise ValueError("Saved predictions differ from the protected data/execution identities")
    data = saved[(saved.epoch == 400) & (saved.parking_ground_truth == 1)].copy()
    keys = ["input_standardised", "target_standardised", "model", "seed", "test_segment"]
    if (set(data.seed) != {20260804, 20260805, 20260806}
            or set(data.model) != {"cnn", "mlp8"}
            or len(data.groupby(keys)) != 48
            or data.duplicated(keys + ["window_index"]).any()):
        raise ValueError("The four-condition, two-model, three-seed pairing differs")
    mapped = data.join(reference[["target_time_s", "target_value_m", "retained_floor_value_mask"]],
                       on=["test_segment", "window_index"], rsuffix="_audit")
    if (mapped.target_time_s_audit.isna().any()
            or not np.array_equal(mapped.target_time_s, mapped.target_time_s_audit)
            or not np.array_equal(mapped.true_z_m, mapped.target_value_m)
            or not mapped.retained_floor_value_mask.astype(bool).all()
            or not np.allclose(data.abs_error_m, np.abs(data.pred_z_m-data.true_z_m),
                               atol=1e-9, rtol=0)):
        raise ValueError("Saved epoch-400 primary targets do not match the preserved audit")
    excluded = np.zeros(len(data), dtype=bool)
    for segment, window, _ in BOUNDARY_IDS:
        selected = (data.test_segment.to_numpy() == segment) & (data.window_index.to_numpy() == window)
        if selected.sum() != 24:
            raise ValueError("A boundary target is missing from a paired model/condition/seed")
        excluded |= selected
    reduced = data.loc[~excluded]
    for segment, original_count, reduced_count in ((2, 2144, 2143), (3, 2305, 2304)):
        if (not data[data.test_segment == segment].groupby(keys).size().eq(original_count).all()
                or not reduced[reduced.test_segment == segment].groupby(keys).size().eq(reduced_count).all()):
            raise ValueError("Segment-specific original or diagnostic target counts differ")
    seed_original = data.groupby(keys).abs_error_m.mean()
    seed_reduced = reduced.groupby(keys).abs_error_m.mean()
    segment_keys = ["input_standardised", "target_standardised", "model", "test_segment"]
    original = seed_original.groupby(segment_keys).mean()
    diagnostic = seed_reduced.groupby(segment_keys).mean()
    metadata = {
        "analysis_id": "MENESES_POSTHOC_TWO_BOUNDARY_TARGET_SENSITIVITY_V1",
        "epoch": 400, "seeds": "20260804;20260805;20260806", "n_seeds": 3,
        "unit": "m", "gap_direction": "small_mlp_minus_cnn", "source_commit": COMMIT,
        "prediction_file": prediction_path.name, "prediction_sha256": before[prediction_path],
        "audit_file": audit_path.name, "audit_sha256": before[audit_path],
        "execution_source_sha256": execution_digest.hexdigest(), "data_sha256": data_digest.hexdigest(),
        "analysis_script_sha256": sha_file(Path(__file__)),
        "raw_csv_sha256": json.dumps({path.name: before[path] for path in raw_paths}, sort_keys=True),
        "label_scope": "released_floor_value_subset;independent_camera_certification_unresolved",
    }
    rows = []
    for input_std, target_std in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for scope, segments in (("segment", (2,)), ("segment", (3,)), ("equal_segment", (2, 3))):
            values = {}
            for model in ("cnn", "mlp8"):
                values["original_" + model + "_mae_m"] = float(np.mean([
                    original.loc[(input_std, target_std, model, segment)] for segment in segments]))
                values["excluded_" + model + "_mae_m"] = float(np.mean([
                    diagnostic.loc[(input_std, target_std, model, segment)] for segment in segments]))
                values["delta_" + model + "_mae_m"] = (
                    values["excluded_" + model + "_mae_m"] - values["original_" + model + "_mae_m"])
            values["original_gap_m"] = values["original_mlp8_mae_m"] - values["original_cnn_mae_m"]
            values["excluded_gap_m"] = values["excluded_mlp8_mae_m"] - values["excluded_cnn_mae_m"]
            values["delta_gap_m"] = values["excluded_gap_m"] - values["original_gap_m"]
            original_counts = {segment: 2144 if segment == 2 else 2305 for segment in segments}
            excluded_counts = {segment: count-1 for segment, count in original_counts.items()}
            rows.append({**metadata, "input_standardised": input_std, "target_standardised": target_std,
                         "aggregation_scope": scope, "segments": ";".join(map(str, segments)),
                         "original_targets_n": sum(original_counts.values()),
                         "excluded_targets_n": sum(excluded_counts.values()),
                         "original_counts_by_segment": ";".join(f"{s}:{n}" for s, n in original_counts.items()),
                         "excluded_counts_by_segment": ";".join(f"{s}:{n}" for s, n in excluded_counts.items()),
                         "excluded_target_ids": ";".join(f"segment{s}:window{w}:targetrow{r}"
                                                         for s, w, r in BOUNDARY_IDS if s in segments),
                         **values, "architecture_rank_changed": bool(
                             np.sign(values["original_gap_m"]) != np.sign(values["excluded_gap_m"]))})
    result = pd.DataFrame(rows)
    # Verify seed-first segment means with an independent sum/count calculation.
    for frame, expected in ((data, original), (reduced, diagnostic)):
        direct = frame.groupby(segment_keys).abs_error_m.agg(["sum", "count"])
        if not np.allclose(direct["sum"] / direct["count"], expected, atol=1e-12, rtol=0):
            raise ValueError("Independent sum/count check differs from seed-first aggregation")
    if any(sha_file(path) != digest for path, digest in before.items()):
        raise RuntimeError("A protected CSV, audit window list or execution source changed")
    result.to_csv(SENSITIVITY_OUTPUT, index=False, float_format="%.17g")
    reread = pd.read_csv(SENSITIVITY_OUTPUT, float_precision="round_trip")
    numeric = list(values)
    if len(reread) != 12 or not np.array_equal(result[numeric].to_numpy(), reread[numeric].to_numpy()):
        raise ValueError("Diagnostic CSV values did not round-trip exactly")
    print("BOUNDARY_SENSITIVITY_PASS: 12 rows, original mask/predictions/audit preserved; no training or online access")
    print(result[result.aggregation_scope == "equal_segment"][["input_standardised", "target_standardised"]
          + numeric].to_string(index=False, float_format=lambda x: f"{x:.12f}"))
    print("MAX_OVERALL_MODEL_CHANGE_M", float(result[result.aggregation_scope == "equal_segment"][[
        "delta_cnn_mae_m", "delta_mlp8_mae_m"]].abs().to_numpy().max()))
    print("MAX_OVERALL_GAP_CHANGE_M", float(result[result.aggregation_scope == "equal_segment"].delta_gap_m.abs().max()))
    print("ARCHITECTURE_RANK_CHANGES", int(result.architecture_rank_changed.sum()))
    print("CSV", SENSITIVITY_OUTPUT.name, "schema", ",".join(result.columns))
    print("PROTECTED_HASHES_UNCHANGED", all(sha_file(path) == digest for path, digest in before.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-sensitivity", action="store_true",
                        help="Only aggregate saved predictions; preserve the existing provenance audit")
    arguments = parser.parse_args()
    boundary_sensitivity() if arguments.boundary_sensitivity else main()
