"""Independently recompute retained coverage, feature, and memory diagnostics.

The script reconstructs spatial-hull classes, repeated-coordinate feature
distances, and process-memory summaries, then writes
``verification_controls.json``. Prediction errors average seeds within each
acquisition row. It imports no diagnostic implementation, fits no model, and
performs no new memory profiling.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
MIB = 1048576


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def turn(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def polygon(points):
    """Return counter-clockwise convex-hull vertices for ``(N,2)`` coordinates."""
    points = sorted(set(map(tuple, points)))
    lower, upper = [], []
    for candidates, chain in ((points, lower), (points[::-1], upper)):
        for point in candidates:
            while len(chain) > 1 and turn(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def classify(points, hull):
    """Classify ``(N,2)`` points as inside, boundary, or outside a convex hull."""
    edge = np.roll(hull, -1, axis=0) - hull
    delta = points[:, None, :] - hull[None, :, :]
    inward = (edge[None, :, 0]*delta[:, :, 1] - edge[None, :, 1]*delta[:, :, 0]) / np.linalg.norm(edge, axis=1)
    minimum = inward.min(axis=1)
    return np.where(minimum < -1e-7, "outside", np.where(np.abs(minimum) <= 1e-7, "boundary", "inside"))


def close(a, b, tolerance=1e-9):
    """Assert elementwise numeric agreement and return the maximum difference."""
    delta = float(np.max(np.abs(np.asarray(a, dtype=float)-np.asarray(b, dtype=float))))
    assert delta <= tolerance, delta
    return delta


def entropy_bits(counts):
    """Return Shannon entropy in bits for nonzero category counts."""
    p = np.asarray(counts, dtype=float)
    p = p[p > 0]/p.sum()
    return float(-np.sum(p*np.log2(p)))


def main():
    """Recompute all diagnostics and write one hash-bound verification JSON file."""
    paths = sorted(set(list(WORK.glob("coverage_*.csv")) + list(WORK.glob("coverage_*.json"))
                       + list(WORK.glob("memory_*.csv")) + list(WORK.glob("memory_*.json"))))
    input_hashes = {p.name: sha(p) for p in paths}
    predicted = pd.read_csv(ROOT / "results/study_l_supplementary/loso_sample_predictions.csv")
    meta = predicted.drop_duplicates("row_index").sort_values("row_index").set_index("row_index")
    assert len(meta) == 819 and meta.index.tolist() == list(range(819))
    assert predicted.groupby(["row_index", "model"]).size().eq(3).all()
    source = ROOT / "study_l/sources/l1_selective"
    original_xy = np.column_stack([np.load(source/"140257c64c_x_pos.npy", allow_pickle=False),
                                   np.load(source/"2a781ba01f_y_pos.npy", allow_pickle=False)]).astype(float)
    close(meta[["true_x_um", "true_y_um"]], original_xy, 0)
    xy = original_xy
    sessions = meta.session.to_numpy()
    errors = predicted.groupby(["row_index", "model"]).euclidean_error_um.mean().unstack("model")
    reconstructed = np.hypot(predicted.pred_x_um-predicted.true_x_um,
                             predicted.pred_y_um-predicted.true_y_um)
    close(reconstructed, predicted.euclidean_error_um)
    exported = pd.read_csv(WORK/"coverage_observations.csv").set_index("row_index").loc[meta.index]
    close(errors.cnn, exported.cnn_error_um)
    close(errors.mlp8, exported.mlp8_error_um)
    classes = np.empty(819, dtype=object)
    overlap = np.zeros(819, dtype=bool)
    nearest = np.empty(819)
    by_acquisition = []
    fold_ids = meta.fold.to_numpy()
    for fold in range(8):
        test, train = np.flatnonzero(fold_ids == fold), np.flatnonzero(fold_ids != fold)
        vertices = polygon(xy[train])
        classes[test] = classify(xy[test], vertices)
        observed = set(map(tuple, xy[train]))
        overlap[test] = [tuple(p) in observed for p in xy[test]]
        nearest[test] = np.sqrt(((xy[test,None,:]-xy[None,train,:])**2).sum(axis=2)).min(axis=1)
        by_acquisition.append({"acquisition": f"A{fold+1}", "test_rows": len(test),
            "inside": int(np.sum(classes[test] == "inside")),
            "boundary": int(np.sum(classes[test] == "boundary")),
            "outside": int(np.sum(classes[test] == "outside")),
            "exact_overlap": int(overlap[test].sum()), "polygon_vertices_um": vertices.tolist()})
    assert classes.tolist() == exported.hull_class.tolist()
    assert overlap.tolist() == exported.exact_train_coordinate_overlap.tolist()
    close(nearest, exported.nearest_train_distance_um, 0)
    assert int(np.sum(classes == "outside")) == 153 and int(overlap.sum()) == 209
    assert not np.any(overlap & (classes == "outside"))
    coverage_json = json.loads((WORK/"coverage_analysis.json").read_text(encoding="utf-8"))
    axis_results = []
    acquisition_entropy = entropy_bits(np.bincount(fold_ids))
    for column, reported in enumerate(coverage_json["axis_acquisition_confounding"]):
        levels = np.unique(xy[:,column])
        conditional, sole_levels, sole_observations = 0.0, 0, 0
        for level in levels:
            part = fold_ids[xy[:,column] == level]
            conditional += len(part)/819*entropy_bits(np.bincount(part,minlength=8))
            if len(np.unique(part)) == 1:
                sole_levels += 1
                sole_observations += len(part)
        explained = 1-conditional/acquisition_entropy
        close(explained,reported["fraction_acquisition_entropy_explained"])
        assert sole_levels == reported["levels_owned_by_one_acquisition"]
        assert sole_observations == reported["observations_on_single_owner_axis_levels"]
        axis_results.append({"axis":"XY"[column],"levels":len(levels),
            "single_owner_levels":sole_levels,"single_owner_observations":sole_observations,
            "fraction_acquisition_entropy_explained":explained})
    a8 = []
    for kind in ("boundary", "outside"):
        ids = np.flatnonzero((fold_ids == 7) & (classes == kind))
        a8.append({"class": kind, "n": len(ids), "stage_X_um": np.unique(xy[ids,0]).tolist(),
                   "cnn_mean_error_um": float(errors.loc[ids,"cnn"].mean()),
                   "mlp8_mean_error_um": float(errors.loc[ids,"mlp8"].mean())})
    parameters = np.load(source/"94e1545b49_para_list.npy", allow_pickle=False)
    uncertainty = np.load(source/"45a5d08325_para_std_list.npy", allow_pickle=False)
    raw = np.stack((np.abs(parameters[:,:,1]-parameters[:,:,4]),
                    uncertainty[:,:,1], uncertainty[:,:,4]), axis=2).reshape(819,24)
    features = np.log(raw)
    distance_table = pd.read_csv(WORK/"coverage_repeat_features_observations.csv").set_index("row_index")
    pair_table = pd.read_csv(WORK/"coverage_repeat_features_pairs.csv").set_index(["test_row","training_row"])
    independently_computed, directed_pairs = [], 0
    case = None
    max_feature_difference = 0.0
    for row in np.flatnonzero(overlap):
        train = np.flatnonzero(fold_ids != fold_ids[row])
        matching = train[np.all(xy[train] == xy[row], axis=1)]
        sd, mean = features[train].std(axis=0), features[train].mean(axis=0)
        assert np.all(sd > 0)
        differences = (features[row]-features[matching])/sd
        norms = np.linalg.norm(differences,axis=1)
        rms = float(np.sqrt(np.mean(differences**2)))
        z = (features[row]-mean)/sd
        outside_features = (features[row]<features[train].min(axis=0)) | (features[row]>features[train].max(axis=0))
        entry = distance_table.loc[row]
        assert len(matching) == int(entry.n_same_coordinate_training_observations)
        assert matching.tolist() == list(map(int,str(entry.training_match_row_indices).split(";")))
        max_feature_difference = max(max_feature_difference,
            close(norms.min(),entry.all24_min_l2), close(rms,entry.all24_per_feature_rms),
            close(np.max(np.abs(z)),entry.max_abs_feature_z))
        assert int(outside_features.sum()) == int(entry.n_features_outside_training_range)
        group_rms = {}
        for offset, group in enumerate(("peak_separation","u1","u2")):
            value = float(np.sqrt(np.mean(differences[:,offset::3]**2)))
            close(value, entry[group+"_per_feature_rms"])
            group_rms[group] = value
        for other, norm in zip(matching,norms):
            close(norm,pair_table.loc[(row,other),"all24_l2"])
            directed_pairs += 1
        independently_computed.append({"row":int(row),"min_l2":float(norms.min()),"rms_all_pairs_features":rms})
        if row == 563:
            case = {"row_index":563,"same_coordinate_training_rows":matching.tolist(),
                "xy_um":xy[row].tolist(),"acquisition":"A6", "minimum_same_coordinate_feature_l2":float(norms.min()),
                "per_feature_rms_over_both_training_references":rms,
                "minimum_pair_l2_divided_by_sqrt24":float(norms.min()/np.sqrt(24)),
                "group_rms_over_both_references":group_rms,
                "outside_feature_indices":np.flatnonzero(outside_features).tolist(),
                "channel8_u1_GHz":float(uncertainty[row,7,1]),
                "channel8_u1_log_z":float(z[22]),
                "peer_channel8_u1_GHz":uncertainty[matching,7,1].tolist(),
                "widths_GHz":parameters[row,:,2].tolist(),
                "channels_at_width_upper_bound_atol1e-10":(np.flatnonzero(np.isclose(parameters[row,:,2],.05,rtol=0,atol=1e-10))+1).tolist(),
                "cnn_error_um":float(errors.loc[row,"cnn"]),"mlp8_error_um":float(errors.loc[row,"mlp8"])}
    assert len(independently_computed) == 209 and directed_pairs == len(pair_table) == 212
    case["descending_min_distance_rank"] = 1+sum(r["min_l2"] > case["minimum_same_coordinate_feature_l2"] for r in independently_computed)
    assert case["descending_min_distance_rank"] == 2
    assert int(errors.cnn.idxmax()) == int(errors.mlp8.idxmax()) == 563
    feature_distances = pd.Series({r["row"]:r["min_l2"] for r in independently_computed})
    feature_associations = {}
    for name, ids in (("all_209",feature_distances.index),
                      ("A6",feature_distances.index[fold_ids[feature_distances.index] == 5])):
        feature_associations[name] = {"observations":len(ids)}
        for model in ("cnn","mlp8"):
            feature_associations[name][model] = float(np.corrcoef(
                feature_distances.loc[ids].rank(), errors.loc[ids,model].rank())[0,1])
    raw_memory = pd.read_csv(WORK/"memory_profiles_raw.csv")
    summary = json.loads((WORK/"memory_profiles_summary.json").read_text(encoding="utf-8"))
    assert len(raw_memory) == 200 and raw_memory.pid.nunique() == 10
    assert raw_memory.groupby(["model","repeat","stage"]).size().eq(5).all()
    assert raw_memory.groupby(["model","repeat"]).pid.nunique().eq(1).all()
    metrics = ["rss_bytes","private_commit_bytes","process_lifetime_peak_rss_bytes",
               "process_lifetime_peak_commit_bytes","rss_increment_from_runtime_baseline_bytes",
               "private_commit_increment_from_runtime_baseline_bytes"]
    memory_results, checked_numbers, changing_peaks = [], 0, []
    for (model,repeat), group in raw_memory.groupby(["model","repeat"]):
        baseline = group[group.stage == "runtime_baseline"]
        close(group.rss_bytes-baseline.rss_bytes.median(),group.rss_increment_from_runtime_baseline_bytes,0)
        close(group.private_commit_bytes-baseline.private_commit_bytes.median(),group.private_commit_increment_from_runtime_baseline_bytes,0)
        for metric in metrics[2:4]:
            sequence = group.sort_values("elapsed_wall_sec")[metric].to_numpy()
            assert np.all(np.diff(sequence) >= 0)
            for stage, phase in group.groupby("stage"):
                if phase[metric].nunique() > 1:
                    changing_peaks.append({"model":model,"repeat":int(repeat),"stage":stage,
                                          "metric":metric,"range_bytes":int(phase[metric].max()-phase[metric].min())})
    for reported in summary["summary"]:
        group = raw_memory[(raw_memory.model == reported["model"]) & (raw_memory.stage == reported["stage"])]
        compact = {"model":reported["model"],"stage":reported["stage"]}
        for metric in metrics:
            values = [float(group[group.repeat == repeat][metric].median()) for repeat in range(1,6)]
            result = {"per_repeat_medians":values,"median":float(np.median(values)),"min":min(values),"max":max(values)}
            for name,value in result.items():
                close(value,reported[metric][name],0)
                scaled = np.asarray(value)/MIB
                close(scaled,reported[metric.replace("_bytes","_MiB")][name],0)
                checked_numbers += int(np.asarray(value).size)*2
            compact[metric.replace("_bytes","_MiB")] = {k:v/MIB for k,v in result.items() if k != "per_repeat_medians"}
        memory_results.append(compact)
    assert np.all(raw_memory.process_lifetime_peak_rss_bytes >= raw_memory.rss_bytes)
    assert np.all(raw_memory.process_lifetime_peak_commit_bytes >= raw_memory.private_commit_bytes)
    raw_json_rows = []
    for repeat in range(1,6):
        for model in ("cnn","mlp8"):
            original = json.loads((WORK/f"memory_raw_{repeat:02}_{model}.json").read_text(encoding="utf-8"))
            assert original["model"] == model and original["repeat"] == repeat
            assert original["runtime"]["torch_threads"] == original["runtime"]["torch_interop_threads"] == 1
            raw_json_rows.extend(original["samples"])
    source_frame = pd.DataFrame(raw_json_rows).sort_values(["model","repeat","stage","sample"])
    exported_frame = raw_memory.sort_values(["model","repeat","stage","sample"])
    close(source_frame[metrics],exported_frame[metrics],0)
    def get(stage, model, metric):
        return next(r for r in memory_results if r["model"] == model and r["stage"] == stage)[metric]["median"]
    peak_rss_ratio = get("post_batch_256","cnn","process_lifetime_peak_rss_MiB")/get("post_batch_256","mlp8","process_lifetime_peak_rss_MiB")
    checked_markdown_cells = 0
    for filename, columns in (("memory_profiles_report.md",["rss_MiB","private_commit_MiB",
             "rss_increment_from_runtime_baseline_MiB","private_commit_increment_from_runtime_baseline_MiB"]),
             ("memory_peak_summary.md",["process_lifetime_peak_rss_MiB","process_lifetime_peak_commit_MiB"])):
        for line in (WORK/filename).read_text(encoding="utf-8").splitlines():
            cells = [v.strip() for v in line.split("|")][1:-1]
            if not cells or cells[0] not in ("cnn","mlp8"):
                continue
            expected_row = next(r for r in memory_results if r["model"] == cells[0] and r["stage"] == cells[1])
            for text,metric in zip(cells[2:],columns):
                numbers = list(map(float,re.findall(r"-?\d+\.\d+",text)))
                close(numbers,[expected_row[metric][k] for k in ("median","min","max")],.000500001)
                checked_markdown_cells += 1
    for p in paths:
        assert sha(p) == input_hashes[p.name], f"Audited input changed during verification: {p.name}"
    report = {"status":"INDEPENDENT_RECOMPUTATION_MATCHES_WITH_SCOPE_LIMITS",
        "performed_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"coverage and memory only; no model training, no fresh memory profiling, no matched-run reading or polling",
        "implementation_independence":"No coverage/memory diagnostic implementation imported; independent monotone-chain hull and raw-table recomputation.",
        "coverage_counts":{"observations":819,"inside":int(np.sum(classes=="inside")),
            "boundary":int(np.sum(classes=="boundary")),"outside":int(np.sum(classes=="outside")),
            "exact_coordinate_overlap":int(overlap.sum()),"directed_same_coordinate_pairs":directed_pairs},
        "acquisitions":by_acquisition,"A8_boundary_outside":a8,"axis_acquisition_association":axis_results,
        "outside_exact_overlap_intersection_n":0,
        "feature_case":case,"feature_distance_error_associations":feature_associations,
        "maximum_feature_table_difference":max_feature_difference,
        "memory":{"processes":10,"steady_rows":200,"summary_values_rechecked":checked_numbers,
            "markdown_numeric_cells_rechecked":checked_markdown_cells,
            "raw_JSON_CSV_byte_metrics_match":True,"recomputed_summaries":memory_results,
            "within_stage_peak_counter_changes":changing_peaks,
            "post_batch256_whole_process_peak_RSS_CNN_over_MLP8":peak_rss_ratio},
        "interpretation_warnings":[
            "Outside-hull (153) and exact-coordinate-overlap (209) are disjoint here but do not form an exhaustive partition of all 819 observations; no acquisition has both strictly inside and outside targets, but A8 has a descriptive boundary/outside contrast.",
            "Case distance 12.45 is a minimum across same-coordinate references; RMS 2.565 averages both references and all features. The minimum-pair RMS is about 2.541, so these two RMS concepts must not be equated.",
            "Centre SE and a fitted width at the original bound are fit-condition clues, not proof of a physical sensor defect, mislabel or causal reason for prediction error.",
            "Memory values belong to the frozen L31 inference implementations, not the new matched training runs; no training-memory result is provided.",
            "Steady RSS increments and whole-process lifetime peaks are different quantities; the latter include runtime imports, model loading and previous stages, and do not represent model-only or isolated per-batch RAM.",
            "Median whole-process peak RSS ratio is about 1.108, not the approximately 670-fold serialised-state ratio. Negative private-commit increments must remain signed.",
            "Repeated-feature diagnostics were recalculated from existing measurements."
        ],
        "protected_input_hashes_unchanged":True,"input_sha256":input_hashes}
    (WORK/"verification_controls.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"coverage":report["coverage_counts"],
        "A8":a8,"case":case,"memory_summary_values_rechecked":checked_numbers,
        "within_stage_peak_counter_changes":changing_peaks,
        "post_batch256_peak_RSS_ratio":peak_rss_ratio},ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()
