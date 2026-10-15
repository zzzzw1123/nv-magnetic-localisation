"""Fixed plan for a post-hoc spatial-coverage diagnostic, recorded before execution.

Only loso_sample_predictions.csv is read. No estimator is fitted and no primary
result, split or protocol is changed. The diagnostic is not new validation.

1. Reconstruct each of the 819 observations once by row_index. Verify invariant
   true x/y, acquisition and fold across three seeds and both nonlinear models.
   Average the three Euclidean errors within an observation for each model.
2. For every held-out acquisition, use all observations from the other seven
   acquisitions as the geometric training support. Report train/test x/y ranges,
   unique-coordinate counts and exact target-coordinate overlap.
3. Construct the actual two-dimensional convex hull of unique training targets.
   Classify every test target by signed distance to its hull half-spaces: outside
   if the maximum distance exceeds 1e-7 micrometres, boundary if its absolute
   value is at most that tolerance, otherwise inside. This tolerance is purely
   numerical, fixed before execution. Bounding rectangles are checked separately
   and are never substituted for the convex hull.
4. Compute Euclidean distance to the nearest observed training target with a
   cKDTree. This is position-space support, not feature-space or physical noise.
5. Report every acquisition and every hull category (inside/boundary/outside),
   including empty categories. Also report exact-overlap and no-overlap strata.
   Report pooled observation-weighted and available-acquisition-weighted means
   separately. Do not treat 819 observations or seed repeats as independent trials.
6. Quantify acquisition/position confounding with complete acquisition-by-X and
   acquisition-by-Y count tables, coordinate-level ownership, conditional entropy
   H(acquisition|axis), and I(acquisition;axis)/H(acquisition). These are empirical
   descriptions, not model performance. Report within-acquisition inside/outside
   contrasts only where both are observed; list all unidentifiable folds.
7. Report descriptive Spearman associations between nearest-position distance
   and errors/differences, both pooled and within each acquisition, without p
   values. Flag constant-distance folds as unavailable. No selected regions,
   outcome-dependent thresholds, interpolation, smoothing or model retraining.

The output distinguishes seed-average errors from errors of an ensemble formed
by averaging predictions. Spatial and acquisition effects may remain confounded.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree


WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[2]
SOURCE = ROOT / "results/study_l_supplementary/loso_sample_predictions.csv"
SEEDS = [20260804, 20260805, 20260806]
TOL_UM = 1e-7
HULL_CLASSES = ["inside", "boundary", "outside"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def entropy(counts):
    values = np.asarray(counts, dtype=float)
    values = values[values > 0]
    if not len(values):
        return 0.0
    probabilities = values / values.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def describe(group):
    entry = {
        "n_observations": len(group),
        "n_coordinates": len(group[["true_x_um", "true_y_um"]].drop_duplicates()),
        "n_acquisitions": int(group.acquisition.nunique()),
    }
    for column in ["cnn_error_um", "mlp8_error_um", "mlp8_minus_cnn_error_um", "nearest_train_distance_um"]:
        values = group[column]
        for label, result in [
            ("mean", values.mean()), ("median", values.median()),
            ("sd", values.std(ddof=1)), ("min", values.min()), ("max", values.max()),
        ]:
            entry[column.replace("_um", "_" + label + "_um")] = float(result) if pd.notna(result) else None
    return entry


def axis_confounding(observations, axis):
    table = pd.crosstab(observations[axis], observations.acquisition).reindex(columns=[f"A{i}" for i in range(1, 9)], fill_value=0)
    total = table.to_numpy().sum()
    row_counts = table.sum(axis=1)
    h_acquisition = entropy(table.sum(axis=0))
    h_conditional = float(sum(float(row_counts.loc[value]) / total * entropy(row) for value, row in table.iterrows()))
    single_owner = (table > 0).sum(axis=1).eq(1)
    summary = {
        "axis": axis, "axis_levels": len(table),
        "levels_owned_by_one_acquisition": int(single_owner.sum()),
        "observations_on_single_owner_axis_levels": int(row_counts[single_owner].sum()),
        "fraction_observations_on_single_owner_axis_levels": float(row_counts[single_owner].sum() / total),
        "acquisition_entropy_bits": h_acquisition,
        "conditional_acquisition_entropy_bits": h_conditional,
        "fraction_acquisition_entropy_explained": (h_acquisition - h_conditional) / h_acquisition,
    }
    table = table.reset_index().rename(columns={axis: "coordinate_um"})
    table.insert(0, "axis", axis)
    return summary, table


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    source_digest = digest(SOURCE)
    script_digest = digest(Path(__file__))
    raw = pd.read_csv(SOURCE)
    assert raw.shape == (4914, 12), raw.shape
    assert not raw.isna().any().any()
    assert np.isfinite(raw.select_dtypes(include=[np.number]).to_numpy()).all()
    assert sorted(raw.seed.unique().tolist()) == SEEDS
    assert set(raw.model) == {"cnn", "mlp8"}
    assert raw.row_index.nunique() == 819 and raw.session.nunique() == 8
    assert not raw.duplicated(["seed", "row_index", "model"]).any()
    assert (raw.groupby(["row_index", "model"]).seed.nunique() == 3).all()
    assert (raw.groupby("row_index")[["true_x_um", "true_y_um", "session", "fold"]].nunique().to_numpy() == 1).all()
    assert sorted(raw.fold.unique().tolist()) == list(range(8))
    assert (raw.groupby("fold").session.nunique() == 1).all()
    residuals = raw[["pred_x_um", "pred_y_um"]].to_numpy() - raw[["true_x_um", "true_y_um"]].to_numpy()
    assert np.max(np.abs(residuals - raw[["residual_x_um", "residual_y_um"]].to_numpy())) < 1e-9
    error_difference = float(np.max(np.abs(np.hypot(residuals[:, 0], residuals[:, 1]) - raw.euclidean_error_um)))
    assert error_difference < 1e-9

    observation_errors = raw.groupby(["row_index", "model"]).euclidean_error_um.mean().unstack("model")
    observations = raw.drop_duplicates("row_index").set_index("row_index")[["fold", "session", "true_x_um", "true_y_um"]].copy()
    observations = observations.join(observation_errors.rename(columns={"cnn": "cnn_error_um", "mlp8": "mlp8_error_um"}))
    observations["acquisition"] = observations.fold.map(lambda fold: "A" + str(int(fold) + 1))
    observations["mlp8_minus_cnn_error_um"] = observations.mlp8_error_um - observations.cnn_error_um
    observations["n_seeds_averaged"] = 3
    assert len(observations[["true_x_um", "true_y_um"]].drop_duplicates()) == 714
    fold_summaries, geometry_records = [], []
    for fold in range(8):
        test = observations.loc[observations.fold.eq(fold)].copy()
        training = observations.loc[~observations.fold.eq(fold)]
        train_xy = training[["true_x_um", "true_y_um"]].drop_duplicates().to_numpy()
        test_xy = test[["true_x_um", "true_y_um"]].to_numpy()
        assert np.linalg.matrix_rank(train_xy - train_xy.mean(axis=0)) == 2
        hull = ConvexHull(train_xy)
        normals, offsets = hull.equations[:, :2], hull.equations[:, 2]
        distances = (test_xy @ normals.T + offsets) / np.linalg.norm(normals, axis=1)
        max_signed = distances.max(axis=1)
        hull_class = np.where(max_signed > TOL_UM, "outside", np.where(np.abs(max_signed) <= TOL_UM, "boundary", "inside"))
        train_set = set(map(tuple, train_xy))
        exact_overlap = np.array([tuple(point) in train_set for point in test_xy])
        nearest, _ = cKDTree(train_xy).query(test_xy, k=1)
        assert np.all(nearest[exact_overlap] == 0)
        assert np.all(nearest[~exact_overlap] > TOL_UM)
        bbox_outside = np.any((test_xy < train_xy.min(axis=0) - TOL_UM) | (test_xy > train_xy.max(axis=0) + TOL_UM), axis=1)
        assert not np.any(bbox_outside & (hull_class != "outside"))
        observations.loc[test.index, "hull_class"] = hull_class
        observations.loc[test.index, "exact_train_coordinate_overlap"] = exact_overlap
        observations.loc[test.index, "nearest_train_distance_um"] = nearest
        observations.loc[test.index, "max_hull_halfspace_distance_um"] = max_signed
        observations.loc[test.index, "outside_train_bounding_rectangle"] = bbox_outside
        row = {
            "acquisition": "A" + str(fold + 1), "fold": fold, "session": test.session.iloc[0],
            "n_test_observations": len(test), "n_train_observations": len(training),
            "n_test_coordinates": len(np.unique(test_xy, axis=0)), "n_train_coordinates": len(train_xy),
            "n_exact_overlap_observations": int(exact_overlap.sum()),
            "n_no_exact_overlap_observations": int((~exact_overlap).sum()),
            "n_hull_inside": int(np.sum(hull_class == "inside")),
            "n_hull_boundary": int(np.sum(hull_class == "boundary")),
            "n_hull_outside": int(np.sum(hull_class == "outside")),
            "n_outside_rectangle": int(bbox_outside.sum()),
            "n_rectangle_inside_but_hull_outside": int(np.sum((~bbox_outside) & (hull_class == "outside"))),
            "nearest_train_distance_min_um": float(nearest.min()),
            "nearest_train_distance_median_um": float(np.median(nearest)),
            "nearest_train_distance_mean_um": float(nearest.mean()),
            "nearest_train_distance_max_um": float(nearest.max()),
            "cnn_mean_error_um": float(test.cnn_error_um.mean()),
            "mlp8_mean_error_um": float(test.mlp8_error_um.mean()),
            "mlp8_minus_cnn_mean_error_um": float(test.mlp8_minus_cnn_error_um.mean()),
            "within_fold_inside_outside_comparison_observed": bool(np.any(hull_class == "inside") and np.any(hull_class == "outside")),
        }
        for dimension, col in enumerate(["x", "y"]):
            for category, points in [("train", train_xy), ("test", test_xy)]:
                row[f"{category}_{col}_min_um"] = float(points[:, dimension].min())
                row[f"{category}_{col}_max_um"] = float(points[:, dimension].max())
                row[f"{category}_{col}_unique_levels"] = len(np.unique(points[:, dimension]))
        fold_summaries.append(row)
        geometry_records.append({"acquisition": row["acquisition"],
                                 "hull_vertices_xy_um": train_xy[hull.vertices].tolist(),
                                 "hull_area_um2": float(hull.volume),
                                 "hull_halfspaces": hull.equations.tolist()})
    assert not observations[["hull_class", "exact_train_coordinate_overlap", "nearest_train_distance_um"]].isna().any().any()
    observations["exact_train_coordinate_overlap"] = observations.exact_train_coordinate_overlap.astype(bool)
    observations["outside_train_bounding_rectangle"] = observations.outside_train_bounding_rectangle.astype(bool)

    strata = []
    classes = [("hull", "hull_class", HULL_CLASSES),
               ("exact_coordinate_overlap", "exact_train_coordinate_overlap", [True, False])]
    for acquisition, group in observations.groupby("acquisition", sort=True):
        strata.append({"scope": "per_acquisition", "acquisition": acquisition, "stratification": "all",
                       "stratum": "all", "weighting": "observations_within_acquisition", **describe(group)})
        for category, field, levels in classes:
            for level in levels:
                selected = group.loc[group[field].eq(level)]
                strata.append({"scope": "per_acquisition", "acquisition": acquisition, "stratification": category,
                               "stratum": str(level).lower(), "weighting": "observations_within_acquisition", **describe(selected)})
    for category, field, levels in classes:
        for level in levels:
            selected = observations.loc[observations[field].eq(level)]
            strata.append({"scope": "pooled", "acquisition": "all", "stratification": category,
                           "stratum": str(level).lower(), "weighting": "observation_weighted", **describe(selected)})
            aggregate = selected.groupby("acquisition")[["cnn_error_um", "mlp8_error_um", "mlp8_minus_cnn_error_um", "nearest_train_distance_um"]].mean()
            equal_entry = {"scope": "pooled", "acquisition": "all", "stratification": category,
                           "stratum": str(level).lower(), "weighting": "equal_available_acquisition",
                           "n_observations": len(selected), "n_coordinates": len(selected[["true_x_um", "true_y_um"]].drop_duplicates()),
                           "n_acquisitions": len(aggregate)}
            for col in aggregate.columns:
                equal_entry[col.replace("_um", "_mean_um")] = float(aggregate[col].mean()) if len(aggregate) else None
            strata.append(equal_entry)

    within_fold = []
    distance_associations = []
    for acquisition, group in [("pooled", observations)] + list(observations.groupby("acquisition", sort=True)):
        entry = {"acquisition": acquisition, "n_observations": len(group),
                 "n_unique_distances": int(group.nearest_train_distance_um.nunique()),
                 "interpretation": "descriptive rank association; pooled estimates confound acquisition and position"}
        for col in ["cnn_error_um", "mlp8_error_um", "mlp8_minus_cnn_error_um"]:
            if group.nearest_train_distance_um.nunique() > 1 and group[col].nunique() > 1:
                entry["spearman_distance_" + col] = float(group.nearest_train_distance_um.rank().corr(group[col].rank()))
            else:
                entry["spearman_distance_" + col] = None
        distance_associations.append(entry)
        if acquisition == "pooled":
            continue
        inside = group.loc[group.hull_class.eq("inside")]
        outside = group.loc[group.hull_class.eq("outside")]
        contrast = {"acquisition": acquisition, "n_inside": len(inside), "n_outside": len(outside),
                    "identifiable_from_observed_groups": bool(len(inside) and len(outside))}
        for col in ["cnn_error_um", "mlp8_error_um", "mlp8_minus_cnn_error_um"]:
            contrast["outside_minus_inside_" + col] = float(outside[col].mean() - inside[col].mean()) if len(inside) and len(outside) else None
        within_fold.append(contrast)
    axis_summaries, axis_tables = [], []
    for axis in ["true_x_um", "true_y_um"]:
        summary, table = axis_confounding(observations, axis)
        axis_summaries.append(summary)
        axis_tables.append(table)
    coordinate_owners = observations.groupby(["true_x_um", "true_y_um"]).acquisition.nunique()
    ownership_by_row = observations.join(coordinate_owners.rename("n_coordinate_acquisitions"), on=["true_x_um", "true_y_um"])

    acquisitions = pd.DataFrame(fold_summaries)
    strata_frame = pd.DataFrame(strata)
    output_paths = {
        "observations": WORK / "coverage_observations.csv",
        "acquisitions": WORK / "coverage_acquisitions.csv",
        "strata": WORK / "coverage_strata.csv",
        "axis_acquisition_counts": WORK / "coverage_axis_acquisition_counts.csv",
    }
    observations.reset_index().to_csv(output_paths["observations"], index=False, float_format="%.15g")
    acquisitions.to_csv(output_paths["acquisitions"], index=False, float_format="%.15g")
    strata_frame.to_csv(output_paths["strata"], index=False, float_format="%.15g")
    pd.concat(axis_tables, ignore_index=True).to_csv(output_paths["axis_acquisition_counts"], index=False)
    report = {
        "diagnostic_role": "Post-hoc geometric coverage description; no new independent validation, no fitted localisation model.",
        "fixed_plan": __doc__, "script_sha256": script_digest,
        "input": {"path": str(SOURCE.relative_to(ROOT)), "sha256": source_digest, "rows": len(raw),
                  "unique_observations": len(observations), "unique_coordinates": len(coordinate_owners),
                  "seeds": SEEDS, "models": ["cnn", "mlp8"], "na_cells": int(raw.isna().sum().sum()),
                  "max_reconstructed_euclidean_error_difference_um": error_difference},
        "geometry_tolerance_um": TOL_UM,
        "aggregation": "Three seed-specific Euclidean errors averaged within observation; no averaged-prediction ensemble.",
        "acquisition_metrics": json_records(acquisitions), "stratified_errors": strata,
        "training_hulls": geometry_records,
        "axis_acquisition_confounding": axis_summaries,
        "coordinate_ownership": {
            "coordinate_counts_by_number_of_acquisitions": {str(int(k)): int(v) for k, v in coordinate_owners.value_counts().sort_index().items()},
            "observations_at_coordinates_owned_by_one_acquisition": int(ownership_by_row.n_coordinate_acquisitions.eq(1).sum()),
        },
        "within_acquisition_inside_outside_contrasts": within_fold,
        "distance_error_rank_associations": distance_associations,
        "pooled_class_counts": {level: int(observations.hull_class.eq(level).sum()) for level in HULL_CLASSES},
        "exact_overlap_count": int(observations.exact_train_coordinate_overlap.sum()),
        "hull_rectangle_classification_differences": int(np.sum(observations.hull_class.eq("outside") != observations.outside_train_bounding_rectangle)),
        "training_performed": False, "p_values_calculated": False,
        "outputs": {name: {"filename": path.name, "sha256": digest(path)} for name, path in output_paths.items()},
    }
    assert digest(SOURCE) == source_digest
    assert digest(Path(__file__)) == script_digest
    report["input_unchanged_after_analysis"] = True
    (WORK / "coverage_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    counts = report["pooled_class_counts"]
    mixed = [row["acquisition"] for row in within_fold if row["identifiable_from_observed_groups"]]
    no_mixed = [row["acquisition"] for row in within_fold if not row["identifiable_from_observed_groups"]]
    outside_acqs = acquisitions.loc[acquisitions.n_hull_outside.gt(0), "acquisition"].tolist()
    x, y = axis_summaries
    md = ["# Post-hoc spatial-coverage diagnostic", "",
          "This diagnostic uses only retained LOAO predictions. It does not refit models or change primary evaluation records.", "",
          f"Of 819 held-out observations, {counts['inside']} lie strictly inside their fold's training convex hull, {counts['boundary']} on its boundary, and {counts['outside']} outside.",
          f"Exact training-coordinate overlap occurs for {report['exact_overlap_count']} observations. Nearest observed training positions are {observations.nearest_train_distance_um.min():.2f}–{observations.nearest_train_distance_um.max():.2f} µm away.",
          f"Outside-hull observations occur in {', '.join(outside_acqs) or 'no acquisitions'}. Within-acquisition inside/outside comparisons are available in {', '.join(mixed) or 'no acquisitions'}; unavailable in {', '.join(no_mixed)}.", "",
          f"Acquisition identity is strongly associated with stage X if the following data support that interpretation: {x['observations_on_single_owner_axis_levels']}/819 observations occur at X levels belonging to a single acquisition. The observed X coordinate explains {100*x['fraction_acquisition_entropy_explained']:.2f}% of acquisition-label entropy, versus {100*y['fraction_acquisition_entropy_explained']:.2f}% for Y.", "",
          "| Acquisition | Test n | Inside | Boundary | Outside | Exact overlap | Nearest distance, min–max (µm) | CNN mean (µm) | MLP8 mean (µm) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in fold_summaries:
        md.append(f"| {row['acquisition']} | {row['n_test_observations']} | {row['n_hull_inside']} | {row['n_hull_boundary']} | {row['n_hull_outside']} | {row['n_exact_overlap_observations']} | {row['nearest_train_distance_min_um']:.2f}–{row['nearest_train_distance_max_um']:.2f} | {row['cnn_mean_error_um']:.2f} | {row['mlp8_mean_error_um']:.2f} |")
    md.extend(["", "All hull and overlap strata, including empty strata, appear in coverage_strata.csv. Pooled observation-weighted and equal-available-acquisition means are different summaries and are labelled separately.", "", "The geometry can establish whether a prediction interpolates within or extrapolates beyond the available target-position support. It cannot by itself separate spatial effects from acquisition, source conditions or fold-specific fitted models. Rank associations are descriptive, with no independent-observation p values.", ""])
    (WORK / "coverage_interpretation.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"counts": counts, "overlap": report["exact_overlap_count"], "mixed_folds": mixed,
                      "axis_confounding": axis_summaries, "folds": json_records(acquisitions),
                      "pooled_hull_strata": [entry for entry in strata if entry["scope"] == "pooled" and entry["stratification"] == "hull"]}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
