"""Check release files and recompute summaries from saved predictions.

Uses the Python standard library and writes no files. Historical source
fingerprints refer to execution_sources/, not the edited reading copies.
Bootstrap intervals are checked by file checksum, not regenerated here.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
REL = Path("paper/figures/scientific_controls_20260912")
HERE = ROOT / REL
ARCHIVE = ROOT / "execution_sources"
FOLDS = tuple(f"A{i}" for i in range(1, 9))
SEEDS = tuple(range(20260804, 20260814))
COUNTS = (102, 104, 102, 204, 51, 52, 102, 102)
MODEL_SOURCES = tuple(Path("src/qml_microrobot") / name for name in (
    "study_l30_author_cnn.py", "study_l31_compact.py", "study_l_session.py", "study_l_nv.py"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, tolerance=1e-8):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), f"Non-finite {label}")
    require(math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance),
            f"{label}: {actual} differs from {expected}")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def source_hash(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def extended_hash(base, path):
    digest = hashlib.sha256(base.encode("ascii"))
    digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_files():
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    entries = manifest["files"]
    require(bool(entries), "Empty release manifest")
    seen = set()
    for entry in entries:
        relative = Path(entry["path"])
        path = (ROOT / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(ROOT.resolve()), "Manifest path escapes release")
        require(entry["path"] not in seen, "Duplicate manifest path")
        seen.add(entry["path"])
        require(path.is_file(), f"Missing release file: {relative}")
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                f"Release checksum mismatch: {relative}")
    print(f"File checksums: {len(entries)} files passed")


def historical_hashes():
    models = [ARCHIVE / path for path in MODEL_SOURCES]
    recipe = source_hash([ARCHIVE / REL / "recipe_full.py", ARCHIVE / REL / "recipe_factorial.py", *models])
    nested = source_hash([
        ARCHIVE / REL / "nested_loao.py", ARCHIVE / REL / "recipe_factorial.py",
        ROOT / "study_l/l31_compact_nonlinear_bridge_protocol.json",
        ARCHIVE / "scripts/run_study_l31_compact_nonlinear_bridge.py",
        models[1], models[0], models[2], models[3]])
    return {
        "recipe_full.csv": recipe,
        "recipe_additional.csv": extended_hash(recipe, ARCHIVE / REL / "recipe_additional.py"),
        "nested_loao.json": nested,
        "nested_outer_additional.csv": extended_hash(nested, ARCHIVE / REL / "nested_outer_additional.py"),
        "meneses_full.csv": source_hash([ARCHIVE / REL / "meneses_external.py", models[0]]),
        "onnx_benchmark.jsonl": sha(ARCHIVE / REL / "onnx_benchmark.py"),
    }


def test_indices():
    ranges, start = {}, 0
    for fold, count in zip(FOLDS, COUNTS):
        ranges[fold] = set(range(start, start + count))
        start += count
    return ranges


def coordinate_error(row):
    truth = (float(row["true_x_um"]), float(row["true_y_um"]))
    prediction = (float(row["pred_x_um"]), float(row["pred_y_um"]))
    error = math.hypot(prediction[0] - truth[0], prediction[1] - truth[1])
    close(row["error_um"], error, "Euclidean prediction error")
    return truth, error


def verify_recipe(hashes):
    groups, indices, orders, truths, data_hashes = defaultdict(list), defaultdict(set), {}, {}, set()
    ranges = test_indices()
    for name, expected_rows, allowed_seeds in (
            ("recipe_full.csv", 39312, SEEDS[:3]), ("recipe_additional.csv", 91728, SEEDS[3:])):
        count = 0
        for row in csv_rows(HERE / name):
            count += 1
            fold, model, seed = row["fold"], row["model"], int(row["seed"])
            x, y, epoch, index = (int(row[key]) for key in (
                "input_standardised", "target_standardised", "epoch", "row_index"))
            require(fold in FOLDS and model in ("cnn", "mlp8") and seed in allowed_seeds
                    and x in (0, 1) and y in (0, 1) and epoch in (1500, 3000), "Unexpected recipe condition")
            require(row["source_sha256"] == hashes[name], f"Historical source mismatch: {name}")
            require(index in ranges[fold], "Recipe row in wrong acquisition")
            data_hashes.add(row["data_sha256"])
            key = (fold, model, seed, x, y, epoch)
            require(index not in indices[key], "Duplicate recipe prediction")
            indices[key].add(index)
            truth, error = coordinate_error(row)
            require(truth == truths.setdefault(index, truth), "Inconsistent coordinate truth")
            groups[key].append(error)
            train = 819 - len(ranges[fold])
            require(int(row["train_rows"]) == train and int(row["test_rows"]) == len(ranges[fold])
                    and int(row["optimizer_updates"]) == epoch * math.ceil(train / 128), "Recipe update/row count differs")
            order_key = (fold, seed, epoch)
            require(row["batch_order_sha256"] == orders.setdefault(order_key, row["batch_order_sha256"]),
                    "Matched recipe batch orders differ")
        require(count == expected_rows, f"Recipe row count differs: {name}")
    require(len(groups) == 1280 and len(truths) == 819 and len(data_hashes) == 1, "Incomplete recipe inventory")
    for key, actual in indices.items():
        require(actual == ranges[key[0]], "Recipe acquisition coverage incomplete")
    fit_means = {key: statistics.fmean(values) for key, values in groups.items()}
    acquisition = {}
    for fold in FOLDS:
        for model in ("cnn", "mlp8"):
            for x in (0, 1):
                for y in (0, 1):
                    for epoch in (1500, 3000):
                        acquisition[fold, model, x, y, epoch] = statistics.fmean(
                            fit_means[fold, model, seed, x, y, epoch] for seed in SEEDS)
    gap = {(fold, x, y, epoch): acquisition[fold, "mlp8", x, y, epoch] - acquisition[fold, "cnn", x, y, epoch]
           for fold in FOLDS for x in (0, 1) for y in (0, 1) for epoch in (1500, 3000)}
    summaries = list(csv_rows(HERE / "recipe_summary.csv"))
    require(len(summaries) == 8 and len({(r["input_standardised"], r["target_standardised"], r["epoch"])
                                      for r in summaries}) == 8, "Recipe summary conditions differ")
    for row in summaries:
        x, y, epoch = (int(row[k]) for k in ("input_standardised", "target_standardised", "epoch"))
        require(int(row["acquisitions"]) == 8 and int(row["optimisation_seeds"]) == 10, "Statistical units differ")
        require(row["old_results_sha256"] == sha(HERE / "recipe_full.csv")
                and row["additional_results_sha256"] == sha(HERE / "recipe_additional.csv"), "Summary input hashes differ")
        for model in ("cnn", "mlp8"):
            values = [acquisition[fold, model, x, y, epoch] for fold in FOLDS]
            close(row[f"{model}_mean"], statistics.fmean(values), f"{model} equal-acquisition mean")
            close(row[f"{model}_acquisition_sd"], statistics.stdev(values), f"{model} acquisition SD")
        d = [gap[fold, x, y, epoch] for fold in FOLDS]
        for prefix, values in (("d_um", d), ("delta_d_um", [value - gap[fold, 0, 0, 3000] for fold, value in zip(FOLDS, d)]),
                               ("same_epoch_delta_d_um", [value - gap[fold, 0, 0, epoch] for fold, value in zip(FOLDS, d)])):
            close(row[f"{prefix}_mean"], statistics.fmean(values), prefix + " mean")
            close(row[f"{prefix}_acquisition_sd"], statistics.stdev(values), prefix + " SD")
    weights = {"Input": (-.5, -.5, .5, .5), "Target": (-.5, .5, -.5, .5), "Interaction": (1, -1, -1, 1),
               "Joint": (-1, 0, 0, 1), "Input_given_target_off": (-1, 0, 1, 0),
               "Input_given_target_on": (0, -1, 0, 1), "Target_given_input_off": (-1, 1, 0, 0),
               "Target_given_input_on": (0, 0, -1, 1)}
    contrasts = list(csv_rows(HERE / "recipe_factorial_effects.csv"))
    expected_contrasts = {(epoch, contrast, kind, fold) for epoch in (1500, 3000) for contrast in weights
                          for kind, fold in [("acquisition", f) for f in FOLDS] + [("summary", "")]}
    require(len(contrasts) == 144 and {
        (int(r["checkpoint"]), r["contrast"], r["record_type"], r["acquisition"]) for r in contrasts
    } == expected_contrasts, "Factorial contrast coverage differs")
    for row in contrasts:
        epoch, vector = int(row["checkpoint"]), weights[row["contrast"]]
        require(tuple(map(float, row["weights_d00_d01_d10_d11"].split(","))) == vector, "Contrast weights differ")
        values = [math.fsum(w * gap[fold, x, y, epoch] for w, (x, y) in zip(vector, ((0, 0), (0, 1), (1, 0), (1, 1))))
                  for fold in FOLDS]
        expected = values[FOLDS.index(row["acquisition"])] if row["record_type"] == "acquisition" else statistics.fmean(values)
        close(row["estimate_um"], expected, "Factorial contrast")
        if row["record_type"] == "summary":
            close(row["acquisition_sd_um"], statistics.stdev(values), "Factorial contrast SD")
    print("Recipe: 640 fits, 1,280 checkpoints, 131,040 errors and all eight recipe summaries passed")
    return acquisition, truths, data_hashes.pop()


def verify_nested(hashes, acquisition, truths, data_hash):
    result = json.loads((HERE / "nested_loao.json").read_text(encoding="utf-8"))
    require(result["source_sha256"] == hashes["nested_loao.json"] and result["data_sha256"] == data_hash,
            "Nested source/data fingerprint differs")
    require(set(result["folds"]) == set(FOLDS), "Nested outer acquisitions differ")
    ranges, means, indices = test_indices(), {}, defaultdict(set)
    expected = {fold: "mlp32_fixed" if fold == "A1" else "mlp16_fixed" if fold == "A6" else "mlp8_fixed" for fold in FOLDS}
    for fold, item in result["folds"].items():
        require(item["selected_candidate"]["candidate"] == expected[fold], "Nested selected identity differs")
        require(len(item["inner_rows"]) == 84 and len(item["candidate_summaries"]) == 12, "Nested inner coverage differs")
        require(all(r["validation_session"] != item["held_out_session"] for r in item["inner_rows"]), "Outer acquisition enters inner validation")
        require(all(r["train_rows"] + r["validation_rows"] == 819 - len(ranges[fold]) for r in item["inner_rows"]), "Nested inner training/validation rows differ")
        minimum = min(r["mean_error_um"] for r in item["candidate_summaries"])
        eligible = [r for r in item["candidate_summaries"] if r["mean_error_um"] <= minimum + 1]
        chosen = min(eligible, key=lambda r: (r["fitted_parameters"], r["mean_error_um"], r["candidate"]))
        require(chosen == item["selected_candidate"], "Nested inner selection rule differs")
        require({int(r["seed"]) for r in item["outer_outcomes"]} == set(SEEDS[:3]), "Nested initial seed inventory differs")
        for outcome in item["outer_outcomes"]:
            values, ids = [], set()
            for row in outcome["predictions"]:
                index = int(row["row_index"])
                require(index not in ids, "Duplicate nested prediction")
                ids.add(index)
                truth, error = coordinate_error(row)
                require(truth == truths[index], "Nested coordinate truth differs")
                values.append(error)
            require(ids == ranges[fold], "Nested initial row coverage differs")
            value = statistics.fmean(values)
            close(outcome["mean_error_um"], value, "Nested initial mean")
            means[fold, int(outcome["seed"])] = value
    selection_hash = sha(HERE / "nested_loao.json")
    extra = defaultdict(list)
    for row in csv_rows(HERE / "nested_outer_additional.csv"):
        fold, seed, index = row["outer_fold"], int(row["seed"]), int(row["row_index"])
        require(fold in FOLDS and seed in SEEDS[3:] and row["selected_candidate"] == expected[fold], "Nested additional identity differs")
        require(row["source_sha256"] == hashes["nested_outer_additional.csv"] and row["data_sha256"] == data_hash
                and row["selection_sha256"] == selection_hash, "Nested additional fingerprints differ")
        require(index in ranges[fold] and index not in indices[fold, seed], "Nested duplicate/wrong outer row")
        indices[fold, seed].add(index)
        truth, error = coordinate_error(row)
        require(truth == truths[index], "Nested additional coordinate truth differs")
        extra[fold, seed].append(error)
    require(len(extra) == 56, "Nested additional fit count differs")
    for key, values in extra.items():
        require(indices[key] == ranges[key[0]], "Nested additional coverage incomplete")
        means[key] = statistics.fmean(values)
    per_acquisition = [statistics.fmean(means[fold, seed] for seed in SEEDS) for fold in FOLDS]
    for fold, value in zip(FOLDS, per_acquisition):
        if expected[fold] == "mlp8_fixed":
            close(value, acquisition[fold, "mlp8", 1, 1, 1500], "Nested/fixed MLP8 matched refits")
    close(statistics.fmean(per_acquisition), 43.33013286380533, "Nested ten-seed mean")
    print(f"Nested: 80 outer fits, selected identities and mean {statistics.fmean(per_acquisition):.10f} um passed")


def verify_meneses(hashes):
    labels = {(int(r["segment"]), int(r["window_index"])): r for r in csv_rows(HERE / "meneses_label_audit.csv")
              if r["role"] == "test_all"}
    require(len(labels) == 5368, "Meneses test-window labels differ")
    digest = hashlib.sha256()
    for segment in range(1, 6):
        name = f"phys_model_data_zAut_segm{segment}.csv"
        digest.update(name.encode("utf-8"))
        with (HERE / "meneses_data" / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    groups, floors, ids, count = defaultdict(list), defaultdict(list), defaultdict(set), 0
    for row in csv_rows(HERE / "meneses_full.csv"):
        count += 1
        x, y, seed, epoch, segment, window = (int(row[k]) for k in (
            "input_standardised", "target_standardised", "seed", "epoch", "test_segment", "window_index"))
        model = row["model"]
        require(x in (0, 1) and y in (0, 1) and model in ("cnn", "mlp8") and seed in SEEDS[:3]
                and epoch in (200, 400) and segment in (2, 3), "Unexpected Meneses condition")
        require(row["source_sha256"] == hashes["meneses_full.csv"] and row["data_sha256"] == digest.hexdigest(), "Meneses fingerprints differ")
        require(int(row["train_windows"]) == 819 and int(row["optimizer_updates"]) == epoch * 4, "Meneses updates differ")
        key = (x, y, model, seed, epoch, segment)
        require(window not in ids[key], "Duplicate Meneses prediction")
        ids[key].add(window)
        label = labels[segment, window]
        truth, prediction = float(row["true_z_m"]), float(row["pred_z_m"])
        close(truth, label["target_value_m"], "Meneses target", 0)
        close(row["target_time_s"], label["target_time_s"], "Meneses target time", 0)
        error = abs(prediction - truth)
        close(row["abs_error_m"], error, "Meneses absolute error")
        groups[key].append(error)
        mask = int(row["parking_ground_truth"])
        require(mask == int(label["retained_floor_value_mask"] == "True"), "Meneses floor-value mask differs")
        if mask:
            floors[key].append(error)
    require(count == 257664 and len(groups) == 96, "Meneses fit/snapshot coverage differs")
    for key, values in groups.items():
        require(ids[key] == set(range(2690 if key[-1] == 2 else 2678)), "Meneses window coverage incomplete")
        require(len(floors[key]) == (2144 if key[-1] == 2 else 2305), "Meneses primary target count differs")
    equal_segment = {(x, y, model): statistics.fmean(
        statistics.fmean(statistics.fmean(floors[x, y, model, seed, 400, segment]) for seed in SEEDS[:3]) for segment in (2, 3))
        for x in (0, 1) for y in (0, 1) for model in ("cnn", "mlp8")}
    rows = [r for r in csv_rows(HERE / "meneses_boundary_sensitivity.csv") if r["aggregation_scope"] == "equal_segment"]
    require(len(rows) == 4, "Meneses equal-segment summary coverage differs")
    for row in rows:
        x, y = int(row["input_standardised"]), int(row["target_standardised"])
        for model in ("cnn", "mlp8"):
            close(row[f"original_{model}_mae_m"], equal_segment[x, y, model], "Meneses equal-segment mean")
        close(row["original_gap_m"], equal_segment[x, y, "mlp8"] - equal_segment[x, y, "cnn"], "Meneses architecture gap")
    print("Meneses: 24 fits, 257,664 absolute errors, label masks and four primary recipe summaries passed")


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q / 100
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def verify_backends(hashes):
    input_hash = sha(HERE / "memory_input_features.npy")
    reference_hash = sha(HERE / "portable_benchmark_reference.npz")
    protocol_hash = sha(HERE / "portable_benchmark_protocol.json")
    records = {}
    with (HERE / "onnx_benchmark.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["backend"], row["model"], int(row["repeat"])
            require(key not in records and row["script_sha256"] == hashes["onnx_benchmark.jsonl"], "Backend process/source identity differs")
            records[key] = row
            require(row["input_sha256"] == input_hash and row["reference_sha256"] == reference_hash
                    and row["source_protocol_sha256"] == protocol_hash, "Backend input/reference/protocol hash differs")
            backend, model, _ = key
            model_path = (ROOT / ("checkpoints/study_l31_author_first5.pt" if model == "cnn"
                                  else "checkpoints/study_l31_selected_compact.pkl")) if backend == "torch" else (
                HERE / (f"portable_{model}.npz" if backend == "numpy" else f"{model}.onnx"))
            require(row["model_sha256"] == sha(model_path)
                    and row["onnx_model_sha256"] == sha(HERE / f"{model}.onnx")
                    and row["serialized_model_bytes"] == model_path.stat().st_size, "Backend model fingerprint/size differs")
            require(row["cpu_affinity"] == [12] and row["process_priority"] == 32
                    and all(v == "1" for v in row["thread_environment"].values())
                    and all(pool["num_threads"] == 1 for pool in row["native_threadpools"]), "Backend thread/affinity settings differ")
            require(len(row["memory_samples"]) == 20, "Backend memory stages incomplete")
            for batch in ("1", "256"):
                require(len(row["latency_ns"][batch]) == 100 and all(math.isfinite(v) and v > 0 for v in row["latency_ns"][batch]), "Backend timed-call inventory differs")
                require(0 <= row["max_abs_coordinate_drift_um"][batch] <= .01, "Backend prediction-consistency threshold failed")
    expected = {(backend, model, repeat) for backend in ("torch", "numpy", "onnx") for model in ("cnn", "mlp8") for repeat in range(1, 6)}
    require(set(records) == expected, "Backend condition/process coverage differs")
    summaries = list(csv_rows(HERE / "onnx_benchmark_summary.csv"))
    require(len(summaries) == 12, "Backend summary coverage differs")
    for row in summaries:
        backend, model, batch = row["backend"], row["model"], row["batch_size"]
        selected = [records[backend, model, repeat] for repeat in range(1, 6)]
        for q in (50, 95, 99):
            value = statistics.median(percentile([v / 1000 for v in r["latency_ns"][batch]], q) for r in selected)
            close(row[f"latency_p{q}_us"], value, "Backend latency percentile")
        for column, stage in (("rss_loaded_mib", "model_loaded"), ("rss_after_batch_mib", "post_batch_" + batch)):
            values = []
            for record in selected:
                samples = [s["rss_bytes"] for s in record["memory_samples"] if s["stage"] == stage]
                require(len(samples) == 5 and all(math.isfinite(v) and v > 0 for v in samples), "Backend RSS snapshots differ")
                values.append(statistics.median(samples) / 1048576)
            close(row[column], statistics.median(values), "Backend process RSS")
        for field in ("model_load_ms", "cold_start_ms", "backend_import_ms"):
            close(row[field], statistics.median(r[field] for r in selected), "Backend loading/startup time")
        require(len({r["serialized_model_bytes"] for r in selected}) == 1, "Backend serialized sizes differ across processes")
        close(row["serialized_model_bytes"], selected[0]["serialized_model_bytes"], "Backend serialized size", 0)
        close(row["max_abs_coordinate_drift_um"], max(r["max_abs_coordinate_drift_um"][batch] for r in selected), "Backend prediction drift", 0)
    print("Backend records: 30 processes, 6,000 timed calls and all twelve summary rows passed")


def verify_matched(truths):
    run_dir = HERE / "matched_runs"
    protocol = json.loads((run_dir / "matched_protocol.json").read_text(encoding="utf-8"))
    source = ast.parse((ARCHIVE / REL / "verify_matched_controls.py").read_text(encoding="utf-8"))
    approved = next(ast.literal_eval(node.value) for node in source.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "APPROVED_SOURCES" for target in node.targets))
    hashes = {name.replace("\\", "/"): value for name, value in protocol["engine"]["source_sha256"].items()}
    require(hashes == approved, "Earlier matched-training source identities differ")
    require(all(sha(ARCHIVE / name) == value for name, value in hashes.items()), "Earlier matched-training archived source differs")
    jobs = protocol["all_job_ids"]
    require(len(jobs) == 54 and len(set(jobs)) == 54, "Earlier matched-training job plan differs")
    combined = {(r["job_id"], int(r["epoch"]), int(r["row_index"])): r
                for r in csv_rows(run_dir / "matched_observation_predictions.csv")}
    require(len(combined) == 12900, "Earlier matched-training combined coverage differs")
    groups, seen = defaultdict(list), set()
    for job in jobs:
        complete = json.loads((run_dir / "jobs" / (job + ".complete.json")).read_text(encoding="utf-8"))
        progress = json.loads((run_dir / "jobs" / (job + ".progress.json")).read_text(encoding="utf-8"))
        path = run_dir / "jobs" / (job + ".predictions.csv")
        require(complete["status"] == progress["status"] == "complete"
                and complete["last_completed_epoch"] == 3000, "Earlier matched-training job incomplete")
        require(complete["prediction_csv_sha256"] == progress["prediction_csv_sha256"] == sha(path), "Earlier matched-training job checksum differs")
        for row in csv_rows(path):
            epoch, seed, index = (int(row[k]) for k in ("epoch", "seed", "row_index"))
            key = (job, epoch, index)
            require(key not in seen and combined.get(key) == row, "Earlier matched-training duplicate/combined mismatch")
            seen.add(key)
            require(epoch in (1500, 3000) and seed in SEEDS[:3], "Earlier matched-training budget/seed differs")
            truth, error = coordinate_error(row)
            require(truth == truths[index], "Earlier matched-training target differs")
            for coordinate in ("x", "y"):
                close(row[f"residual_{coordinate}_um"], float(row[f"pred_{coordinate}_um"]) - float(row[f"true_{coordinate}_um"]), "Earlier matched-training residual")
            groups[row["evaluation"], epoch, row["model"], seed, row["acquisition"]].append(error)
    require(seen == set(combined) and len(groups) == 132, "Earlier matched-training prediction/group coverage differs")
    means = {key: statistics.fmean(values) for key, values in groups.items()}
    acquisition_rows = list(csv_rows(run_dir / "matched_acquisition_seed_summary.csv"))
    require(len(acquisition_rows) == 132, "Earlier matched-training seed summary coverage differs")
    for row in acquisition_rows:
        key = row["evaluation"], int(row["epoch"]), row["model"], int(row["seed"]), row["acquisition"]
        require(int(row["observations"]) == len(groups[key]), "Earlier matched-training observation count differs")
        close(row["mean_error_um"], means[key], "Earlier matched-training acquisition/seed mean")
    summaries = list(csv_rows(run_dir / "matched_equal_acquisition_summary.csv"))
    require(len(summaries) == 8, "Earlier matched-training summary coverage differs")
    for row in summaries:
        evaluation, epoch, model = row["evaluation"], int(row["epoch"]), row["model"]
        folds = FOLDS if evaluation == "loao" else FOLDS[5:]
        values = [statistics.fmean(means[evaluation, epoch, model, seed, fold] for seed in SEEDS[:3]) for fold in folds]
        require(int(row["acquisitions"]) == len(folds) and int(row["seeds_per_acquisition"]) == 3, "Earlier matched-training statistical units differ")
        close(row["equal_acquisition_mean_error_um"], statistics.fmean(values), "Earlier matched-training group mean")
        close(row["acquisition_sd_um"], statistics.stdev(values), "Earlier matched-training acquisition SD")
    print("Earlier matched training: 54 fits, 12,900 errors and eight separate historical summaries passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files-only", action="store_true", help="Check release file hashes without recomputing prediction summaries")
    args = parser.parse_args()
    verify_files()
    if args.files_only:
        return
    hashes = historical_hashes()
    acquisition, truths, data_hash = verify_recipe(hashes)
    verify_nested(hashes, acquisition, truths, data_hash)
    verify_meneses(hashes)
    verify_backends(hashes)
    verify_matched(truths)
    print("Passed. No training, inference, resource measurements or file writes were performed.")
    print("Bootstrap interval files passed checksum checks; intervals were not resampled.")


if __name__ == "__main__":
    main()
