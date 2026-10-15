"""Independent complete-run audit and four-row manuscript-table generator.

This module does not import matched_controls or execute any training. Invoke only
after the coordinator confirms all 54 fits have finished. Missing/incomplete jobs
produce no scientific summary or manuscript table. This verifier writes only its
own report, full legacy-comparison table and Table 2 outputs beside this script;
the runner, run directory, historical evidence and manuscript remain read-only.

Example:
  python verify_matched_controls.py --run-dir <absolute matched_runs directory>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[2]
SEEDS = (20260804, 20260805, 20260806)
EPOCHS = (1500, 3000)
MODELS = ("cnn", "mlp8")
PROTOCOL = "STUDY_L_MATCHED_CONTROLS_POSTHOC_20260912_V1"
NUMERIC_ATOL_UM = 1e-9
LEGACY_MATCH_ATOL = 0.0
LEGACY_PREDICTION_SHA256 = "840047a032e051e1fd24ada67bb88ed74ae2906075e6fd8c3806b919460543cc"
APPROVED_SOURCES = {
    "paper/figures/scientific_controls_20260912/matched_controls.py":
        "8d140cf629ef83ead9ceca6a8a96f59bf06788056cf8649de9d55433fb9ed72f",
    "src/qml_microrobot/study_l30_author_cnn.py":
        "3d7b0f1ce42edbb0372321eb4027c4bd4a8246743c7c68dbb879bb694d88c49b",
    "src/qml_microrobot/study_l31_compact.py":
        "e129d1abe402844d1835b86385782b3872524da53a4390af0ab235c2620cc53c",
    "src/qml_microrobot/study_l_session.py":
        "449ffc341f7e4b0048597666e4cbd5d091edc36013accc35aea3d7133fee0c77",
    "src/qml_microrobot/study_l_nv.py":
        "9c8eebe914aee89c9db49035ece3d8817675a537dcd5afb161bb1b508e5978f6",
}
REQUIRED_SPEC = {
    "protocol_id": PROTOCOL, "epochs": 3000, "snapshots": [1500,3000],
    "batch_size": 128, "learning_rate": .005, "betas": [.9,.999],
    "epsilon": 1e-7, "weight_decay": 0.0,
    "initialisation": "existing architecture classes: Xavier-uniform weights, zero biases",
    "standardisation": "training-only means and population SD (ddof=0) for all 24 inputs and 2 targets; zero SD becomes 1",
    "batch_order": "dedicated CPU torch.Generator(seed+1), randperm each epoch; no dropped final batch",
    "checkpoint_selection": "none; both fixed snapshots must be reported",
    "historical_states_reused": False, "frozen_L34_overwritten": False,
    "old_failed_gate_retested_or_replaced": False,
    "seeds_are_independent_units": False, "equal_compute_or_wall_time_claim": False,
}
PRED_FIELDS = ["protocol_id","job_id","model","evaluation","fold","seed","epoch",
               "row_index","acquisition","archive_session","train_rows","true_x_um",
               "true_y_um","pred_x_um","pred_y_um","residual_x_um","residual_y_um","error_um"]
NUMERIC_DIFFERENCES = {}


class IncompleteRun(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8",newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows, reader.fieldnames


def close(actual, expected, context, tolerance=NUMERIC_ATOL_UM):
    actual, expected = np.asarray(actual,dtype=float), np.asarray(expected,dtype=float)
    require(actual.shape == expected.shape, "Shape mismatch: " + context)
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Non-finite comparison: " + context)
    difference = float(np.max(np.abs(actual-expected))) if actual.size else 0.0
    NUMERIC_DIFFERENCES[context] = max(NUMERIC_DIFFERENCES.get(context,0.0),difference)
    require(difference <= tolerance, f"{context}: difference {difference} exceeds fixed tolerance {tolerance}")


def array_hash(values):
    values = np.ascontiguousarray(values)
    header = {"shape": list(values.shape), "dtype": str(values.dtype)}
    return hashlib.sha256(canonical(header)+values.tobytes()).hexdigest()


def load_inputs():
    # Reuse only the original read-only data loader, not the matched runner.
    sys.path.insert(0,str(ROOT/"src"))
    from qml_microrobot.study_l_session import load_l1_session_data
    data = load_l1_session_data(ROOT)
    x, y = np.asarray(data.features,dtype=np.float64), np.asarray(data.positions_um,dtype=np.float64)
    sessions = np.asarray(data.session_ids,dtype=str)
    order = list(dict.fromkeys(sessions.tolist()))
    require(x.shape == (819,24) and y.shape == (819,2), "Unexpected released array shape")
    require(len(order) == 8, "Expected eight acquisition groups")
    require([int(np.sum(sessions == s)) for s in order] == [102,104,102,204,51,52,102,102],
            "Acquisition membership/count drift")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Non-finite source features/targets")
    fingerprint = hashlib.sha256(canonical({"x":array_hash(x),"y":array_hash(y),
                                           "sessions":sessions.tolist()})).hexdigest()
    return x,y,sessions,order,fingerprint


def expected_jobs(sessions, order):
    groups = []
    for i, group in enumerate(order):
        groups.append(("loao",f"A{i+1}", np.flatnonzero(sessions != group),
                       np.flatnonzero(sessions == group)))
    development = np.isin(sessions,order[:5])
    groups.append(("archive_replay","A6-A8",np.flatnonzero(development),np.flatnonzero(~development)))
    jobs = {}
    for evaluation,fold,train,test in groups:
        require(len(np.intersect1d(train,test)) == 0, "Outer split overlap")
        require(np.array_equal(np.sort(np.r_[train,test]),np.arange(819)), "Outer split does not cover all rows")
        for seed in SEEDS:
            for model in MODELS:
                name = f"{evaluation}_{fold}_{model}_s{seed}"
                jobs[name] = {"job_id":name,"evaluation":evaluation,"fold":fold,
                              "model":model,"seed":seed,"train":train.tolist(),"test":test.tolist()}
    require(len(jobs) == 54, "Expected 54 fixed fits")
    require(sum(len(j["test"])*len(EPOCHS) for j in jobs.values()) == 12900,
            "Expected 12,900 per-observation predictions")
    return jobs


def expected_scales(x,y,train):
    result = {}
    for name,values in (("feature",x[train]),("target",y[train])):
        result[name+"_mean"] = values.mean(axis=0)
        scale = values.std(axis=0,ddof=0)
        result[name+"_scale"] = np.where(scale == 0,1.0,scale)
    return result


def expected_batch_records(train,seed):
    # Independent regeneration from the declared schedule, not from job metadata.
    import torch
    require(str(torch.__version__) == "2.8.0+cpu", "Verify with the recorded Torch runtime")
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(seed+1)
    digest, records = hashlib.sha256(), {}
    global_rows = np.asarray(train,dtype=np.int64)
    for epoch in range(1,3001):
        permutation = torch.randperm(len(global_rows),generator=generator).numpy()
        digest.update(global_rows[permutation].tobytes())
        if epoch in EPOCHS:
            records[epoch] = {"batch_order_sha256":digest.hexdigest(),
                              "optimizer_updates":epoch*math.ceil(len(train)/128)}
    return records


def keyed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(str(row[f]) for f in fields)
        require(key not in result, "Duplicate row key: " + str(key))
        result[key] = row
    return result


def compare_summary(path, rows, keys, numeric_fields, exact_fields=()):
    stored,_ = read_csv(path)
    actual, expected = keyed(stored,keys), keyed(rows,keys)
    require(set(actual) == set(expected), "Summary row coverage differs: " + path.name)
    for key in expected:
        for field in numeric_fields:
            close(float(actual[key][field]),float(expected[key][field]),path.name+":"+field)
        for field in exact_fields:
            require(str(actual[key][field]) == str(expected[key][field]),
                    path.name+": metadata differs: "+field)


def write_csv(path, rows):
    require(bool(rows), "Refuse to emit an empty evidence table")
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table_tex(table):
    lines = [r"% Generated only after the complete-run audit and exact legacy-MLP sanity check.",
        r"% Insert after source-split Table 1; leave the resource table label tab:resources unchanged.",
        r"\begin{table}[!htbp]",r"\centering",r"\small",
        r"\caption{Fixed post-hoc comparison under matched preprocessing and training updates. Errors are equal-acquisition means $\pm$ sample standard deviations across acquisitions after averaging three optimisation seeds within each acquisition ($n=8$ for LOAO; $n=3$ for replay). $\Delta$ is the paired mean error difference, MLP8 minus CNN. Both fixed budgets are reported; matched updates do not imply equal computational cost.}",
        r"\label{tab:matched_controls}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xrccc@{}}",
        r"\toprule",
        r"Evaluation & Epochs & CNN ($\mu$m) & MLP8 ($\mu$m) & $\Delta$ ($\mu$m) \\",
        r"\midrule"]
    for row in table:
        label = "Internal LOAO" if row["evaluation"] == "loao" else "Archive-ordered replay"
        cnn = f"${row['cnn_mean_um']:.2f} \\pm {row['cnn_between_acquisition_sd_um']:.2f}$"
        mlp = f"${row['mlp8_mean_um']:.2f} \\pm {row['mlp8_between_acquisition_sd_um']:.2f}$"
        delta = f"${row['paired_mlp8_minus_cnn_mean_um']:+.2f}$"
        lines.append(f"{label} & {row['epoch']:,} & {cnn} & {mlp} & {delta} \\\\")
    lines += [r"\bottomrule",r"\end{tabularx}",r"\end{table}",""]
    return "\n".join(lines)


def verify(run_dir):
    run_dir = run_dir.resolve()
    require(run_dir.is_relative_to(WORK), "Read only the declared new scientific-controls run directory")
    required = ["matched_protocol.json","matched_run_report.json","matched_observation_predictions.csv",
                "matched_acquisition_seed_summary.csv","matched_equal_acquisition_summary.csv",
                "matched_paired_acquisition_differences.csv"]
    if any(not (run_dir/name).is_file() for name in required):
        raise IncompleteRun("Complete-run outputs are not yet all present; no summaries or Table 2 are produced.")
    initial_hashes = {str(run_dir/name):sha(run_dir/name) for name in required}
    protocol, run_report = read_json(run_dir/required[0]), read_json(run_dir/required[1])
    require(run_report["status"] == "COMPLETE_FIXED_POSTHOC_CONTROLS_NOT_INDEPENDENT_VALIDATION",
            "Runner has not declared the fixed run complete")
    require(protocol["status"] == "AUTHOR_AUTHORIZED_FIXED_POSTHOC_RUN", "Unexpected protocol status")
    spec, engine = protocol["training"], protocol["engine"]
    for key,value in REQUIRED_SPEC.items():
        require(spec.get(key) == value, "Training protocol changed: "+key)
    require("post-hoc" in spec["role"] and "no new independent validation" in spec["role"], "Claim-boundary drift")
    sources = {p.replace("\\","/"):h.lower() for p,h in engine["source_sha256"].items()}
    require(sources == APPROVED_SOURCES, "Source hashes differ from the approved runner/model/loader sources")
    for name,expected_hash in sources.items():
        require(sha(ROOT/name) == expected_hash, "Current frozen source changed: "+name)
    for key,value in {"python":"3.9.13","numpy":"1.26.4","torch":"2.8.0+cpu",
                      "device":"cpu","intraop_threads":1,"interop_threads":1}.items():
        require(engine.get(key) == value, "Recorded execution environment changed: "+key)
    x,y,sessions,order,dataset_hash = load_inputs()
    require(dataset_hash == protocol["dataset_fingerprint"], "Raw features/targets/acquisition identity changed")
    jobs = expected_jobs(sessions,order)
    require(set(protocol["all_job_ids"]) == set(jobs) and len(protocol["all_job_ids"]) == 54, "Protocol job plan differs")
    require(protocol["planned_fits"] == 54 and protocol["snapshots_per_fit"] == 2
            and protocol["expected_prediction_rows"] == 12900, "Protocol planned totals differ")
    markers = list((run_dir/"jobs").glob("*.complete.json"))
    if len(markers) != 54:
        raise IncompleteRun(f"Only {len(markers)} complete job markers are present; expected 54")
    require({p.name[:-len('.complete.json')] for p in markers} == set(jobs), "Unexpected/missing complete-job identity")
    require(not list((run_dir/"jobs").glob("*.numerical_failure.json")), "Terminal numerical failure exists; no promotion")
    require(not list((run_dir/"jobs").glob("*.lock")), "An active/stale job lock remains")
    require(not list((run_dir/"jobs").glob("*.predictions.partial.csv")), "An ordinary incomplete prediction file remains")
    input_files = [run_dir/n for n in required] + markers
    initial_hashes.update({str(path):sha(path) for path in markers})
    predictions, sequence_cache, scale_cache, pair_records = [], {}, {}, {}
    # Analytical counts, independently evaluated from the approved layer widths.
    cnn_count = 64*(3+1)+128*(64*3+1)+256*(128*3+1)+128*(18*256+1)+64*(128+1)+2*(64+1)
    parameter_counts = {"cnn":cnn_count,"mlp8":8*(24+1)+2*(8+1)}
    require(parameter_counts == {"cnn":721858,"mlp8":218}, "Analytical parameter-count error")
    acquisition_ids = {s:f"A{i+1}" for i,s in enumerate(order)}
    for name,expected_job in jobs.items():
        record_path = run_dir/"jobs"/(name+".complete.json")
        prediction_path = run_dir/"jobs"/(name+".predictions.csv")
        progress_path = run_dir/"jobs"/(name+".progress.json")
        require(prediction_path.is_file() and progress_path.is_file(), "Missing complete-job files: "+name)
        input_files += [prediction_path,progress_path]
        initial_hashes.update({str(path):sha(path) for path in (prediction_path,progress_path)})
        record, progress = read_json(record_path), read_json(progress_path)
        configuration = record["configuration"]
        require(record["status"] == progress["status"] == "complete", "Incomplete job promoted: "+name)
        require(record["last_completed_epoch"] == 3000 and record["job_id"] == name, "Final epoch/identity differs")
        require(configuration["job"] == expected_job, "Actual training/test membership differs: "+name)
        require(configuration["training"] == spec and configuration["engine"] == engine, "Per-job settings differ")
        require(configuration["dataset_fingerprint"] == dataset_hash, "Per-job dataset fingerprint differs")
        job_hash = hashlib.sha256(canonical(configuration)).hexdigest()
        require(record["job_hash"] == progress["job_hash"] == job_hash, "Job/configuration hash mismatch")
        require(record["prediction_csv_sha256"] == progress["prediction_csv_sha256"] == sha(prediction_path),
                "Completed CSV is not the hash-bound prediction artifact")
        require(record["fitted_parameters"] == parameter_counts[expected_job["model"]], "Parameter count differs")
        fold_key = (expected_job["evaluation"],expected_job["fold"])
        if fold_key not in scale_cache:
            scale_cache[fold_key] = expected_scales(x,y,expected_job["train"])
        require(set(record["scales"]) == set(scale_cache[fold_key]), "Scaler field set differs")
        for field,expected in scale_cache[fold_key].items():
            close(record["scales"][field],expected,"training_scaler:"+field,0.0)
        sequence_key = (*fold_key,expected_job["seed"])
        if sequence_key not in sequence_cache:
            sequence_cache[sequence_key] = expected_batch_records(expected_job["train"],expected_job["seed"])
        require([v["epoch"] for v in record["snapshots"]] == list(EPOCHS), "A fixed snapshot is missing or replaced")
        for snapshot in record["snapshots"]:
            expected = sequence_cache[sequence_key][snapshot["epoch"]]
            require(snapshot["batch_order_sha256"] == expected["batch_order_sha256"]
                    and snapshot["optimizer_updates"] == expected["optimizer_updates"],
                    "Recorded permutation digest/update count does not match declared schedule")
            require(math.isfinite(snapshot["standardised_training_mse"])
                    and snapshot["standardised_training_mse"] >= 0, "Invalid training-loss metadata")
            key = (*sequence_key,snapshot["epoch"])
            pair_records.setdefault(key,{})[expected_job["model"]] = expected
        rows,fields = read_csv(prediction_path)
        require(fields == PRED_FIELDS, "Prediction schema drift")
        ids = [(int(r["epoch"]),int(r["row_index"])) for r in rows]
        expected_ids = {(epoch,row) for epoch in EPOCHS for row in expected_job["test"]}
        require(len(ids) == len(expected_ids) and set(ids) == expected_ids, "Missing/duplicate snapshot rows")
        for row in rows:
            row_index = int(row["row_index"])
            require(row["protocol_id"] == PROTOCOL and row["job_id"] == name
                    and row["model"] == expected_job["model"] and row["evaluation"] == expected_job["evaluation"]
                    and row["fold"] == expected_job["fold"] and int(row["seed"]) == expected_job["seed"],
                    "Per-observation model/seed/fold identity differs")
            require(row["archive_session"] == sessions[row_index]
                    and row["acquisition"] == acquisition_ids[sessions[row_index]]
                    and int(row["train_rows"]) == len(expected_job["train"]), "Per-observation acquisition/train size differs")
            truth = np.array([float(row["true_x_um"]),float(row["true_y_um"])])
            pred = np.array([float(row["pred_x_um"]),float(row["pred_y_um"])])
            residual = np.array([float(row["residual_x_um"]),float(row["residual_y_um"])])
            close(truth,y[row_index],"held_out_target_identity",0.0)
            close(pred-truth,residual,"residual_identity")
            close(math.hypot(*residual),float(row["error_um"]),"Euclidean_error_identity")
        predictions.extend(rows)
    require(len(predictions) == 12900 and len(pair_records) == 54, "Snapshot/observation total differs")
    require(all(set(v) == set(MODELS) and v["cnn"] == v["mlp8"] for v in pair_records.values()),
            "Architecture pair batch/update mismatch")
    combined,_ = read_csv(run_dir/"matched_observation_predictions.csv")
    require(keyed(combined,["job_id","epoch","row_index"]) == keyed(predictions,["job_id","epoch","row_index"]),
            "Runner combined CSV differs from the complete job CSVs")
    grouped = defaultdict(list)
    for row in predictions:
        key = (row["evaluation"],int(row["epoch"]),row["model"],int(row["seed"]),row["archive_session"])
        grouped[key].append((int(row["row_index"]),float(row["error_um"])))
    acquisition_rows, error_by_group = [], {}
    for key,values in sorted(grouped.items()):
        evaluation,epoch,model,seed,session = key
        mean = float(np.mean([v for _,v in sorted(values)]))
        error_by_group[key] = mean
        acquisition_rows.append({"evaluation":evaluation,"epoch":epoch,"model":model,"seed":seed,
            "acquisition":acquisition_ids[session],"archive_session":session,
            "observations":len(values),"mean_error_um":mean})
    require(len(acquisition_rows) == 132, "Expected 132 acquisition/seed/budget rows")
    compare_summary(run_dir/"matched_acquisition_seed_summary.csv",acquisition_rows,
                    ["evaluation","epoch","model","seed","archive_session"],["mean_error_um"],
                    ["acquisition","observations"])
    summary_rows, paired_rows, table = [], [], []
    for evaluation in ("loao","archive_replay"):
        groups = order if evaluation == "loao" else order[5:]
        for epoch in EPOCHS:
            means = {model:np.array([np.mean([error_by_group[(evaluation,epoch,model,seed,s)]
                          for seed in SEEDS]) for s in groups]) for model in MODELS}
            for model in MODELS:
                summary_rows.append({"evaluation":evaluation,"epoch":epoch,"model":model,
                    "acquisitions":len(groups),"seeds_per_acquisition":3,
                    "equal_acquisition_mean_error_um":float(means[model].mean()),
                    "acquisition_sd_um":float(means[model].std(ddof=1))})
            differences = means["mlp8"]-means["cnn"]
            for group,delta in zip(groups,differences):
                paired_rows.append({"evaluation":evaluation,"epoch":epoch,"archive_session":group,
                                    "mlp8_minus_cnn_seed_mean_error_um":float(delta)})
            table.append({"evaluation":evaluation,"epoch":epoch,"acquisitions":len(groups),
                "seeds_per_acquisition":3,"cnn_mean_um":float(means["cnn"].mean()),
                "cnn_between_acquisition_sd_um":float(means["cnn"].std(ddof=1)),
                "mlp8_mean_um":float(means["mlp8"].mean()),
                "mlp8_between_acquisition_sd_um":float(means["mlp8"].std(ddof=1)),
                "paired_mlp8_minus_cnn_mean_um":float(differences.mean()),
                "paired_between_acquisition_sd_um":float(differences.std(ddof=1))})
    compare_summary(run_dir/"matched_equal_acquisition_summary.csv",summary_rows,
        ["evaluation","epoch","model"],["equal_acquisition_mean_error_um","acquisition_sd_um"],
        ["acquisitions","seeds_per_acquisition"])
    compare_summary(run_dir/"matched_paired_acquisition_differences.csv",paired_rows,
        ["evaluation","epoch","archive_session"],["mlp8_minus_cnn_seed_mean_error_um"])
    require(len(summary_rows) == 8 and len(paired_rows) == 22 and len(table) == 4, "All fixed budgets must remain reported")
    for key,value in {"planned_fits":54,"snapshots_per_fit":2,"prediction_rows":12900,
                      "all_batch_sequence_pairs_equal":True,"all_budgets_reported":[1500,3000],
                      "failed_historical_gate_replaced":False}.items():
        require(run_report[key] == value, "Runner report field differs: "+key)
    reported = keyed(run_report["summaries"],["evaluation","epoch","model"])
    expected = keyed(summary_rows,["evaluation","epoch","model"])
    require(set(reported) == set(expected), "Runner JSON summary coverage differs")
    for key in expected:
        for field in ("equal_acquisition_mean_error_um","acquisition_sd_um"):
            close(reported[key][field],expected[key][field],"runner_JSON:"+field)
    legacy_path = ROOT/"results/study_l_supplementary/loso_sample_predictions.csv"
    require(sha(legacy_path) == LEGACY_PREDICTION_SHA256, "The retained legacy prediction table changed")
    initial_hashes[str(legacy_path)] = LEGACY_PREDICTION_SHA256
    legacy,_ = read_csv(legacy_path)
    legacy = keyed([r for r in legacy if r["model"] == "mlp8"],["seed","row_index"])
    current = keyed([r for r in predictions if r["evaluation"] == "loao"
                     and r["model"] == "mlp8" and int(r["epoch"]) == 1500],["seed","row_index"])
    require(set(legacy) == set(current) and len(current) == 2457, "Legacy-MLP comparison coverage differs")
    legacy_differences = []
    for key in sorted(current,key=lambda k:tuple(map(int,k))):
        new,old = current[key],legacy[key]
        delta = [float(new[f])-float(old[f]) for f in ("pred_x_um","pred_y_um")]
        require(new["archive_session"] == old["session"], "Legacy comparison acquisition differs")
        close([float(new[f]) for f in ("true_x_um","true_y_um")],
              [float(old[f]) for f in ("true_x_um","true_y_um")],"legacy_target_identity",0.0)
        legacy_differences.append({"seed":int(key[0]),"row_index":int(key[1]),
            "acquisition":new["acquisition"],"matched_pred_x_um":float(new["pred_x_um"]),
            "matched_pred_y_um":float(new["pred_y_um"]),"legacy_pred_x_um":float(old["pred_x_um"]),
            "legacy_pred_y_um":float(old["pred_y_um"]),"delta_x_um":delta[0],"delta_y_um":delta[1],
            "delta_euclidean_error_um":float(new["error_um"])-float(old["euclidean_error_um"]),
            "exact_coordinate_match":delta == [0.0,0.0]})
    delta_values = np.array([[r["delta_x_um"],r["delta_y_um"]] for r in legacy_differences])
    maximum = float(np.abs(delta_values).max())
    exact = maximum == LEGACY_MATCH_ATOL
    legacy_by_group = []
    for seed in SEEDS:
        for acquisition in [f"A{i+1}" for i in range(8)]:
            part = [r for r in legacy_differences if r["seed"] == seed and r["acquisition"] == acquisition]
            legacy_by_group.append({"seed":seed,"acquisition":acquisition,"observations":len(part),
                "nonexact_rows":sum(not r["exact_coordinate_match"] for r in part),
                "maximum_absolute_coordinate_difference_um":max(max(abs(r["delta_x_um"]),abs(r["delta_y_um"])) for r in part),
                "mean_euclidean_error_difference_um":float(np.mean([r["delta_euclidean_error_um"] for r in part]))})
    close(run_report["legacy_mlp8_1500_sanity"]["maximum_coordinate_difference_um"],maximum,
          "runner_legacy_maximum_difference",0.0)
    require(run_report["legacy_mlp8_1500_sanity"]["rows"] == 2457
            and run_report["legacy_mlp8_1500_sanity"]["exact_coordinate_match"] == exact,
            "Runner legacy-comparison coverage/exact-match flag differs")
    input_files.append(legacy_path)
    hashes = {str(p):sha(p) for p in input_files}
    require(hashes == initial_hashes, "An input artifact changed during the independent verification")
    for name,expected_hash in APPROVED_SOURCES.items():
        require(sha(ROOT/name) == expected_hash, "A frozen source changed during verification: "+name)
    report = {"status":"VERIFIED_COMPLETE_FIXED_CONTROLS" if exact else "CORE_VERIFIED_LEGACY_DIFFERENCES_REQUIRE_INVESTIGATION",
        "verified_utc":datetime.now(timezone.utc).isoformat(),"verifier_sha256":sha(Path(__file__)),
        "approved_runner_sha256":APPROVED_SOURCES["paper/figures/scientific_controls_20260912/matched_controls.py"],
        "fit_count":54,"snapshot_count":108,"prediction_rows":12900,"acquisition_seed_rows":132,
        "equal_acquisition_rows":8,"paired_acquisition_rows":22,
        "fixed_budgets":list(EPOCHS),"parameter_counts":parameter_counts,
        "training_scaler_comparison":"exact equality to training-only mean and population SD for every job",
        "batch_order_verification":"27 independent 3000-epoch permutation schedules regenerated; 54 paired snapshot digests/update counts match",
        "all_source_hashes_match_approved":True,"no_numerical_failure_or_active_partial_promoted":True,
        "retained_interrupted_artifact_count":len(list((run_dir/"jobs").glob("*.interrupted-*"))),
        "numeric_tolerance_um":NUMERIC_ATOL_UM,"maximum_observed_differences":NUMERIC_DIFFERENCES,
        "legacy_mlp8_1500":{"rows":2457,"coordinate_values":4914,"exact_match_required_atol":LEGACY_MATCH_ATOL,
            "exact_match":exact,"maximum_absolute_coordinate_difference_um":maximum,
            "coordinate_difference_rms_um":float(np.sqrt(np.mean(delta_values**2))),
            "nonexact_observation_rows":sum(not r["exact_coordinate_match"] for r in legacy_differences),
            "maximum_error_difference_um":max(abs(r["delta_euclidean_error_um"]) for r in legacy_differences),
            "by_seed_acquisition":legacy_by_group,
            "action_if_nonzero":"Retain all differences, inspect environment/batch/scalers and source evidence; no threshold relaxation, checkpoint selection or silent retraining."},
        "table2_rows":table,"table2_written":exact,"input_sha256":hashes,
        "claim_boundary":"Same training-only preprocessing and per-fold updates, not equal FLOPs/wall time; fixed post-hoc comparison on known acquisitions, not independent validation or rehabilitation of the old failed gate.",
        "source_files_or_manuscript_modified":False,"model_training_performed":False}
    # Only verifier-owned outputs are written after the entire core audit succeeds.
    write_csv(WORK/"matched_controls_legacy_mlp1500_differences.csv",legacy_differences)
    (WORK/"matched_controls_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if exact:
        write_csv(WORK/"matched_controls_table2.csv",table)
        (WORK/"matched_controls_table2.tex").write_text(table_tex(table),encoding="utf-8")
    else:
        print("Legacy MLP1500 predictions are not exactly identical. Full differences were retained; no manuscript table was generated.",file=sys.stderr)
    print(json.dumps({"status":report["status"],"fits":54,"snapshots":108,"prediction_rows":12900,
                     "legacy_exact_match":exact,"legacy_maximum_difference_um":maximum,
                     "table2_written":exact},indent=2))
    return 0 if exact else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True,
                        help="Absolute path to the coordinator-confirmed complete matched_runs directory")
    args = parser.parse_args()
    try:
        return verify(args.run_dir)
    except IncompleteRun as error:
        print("INCOMPLETE: "+str(error),file=sys.stderr)
        return 2
    except (ValueError,KeyError,FileNotFoundError) as error:
        print("VERIFICATION FAILED: "+str(error),file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
