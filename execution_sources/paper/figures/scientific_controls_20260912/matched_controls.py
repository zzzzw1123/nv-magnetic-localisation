"""Fixed post-hoc Study L architecture controls; never replaces frozen L34.

Commands:
  python matched_controls.py plan
  python matched_controls.py self-test
  python matched_controls.py run --confirm-formal-run --workers 6

Only ``run --confirm-formal-run`` fits the released dataset. Both architectures
use training-only input/target standardisation, the same Adam settings, epochs,
minibatches and per-seed batch sequence. Every fit reports both fixed snapshots;
there is no test-based checkpoint selection or reuse of historical weights.
This matches data exposure/optimizer updates, not FLOPs or wall-clock cost.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import signal
import sys
import tempfile
import threading
import time

# Set before scientific-library imports; worker count is not a BLAS thread count.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError("The process has already started a non-single-thread PyTorch interop runtime.")

WORK = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from qml_microrobot.study_l30_author_cnn import (  # noqa: E402
    AuthorLocalizationCNN, fit_author_cnn, load_author_data,
    predict_author_cnn, set_deterministic_seed, trainable_parameters,
)
from qml_microrobot.study_l31_compact import (  # noqa: E402
    CompactMLP, fit_compact_mlp, predict_compact_mlp,
)

PROTOCOL_ID = "STUDY_L_MATCHED_CONTROLS_POSTHOC_20260912_V1"
SEEDS = (20260804, 20260805, 20260806)
MODELS = ("cnn", "mlp8")
PARAMETERS = {"cnn": 721858, "mlp8": 218}
SPEC = {
    "protocol_id": PROTOCOL_ID,
    "role": "fixed post-hoc controls; no new independent validation",
    "epochs": 3000,
    "snapshots": [1500, 3000],
    "batch_size": 128,
    "learning_rate": 0.005,
    "betas": [0.9, 0.999],
    "epsilon": 1e-7,
    "weight_decay": 0.0,
    "initialisation": "existing architecture classes: Xavier-uniform weights, zero biases",
    "standardisation": "training-only means and population SD (ddof=0) for all 24 inputs and 2 targets; zero SD becomes 1",
    "batch_order": "dedicated CPU torch.Generator(seed+1), randperm each epoch; no dropped final batch",
    "checkpoint_selection": "none; both fixed snapshots must be reported",
    "historical_states_reused": False,
    "frozen_L34_overwritten": False,
    "old_failed_gate_retested_or_replaced": False,
    "seeds_are_independent_units": False,
    "equal_compute_or_wall_time_claim": False,
}
PRED_FIELDS = [
    "protocol_id", "job_id", "model", "evaluation", "fold", "seed", "epoch",
    "row_index", "acquisition", "archive_session", "train_rows",
    "true_x_um", "true_y_um", "pred_x_um", "pred_y_um",
    "residual_x_um", "residual_y_um", "error_um",
]
_WORKER_DATA = None
_WORKER_STOP = None
_WORKER_ENGINE = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(canonical({"shape": values.shape, "dtype": str(values.dtype)})
                          + values.tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path, rows, fields):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(WORK.resolve()) or not path.name.startswith("matched_"):
        raise ValueError("Output must be a matched_* directory inside this new scientific-controls directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Data:
    x: np.ndarray
    y: np.ndarray
    sessions: np.ndarray
    session_order: list[str]
    fingerprint: str


def validated_data(x, y, sessions):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    sessions = np.asarray(sessions, dtype=str)
    if x.ndim != 2 or x.shape[1] != 24 or y.shape != (len(x), 2) or sessions.shape != (len(x),):
        raise ValueError("Expected N×24 features, N×2 targets, and one acquisition id per observation.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite features or targets.")
    order = list(dict.fromkeys(sessions.tolist()))
    if len(order) != 8:
        raise ValueError("This protocol requires exactly eight archive-ordered acquisitions.")
    fingerprint = hashlib.sha256(canonical({
        "x": array_digest(x), "y": array_digest(y), "sessions": sessions.tolist(),
    })).hexdigest()
    x.setflags(write=False)
    y.setflags(write=False)
    sessions.setflags(write=False)
    return Data(x, y, sessions, order, fingerprint)


def released_data():
    source = load_author_data(ROOT)
    data = validated_data(source.features, source.positions_um, source.session_ids)
    counts = [int(np.sum(data.sessions == s)) for s in data.session_order]
    if len(data.x) != 819 or counts != [102, 104, 102, 204, 51, 52, 102, 102]:
        raise ValueError("Frozen Study L acquisition inventory changed.")
    return data


def engine_record():
    files = [Path(__file__), ROOT / "src/qml_microrobot/study_l30_author_cnn.py",
             ROOT / "src/qml_microrobot/study_l31_compact.py",
             ROOT / "src/qml_microrobot/study_l_session.py",
             ROOT / "src/qml_microrobot/study_l_nv.py"]
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__),
            "device": "cpu", "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "source_sha256": {str(p.relative_to(ROOT)): digest_file(p) for p in files}}


def make_jobs(data):
    folds = []
    for i, session in enumerate(data.session_order):
        folds.append(("loao", "A" + str(i + 1),
                      np.flatnonzero(data.sessions != session).tolist(),
                      np.flatnonzero(data.sessions == session).tolist()))
    development = np.isin(data.sessions, data.session_order[:5])
    folds.append(("archive_replay", "A6-A8", np.flatnonzero(development).tolist(),
                  np.flatnonzero(~development).tolist()))
    jobs = []
    for evaluation, fold, train, test in folds:
        if set(train) & set(test) or sorted(train + test) != list(range(len(data.x))):
            raise ValueError("Invalid outer split.")
        for seed in SEEDS:
            for model in MODELS:
                jobs.append({"job_id": f"{evaluation}_{fold}_{model}_s{seed}",
                             "evaluation": evaluation, "fold": fold, "model": model,
                             "seed": seed, "train": train, "test": test})
    assert len(jobs) == 54 and len({j["job_id"] for j in jobs}) == 54
    return jobs


def job_config(job, data, spec, engine):
    return {"job": job, "training": spec, "dataset_fingerprint": data.fingerprint,
            "engine": engine}


def config_hash(config):
    return hashlib.sha256(canonical(config)).hexdigest()


def mean_scale(values):
    mean, scale = values.mean(axis=0), values.std(axis=0, ddof=0)
    scale = np.where(scale == 0, 1.0, scale)
    return mean, scale


def standardised_training(data, train):
    train = np.asarray(train, dtype=np.int64)
    xm, xs = mean_scale(data.x[train])
    ym, ys = mean_scale(data.y[train])
    x = torch.as_tensor((data.x[train] - xm) / xs, dtype=torch.float32)
    y = torch.as_tensor((data.y[train] - ym) / ys, dtype=torch.float32)
    return x, y, {"feature_mean": xm, "feature_scale": xs,
                  "target_mean": ym, "target_scale": ys}


def make_model(name, seed):
    set_deterministic_seed(seed)
    torch.set_num_threads(1)
    model = AuthorLocalizationCNN() if name == "cnn" else CompactMLP(8)
    if name not in MODELS or trainable_parameters(model) != PARAMETERS[name]:
        raise ValueError("Architecture or parameter-count drift.")
    return model


@torch.inference_mode()
def predict(model, features, scales):
    model.eval()
    x = torch.as_tensor((features - scales["feature_mean"]) / scales["feature_scale"],
                        dtype=torch.float32)
    return model(x).cpu().numpy().astype(float) * scales["target_scale"] + scales["target_mean"]


def train_snapshots(data, job, spec, stop, on_snapshot, heartbeat=100):
    """Common loop; no model-specific optimiser, scaling or batch-order branch."""
    x, y, scales = standardised_training(data, job["train"])
    model = make_model(job["model"], job["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"],
                                 betas=tuple(spec["betas"]), eps=spec["epsilon"],
                                 weight_decay=spec["weight_decay"])
    generator = torch.Generator(device="cpu").manual_seed(job["seed"] + 1)
    order_digest = hashlib.sha256()
    snapshots = []
    updates = 0
    started = time.perf_counter()
    for epoch in range(1, spec["epochs"] + 1):
        if stop.is_set():
            return {"status": "interrupted", "last_completed_epoch": epoch - 1,
                    "snapshots": snapshots, "scales": {k: v.tolist() for k, v in scales.items()}}
        model.train()
        order = torch.randperm(len(x), generator=generator)
        # Global row ids make the digest independently comparable between models.
        order_digest.update(np.asarray(job["train"], dtype=np.int64)[order.numpy()].tobytes())
        for start in range(0, len(order), spec["batch_size"]):
            batch = order[start:start + spec["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x[batch]) - y[batch]) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {job['job_id']} at epoch {epoch}")
            loss.backward()
            optimizer.step()
            updates += 1
        if epoch in spec["snapshots"]:
            prediction = predict(model, data.x[job["test"]], scales)
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"Non-finite snapshot prediction at epoch {epoch}.")
            with torch.inference_mode():
                train_mse = float(torch.mean((model(x) - y) ** 2).item())
            if not math.isfinite(train_mse):
                raise FloatingPointError(f"Non-finite snapshot training MSE at epoch {epoch}.")
            snapshot = {"epoch": epoch, "optimizer_updates": updates,
                        "batch_order_sha256": order_digest.hexdigest(),
                        "standardised_training_mse": train_mse,
                        "concurrent_fit_wall_sec_not_resource_benchmark": time.perf_counter() - started}
            on_snapshot(epoch, prediction, snapshot)
            snapshots.append(snapshot)
        if heartbeat and (epoch % heartbeat == 0 or epoch == spec["epochs"]):
            print(f"{job['job_id']} epoch {epoch}/{spec['epochs']} elapsed {time.perf_counter()-started:.1f}s", flush=True)
    return {"status": "complete", "last_completed_epoch": spec["epochs"],
            "snapshots": snapshots, "scales": {k: v.tolist() for k, v in scales.items()}}


def job_paths(output, job):
    base = output / "jobs" / job["job_id"]
    return {"csv": base.with_suffix(".predictions.csv"),
            "complete": base.with_suffix(".complete.json"),
            "partial": base.with_suffix(".predictions.partial.csv"),
            "progress": base.with_suffix(".progress.json"),
            "failure": base.with_suffix(".numerical_failure.json"),
            "lock": base.with_suffix(".lock")}


def completed_job(output, job, data, spec, engine):
    paths = job_paths(output, job)
    if paths["failure"].exists():
        raise RuntimeError(f"Terminal numerical failure recorded for {job['job_id']}; automatic retry is prohibited. Inspect {paths['failure']} and retain any earlier snapshot.")
    if not paths["complete"].exists():
        return None
    record = json.loads(paths["complete"].read_text(encoding="utf-8"))
    expected_hash = config_hash(job_config(job, data, spec, engine))
    if record.get("status") != "complete" or record.get("job_hash") != expected_hash:
        raise ValueError(f"Completed job has incompatible configuration: {job['job_id']}. Use a new matched_* output directory.")
    if not paths["csv"].exists() or digest_file(paths["csv"]) != record.get("prediction_csv_sha256"):
        raise ValueError(f"Completed prediction file is missing or changed: {job['job_id']}")
    rows = read_csv(paths["csv"])
    expected = {(epoch, row) for epoch in spec["snapshots"] for row in job["test"]}
    keys = [(int(r["epoch"]), int(r["row_index"])) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Completed snapshot coverage is incomplete or duplicated.")
    if [s["epoch"] for s in record["snapshots"]] != spec["snapshots"]:
        raise ValueError("Completed snapshot metadata does not match the fixed budgets.")
    for row in rows:
        if row["job_id"] != job["job_id"] or row["model"] != job["model"] or int(row["seed"]) != job["seed"]:
            raise ValueError("Prediction row identity mismatch.")
        numeric = [float(row[k]) for k in PRED_FIELDS[11:]]
        if not np.isfinite(numeric).all():
            raise ValueError("Non-finite completed prediction record.")
        row_id = int(row["row_index"])
        truth = np.array([float(row["true_x_um"]), float(row["true_y_um"])])
        pred = np.array([float(row["pred_x_um"]), float(row["pred_y_um"])])
        residual = np.array([float(row["residual_x_um"]), float(row["residual_y_um"])])
        if (not np.array_equal(truth, data.y[row_id])
                or row["archive_session"] != data.sessions[row_id]
                or not np.allclose(pred-truth, residual, rtol=0, atol=1e-9)
                or not np.isclose(np.linalg.norm(residual), float(row["error_um"]), rtol=0, atol=1e-9)):
            raise ValueError("Completed prediction truth/residual identity mismatch.")
    return record


def run_job(output, job, data, spec, engine, stop, heartbeat=100):
    paths = job_paths(output, job)
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)
    existing = completed_job(output, job, data, spec, engine)
    if existing is not None:
        return {"job_id": job["job_id"], "status": "resumed_complete"}
    if stop.is_set():
        return {"job_id": job["job_id"], "status": "interrupted_before_start"}
    jobhash = config_hash(job_config(job, data, spec, engine))
    # Exclusive lock prevents two invocations from writing the same job. A stale
    # lock after forced termination requires explicit operator inspection/removal.
    with paths["lock"].open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_utc": utc_now(), "job_hash": jobhash}, handle)
    recorded_snapshots = []
    try:
        # Incomplete runs restart from epoch zero; retain, never promote, their files.
        stamp = str(time.time_ns())
        for key in ("partial", "progress", "csv"):
            if paths[key].exists():
                paths[key].rename(paths[key].with_name(paths[key].name + ".interrupted-" + stamp))
        session_lookup = {s: "A" + str(i + 1) for i, s in enumerate(data.session_order)}
        with paths["partial"].open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PRED_FIELDS)
            writer.writeheader()

            def snapshot(epoch, prediction, stats):
                for row_id, pred in zip(job["test"], prediction):
                    truth = data.y[row_id]
                    residual = pred - truth
                    writer.writerow(dict(zip(PRED_FIELDS, [
                        PROTOCOL_ID, job["job_id"], job["model"], job["evaluation"],
                        job["fold"], job["seed"], epoch, row_id,
                        session_lookup[data.sessions[row_id]], data.sessions[row_id], len(job["train"]),
                        *truth.tolist(), *pred.tolist(), *residual.tolist(), float(np.linalg.norm(residual)),
                    ])))
                handle.flush()
                os.fsync(handle.fileno())
                atomic_json(paths["progress"], {"status": "incomplete", "job_hash": jobhash,
                            "last_snapshot": stats, "written_utc": utc_now()})
                recorded_snapshots.append(stats)

            outcome = train_snapshots(data, job, spec, stop, snapshot, heartbeat)
        outcome.update({"job_id": job["job_id"], "job_hash": jobhash,
                        "configuration": job_config(job, data, spec, engine),
                        "completed_or_interrupted_utc": utc_now(),
                        "fitted_parameters": PARAMETERS[job["model"]]})
        if outcome["status"] != "complete":
            atomic_json(paths["progress"], outcome)
            return {"job_id": job["job_id"], "status": "interrupted"}
        os.replace(paths["partial"], paths["csv"])
        outcome["prediction_csv_sha256"] = digest_file(paths["csv"])
        atomic_json(paths["complete"], outcome)
        completed_job(output, job, data, spec, engine)
        atomic_json(paths["progress"], {"status": "complete", "job_hash": jobhash,
                    "last_snapshot": outcome["snapshots"][-1],
                    "prediction_csv_sha256": outcome["prediction_csv_sha256"]})
        return {"job_id": job["job_id"], "status": "complete"}
    except FloatingPointError as error:
        failure = {"status": "numerical_failure", "terminal": True,
                   "automatic_retry_allowed": False, "job_id": job["job_id"],
                   "job_hash": jobhash, "configuration": job_config(job, data, spec, engine),
                   "message": str(error), "recorded_snapshots": recorded_snapshots,
                   "earlier_snapshot_retained": bool(recorded_snapshots),
                   "partial_prediction_csv_sha256": digest_file(paths["partial"]) if paths["partial"].exists() else None,
                   "recorded_utc": utc_now()}
        atomic_json(paths["failure"], failure)
        atomic_json(paths["progress"], failure)
        raise
    finally:
        # This process created this exact lock file; no directory deletion is used.
        paths["lock"].unlink(missing_ok=True)


def initialise_worker(stop, expected_data_hash, expected_engine):
    global _WORKER_DATA, _WORKER_STOP, _WORKER_ENGINE
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    _WORKER_DATA, _WORKER_STOP, _WORKER_ENGINE = released_data(), stop, engine_record()
    if _WORKER_DATA.fingerprint != expected_data_hash or _WORKER_ENGINE != expected_engine:
        raise RuntimeError("Worker data/code/environment differs from the reviewed run manifest.")


def worker_entry(output, job, heartbeat):
    return run_job(Path(output), job, _WORKER_DATA, SPEC, _WORKER_ENGINE,
                   _WORKER_STOP, heartbeat)


def toy_pool_probe(payload):
    """Spawn/persistence check using synthetic arrays only."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    output, x, y, sessions, job, spec, engine = payload
    data = validated_data(x, y, sessions)
    if engine_record() != engine:
        raise RuntimeError("Synthetic worker environment mismatch.")
    outcome = run_job(Path(output), job, data, spec, engine, threading.Event(), heartbeat=0)
    return {**outcome, "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads()}


def aggregate(output, jobs, data, spec, engine):
    """All models, seeds, folds and both budgets; no winning-budget selection."""
    predictions, acquisition_rows, batch_sequences = [], [], {}
    for job in jobs:
        record = completed_job(output, job, data, spec, engine)
        if record is None:
            raise ValueError("Aggregation requires every planned complete job.")
        rows = read_csv(job_paths(output, job)["csv"])
        predictions.extend(rows)
        for snapshot in record["snapshots"]:
            key = (job["evaluation"], job["fold"], job["seed"], snapshot["epoch"])
            identity = (snapshot["batch_order_sha256"], snapshot["optimizer_updates"])
            if key in batch_sequences and batch_sequences[key] != identity:
                raise ValueError("CNN and MLP8 did not use identical batch order/update count.")
            batch_sequences[key] = identity
        for epoch in spec["snapshots"]:
            for session in sorted({r["archive_session"] for r in rows}):
                part = [r for r in rows if int(r["epoch"]) == epoch and r["archive_session"] == session]
                acquisition_rows.append({"evaluation": job["evaluation"], "epoch": epoch,
                    "model": job["model"], "seed": job["seed"],
                    "acquisition": part[0]["acquisition"], "archive_session": session,
                    "observations": len(part), "mean_error_um": float(np.mean([float(r["error_um"]) for r in part]))})
    summaries, differences = [], []
    for evaluation in ("loao", "archive_replay"):
        for epoch in spec["snapshots"]:
            selected = [r for r in acquisition_rows if r["evaluation"] == evaluation and r["epoch"] == epoch]
            sessions = sorted({r["archive_session"] for r in selected})
            model_means = {}
            for model in MODELS:
                means = [float(np.mean([r["mean_error_um"] for r in selected
                          if r["archive_session"] == s and r["model"] == model])) for s in sessions]
                model_means[model] = means
                summaries.append({"evaluation": evaluation, "epoch": epoch, "model": model,
                    "acquisitions": len(sessions), "seeds_per_acquisition": len(SEEDS),
                    "equal_acquisition_mean_error_um": float(np.mean(means)),
                    "acquisition_sd_um": float(np.std(means, ddof=1))})
            for i, session in enumerate(sessions):
                differences.append({"evaluation": evaluation, "epoch": epoch, "archive_session": session,
                    "mlp8_minus_cnn_seed_mean_error_um": model_means["mlp8"][i] - model_means["cnn"][i]})
    atomic_csv(output / "matched_observation_predictions.csv", predictions, PRED_FIELDS)
    atomic_csv(output / "matched_acquisition_seed_summary.csv", acquisition_rows, list(acquisition_rows[0]))
    atomic_csv(output / "matched_equal_acquisition_summary.csv", summaries, list(summaries[0]))
    atomic_csv(output / "matched_paired_acquisition_differences.csv", differences, list(differences[0]))
    legacy = read_csv(ROOT / "results/study_l_supplementary/loso_sample_predictions.csv")
    legacy = {(int(r["seed"]), int(r["row_index"])): r for r in legacy if r["model"] == "mlp8"}
    comparison = [r for r in predictions if r["evaluation"] == "loao" and r["model"] == "mlp8" and int(r["epoch"]) == 1500]
    gap = [abs(float(r[k]) - float(legacy[(int(r["seed"]), int(r["row_index"]))][k]))
           for r in comparison for k in ("pred_x_um", "pred_y_um")]
    report = {"status": "COMPLETE_FIXED_POSTHOC_CONTROLS_NOT_INDEPENDENT_VALIDATION",
              "planned_fits": len(jobs), "snapshots_per_fit": len(spec["snapshots"]),
              "prediction_rows": len(predictions), "all_batch_sequence_pairs_equal": True,
              "all_budgets_reported": spec["snapshots"], "summaries": summaries,
              "legacy_mlp8_1500_sanity": {"purpose": "comparison only; no state reused or outcome selected",
                  "rows": len(comparison), "maximum_coordinate_difference_um": max(gap),
                  "exact_coordinate_match": bool(all(v == 0 for v in gap))},
              "old_L34_results_unchanged_by_this_script": True,
              "failed_historical_gate_replaced": False,
              "completed_utc": utc_now()}
    atomic_json(output / "matched_run_report.json", report)
    return report


def plan(data, engine):
    jobs = make_jobs(data)
    return {"status": "PROPOSAL_REQUIRES_EXPLICIT_FORMAL_START", "training": SPEC,
            "dataset_fingerprint": data.fingerprint, "engine": engine,
            "model_parameters": PARAMETERS, "planned_fits": len(jobs),
            "snapshots_per_fit": 2,
            "expected_prediction_rows": sum(len(j["test"]) for j in jobs) * 2,
            "all_job_ids": [j["job_id"] for j in jobs],
            "fold_rows": [{"evaluation": j["evaluation"], "fold": j["fold"],
                           "train": len(j["train"]), "test": len(j["test"]),
                           "updates_per_epoch": math.ceil(len(j["train"])/SPEC["batch_size"])}
                          for j in jobs if j["seed"] == SEEDS[0] and j["model"] == "cnn"],
            "claim_boundary": "Diagnostic fixed-setting architecture comparison; architecture choices and historical data were already known. No new arrays, prospectivity, formal equivalence, causal capacity attribution, or replacement of old gates.",
            "resume": "Only complete job with identical configuration/data/code/environment and matching CSV hash; ordinary partial jobs restart from zero and are retained as interrupted artifacts. Numerical failures are terminal, retain earlier snapshots, and reject automatic retry.",
            "timing": "Concurrent wall times are progress metadata, not resource benchmarks."}


def formal_run(args):
    if not args.confirm_formal_run:
        raise ValueError("Formal training is disabled without --confirm-formal-run after protocol/code review.")
    data, engine = released_data(), engine_record()
    output, jobs = safe_output(args.output), make_jobs(data)
    manifest = plan(data, engine)
    manifest["status"] = "AUTHOR_AUTHORIZED_FIXED_POSTHOC_RUN"
    manifest_path = output / "matched_protocol.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing output directory belongs to a different protocol/code/data configuration.")
    else:
        atomic_json(manifest_path, manifest)
    pending = [j for j in jobs if completed_job(output, j, data, SPEC, engine) is None]
    print(f"54 fits planned; {54-len(pending)} verified complete; {len(pending)} pending.", flush=True)
    context = mp.get_context("spawn")
    stop = context.Event()
    interrupted = False
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
             initializer=initialise_worker, initargs=(stop, data.fingerprint, engine))
    futures = []
    try:
        futures = [pool.submit(worker_entry, str(output), j, args.heartbeat_every) for j in pending]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping after the current epoch; incomplete jobs will not be resumed as checkpoints.", flush=True)
        stop.set()
        for future in futures:
            future.cancel()
    except BaseException:
        stop.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if interrupted:
        return 130
    report = aggregate(output, jobs, data, SPEC, engine)
    print(json.dumps(report, indent=2), flush=True)
    return 0


def self_test():
    """Small synthetic-only fits; no released-data fitting or old-state reuse."""
    rng = np.random.default_rng(7381)
    x = rng.normal(size=(24, 24))
    x[:, 23] = 7.0
    y = np.column_stack([1300 + 5*x[:, 0], 1800 + 9*x[:, 1]])
    data = validated_data(x, y, np.repeat(["toy_A" + str(i) for i in range(8)], 3))
    jobs = make_jobs(data)
    assert len(jobs) == 54
    spec = {**SPEC, "epochs": 4, "snapshots": [2, 4], "batch_size": 5}
    a = next(j for j in jobs if j["model"] == "cnn" and j["fold"] == "A1" and j["seed"] == SEEDS[0])
    b = {**a, "model": "mlp8", "job_id": a["job_id"].replace("cnn", "mlp8")}
    tx, ty, scales = standardised_training(data, a["train"])
    assert np.allclose(tx.numpy().mean(axis=0), 0, atol=1e-7)
    assert scales["feature_scale"][23] == 1 and torch.all(tx[:, 23] == 0)
    altered_x, altered_y = data.x.copy(), data.y.copy()
    altered_x[a["test"]] += 1e6
    altered_y[a["test"]] -= 1e6
    changed = validated_data(altered_x, altered_y, data.sessions)
    cx, cy, other_scales = standardised_training(changed, a["train"])
    assert torch.equal(tx, cx) and torch.equal(ty, cy)
    assert all(np.array_equal(scales[k], other_scales[k]) for k in scales)
    engine = engine_record()
    checked = ["54-job split plan", "training-only standardisation", "constant-feature handling"]
    stop = threading.Event()
    # The temporary directory is explicitly constrained before automatic cleanup.
    with tempfile.TemporaryDirectory(prefix="matched_selftest_", dir=WORK) as temporary:
        temporary = Path(temporary).resolve()
        assert temporary.parent == WORK.resolve() and temporary.name.startswith("matched_selftest_")
        output = safe_output(temporary / "matched_runs")
        records = []
        for job in [a, b]:
            assert run_job(output, job, data, spec, engine, stop, heartbeat=0)["status"] == "complete"
            record = completed_job(output, job, data, spec, engine)
            records.append(record)
            rows = read_csv(job_paths(output, job)["csv"])
            assert len(rows) == 2*len(job["test"])
            assert run_job(output, job, data, spec, engine, stop, heartbeat=0)["status"] == "resumed_complete"
            first = np.asarray([[float(r["pred_x_um"]), float(r["pred_y_um"])]
                                for r in rows if int(r["epoch"]) == 4])
            duplicate = []
            train_snapshots(data, job, spec, stop, lambda e, p, s: duplicate.append((e,p)), heartbeat=0)
            assert np.array_equal(first, duplicate[-1][1])
            if job["model"] == "mlp8":
                legacy = fit_compact_mlp(data.x[job["train"]], data.y[job["train"]], 8,
                           seed=job["seed"], epochs=4, batch_size=5, learning_rate=.005)
                legacy_prediction = predict_compact_mlp(legacy, data.x[job["test"]])
            else:
                legacy, _ = fit_author_cnn(tx.numpy(), ty.numpy(), np.arange(len(tx)),
                           seed=job["seed"], epochs=4, batch_size=5, learning_rate=.005)
                test_x = (data.x[job["test"]]-scales["feature_mean"])/scales["feature_scale"]
                legacy_prediction = predict_author_cnn(legacy, test_x)*scales["target_scale"]+scales["target_mean"]
            assert np.array_equal(first, legacy_prediction), job["model"]
        assert [(s["batch_order_sha256"],s["optimizer_updates"]) for s in records[0]["snapshots"]] == [(s["batch_order_sha256"],s["optimizer_updates"]) for s in records[1]["snapshots"]]
        checked += ["two fixed snapshots and prediction coverage", "cross-model identical batch order/update count",
                    "deterministic repeat predictions", "exact short-run agreement with both reused legacy loops",
                    "complete-job hash resume"]
        try:
            completed_job(output, a, data, {**spec,"learning_rate":.004}, engine)
            raise AssertionError("Mismatched configuration was accepted")
        except ValueError:
            pass
        with job_paths(output,a)["csv"].open("a",encoding="utf-8") as handle:
            handle.write("corrupted\n")
        try:
            completed_job(output, a, data, spec, engine)
            raise AssertionError("Corrupted snapshot was accepted")
        except ValueError:
            pass
        checked += ["configuration mismatch rejected", "CSV corruption rejected"]
        stopped = threading.Event()
        stopped.set()
        assert train_snapshots(data,b,spec,stopped,lambda *args: None,heartbeat=0)["status"] == "interrupted"
        checked.append("cooperative stop does not mark a job complete")
        pool_output = safe_output(temporary / "matched_pool_runs")
        payloads = [(str(pool_output),data.x,data.y,data.sessions,job,spec,engine) for job in (a,b)]
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
            pool_results = list(pool.map(toy_pool_probe, payloads))
        assert all(r["status"] == "complete" and r["intraop_threads"] == r["interop_threads"] == 1
                   for r in pool_results)
        checked.append("spawned synthetic workers are single-thread and persist complete jobs")
        from unittest.mock import patch
        original_train = train_snapshots

        def fail_after_first_snapshot(data, job, spec, stop, callback, heartbeat):
            def inject_failure(epoch, prediction, stats):
                callback(epoch, prediction, stats)
                raise FloatingPointError("Synthetic numerical failure after the first fixed snapshot")
            return original_train(data, job, spec, stop, inject_failure, heartbeat=0)

        failure_output = safe_output(temporary / "matched_failure_probe")
        with patch.dict(globals(), {"train_snapshots": fail_after_first_snapshot}):
            try:
                run_job(failure_output,b,data,spec,engine,stop,heartbeat=0)
                raise AssertionError("Synthetic numerical failure was swallowed")
            except FloatingPointError:
                pass
        failure_paths = job_paths(failure_output,b)
        failure = json.loads(failure_paths["failure"].read_text(encoding="utf-8"))
        assert failure["status"] == "numerical_failure" and failure["terminal"]
        assert [s["epoch"] for s in failure["recorded_snapshots"]] == [2]
        assert len(read_csv(failure_paths["partial"])) == len(b["test"])
        assert not failure_paths["complete"].exists()
        try:
            run_job(failure_output,b,data,spec,engine,stop,heartbeat=0)
            raise AssertionError("Terminal numerical failure was automatically retried")
        except RuntimeError as error:
            assert "Terminal numerical failure" in str(error)
        checked.append("numerical failure retains first snapshot and rejects automatic retry")
    print(json.dumps({"status":"SYNTHETIC_SELF_TEST_PASSED", "checks":checked,
                      "formal_data_training_performed":False},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("plan", help="Read released data and print the proposed fixed plan; no fitting or output writes")
    actions.add_parser("self-test", help="Only small synthetic tests; never fit the released dataset")
    run = actions.add_parser("run", help="Explicit authorised formal run; old Study L files are read-only")
    run.add_argument("--confirm-formal-run", action="store_true")
    run.add_argument("--workers", type=int, choices=(6,8), default=6)
    run.add_argument("--heartbeat-every", type=int, choices=(50,100), default=100)
    run.add_argument("--output", type=Path, default=WORK/"matched_runs")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(released_data(),engine_record()),indent=2))
        return 0
    if args.command == "self-test":
        self_test()
        return 0
    return formal_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
