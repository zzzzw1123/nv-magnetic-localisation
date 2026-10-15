"""Post-hoc 2x2x2 recipe decomposition on eight NV acquisitions.

The three-seed stage retains every fixed recipe, acquisition, and epoch
snapshot. It is an internal diagnostic on the already released chip15 data,
not prospective or cross-array validation. Frozen L34/L35 outputs are read
only and never replaced. Run ``check`` before ``run``; ``run`` appends only
complete fit records to one resumable CSV.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import hashlib
import itertools
import os
from pathlib import Path
import platform
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    if torch.get_num_interop_threads() != 1:
        raise

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from qml_microrobot.study_l30_author_cnn import (  # noqa: E402
    AuthorLocalizationCNN, set_deterministic_seed,
)
from qml_microrobot.study_l31_compact import CompactMLP  # noqa: E402
from recipe_factorial import data_and_folds, scale_from_training  # noqa: E402

PROTOCOL = "NV_RECIPE_FACTORIAL_POSTHOC_ALL8_THREESEED_V1"
SEEDS = (20260804, 20260805, 20260806)
FOLDS = tuple(f"A{i}" for i in range(1, 9))
MODELS = ("cnn", "mlp8")
SNAPSHOTS = (1500, 3000)
EPOCHS = 3000
BATCH = 128
LR = 0.005
OUTPUT = HERE / "recipe_full.csv"
FIELDS = (
    "protocol", "source_sha256", "data_sha256", "engine", "job_id",
    "fold", "model", "seed", "input_standardised", "target_standardised",
    "epoch", "train_rows", "test_rows", "optimizer_updates",
    "batch_order_sha256", "fit_wall_sec_diagnostic", "row_index",
    "true_x_um", "true_y_um", "pred_x_um", "pred_y_um", "error_um",
)


def source_hash():
    digest = hashlib.sha256()
    for path in (Path(__file__), HERE / "recipe_factorial.py",
                 ROOT / "src/qml_microrobot/study_l30_author_cnn.py",
                 ROOT / "src/qml_microrobot/study_l31_compact.py",
                 ROOT / "src/qml_microrobot/study_l_session.py",
                 ROOT / "src/qml_microrobot/study_l_nv.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def engine():
    return f"Python {platform.python_version()}; NumPy {np.__version__}; PyTorch {torch.__version__}; CPU"


def jobs():
    return list(itertools.product(FOLDS, MODELS, (False, True), (False, True), SEEDS))


def job_id(job):
    fold, model, input_std, target_std, seed = job
    return f"{fold}_{model}_x{int(input_std)}_y{int(target_std)}_s{seed}"


@lru_cache(maxsize=1)
def cached_data():
    return data_and_folds()


def fit_job(job):
    fold, model_name, input_std, target_std, seed = job
    x, y, sessions, acquisition_order, data_hash = cached_data()
    held_out = acquisition_order[int(fold[1:]) - 1]
    train = np.flatnonzero(sessions != held_out)
    test = np.flatnonzero(sessions == held_out)
    if len(set(train.tolist()) & set(test.tolist())) or len(train) + len(test) != len(x):
        raise ValueError("Acquisition split overlap or omission")
    xm, xs = scale_from_training(x[train], input_std)
    ym, ys = scale_from_training(y[train], target_std)
    xtrain = torch.as_tensor((x[train] - xm) / xs, dtype=torch.float32)
    ytrain = torch.as_tensor((y[train] - ym) / ys, dtype=torch.float32)
    xtest = torch.as_tensor((x[test] - xm) / xs, dtype=torch.float32)
    set_deterministic_seed(seed)
    model = AuthorLocalizationCNN() if model_name == "cnn" else CompactMLP(8)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999),
                                 eps=1e-7, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    order_digest = hashlib.sha256()
    code_hash = source_hash()
    runtime = engine()
    started = time.perf_counter()
    updates = 0
    rows = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        order_digest.update(train[order.numpy()].astype(np.int64, copy=False).tobytes())
        for start in range(0, len(train), BATCH):
            batch = order[start:start + BATCH]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xtrain[batch]) - ytrain[batch]) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {job_id(job)} at epoch {epoch}")
            loss.backward()
            optimizer.step()
            updates += 1
        if epoch in SNAPSHOTS:
            model.eval()
            with torch.inference_mode():
                prediction = model(xtest).cpu().numpy().astype(float) * ys + ym
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"Non-finite prediction in {job_id(job)} at epoch {epoch}")
            elapsed = time.perf_counter() - started
            common = {
                "protocol": PROTOCOL, "source_sha256": code_hash,
                "data_sha256": data_hash, "engine": runtime,
                "job_id": job_id(job), "fold": fold, "model": model_name,
                "seed": seed, "input_standardised": int(input_std),
                "target_standardised": int(target_std), "epoch": epoch,
                "train_rows": len(train), "test_rows": len(test),
                "optimizer_updates": updates,
                "batch_order_sha256": order_digest.hexdigest(),
                "fit_wall_sec_diagnostic": repr(elapsed),
            }
            for index, estimate in zip(test, prediction):
                truth = y[index]
                rows.append({**common, "row_index": int(index),
                             "true_x_um": repr(float(truth[0])),
                             "true_y_um": repr(float(truth[1])),
                             "pred_x_um": repr(float(estimate[0])),
                             "pred_y_um": repr(float(estimate[1])),
                             "error_um": repr(float(np.linalg.norm(estimate - truth)))})
    return job, rows


def completed_jobs():
    if not OUTPUT.exists():
        return set()
    with OUTPUT.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Existing result CSV has a different schema")
        rows = list(reader)
    _, y, sessions, acquisition_order, data_hash = cached_data()
    expected_hash = source_hash()
    expected_engine = engine()
    groups = {}
    for row in rows:
        if (row["protocol"] != PROTOCOL or row["source_sha256"] != expected_hash
                or row["data_sha256"] != data_hash or row["engine"] != expected_engine):
            raise ValueError("Existing result CSV has incompatible protocol, source, data or engine")
        key = row["job_id"]
        groups.setdefault(key, []).append(row)
    admissible = {job_id(job) for job in jobs()}
    if not set(groups).issubset(admissible):
        raise ValueError("Unknown job in existing result CSV")
    for key, group in groups.items():
        fold = group[0]["fold"]
        held_out = acquisition_order[int(fold[1:]) - 1]
        expected_indices = set(np.flatnonzero(sessions == held_out).tolist())
        actual = [(int(row["epoch"]), int(row["row_index"])) for row in group]
        wanted = {(epoch, index) for epoch in SNAPSHOTS for index in expected_indices}
        if len(actual) != len(wanted) or set(actual) != wanted:
            raise ValueError(f"Incomplete or duplicate rows in {key}")
        for row in group:
            index = int(row["row_index"])
            truth = np.array([float(row["true_x_um"]), float(row["true_y_um"])])
            prediction = np.array([float(row["pred_x_um"]), float(row["pred_y_um"])])
            error = float(row["error_um"])
            if (not np.array_equal(truth, y[index]) or not np.isfinite(prediction).all()
                    or not np.isfinite(error)
                    or not np.isclose(np.linalg.norm(prediction - truth), error, atol=1e-9)):
                raise ValueError(f"Prediction/truth/error mismatch in {key}")
    return set(groups)


def check():
    x, y, sessions, acquisition_order, data_hash = cached_data()
    if len(jobs()) != 192 or len({job_id(job) for job in jobs()}) != 192:
        raise ValueError("Expected exactly 192 unique three-seed jobs")
    if sum(np.sum(sessions == s) for s in acquisition_order) != len(x):
        raise ValueError("Incomplete acquisition partition")
    pilot = HERE / "recipe_pilot.csv"
    with pilot.open(encoding="utf-8", newline="") as handle:
        pilot_rows = list(csv.DictReader(handle))
    if len(pilot_rows) != 32 or len({r["code_sha256"] for r in pilot_rows}) != 1:
        raise ValueError("One-seed pilot is missing or inconsistent")
    if pilot_rows[0]["code_sha256"] != hashlib.sha256((HERE / "recipe_factorial.py").read_bytes()).hexdigest():
        raise ValueError("Pilot implementation changed since its verification")
    if {r["data_sha256"] for r in pilot_rows} != {data_hash}:
        raise ValueError("Pilot and full experiment data differ")
    print(f"CHECK OK: 819 rows, eight acquisitions, 192 jobs, two snapshots, data_sha256={data_hash}", flush=True)


def run(workers):
    check()
    done = completed_jobs()
    pending = [job for job in jobs() if job_id(job) not in done]
    print(f"RUN {PROTOCOL}: {len(done)} complete, {len(pending)} pending, workers={workers}", flush=True)
    if not pending:
        return
    if not OUTPUT.exists():
        with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
            handle.flush()
            os.fsync(handle.fileno())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fit_job, job): job for job in pending}
        for future in as_completed(futures):
            job, rows = future.result()
            with OUTPUT.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            means = [np.mean([float(r["error_um"]) for r in rows if r["epoch"] == epoch])
                     for epoch in SNAPSHOTS]
            print(f"DONE {job_id(job)}: 1500={means[0]:.6f}, 3000={means[1]:.6f} um", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("workers must be between one and six")
    check() if args.action == "check" else run(args.workers)
