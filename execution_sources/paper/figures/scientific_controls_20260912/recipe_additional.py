"""Complete the fixed Weng recipe factorial from three to ten seeds.

The existing three-seed CSV and its source fingerprint are immutable inputs.
This runner appends only the seven consecutive additional seeds to one CSV;
it reuses the exact model, data split, preprocessing, optimizer and fit loop
from recipe_full.py. Epoch 1500 and 3000 are checkpoints of one trajectory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import math
import os
from pathlib import Path

import numpy as np

import recipe_full as base

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "recipe_additional.csv"
PROTOCOL = "NV_RECIPE_FACTORIAL_POSTHOC_FINAL10_EXTRA7_V1"
SEEDS = tuple(range(20260807, 20260814))
FIELDS = base.FIELDS


def source_hash() -> str:
    digest = hashlib.sha256()
    digest.update(base.source_hash().encode("ascii"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def jobs():
    return list(itertools.product(base.FOLDS, base.MODELS,
                                  (False, True), (False, True), SEEDS))


def fit_job(job):
    before = source_hash()
    task, rows = base.fit_job(job)
    if source_hash() != before:
        raise RuntimeError("Recipe source changed during a fit")
    for row in rows:
        row["protocol"] = PROTOCOL
        row["source_sha256"] = before
    return task, rows


def completed_jobs():
    if not OUTPUT.exists():
        return set()
    with OUTPUT.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Additional-seed CSV schema differs")
        rows = list(reader)
    _, y, sessions, order, data_hash = base.cached_data()
    allowed = {base.job_id(job): job for job in jobs()}
    code_hash = source_hash()
    groups = {}
    for row in rows:
        key = row["job_id"]
        if (key not in allowed or row["protocol"] != PROTOCOL
                or row["source_sha256"] != code_hash
                or row["data_sha256"] != data_hash
                or row["engine"] != base.engine()):
            raise ValueError("Additional-seed result has incompatible provenance")
        fold, model, input_std, target_std, seed = allowed[key]
        if (row["fold"] != fold or row["model"] != model
                or int(row["input_standardised"]) != int(input_std)
                or int(row["target_standardised"]) != int(target_std)
                or int(row["seed"]) != seed):
            raise ValueError(f"Job metadata mismatch: {key}")
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        fold = allowed[key][0]
        test = set(np.flatnonzero(sessions == order[int(fold[1:]) - 1]).tolist())
        wanted = {(epoch, index) for epoch in base.SNAPSHOTS for index in test}
        got = [(int(row["epoch"]), int(row["row_index"])) for row in group]
        if len(got) != len(wanted) or set(got) != wanted:
            raise ValueError(f"Incomplete or duplicated job: {key}")
        train_rows = len(sessions) - len(test)
        for row in group:
            index, epoch = int(row["row_index"]), int(row["epoch"])
            truth = np.array([float(row["true_x_um"]), float(row["true_y_um"])])
            prediction = np.array([float(row["pred_x_um"]), float(row["pred_y_um"])])
            error = float(row["error_um"])
            if (int(row["train_rows"]) != train_rows
                    or int(row["test_rows"]) != len(test)
                    or int(row["optimizer_updates"]) != epoch * math.ceil(train_rows / base.BATCH)
                    or not np.array_equal(truth, y[index])
                    or not np.isfinite(prediction).all()
                    or not math.isclose(float(np.linalg.norm(prediction - truth)),
                                        error, rel_tol=0.0, abs_tol=1e-9)):
                raise ValueError(f"Prediction, target or updates mismatch: {key}")
    return set(groups)


def check():
    base.check()
    if len(base.completed_jobs()) != 192:
        raise ValueError("Original three-seed experiment is not complete")
    if len(jobs()) != 448 or len({base.job_id(job) for job in jobs()}) != 448:
        raise ValueError("Expected 448 distinct additional-seed fits")
    if set(SEEDS) & set(base.SEEDS):
        raise ValueError("Additional seeds overlap the frozen three-seed stage")
    print(f"CHECK OK: seven additional seeds, 448 fits, source_sha256={source_hash()}", flush=True)


def run(workers):
    check()
    done = completed_jobs()
    pending = [job for job in jobs() if base.job_id(job) not in done]
    print(f"RUN {PROTOCOL}: {len(done)} complete, {len(pending)} pending, workers={workers}", flush=True)
    if not pending:
        return
    if not OUTPUT.exists():
        with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
            handle.flush()
            os.fsync(handle.fileno())
    expected_hash = source_hash()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fit_job, job): job for job in pending}
        for future in as_completed(futures):
            job, rows = future.result()
            if source_hash() != expected_hash:
                raise RuntimeError("Recipe source changed before result write")
            with OUTPUT.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            means = [np.mean([float(row["error_um"]) for row in rows
                              if row["epoch"] == epoch]) for epoch in base.SNAPSHOTS]
            print(f"DONE {base.job_id(job)}: 1500={means[0]:.6f}, 3000={means[1]:.6f} um", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("workers must be between one and six")
    check() if args.action == "check" else run(args.workers)
