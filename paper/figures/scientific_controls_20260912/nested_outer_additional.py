"""Extend nested-LOAO outer refits from three to ten seeds.

``nested_loao.json`` supplies one selected candidate per acquisition. This
runner refits it for seven additional seeds on ``(n_train, 24)`` features and
writes ``(n_test, 2)`` predictions and errors in micrometres to
``nested_outer_additional.csv``. Seeds are repeated fits within acquisitions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
import nested_loao as nested  # noqa: E402
from qml_microrobot.study_l31_compact import fit_compact_mlp, predict_compact_mlp  # noqa: E402

PROTOCOL = "NV_L31_NESTED_OUTER_FINAL10_EXTRA7_V1"
SEEDS = tuple(range(20260807, 20260814))
OUTPUT = HERE / "nested_outer_additional.csv"
FIELDS = (
    "protocol", "source_sha256", "selection_sha256", "data_sha256", "engine",
    "outer_fold", "held_out_session", "selected_candidate", "seed",
    "train_rows", "test_rows", "optimizer_updates", "batch_order_sha256",
    "fit_wall_sec_diagnostic", "row_index", "true_x_um", "true_y_um",
    "pred_x_um", "pred_y_um", "error_um",
)


def source_hash():
    """Hash this runner with the nested selector sources."""
    digest = hashlib.sha256()
    digest.update(nested.source_hash().encode("ascii"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def selection_hash():
    """Return the SHA-256 digest of the completed nested-selection record."""
    return hashlib.sha256(nested.OUTPUT.read_bytes()).hexdigest()


def engine():
    return f"Python {platform.python_version()}; NumPy {np.__version__}; PyTorch {torch.__version__}; CPU"


def frozen_selection():
    """Load features, targets, and the eight selected neural candidates."""
    x, y, sessions, order, data_hash = nested.data_and_folds()
    result = nested.existing_result(data_hash, sessions, order)
    if set(result["folds"]) != set(nested.OUTER_FOLDS):
        raise ValueError("Original nested selection is not complete")
    if result["selection_seed"] != 20260804 or result["outer_mlp_seeds"] != [20260804, 20260805, 20260806]:
        raise ValueError("Original nested seeds changed")
    candidates = {}
    for fold in nested.OUTER_FOLDS:
        item = result["folds"][fold]
        candidate = item["selected_candidate"]["candidate"]
        if candidate not in {"mlp8_fixed", "mlp16_fixed", "mlp32_fixed"}:
            raise ValueError(f"Non-neural candidate {candidate} needs no repeated outer fit")
        candidates[fold] = candidate
    return x, y, sessions, order, data_hash, result, candidates


def job_id(fold, seed):
    """Encode one outer acquisition and optimisation seed as a stable key."""
    return f"{fold}_s{seed}"


def batch_order_hash(train, seed):
    """Hash the 1500 deterministic training-index permutations for one refit."""
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    digest = hashlib.sha256()
    for _ in range(1500):
        permutation = torch.randperm(len(train), generator=generator).numpy()
        digest.update(train[permutation].astype(np.int64, copy=False).tobytes())
    return digest.hexdigest()


def fit_outer(fold, seed):
    """Refit one selected MLP and return ``n_test`` prediction rows in µm."""
    code_hash, selected_hash = source_hash(), selection_hash()
    x, y, sessions, order, data_hash, _, candidates = frozen_selection()
    candidate = candidates[fold]
    held_out = order[int(fold[1:]) - 1]
    train = np.flatnonzero(sessions != held_out)
    test = np.flatnonzero(sessions == held_out)
    if len(set(train) & set(test)) or len(train) + len(test) != len(x):
        raise ValueError("Outer fold overlap or omission")
    settings = nested.protocol()["candidates"]["compact_mlp"]
    width = int(candidate.split("_", 1)[0][3:])
    started = time.perf_counter()
    state = fit_compact_mlp(x[train], y[train], width, seed=seed,
                            epochs=int(settings["epochs"]),
                            batch_size=int(settings["batch_size"]),
                            learning_rate=float(settings["learning_rate"]))
    prediction = predict_compact_mlp(state, x[test])
    if not np.isfinite(prediction).all() or source_hash() != code_hash or selection_hash() != selected_hash:
        raise RuntimeError("Non-finite prediction or source/selection drift")
    common = {"protocol": PROTOCOL, "source_sha256": code_hash,
              "selection_sha256": selected_hash, "data_sha256": data_hash,
              "engine": engine(), "outer_fold": fold,
              "held_out_session": held_out, "selected_candidate": candidate,
              "seed": seed, "train_rows": len(train), "test_rows": len(test),
              "optimizer_updates": int(settings["epochs"]) * math.ceil(len(train) / int(settings["batch_size"])),
              "batch_order_sha256": batch_order_hash(train, seed),
              "fit_wall_sec_diagnostic": repr(time.perf_counter() - started)}
    rows = []
    for index, estimate in zip(test, prediction):
        truth = y[index]
        rows.append({**common, "row_index": int(index),
                     "true_x_um": repr(float(truth[0])),
                     "true_y_um": repr(float(truth[1])),
                     "pred_x_um": repr(float(estimate[0])),
                     "pred_y_um": repr(float(estimate[1])),
                     "error_um": repr(float(np.linalg.norm(estimate - truth)))})
    return rows


def completed_jobs():
    """Validate stored source hashes, row coverage, batch order, and errors."""
    if not OUTPUT.exists():
        return set()
    with OUTPUT.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Additional nested outer CSV schema differs")
        rows = list(reader)
    _, y, sessions, order, data_hash, _, candidates = frozen_selection()
    code_hash, selected_hash = source_hash(), selection_hash()
    groups = {}
    for row in rows:
        fold, seed = row["outer_fold"], int(row["seed"])
        if (fold not in candidates or seed not in SEEDS or row["protocol"] != PROTOCOL
                or row["source_sha256"] != code_hash
                or row["selection_sha256"] != selected_hash
                or row["data_sha256"] != data_hash
                or row["engine"] != engine()
                or row["selected_candidate"] != candidates[fold]
                or row["held_out_session"] != order[int(fold[1:]) - 1]):
            raise ValueError("Additional nested outer provenance mismatch")
        groups.setdefault(job_id(fold, seed), []).append(row)
    for key, group in groups.items():
        fold, seed = group[0]["outer_fold"], int(group[0]["seed"])
        test = set(np.flatnonzero(sessions == order[int(fold[1:]) - 1]).tolist())
        train = np.flatnonzero(sessions != order[int(fold[1:]) - 1])
        got = [int(row["row_index"]) for row in group]
        if len(got) != len(test) or set(got) != test:
            raise ValueError(f"Incomplete or duplicate outer predictions: {key}")
        expected_order = batch_order_hash(train, seed)
        for row in group:
            index = int(row["row_index"])
            truth = np.array([float(row["true_x_um"]), float(row["true_y_um"])])
            prediction = np.array([float(row["pred_x_um"]), float(row["pred_y_um"])])
            if (int(row["train_rows"]) != len(train) or int(row["test_rows"]) != len(test)
                    or int(row["optimizer_updates"]) != 1500 * math.ceil(len(train) / 128)
                    or row["batch_order_sha256"] != expected_order
                    or not np.array_equal(truth, y[index])
                    or not np.isfinite(prediction).all()
                    or not np.isclose(np.linalg.norm(prediction - truth), float(row["error_um"]), atol=1e-9)):
                raise ValueError(f"Outer prediction or training update mismatch: {key}")
    return set(groups)


def smoke():
    """Reproduce the historical A1/seed-20260804 predictions within 1e-9 µm."""
    x, y, sessions, order, _, result, candidates = frozen_selection()
    fold, seed = "A1", 20260804
    train = np.flatnonzero(sessions != order[0])
    test = np.flatnonzero(sessions == order[0])
    settings = nested.protocol()["candidates"]["compact_mlp"]
    state = fit_compact_mlp(x[train], y[train], 32, seed=seed,
                            epochs=int(settings["epochs"]),
                            batch_size=int(settings["batch_size"]),
                            learning_rate=float(settings["learning_rate"]))
    predicted = predict_compact_mlp(state, x[test])
    old = next(outcome for outcome in result["folds"][fold]["outer_outcomes"] if outcome["seed"] == seed)
    indexed = {row["row_index"]: [row["pred_x_um"], row["pred_y_um"]] for row in old["predictions"]}
    expected = np.array([indexed[index] for index in test])
    drift = float(np.max(np.abs(predicted - expected)))
    if candidates[fold] != "mlp32_fixed" or drift > 1e-9:
        raise ValueError(f"Historical A1 outer fit not reproduced: {drift} um")
    print(f"SMOKE OK: A1 selected MLP32, seed {seed}, max prediction drift={drift:.3g} um", flush=True)


def check():
    """Confirm eight selections and 56 disjoint additional-seed fits."""
    nested.check()
    _, _, _, _, data_hash, _, candidates = frozen_selection()
    if len(candidates) != 8 or len(SEEDS) != 7 or set(SEEDS) & set(nested.OUTER_MLP_SEEDS):
        raise ValueError("Outer selection or additional seeds changed")
    print(f"CHECK OK: 8 fixed selections, 56 extra fits, data_sha256={data_hash}, selection_sha256={selection_hash()}", flush=True)


def run():
    """Run missing outer refits serially and append each complete fit."""
    check()
    done = completed_jobs()
    pending = [(fold, seed) for fold in nested.OUTER_FOLDS for seed in SEEDS
               if job_id(fold, seed) not in done]
    print(f"NESTED OUTER: {len(done)} complete, {len(pending)} pending, workers=1", flush=True)
    if not pending:
        return
    if not OUTPUT.exists():
        with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
            handle.flush()
            os.fsync(handle.fileno())
    code_hash, selected_hash = source_hash(), selection_hash()
    for fold, seed in pending:
        rows = fit_outer(fold, seed)
        if source_hash() != code_hash or selection_hash() != selected_hash:
            raise RuntimeError("Source or selection changed before outer result write")
        with OUTPUT.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        print(f"DONE {job_id(fold, seed)}: {np.mean([float(row['error_um']) for row in rows]):.6f} um", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "smoke", "run"))
    args = parser.parse_args()
    if args.action == "check":
        check()
    elif args.action == "smoke":
        check()
        smoke()
    else:
        run()
