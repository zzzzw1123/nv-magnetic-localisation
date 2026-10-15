"""Pilot the input/target-standardisation factorial for CNN and MLP8.

Data comprise ``X(819, 24)``, ``y(819, 2)`` in micrometres, and eight
acquisition labels. A1 and A2 are held out in turn while the other seven
acquisitions train each model. The 16 jobs write mean held-out errors at paired
epoch-1500 and epoch-3000 checkpoints to ``recipe_pilot.csv``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import os
from pathlib import Path
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
    AuthorLocalizationCNN, load_author_data, set_deterministic_seed,
)
from qml_microrobot.study_l31_compact import CompactMLP  # noqa: E402

PROTOCOL = "NV_RECIPE_FACTORIAL_POSTHOC_PILOT_V1"
SEED = 20260804
FOLDS = ("A1", "A2")
MODELS = ("cnn", "mlp8")
EPOCHS = 3000
SNAPSHOTS = (1500, 3000)
BATCH = 128
LR = 0.005
OUTPUT = HERE / "recipe_pilot.csv"
FIELDS = (
    "protocol", "code_sha256", "data_sha256", "fold", "model", "seed",
    "input_standardised", "target_standardised", "epoch", "test_rows",
    "optimizer_updates", "mean_error_um", "fit_wall_sec_diagnostic",
)


def data_and_folds():
    """Load features and return ``X(819,24)``, ``y(819,2)`` in µm,
    acquisition labels, acquisition order, and a content SHA-256 digest."""
    data = load_author_data(ROOT)
    x = np.asarray(data.features, dtype=np.float64)
    y = np.asarray(data.positions_um, dtype=np.float64)
    sessions = np.asarray(data.session_ids, dtype=str)
    order = list(dict.fromkeys(sessions.tolist()))
    counts = [int(np.sum(sessions == s)) for s in order]
    if x.shape != (819, 24) or y.shape != (819, 2) or len(order) != 8:
        raise ValueError("Frozen NV data shape or acquisition count changed")
    if counts != [102, 104, 102, 204, 51, 52, 102, 102]:
        raise ValueError("Frozen acquisition membership changed")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite NV values")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    digest.update("\n".join(sessions).encode("utf-8"))
    return x, y, sessions, order, digest.hexdigest()


def scale_from_training(values, enabled):
    """Return column means and population scales fitted on training rows only.

    ``values`` has shape ``(n, p)``. Disabled scaling returns zero offsets and
    unit scales of shape ``(p,)`` so callers can use one transformation path.
    """
    if not enabled:
        return np.zeros(values.shape[1]), np.ones(values.shape[1])
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return mean, scale


def tasks():
    """Enumerate the 16 pilot jobs: 2 folds × 2 models × 2 × 2 recipes."""
    return list(itertools.product(FOLDS, MODELS, (False, True), (False, True)))


def run_one(task):
    """Fit one pilot job and return mean-error rows at both epoch checkpoints.

    Predictions have shape ``(n_test, 2)`` and are converted back to
    micrometres before Euclidean localisation error is averaged.
    """
    fold, model_name, input_std, target_std = task
    x, y, sessions, order, data_hash = data_and_folds()
    held_out = order[int(fold[1:]) - 1]
    train = np.flatnonzero(sessions != held_out)
    test = np.flatnonzero(sessions == held_out)
    xm, xs = scale_from_training(x[train], input_std)
    ym, ys = scale_from_training(y[train], target_std)
    xtrain = torch.as_tensor((x[train] - xm) / xs, dtype=torch.float32)
    ytrain = torch.as_tensor((y[train] - ym) / ys, dtype=torch.float32)
    xtest = torch.as_tensor((x[test] - xm) / xs, dtype=torch.float32)
    set_deterministic_seed(SEED)
    model = AuthorLocalizationCNN() if model_name == "cnn" else CompactMLP(8)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-7)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started = time.perf_counter()
    updates = 0
    rows = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order_in_epoch = torch.randperm(len(train), generator=generator)
        for start in range(0, len(train), BATCH):
            batch = order_in_epoch[start:start + BATCH]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xtrain[batch]) - ytrain[batch]) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {task}, epoch {epoch}")
            loss.backward()
            optimizer.step()
            updates += 1
        if epoch in SNAPSHOTS:
            model.eval()
            with torch.inference_mode():
                prediction = model(xtest).cpu().numpy().astype(float) * ys + ym
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"Non-finite prediction: {task}, epoch {epoch}")
            error = float(np.linalg.norm(prediction - y[test], axis=1).mean())
            rows.append({
                "protocol": PROTOCOL, "code_sha256": code_hash,
                "data_sha256": data_hash, "fold": fold, "model": model_name,
                "seed": SEED, "input_standardised": int(input_std),
                "target_standardised": int(target_std), "epoch": epoch,
                "test_rows": len(test), "optimizer_updates": updates,
                "mean_error_um": repr(error),
                "fit_wall_sec_diagnostic": repr(time.perf_counter() - started),
            })
    return task, rows


def existing_keys():
    """Validate the resumable pilot CSV and return completed task keys."""
    if not OUTPUT.exists():
        return set()
    with OUTPUT.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Existing pilot CSV schema differs")
        rows = list(reader)
    current_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _, _, _, _, data_hash = data_and_folds()
    if any(row["protocol"] != PROTOCOL or row["code_sha256"] != current_hash
           or row["data_sha256"] != data_hash for row in rows):
        raise ValueError("Existing pilot CSV has different code, protocol, or data")
    keys = {}
    for row in rows:
        key = (row["fold"], row["model"], bool(int(row["input_standardised"])),
               bool(int(row["target_standardised"])))
        keys.setdefault(key, []).append(int(row["epoch"]))
    if any(sorted(v) != list(SNAPSHOTS) for v in keys.values()):
        raise ValueError("Existing pilot CSV contains incomplete/duplicate jobs")
    return set(keys)


def check():
    """Validate shapes, acquisition membership, task count, and outputs."""
    x, y, sessions, order, data_hash = data_and_folds()
    assert len(tasks()) == 16
    assert all(np.sum(sessions == order[int(f[1:]) - 1]) > 0 for f in FOLDS)
    assert AuthorLocalizationCNN()(torch.zeros(2, 24)).shape == (2, 2)
    assert CompactMLP(8)(torch.zeros(2, 24)).shape == (2, 2)
    print(f"CHECK OK: {x.shape}, {y.shape}, folds={FOLDS}, tasks=16, data_sha256={data_hash}")


def run(workers):
    """Execute missing pilot jobs and append each completed pair of snapshots."""
    check()
    done = existing_keys()
    pending = [task for task in tasks() if task not in done]
    print(f"PILOT: {len(done)} complete, {len(pending)} pending; workers={workers}", flush=True)
    if not pending:
        return
    if not OUTPUT.exists():
        with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
            handle.flush()
            os.fsync(handle.fileno())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, task): task for task in pending}
        for future in as_completed(futures):
            task, rows = future.result()
            with OUTPUT.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            print(f"DONE {task}: 1500={rows[0]['mean_error_um']} um; "
                  f"3000={rows[1]['mean_error_um']} um", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("workers must be 1–4")
    check() if args.action == "check" else run(args.workers)
