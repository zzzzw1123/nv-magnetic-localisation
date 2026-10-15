"""Apply four preprocessing recipes to the Meneses NV tracking task.

Five segments provide time in seconds, three magnetic channels in nanotesla,
and vertical position in metres. Non-overlapping ``(40, 3)`` histories predict
the next position; segments 1, 4, and 5 train while segments 2 and 3 test.
Output CSVs store row-level truth, predictions, and absolute errors for CNN
and MLP8.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import platform
import sys
import time
from urllib.request import Request, urlopen

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    if torch.get_num_interop_threads() != 1:
        raise

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = HERE / "meneses_data"
sys.path.insert(0, str(ROOT / "src"))
from qml_microrobot.study_l30_author_cnn import set_deterministic_seed  # noqa: E402

PROTOCOL = "NV_MENESES_CROSS_TASK_POSTHOC_V1"
SOURCE_COMMIT = "7e8a78a33078a1479fe66b9c73bf3c6d7e14ca02"
SOURCE_BASE = f"https://raw.githubusercontent.com/Fertmeneses/ML_QDM_Meneses_et_al/{SOURCE_COMMIT}/Data/"
SOURCE_NAMES = tuple(f"phys_model_data_zAut_segm{i}.csv" for i in range(1, 6))
SOURCE_ROWS = (35719, 107601, 107138, 105475, 70561)
TRAIN_SEGMENTS = (1, 4, 5)
TEST_SEGMENTS = (2, 3)
PARKING_LEVELS_M = np.array([0.0, 4.1, 7.8, 11.5, 15.2, 18.9, 22.6, 26.3])
WINDOW = 40                   # Four seconds at 10 Hz; windows do not overlap.
TRAIN_WINDOWS = 819           # 273 label-blind, equally spaced windows per full-training segment.
SEEDS = (20260804, 20260805, 20260806)
SNAPSHOTS = (200, 400)
BATCH = 256
LR = 1e-4
MODEL_NAMES = ("cnn", "mlp8")
FIELDS = (
    "protocol", "mode", "source_commit", "source_sha256", "data_sha256", "engine",
    "job_id", "model", "seed", "input_standardised", "target_standardised",
    "epoch", "train_windows", "optimizer_updates", "batch_order_sha256",
    "fit_wall_sec_diagnostic", "test_segment", "window_index", "target_time_s",
    "true_z_m", "pred_z_m", "abs_error_m", "parking_ground_truth",
)


def source_hash():
    """Hash this runner and the shared deterministic-seed implementation."""
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "src/qml_microrobot/study_l30_author_cnn.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def engine():
    return f"Python {platform.python_version()}; NumPy {np.__version__}; PyTorch {torch.__version__}; CPU"


def ensure_data():
    """Download missing pinned-commit CSVs atomically into ``meneses_data``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_NAMES:
        path = DATA_DIR / name
        if path.is_file():
            continue
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            raise RuntimeError(f"Interrupted download needs inspection: {temporary}")
        request = Request(SOURCE_BASE + name, headers={"User-Agent": "NV-magnetic-sensing-benchmarks"})
        try:
            with urlopen(request, timeout=60) as response, temporary.open("xb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise


@lru_cache(maxsize=1)
def load_data():
    """Load five finite, time-ordered arrays and return a byte-level digest.

    Segments 1--3 have shape ``(n, 6)`` and include camera truth; segments 4--5
    have shape ``(n, 5)``. Time is in seconds, fields are in nanotesla, and
    vertical-position columns are in metres.
    """
    arrays = {}
    digest = hashlib.sha256()
    for segment, (name, expected_rows) in enumerate(zip(SOURCE_NAMES, SOURCE_ROWS), start=1):
        path = DATA_DIR / name
        with path.open(encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
        expected_header = ["Time_s", "Bx_nT", "By_nT", "Bz_nT", "zAut_m"]
        if segment <= 3:
            expected_header.append("zTrue_m")
        if header != expected_header:
            raise ValueError(f"Unexpected Meneses schema in segment {segment}: {header}")
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        if values.shape != (expected_rows, len(header)) or not np.isfinite(values).all():
            raise ValueError(f"Meneses segment {segment} shape or finite-value check failed")
        if not np.all(np.diff(values[:, 0]) > 0):
            raise ValueError(f"Meneses segment {segment} time is not strictly increasing")
        arrays[segment] = values
        digest.update(name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return arrays, digest.hexdigest()


def windows(values, segment):
    """Convert one segment into non-overlapping next-sample prediction windows.

    Returns ``X(n,40,3)`` in nanotesla, ``y(n,)`` in metres, target times in
    seconds, and a Boolean parking-level mask. The target immediately follows
    each 40-sample history, preserving causal order.
    """
    starts = np.arange(0, len(values) - WINDOW, WINDOW, dtype=np.int64)
    # Use camera truth when released and the automated position otherwise; the
    # target is the sample immediately after each 40-point magnetic history.
    zcol = 5 if values.shape[1] == 6 else 4
    x = np.stack([values[start:start + WINDOW, 1:4] for start in starts]).astype(np.float64)
    y = values[starts + WINDOW, zcol].astype(np.float64)
    t = values[starts + WINDOW, 0].astype(np.float64)
    park = np.isin(y, PARKING_LEVELS_M) if segment in TEST_SEGMENTS else np.zeros(len(y), dtype=bool)
    return x, y, t, park


@lru_cache(maxsize=2)
def dataset(mode):
    """Assemble the 819-window training set and held-out segment mapping.

    The returned training shapes are ``(819, 40, 3)`` and ``(819,)``. Window
    indices are equally spaced without reading labels, preventing target-driven
    subsampling.
    """
    arrays, fingerprint = load_data()
    windowed = {segment: windows(values, segment) for segment, values in arrays.items()}
    if mode == "pilot":
        train_segments, test_segments = (1, 4), (5,)
        counts = {1: 410, 4: 409}
    elif mode == "full":
        train_segments, test_segments = TRAIN_SEGMENTS, TEST_SEGMENTS
        counts = {segment: TRAIN_WINDOWS // 3 for segment in TRAIN_SEGMENTS}
    else:
        raise ValueError(mode)
    selected_x, selected_y = [], []
    for segment in train_segments:
        x, y, _, _ = windowed[segment]
        indices = np.linspace(0, len(x) - 1, counts[segment], dtype=np.int64)
        if len(np.unique(indices)) != counts[segment]:
            raise ValueError("Label-blind training-window selection contains duplicates")
        selected_x.append(x[indices])
        selected_y.append(y[indices])
    xtrain = np.concatenate(selected_x)
    ytrain = np.concatenate(selected_y)
    if xtrain.shape != (TRAIN_WINDOWS, WINDOW, 3) or ytrain.shape != (TRAIN_WINDOWS,):
        raise ValueError("Frozen 819-window training interface changed")
    return xtrain, ytrain, {s: windowed[s] for s in test_segments}, fingerprint


class CompactMLP(nn.Module):
    """Map ``(batch, 40, 3)`` magnetic windows to ``(batch,)`` positions."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(WINDOW * 3, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, x):
        """Return one scaled vertical-position estimate per input window."""
        return self.network(x).squeeze(-1)


class TemporalCNN(nn.Module):
    """Temporal convolutional regressor with ``(batch,40,3)`` input."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=8), nn.Tanh(),
            nn.Conv1d(16, 16, kernel_size=4), nn.Tanh(), nn.Flatten(),
            nn.Linear(30 * 16, 1024), nn.Tanh(),
            nn.Linear(1024, 512), nn.Tanh(), nn.Linear(512, 1),
        )

    def forward(self, x):
        """Transpose channels for Conv1d and return ``(batch,)`` positions."""
        return self.network(x.transpose(1, 2)).squeeze(-1)


def make_model(name, seed):
    """Construct and deterministically initialise one CNN or MLP8 regressor."""
    set_deterministic_seed(seed)
    model = TemporalCNN() if name == "cnn" else CompactMLP()
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def jobs(mode):
    """Enumerate 8 pilot jobs or 24 full jobs across recipes and seeds."""
    seeds = SEEDS[:1] if mode == "pilot" else SEEDS
    return [(mode, model, xs, ys, seed) for model in MODEL_NAMES
            for xs in (False, True) for ys in (False, True) for seed in seeds]


def job_id(job):
    """Encode mode, model, preprocessing switches, and seed as a stable key."""
    mode, model, xs, ys, seed = job
    return f"{mode}_{model}_x{int(xs)}_y{int(ys)}_s{seed}"


def output_path(mode):
    """Return the pilot or held-out row-level CSV destination."""
    return HERE / ("meneses_pilot.csv" if mode == "pilot" else "meneses_full.csv")


def run_job(job):
    """Fit one recipe and return row-level predictions at both checkpoints.

    Training tensors have shape ``(819,40,3)`` and targets ``(819,)``.
    Predictions are transformed back to metres before absolute error is stored.
    """
    mode, name, input_std, target_std, seed = job
    xtrain, ytrain, tests, data_hash = dataset(mode)
    if input_std:
        xm = xtrain.mean(axis=(0, 1))
        xs = xtrain.std(axis=(0, 1), ddof=0)
        xs = np.where(xs == 0, 1.0, xs)
    else:
        xm, xs = np.zeros(3), np.ones(3)
    if target_std:
        ym, ys = float(ytrain.mean()), float(ytrain.std(ddof=0))
        ys = 1.0 if ys == 0 else ys
    else:
        ym, ys = 0.0, 1.0
    x = torch.as_tensor((xtrain - xm) / xs, dtype=torch.float32)
    y = torch.as_tensor((ytrain - ym) / ys, dtype=torch.float32)
    model = make_model(name, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999),
                                 eps=1e-7, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    order_digest = hashlib.sha256()
    started = time.perf_counter()
    updates = 0
    result = []
    code_hash = source_hash()
    runtime = engine()
    for epoch in range(1, SNAPSHOTS[-1] + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        order_digest.update(order.numpy().astype(np.int64, copy=False).tobytes())
        for first in range(0, len(x), BATCH):
            batch = order[first:first + BATCH]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(torch.abs(model(x[batch]) - y[batch]))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {job_id(job)} at epoch {epoch}")
            loss.backward()
            optimizer.step()
            updates += 1
        if epoch not in SNAPSHOTS:
            continue
        model.eval()
        common = {"protocol": PROTOCOL, "mode": mode, "source_commit": SOURCE_COMMIT,
                  "source_sha256": code_hash, "data_sha256": data_hash, "engine": runtime,
                  "job_id": job_id(job), "model": name, "seed": seed,
                  "input_standardised": int(input_std), "target_standardised": int(target_std),
                  "epoch": epoch, "train_windows": len(x), "optimizer_updates": updates,
                  "batch_order_sha256": order_digest.hexdigest(),
                  "fit_wall_sec_diagnostic": repr(time.perf_counter() - started)}
        with torch.inference_mode():
            for segment, (xtest, truth, times, parking) in tests.items():
                estimates = []
                for first in range(0, len(xtest), 1024):
                    features = torch.as_tensor((xtest[first:first + 1024] - xm) / xs, dtype=torch.float32)
                    estimates.append(model(features).cpu().numpy().astype(float) * ys + ym)
                pred = np.concatenate(estimates)
                if not np.isfinite(pred).all():
                    raise FloatingPointError(f"Non-finite prediction in {job_id(job)} at epoch {epoch}")
                for index, (time_s, actual, estimate, park) in enumerate(zip(times, truth, pred, parking)):
                    result.append({**common, "test_segment": segment, "window_index": index,
                                   "target_time_s": repr(float(time_s)), "true_z_m": repr(float(actual)),
                                   "pred_z_m": repr(float(estimate)),
                                   "abs_error_m": repr(float(abs(estimate - actual))),
                                   "parking_ground_truth": int(park) if mode == "full" else ""})
    return job, result


def completed_jobs(mode):
    """Validate source hashes, result coverage, units, and error arithmetic."""
    path = output_path(mode)
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Existing Meneses CSV schema differs")
        rows = list(reader)
    _, _, tests, data_hash = dataset(mode)
    expected_hash = source_hash()
    allowed = {job_id(job) for job in jobs(mode)}
    grouped = {}
    for row in rows:
        if (row["protocol"] != PROTOCOL or row["mode"] != mode
                or row["source_commit"] != SOURCE_COMMIT or row["source_sha256"] != expected_hash
                or row["data_sha256"] != data_hash or row["engine"] != engine()
                or row["job_id"] not in allowed):
            raise ValueError("Existing Meneses CSV has incompatible source, data or protocol")
        grouped.setdefault(row["job_id"], []).append(row)
    wanted = {(epoch, segment, index) for epoch in SNAPSHOTS
              for segment, (x, _, _, _) in tests.items() for index in range(len(x))}
    for key, group in grouped.items():
        got = [(int(r["epoch"]), int(r["test_segment"]), int(r["window_index"])) for r in group]
        if len(got) != len(wanted) or set(got) != wanted:
            raise ValueError(f"Incomplete or duplicate results for {key}")
        for row in group:
            segment, index = int(row["test_segment"]), int(row["window_index"])
            _, truth, times, parking = tests[segment]
            if (float(row["true_z_m"]) != float(truth[index])
                    or float(row["target_time_s"]) != float(times[index])
                    or not np.isfinite(float(row["pred_z_m"]))
                    or not math.isclose(abs(float(row["pred_z_m"]) - float(row["true_z_m"])),
                                        float(row["abs_error_m"]), abs_tol=1e-9)
                    or (mode == "full" and int(row["parking_ground_truth"]) != int(parking[index]))):
                raise ValueError(f"Meneses result row mismatch in {key}")
    return set(grouped)


def check():
    """Validate source schemas, causal windows, job counts, and model shapes."""
    ensure_data()
    arrays, data_hash = load_data()
    for segment in range(1, 6):
        x, y, times, park = windows(arrays[segment], segment)
        if x.shape[1:] != (WINDOW, 3) or len(x) != len(y) or len(y) != len(times):
            raise ValueError("Meneses window shape changed")
        if segment in TEST_SEGMENTS and not np.any(park):
            raise ValueError("Held-out segment contains no camera-anchored parking labels")
    assert len(jobs("pilot")) == 8 and len(jobs("full")) == 24
    assert dataset("pilot")[0].shape == dataset("full")[0].shape == (819, 40, 3)
    cnn, mlp = TemporalCNN(), CompactMLP()
    assert cnn(torch.zeros(2, 40, 3)).shape == mlp(torch.zeros(2, 40, 3)).shape == (2,)
    print(f"CHECK OK: rows={sum(SOURCE_ROWS)}, data_sha256={data_hash}, CNN_params={sum(p.numel() for p in cnn.parameters())}, MLP8_params={sum(p.numel() for p in mlp.parameters())}", flush=True)


def run(mode, workers):
    """Execute missing pilot or full jobs and append complete result blocks."""
    check()
    if mode == "full":
        pilot_done = completed_jobs("pilot")
        if len(pilot_done) != 8:
            raise ValueError("Training-side numerical pilot must be complete before held-out evaluation")
    done = completed_jobs(mode)
    pending = [job for job in jobs(mode) if job_id(job) not in done]
    print(f"{mode}: {len(done)} complete, {len(pending)} pending, workers={workers}", flush=True)
    if not pending:
        return
    path = output_path(mode)
    if not path.exists():
        with path.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
            handle.flush()
            os.fsync(handle.fileno())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, job): job for job in pending}
        for future in as_completed(futures):
            job, rows = future.result()
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            means = [np.mean([float(r["abs_error_m"]) for r in rows if r["epoch"] == epoch])
                     for epoch in SNAPSHOTS]
            print(f"DONE {job_id(job)}: 200={means[0]:.4f}, 400={means[1]:.4f} m", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "pilot", "full"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be 1-4")
    check() if args.action == "check" else run(args.action, args.workers)
