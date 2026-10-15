"""Benchmark L31 Torch, NumPy, and ONNX inference on one CPU core.

CNN and MLP8 map ``(256, 24)`` features to ``(256, 2)`` positions in
micrometres. Batch 1 processes one row per call; batch 256 processes the full
matrix. Each backend/model condition uses five isolated processes with 50
warm-up and 100 timed calls per batch size. Results are written to JSONL and a
summary CSV.
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
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
RAW = HERE / "onnx_benchmark.jsonl"
SUMMARY = HERE / "onnx_benchmark_summary.csv"
MODELS = ("cnn", "mlp8")
BACKENDS = ("torch", "numpy", "onnx")
CONDITIONS = tuple((backend, model) for model in MODELS for backend in BACKENDS)
REPEATS = 5
WARMUP = 50
TIMED = 100
CORE = 12
TOLERANCE_UM = .01
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(path):
    """Return the streaming SHA-256 digest of a file without loading it at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sources():
    """Validate model sources and the same-batch prediction reference."""
    from portable_benchmark import check_sources as prior_check
    checked = prior_check()
    reference = json.loads((HERE / "portable_benchmark_reference.json").read_text(encoding="utf-8"))
    if sha(HERE / "portable_benchmark_reference.npz") != reference["reference_npz_sha256"]:
        raise ValueError("Frozen same-batch Torch reference changed")
    return checked


def packages():
    """Return pinned runtime versions after rejecting environment drift."""
    import numpy
    import torch
    import onnx
    import onnxruntime
    import onnxscript
    versions = {"python": platform.python_version(), "numpy": numpy.__version__, "torch": torch.__version__,
                "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
                "onnxscript": onnxscript.__version__}
    if (versions["python"] != "3.9.13" or versions["numpy"] != "1.26.4"
            or versions["torch"] != "2.8.0+cpu" or versions["onnx"] != "1.19.1"
            or versions["onnxruntime"] != "1.19.2" or versions["onnxscript"] != "0.7.2"):
        raise ValueError(f"Benchmark environment drift: {versions}")
    return versions


def onnx_options():
    """Create a sequential, single-threaded ONNX Runtime session configuration."""
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options


def onnx_session(model, path=None):
    """Load one CPU-only ONNX session from the canonical or supplied path."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(path or HERE / f"{model}.onnx"),
                                   sess_options=onnx_options(), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("ONNX provider differs from CPU-only protocol")
    return session


def same_batch_reference(model):
    """Load ``(256,2)`` micrometre references for batch sizes 1 and 256."""
    import numpy as np
    with np.load(HERE / "portable_benchmark_reference.npz", allow_pickle=False) as source:
        return {batch: np.array(source[f"{model}_batch{batch}"], copy=True) for batch in (1, 256)}


def parity(model, path=None):
    """Compare ONNX ``(256,2)`` outputs with saved Torch coordinates in µm."""
    import numpy as np
    x = np.load(HERE / "memory_input_features.npy", allow_pickle=False)
    expected = same_batch_reference(model)
    session = onnx_session(model, path)
    def predict(values):
        return session.run(None, {session.get_inputs()[0].name:
                                  np.asarray(values, dtype=np.float64)})[0]
    batch1 = np.concatenate([predict(x[i:i + 1]) for i in range(256)])
    batch256 = predict(x)
    observed = {1: batch1, 256: batch256}
    differences = {}
    for batch in (1, 256):
        output = observed[batch]
        if output.shape != (256, 2) or not np.isfinite(output).all():
            raise ValueError(f"ONNX {model} batch{batch} output shape/finite failure")
        differences[batch] = float(np.max(np.abs(output - expected[batch])))
        if differences[batch] > TOLERANCE_UM:
            raise ValueError(f"ONNX {model} batch{batch} drift {differences[batch]:.9g} um")
    return differences


def export():
    """Export both saved models to ONNX and accept parity-checked files."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    check_sources()
    versions = packages()
    import onnx
    import torch
    from memory_profile import load_model, runtime
    state_runtime = runtime()

    class CNN(torch.nn.Module):
        """Adapt float64 ``(batch,24)`` inputs to the saved float32 CNN."""
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, values):
            return self.model(values.to(torch.float32)).to(torch.float64)

    class MLP8(torch.nn.Module):
        """Embed MLP8 scaling around ``(batch,24)`` to ``(batch,2)`` inference."""
        def __init__(self, state):
            super().__init__()
            self.model = state.model
            for key, value in (("feature_mean", state.feature_mean),
                               ("feature_scale", state.feature_scale),
                               ("target_mean", state.target_mean),
                               ("target_scale", state.target_scale)):
                self.register_buffer(key, torch.as_tensor(value, dtype=torch.float64))
        def forward(self, values):
            x = ((values - self.feature_mean) / self.feature_scale).to(torch.float32)
            scaled = self.model(x).to(torch.float64)
            return scaled * self.target_scale + self.target_mean

    exported = {}
    for model in MODELS:
        state = load_model(model, state_runtime)
        wrapper = CNN(state) if model == "cnn" else MLP8(state)
        wrapper.eval()
        dummy = torch.zeros(2, 24, dtype=torch.float64)
        dimension = torch.export.Dim("batch", min=1, max=256)
        with tempfile.TemporaryDirectory(prefix="onnx_export_", dir=HERE) as temporary:
            candidate = Path(temporary) / f"{model}.onnx"
            program = torch.onnx.export(
                wrapper, (dummy,), dynamo=True, opset_version=18,
                input_names=["features"], output_names=["position_um"],
                dynamic_shapes=({0: dimension},), external_data=False)
            program.save(str(candidate))
            onnx.checker.check_model(str(candidate))
            drift = parity(model, candidate)
            final = HERE / f"{model}.onnx"
            if final.exists():
                if sha(final) != sha(candidate):
                    raise ValueError(f"Existing ONNX export differs: {final}")
            else:
                os.replace(candidate, final)
            exported[model] = {"sha256": sha(final), "bytes": final.stat().st_size,
                               "max_abs_coordinate_drift_um": drift}
    print(json.dumps({"status": "EXPORT_AND_PARITY_PASS", "packages": versions,
                      "models": exported}, ensure_ascii=False), flush=True)


def check():
    """Validate sources, package pins, ONNX files, and numerical parity."""
    provenance = check_sources()
    versions = packages()
    for model in MODELS:
        path = HERE / f"{model}.onnx"
        if not path.exists():
            raise ValueError(f"ONNX export missing: {model}")
        parity(model)
    return {"sources": provenance, "packages": versions,
            "model_sha256": {model: sha(HERE / f"{model}.onnx") for model in MODELS},
            "input_sha256": sha(HERE / "memory_input_features.npy"),
            "reference_sha256": sha(HERE / "portable_benchmark_reference.npz"),
            "script_sha256": sha(__file__)}


def input_indices(repeat, warmup=False):
    """Return deterministic indices for one repeat's warm-up or timed calls."""
    count = WARMUP if warmup else TIMED
    start = (repeat - 1) * TIMED - (WARMUP if warmup else 0)
    return [(start + i) % 256 for i in range(count)]


def model_file(backend, model):
    """Resolve the serialized artifact used by one backend/model condition."""
    if backend == "torch":
        return ROOT / ("checkpoints/study_l31_author_first5.pt" if model == "cnn"
                       else "checkpoints/study_l31_selected_compact.pkl")
    return HERE / (f"portable_{model}.npz" if backend == "numpy" else f"{model}.onnx")


def worker(backend, model, repeat):
    """Measure one backend/model/repeat in an isolated, single-core process.

    Returns 100 latency samples in nanoseconds for batch sizes 1 and 256,
    twenty RSS snapshots in bytes, load timings in milliseconds, and maximum
    coordinate drift in micrometres.
    """
    for name in THREAD_ENV:
        os.environ[name] = "1"
    import gc
    import numpy as np
    import psutil
    import threadpoolctl
    from memory_profile import counters
    process = psutil.Process()
    process.cpu_affinity([CORE])
    process.nice(psutil.NORMAL_PRIORITY_CLASS)
    started = time.perf_counter()
    source = check_sources()
    x = np.load(HERE / "memory_input_features.npy", allow_pickle=False)
    reference = same_batch_reference(model)
    single = [x[i:i + 1] for i in range(256)]
    import_started = time.perf_counter()
    if backend == "torch":
        from memory_profile import load_model, predict, runtime
        rt = runtime()
        loader = lambda: load_model(model, rt)
        predictor = lambda state, values: predict(model, state, values, rt)
        if rt["torch"].get_num_threads() != 1 or rt["torch"].get_num_interop_threads() != 1:
            raise ValueError("PyTorch threads differ from one")
    elif backend == "numpy":
        from portable_inference import PortableEstimator
        loader = lambda: PortableEstimator(model, model_file(backend, model))
        predictor = lambda state, values: state.predict(values)
    else:
        import onnxruntime as ort
        if ort.__version__ != "1.19.2":
            raise ValueError("ONNX Runtime version drift")
        loader = lambda: onnx_session(model)
        predictor = lambda state, values: state.run(
            None, {state.get_inputs()[0].name: np.asarray(values, dtype=np.float64)})[0]
    import_ms = (time.perf_counter() - import_started) * 1000
    # Run the same instrumented NumPy workload before sampling every backend.
    for _ in range(20):
        np.ones((8, 8), dtype=np.float32) @ np.ones((8, 8), dtype=np.float32)
    def sample(stage):
        gc.collect()
        values = []
        for _ in range(5):
            time.sleep(.04)
            values.append({"stage": stage, **counters()})
        return values
    memory = sample("runtime_baseline")
    load_started = time.perf_counter()
    state = loader()
    load_ms = (time.perf_counter() - load_started) * 1000
    ready_wall_ns = time.time_ns()
    memory += sample("model_loaded")
    latency = {}
    drift = {}
    for batch in (1, 256):
        for index in input_indices(repeat, warmup=True):
            predictor(state, single[index] if batch == 1 else x)
        elapsed = []
        maximum = 0.0
        for index in input_indices(repeat):
            values = single[index] if batch == 1 else x
            before = time.perf_counter_ns()
            prediction = predictor(state, values)
            elapsed.append(time.perf_counter_ns() - before)
            expected = reference[1][index:index + 1] if batch == 1 else reference[256]
            difference = float(np.max(np.abs(prediction - expected)))
            if not np.isfinite(prediction).all() or difference > TOLERANCE_UM:
                raise ValueError(f"Numeric parity failed {backend}/{model}/batch{batch}: {difference}")
            maximum = max(maximum, difference)
        latency[str(batch)] = elapsed
        drift[str(batch)] = maximum
        memory += sample(f"post_batch_{batch}")
    native = threadpoolctl.threadpool_info()
    if any(pool.get("num_threads") != 1 for pool in native):
        raise ValueError("Native BLAS thread count differs from one")
    if backend == "numpy" and ("torch" in sys.modules or "sklearn" in sys.modules):
        raise ValueError("NumPy worker imported Torch/sklearn")
    if backend == "onnx" and "torch" in sys.modules:
        raise ValueError("ONNX worker imported Torch")
    return {"backend": backend, "model": model, "repeat": repeat,
            "script_sha256": sha(__file__), "input_sha256": sha(HERE / "memory_input_features.npy"),
            "reference_sha256": sha(HERE / "portable_benchmark_reference.npz"),
            "model_sha256": sha(model_file(backend, model)),
            "source_protocol_sha256": source["protocol_sha256"],
            "onnx_model_sha256": sha(HERE / f"{model}.onnx"),
            "python": platform.python_version(), "numpy": np.__version__,
            "torch_version": "2.8.0+cpu" if backend == "torch" else None,
            "onnxruntime_version": "1.19.2" if backend == "onnx" else None,
            "cpu_affinity": process.cpu_affinity(), "process_priority": int(process.nice()),
            "thread_environment": {name: os.environ[name] for name in THREAD_ENV},
            "native_threadpools": native, "backend_import_ms": import_ms,
            "model_load_ms": load_ms, "ready_wall_ns": ready_wall_ns,
            "serialized_model_bytes": model_file(backend, model).stat().st_size,
            "latency_ns": latency, "max_abs_coordinate_drift_um": drift,
            "memory_samples": memory, "elapsed_wall_sec": time.perf_counter() - started}


def existing():
    """Load and validate completed JSONL process records for safe resumption."""
    if not RAW.exists():
        return {}
    seen = {}
    with RAW.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["backend"], row["model"], row["repeat"])
            if (key in seen or row["backend"] not in BACKENDS or row["model"] not in MODELS
                    or row["repeat"] not in range(1, REPEATS + 1)
                    or row["script_sha256"] != sha(__file__)
                    or row["input_sha256"] != sha(HERE / "memory_input_features.npy")
                    or row["reference_sha256"] != sha(HERE / "portable_benchmark_reference.npz")
                    or row["model_sha256"] != sha(model_file(*key[:2]))
                    or len(row["latency_ns"]["1"]) != TIMED
                    or len(row["latency_ns"]["256"]) != TIMED
                    or len(row["memory_samples"]) != 20
                    or max(row["max_abs_coordinate_drift_um"].values()) > TOLERANCE_UM):
                raise ValueError(f"Existing ONNX benchmark row incompatible: {key}")
            seen[key] = row
    return seen


def run():
    """Launch missing isolated workers in rotated order, then summarize them."""
    checks = check()
    import psutil
    if any("recipe_additional.py" in " ".join(p.info.get("cmdline") or [])
           for p in psutil.process_iter(["cmdline"])):
        raise RuntimeError("Recipe training still active; resource benchmark must wait")
    prior = existing()
    conditions = [(backend, model, repeat) for repeat in range(1, REPEATS + 1)
                  for backend, model in CONDITIONS[(repeat - 1) % len(CONDITIONS):]
                  + CONDITIONS[:(repeat - 1) % len(CONDITIONS)]]
    print(f"BENCHMARK: {len(prior)} complete, {30-len(prior)} pending", flush=True)
    if len(prior) == 30:
        summarize()
        return
    for backend, model, repeat in conditions:
        key = backend, model, repeat
        if key in prior:
            continue
        env = os.environ.copy()
        env.update({name: "1" for name in THREAD_ENV})
        command = [sys.executable, str(Path(__file__).resolve()), "worker", backend, model, str(repeat)]
        launched_wall_ns = time.time_ns()
        result = subprocess.run(command, capture_output=True, text=True, env=env, cwd=ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"Worker failed {key}: {result.stderr[-2500:]}")
        row = json.loads(result.stdout.strip().splitlines()[-1])
        row["cold_start_ms"] = (row["ready_wall_ns"] - launched_wall_ns) / 1_000_000
        if (row["backend"], row["model"], row["repeat"]) != key:
            raise ValueError(f"Worker identity mismatch {key}")
        with RAW.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"DONE {backend}/{model}/repeat{repeat}", flush=True)
    if check() != checks:
        raise ValueError("Model or source changed during benchmark")
    summarize()


def summarize():
    """Aggregate 30 process records into 12 backend/model/batch summary rows."""
    import numpy as np
    records = existing()
    if len(records) != 30:
        raise ValueError("Cannot summarize incomplete 30-process benchmark")
    rows = []
    for model in MODELS:
        for backend in BACKENDS:
            selected = [records[(backend, model, repeat)] for repeat in range(1, REPEATS + 1)]
            for batch in (1, 256):
                calls = [np.array(row["latency_ns"][str(batch)], dtype=np.float64) / 1000
                         for row in selected]
                quantiles = {f"p{int(q)}_us": [float(np.percentile(values, q)) for values in calls]
                             for q in (50, 95, 99)}
                stage = f"post_batch_{batch}"
                rss = [statistics.median(sample["rss_bytes"] for sample in row["memory_samples"]
                                         if sample["stage"] == stage) / 1048576 for row in selected]
                load_rss = [statistics.median(sample["rss_bytes"] for sample in row["memory_samples"]
                                              if sample["stage"] == "model_loaded") / 1048576 for row in selected]
                rows.append({"backend": backend, "model": model, "batch_size": batch,
                             "process_repeats": REPEATS, "timed_calls_per_repeat": TIMED,
                             "latency_p50_us": statistics.median(quantiles["p50_us"]),
                             "latency_p95_us": statistics.median(quantiles["p95_us"]),
                             "latency_p99_us": statistics.median(quantiles["p99_us"]),
                             "rss_loaded_mib": statistics.median(load_rss),
                             "rss_after_batch_mib": statistics.median(rss),
                             "model_load_ms": statistics.median(row["model_load_ms"] for row in selected),
                             "cold_start_ms": statistics.median(row["cold_start_ms"] for row in selected),
                             "backend_import_ms": statistics.median(row["backend_import_ms"] for row in selected),
                             "serialized_model_bytes": selected[0]["serialized_model_bytes"],
                             "max_abs_coordinate_drift_um": max(row["max_abs_coordinate_drift_um"][str(batch)]
                                                                  for row in selected),
                             "source_jsonl_sha256": sha(RAW),
                             "script_sha256": sha(__file__)})
    with SUMMARY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"SUMMARY OK: {len(rows)} backend/model/batch combinations, 30 isolated processes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("export", "check", "parity", "run", "worker", "summarize"))
    parser.add_argument("backend", nargs="?")
    parser.add_argument("model", nargs="?")
    parser.add_argument("repeat", nargs="?", type=int)
    args = parser.parse_args()
    if args.action == "export":
        export()
    elif args.action == "check":
        print(json.dumps(check()["model_sha256"]), flush=True)
    elif args.action == "parity":
        check_sources()
        for name in MODELS:
            print(name, parity(name), flush=True)
    elif args.action == "run":
        run()
    elif args.action == "summarize":
        summarize()
    else:
        if args.backend not in BACKENDS or args.model not in MODELS or args.repeat not in range(1, 6):
            parser.error("worker requires backend, model and repeat 1..5")
        print(json.dumps(worker(args.backend, args.model, args.repeat), ensure_ascii=False), flush=True)
