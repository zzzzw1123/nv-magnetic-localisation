"""Validate and summarize the retained 30-process backend campaign.

The script reads hash-bound Torch, NumPy, and ONNX process records and writes
``onnx_detailed_summary.csv`` in long format. Each median and range is computed
from five process-level values. Existing measurements and protected inputs are
verified without repeating inference, timing, or memory profiling.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE / "onnx_benchmark.jsonl"
LEGACY_SUMMARY = HERE / "onnx_benchmark_summary.csv"
OUTPUT = HERE / "onnx_detailed_summary.csv"
EXPECTED_RAW_SHA256 = "7e4f95aa272f53be49d8f8b549454f1f04185b37814c2cbac8c94a773f22992e"
EXPECTED_SCRIPT_SHA256 = "cc8d47ac541e3471c751b9b54c32734f9d94eedfa3c36b42ac64c1db4b3c4482"
BACKENDS = ("torch", "numpy", "onnx")
MODELS = ("cnn", "mlp8")
STAGES = ("runtime_baseline", "model_loaded", "post_batch_1", "post_batch_256")
FIELDS = (
    "record_type", "backend", "model", "batch_or_stage", "metric", "unit",
    "median", "min", "max", "processes", "sourcehash",
)
MIB = 1048576


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def model_path(backend, model):
    """Resolve the serialized state used by one backend/model condition."""
    if backend == "torch":
        return ROOT / ("checkpoints/study_l31_author_first5.pt" if model == "cnn"
                       else "checkpoints/study_l31_selected_compact.pkl")
    return HERE / (f"portable_{model}.npz" if backend == "numpy" else f"{model}.onnx")


def load_and_validate():
    """Load 30 process records and validate their inputs, states, and measurements."""
    if sha(RAW) != EXPECTED_RAW_SHA256:
        raise ValueError("Retained raw benchmark fingerprint changed")
    script = HERE / "onnx_benchmark.py"
    if sha(script) != EXPECTED_SCRIPT_SHA256:
        raise ValueError("Hash-bound benchmark source changed")
    protected = {RAW: sha(RAW), script: sha(script), LEGACY_SUMMARY: sha(LEGACY_SUMMARY)}
    records = {}
    for line_number, line in enumerate(RAW.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["backend"], record["model"], record["repeat"])
        if key in records:
            raise ValueError(f"Duplicate process record on line {line_number}: {key}")
        records[key] = record
    expected_keys = {(b, m, r) for b in BACKENDS for m in MODELS for r in range(1, 6)}
    if set(records) != expected_keys:
        raise ValueError("Expected all 30 backend/model/repeat combinations")
    input_path = HERE / "memory_input_features.npy"
    reference_path = HERE / "portable_benchmark_reference.npz"
    protocol_path = HERE / "portable_benchmark_protocol.json"
    for path in (input_path, reference_path, protocol_path):
        protected[path] = sha(path)
    inputs = np.load(input_path, allow_pickle=False)
    if inputs.shape != (256, 24) or not np.isfinite(inputs).all():
        raise ValueError("Retained benchmark input shape/finite-value check failed")
    with np.load(reference_path, allow_pickle=False) as reference:
        if any(reference[f"{m}_batch{b}"].shape != (256, 2)
               or not np.isfinite(reference[f"{m}_batch{b}"]).all()
               for m in MODELS for b in (1, 256)):
            raise ValueError("Retained prediction-reference shape check failed")
    for backend, model, repeat in sorted(records):
        record = records[backend, model, repeat]
        if (record["script_sha256"] != EXPECTED_SCRIPT_SHA256
                or record["input_sha256"] != protected[input_path]
                or record["reference_sha256"] != protected[reference_path]
                or record["source_protocol_sha256"] != protected[protocol_path]):
            raise ValueError("Process fingerprint mismatch")
        source_model = model_path(backend, model)
        onnx_model = HERE / f"{model}.onnx"
        for path in (source_model, onnx_model):
            protected[path] = sha(path)
        if record["model_sha256"] != protected[source_model] or record["onnx_model_sha256"] != protected[onnx_model]:
            raise ValueError("Retained weight/export fingerprint mismatch")
        if record["serialized_model_bytes"] != source_model.stat().st_size:
            raise ValueError("Retained serialised-size mismatch")
        if record["cpu_affinity"] != [12] or record["process_priority"] != 32:
            raise ValueError("Retained affinity/priority setting mismatch")
        if any(v != "1" for v in record["thread_environment"].values()):
            raise ValueError("Retained thread setting mismatch")
        if any(pool["num_threads"] != 1 for pool in record["native_threadpools"]):
            raise ValueError("Retained native threadpool setting mismatch")
        memory = record["memory_samples"]
        if len(memory) != 20 or {s["stage"] for s in memory} != set(STAGES):
            raise ValueError("Incomplete memory-stage record")
        for stage in STAGES:
            samples = [s["rss_bytes"] for s in memory if s["stage"] == stage]
            if len(samples) != 5 or not all(finite(v) and v >= 0 for v in samples):
                raise ValueError("Expected five valid RSS samples in each stage")
        if set(record["latency_ns"]) != {"1", "256"}:
            raise ValueError("Incomplete batch-size record")
        for batch in ("1", "256"):
            values = record["latency_ns"][batch]
            drift = record["max_abs_coordinate_drift_um"][batch]
            if len(values) != 100 or not all(finite(v) and v > 0 for v in values):
                raise ValueError("Expected 100 retained timed calls in each batch")
            if not finite(drift) or not 0 <= drift <= 0.01:
                raise ValueError("Retained coordinate-consistency gate failed")
        for field in ("backend_import_ms", "model_load_ms", "cold_start_ms"):
            if not finite(record[field]) or record[field] < 0:
                raise ValueError("Invalid retained loading/startup time")
    return records, protected


def summarise(records):
    """Return long-format rows built from five process-level values per condition."""
    rows = []

    def add(record_type, backend, model, batch_or_stage, metric, unit, values):
        if len(values) != 5 or not all(finite(v) for v in values):
            raise ValueError("Each statistical row must retain five process values")
        rows.append({
            "record_type": record_type, "backend": backend, "model": model,
            "batch_or_stage": batch_or_stage, "metric": metric, "unit": unit,
            "median": statistics.median(values), "min": min(values), "max": max(values),
            "processes": 5, "sourcehash": EXPECTED_RAW_SHA256,
        })

    for model in MODELS:
        for backend in BACKENDS:
            processes = [records[backend, model, repeat] for repeat in range(1, 6)]
            # First take five stage snapshots' median within each process.
            stage_rss = {
                stage: [statistics.median(s["rss_bytes"] for s in record["memory_samples"]
                                          if s["stage"] == stage) for record in processes]
                for stage in STAGES
            }
            for stage in STAGES:
                add("memory", backend, model, stage, "process_rss", "MiB",
                    [v / MIB for v in stage_rss[stage]])
                # Baseline subtraction is paired within each process; retain negatives.
                add("memory", backend, model, stage, "rss_increment_from_runtime_baseline", "MiB",
                    [(stage_value - baseline) / MIB
                     for stage_value, baseline in zip(stage_rss[stage], stage_rss["runtime_baseline"])])
            for batch in (1, 256):
                for q in (50, 95, 99):
                    # Reproduce the legacy NumPy default linear percentile exactly.
                    per_process = [float(np.percentile(np.asarray(record["latency_ns"][str(batch)], dtype=np.float64) / 1000, q))
                                   for record in processes]
                    add("latency", backend, model, f"batch_{batch}", f"latency_p{q}", "us", per_process)
                add("parity", backend, model, f"batch_{batch}", "max_abs_coordinate_drift", "um",
                    [record["max_abs_coordinate_drift_um"][str(batch)] for record in processes])
                # This is coverage from the hash-bound schedule, not additional inference.
                add("coverage", backend, model, f"batch_{batch}", "distinct_input_rows_checked_per_process", "rows",
                    [100 if batch == 1 else 256] * 5)
            for source_field, metric in (
                ("backend_import_ms", "backend_import"),
                ("model_load_ms", "model_load"),
                # Raw field was named cold_start_ms, but measures launch-to-ready.
                ("cold_start_ms", "launch_to_ready"),
            ):
                add("loading", backend, model, "process", metric, "ms",
                    [record[source_field] for record in processes])
            add("serialization", backend, model, "model_state", "serialized_model_size", "bytes",
                [record["serialized_model_bytes"] for record in processes])
    return rows


def verify_legacy_summary(rows):
    """Match long-format medians to the retained twelve-row summary."""
    lookup = {(r["record_type"], r["backend"], r["model"], r["batch_or_stage"], r["metric"]): r for r in rows}
    with LEGACY_SUMMARY.open(encoding="utf-8", newline="") as handle:
        original = list(csv.DictReader(handle))
    if len(original) != 12:
        raise ValueError("Expected the retained twelve-row main summary")
    for old in original:
        b, m, batch = old["backend"], old["model"], old["batch_size"]
        for percentile in (50, 95, 99):
            new = lookup["latency", b, m, "batch_" + batch, f"latency_p{percentile}"]["median"]
            if not math.isclose(new, float(old[f"latency_p{percentile}_us"]), rel_tol=0, abs_tol=1e-8):
                raise ValueError("Retained latency summary does not match raw records")
        for old_field, stage in (("rss_loaded_mib", "model_loaded"), ("rss_after_batch_mib", "post_batch_" + batch)):
            if lookup["memory", b, m, stage, "process_rss"]["median"] != float(old[old_field]):
                raise ValueError("Retained RSS summary does not match raw records")
        for old_field, metric in (("backend_import_ms", "backend_import"), ("model_load_ms", "model_load"), ("cold_start_ms", "launch_to_ready")):
            if lookup["loading", b, m, "process", metric]["median"] != float(old[old_field]):
                raise ValueError("Retained loading/startup summary mismatch")
        if lookup["serialization", b, m, "model_state", "serialized_model_size"]["median"] != int(old["serialized_model_bytes"]):
            raise ValueError("Retained size summary mismatch")
        if lookup["parity", b, m, "batch_" + batch, "max_abs_coordinate_drift"]["max"] != float(old["max_abs_coordinate_drift_um"]):
            raise ValueError("Retained parity summary mismatch")
        if old["source_jsonl_sha256"] != EXPECTED_RAW_SHA256 or old["script_sha256"] != EXPECTED_SCRIPT_SHA256:
            raise ValueError("Retained main-summary fingerprint mismatch")


def main():
    """Validate inputs, write the long-format CSV, and recheck protected hashes."""
    records, protected = load_and_validate()
    rows = summarise(records)
    verify_legacy_summary(rows)
    for path, expected_hash in protected.items():
        if sha(path) != expected_hash:
            raise ValueError("Protected source changed during postprocessing")
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for path, expected_hash in protected.items():
        if sha(path) != expected_hash:
            raise ValueError("Protected source changed after CSV write")
    all_single_rows = {((repeat - 1) * 100 + index) % 256 for repeat in range(1, 6) for index in range(100)}
    if all_single_rows != set(range(256)):
        raise ValueError("Retained single-row schedule coverage mismatch")
    negative = [r for r in rows if r["metric"] == "rss_increment_from_runtime_baseline" and r["min"] < 0]
    print(f"Wrote {len(rows)} long-format rows from 30 processes; no measurement rerun")
    print("CSV:", OUTPUT)
    print("Raw/source/weights/input/reference/legacy summary fingerprints preserved")
    print("Parity coverage: 256 input rows per backend/model; batch 1 union across five processes, batch 256 in every timed call")
    print("Negative process-relative RSS ranges retained:", len(negative))
    for row in negative:
        print(row["backend"], row["model"], row["batch_or_stage"], "median/min/max", row["median"], row["min"], row["max"])


if __name__ == "__main__":
    main()
