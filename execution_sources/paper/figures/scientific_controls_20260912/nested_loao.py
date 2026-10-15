"""Post-hoc nested acquisition evaluation of the original L31 selector.

All twelve L31 candidates and its executed one-micrometre tie rule are kept.
Each of eight outer acquisitions is absent from every inner fit and decision.
The original A6-A8 archive replay is a separate historical evaluation and is
not replaced by this internal, retrospective nested LOAO diagnostic.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from qml_microrobot.study_l31_compact import (  # noqa: E402
    fit_compact_mlp, fitted_parameter_count, predict_compact_mlp,
)
from scripts.run_study_l31_compact_nonlinear_bridge import (  # noqa: E402
    candidate_ids, fit_candidate, predict_candidate,
)
from recipe_factorial import data_and_folds  # noqa: E402

PROTOCOL_ID = "NV_L31_NESTED_LOAO_POSTHOC_V1"
PROTOCOL_PATH = ROOT / "study_l/l31_compact_nonlinear_bridge_protocol.json"
OUTPUT = HERE / "nested_loao.json"
OUTER_FOLDS = tuple(f"A{i}" for i in range(1, 9))
OUTER_MLP_SEEDS = (20260804, 20260805, 20260806)


def source_hash():
    digest = hashlib.sha256()
    for path in (Path(__file__), HERE / "recipe_factorial.py", PROTOCOL_PATH,
                 ROOT / "scripts/run_study_l31_compact_nonlinear_bridge.py",
                 ROOT / "src/qml_microrobot/study_l31_compact.py",
                 ROOT / "src/qml_microrobot/study_l30_author_cnn.py",
                 ROOT / "src/qml_microrobot/study_l_session.py",
                 ROOT / "src/qml_microrobot/study_l_nv.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def engine():
    return f"Python {platform.python_version()}; NumPy {np.__version__}; PyTorch {torch.__version__}; CPU"


def protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def original_selection_check(config):
    path = ROOT / "results/study_l31_compact_candidate_cv.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = candidate_ids(config)
    if len(candidates) != 12 or len(rows) != 60:
        raise ValueError("Historical L31 candidate inventory changed")
    summaries = []
    for candidate in candidates:
        values = [float(row["error_um"]) for row in rows if row["candidate"] == candidate]
        if len(values) != 5:
            raise ValueError(f"Historical L31 folds missing for {candidate}")
        summaries.append({"candidate": candidate, "mean_error_um": float(np.mean(values)),
                          "fitted_parameters": fitted_parameter_count(candidate)})
    minimum = min(row["mean_error_um"] for row in summaries)
    eligible = [row for row in summaries if row["mean_error_um"] <= minimum + 1.0]
    chosen = min(eligible, key=lambda row: (row["fitted_parameters"],
                                           row["mean_error_um"], row["candidate"]))
    historical = json.loads((ROOT / "results/study_l31_compact_nonlinear_bridge_result.json").read_text(encoding="utf-8"))
    if chosen["candidate"] != "mlp8_fixed" or chosen["candidate"] != historical["selected_candidate"]["candidate"]:
        raise ValueError("Original L31 selection rule did not reproduce its frozen choice")
    if not np.isclose(chosen["mean_error_um"], float(historical["selected_candidate"]["mean_error_um"]), atol=1e-10):
        raise ValueError("Original L31 selection mean changed")
    return candidates


def check():
    x, y, sessions, order, data_hash = data_and_folds()
    if len(order) != 8 or x.shape != (819, 24) or y.shape != (819, 2):
        raise ValueError("Frozen NV data changed")
    config = protocol()
    candidates = original_selection_check(config)
    if config["candidates"]["compact_mlp"] != {"hidden_widths": [8, 16, 32],
            "epochs": 1500, "batch_size": 128, "learning_rate": 0.005,
            "seed": 20260804}:
        raise ValueError("Original compact MLP training protocol changed")
    print(f"CHECK OK: eight outer acquisitions, twelve L31 candidates, original selection reproduced, data_sha256={data_hash}", flush=True)


def empty_result(data_hash):
    return {"protocol_id": PROTOCOL_ID,
            "role": "retrospective internal nested LOAO; separate from archive-ordered replay",
            "source_sha256": source_hash(), "data_sha256": data_hash,
            "engine": engine(), "original_l31_protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            "selection_rule": "within 1 um of minimum inner equal-acquisition mean; then fewest parameters, lower mean, lexical candidate id",
            "selection_seed": 20260804, "outer_mlp_seeds": list(OUTER_MLP_SEEDS),
            "folds": {}}


def existing_result(data_hash, sessions, order):
    if not OUTPUT.exists():
        return empty_result(data_hash)
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    expected = empty_result(data_hash)
    for key in expected:
        if key != "folds" and result.get(key) != expected[key]:
            raise ValueError(f"Existing nested output incompatible: {key}")
    if not isinstance(result.get("folds"), dict):
        raise ValueError("Existing nested output has no fold dictionary")
    for fold, item in result["folds"].items():
        if fold not in OUTER_FOLDS or item["outer_fold"] != fold:
            raise ValueError("Unknown outer fold in existing nested output")
        test_ids = set(np.flatnonzero(sessions == order[int(fold[1:]) - 1]).tolist())
        if len(item["inner_rows"]) != 84 or len(item["candidate_summaries"]) != 12:
            raise ValueError("Incomplete inner selection in existing nested output")
        if any(row["train_rows"] + row["validation_rows"] != 819-len(test_ids)
               or row["validation_session"] == order[int(fold[1:]) - 1]
               for row in item["inner_rows"]):
            raise ValueError("Outer acquisition leaked into inner selection")
        for outcome in item["outer_outcomes"]:
            records = outcome["predictions"]
            if len(records) != len(test_ids) or {r["row_index"] for r in records} != test_ids:
                raise ValueError("Incomplete outer predictions")
            if not np.isclose(np.mean([r["error_um"] for r in records]), outcome["mean_error_um"], atol=1e-10):
                raise ValueError("Outer mean does not match raw predictions")
    return result


def write_result(result):
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, OUTPUT)


def fit_outer(fold):
    x, y, sessions, order, _ = data_and_folds()
    config = protocol()
    candidates = candidate_ids(config)
    held_out_session = order[int(fold[1:]) - 1]
    outer_test = np.flatnonzero(sessions == held_out_session)
    outer_train = np.flatnonzero(sessions != held_out_session)
    inner_sessions = [session for session in order if session != held_out_session]
    if len(inner_sessions) != 7 or set(outer_train) & set(outer_test):
        raise ValueError("Outer split invalid")
    started = time.perf_counter()
    inner_rows = []
    for candidate in candidates:
        for validation_session in inner_sessions:
            inner_train = np.flatnonzero((sessions != held_out_session) & (sessions != validation_session))
            inner_val = np.flatnonzero(sessions == validation_session)
            if set(inner_train) & set(outer_test) or set(inner_val) & set(outer_test):
                raise ValueError("Outer test leaked into inner fit")
            model = fit_candidate(candidate, x[inner_train], y[inner_train], config)
            prediction = predict_candidate(candidate, model, x[inner_val])
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"Non-finite inner prediction: {fold}, {candidate}")
            inner_rows.append({"candidate": candidate, "validation_session": validation_session,
                               "train_rows": len(inner_train), "validation_rows": len(inner_val),
                               "mean_error_um": float(np.linalg.norm(prediction-y[inner_val], axis=1).mean())})
    summaries = []
    for candidate in candidates:
        errors = [row["mean_error_um"] for row in inner_rows if row["candidate"] == candidate]
        if len(errors) != 7:
            raise ValueError("Candidate inner-fold coverage incomplete")
        summaries.append({"candidate": candidate, "fitted_parameters": fitted_parameter_count(candidate),
                          "mean_error_um": float(np.mean(errors)), "worst_error_um": float(np.max(errors))})
    minimum = min(row["mean_error_um"] for row in summaries)
    eligible = [row for row in summaries if row["mean_error_um"] <= minimum + 1.0]
    selected = min(eligible, key=lambda row: (row["fitted_parameters"],
                                             row["mean_error_um"], row["candidate"]))
    candidate = selected["candidate"]
    seeds = OUTER_MLP_SEEDS if candidate.startswith("mlp") else (None,)
    outer_outcomes = []
    for seed in seeds:
        if seed is None:
            model = fit_candidate(candidate, x[outer_train], y[outer_train], config)
            prediction = predict_candidate(candidate, model, x[outer_test])
        else:
            width = int(candidate.split("_", 1)[0][3:])
            settings = config["candidates"]["compact_mlp"]
            model = fit_compact_mlp(x[outer_train], y[outer_train], width,
                                    seed=seed, epochs=int(settings["epochs"]),
                                    batch_size=int(settings["batch_size"]),
                                    learning_rate=float(settings["learning_rate"]))
            prediction = predict_compact_mlp(model, x[outer_test])
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"Non-finite outer prediction: {fold}, {candidate}, {seed}")
        records = []
        for row_index, estimate in zip(outer_test, prediction):
            truth = y[row_index]
            records.append({"row_index": int(row_index),
                            "true_x_um": float(truth[0]), "true_y_um": float(truth[1]),
                            "pred_x_um": float(estimate[0]), "pred_y_um": float(estimate[1]),
                            "error_um": float(np.linalg.norm(estimate - truth))})
        outer_outcomes.append({"seed": seed, "mean_error_um": float(np.mean([r["error_um"] for r in records])),
                               "predictions": records})
    return {"outer_fold": fold, "held_out_session": held_out_session,
            "outer_train_rows": len(outer_train), "outer_test_rows": len(outer_test),
            "inner_rows": inner_rows, "candidate_summaries": summaries,
            "selected_candidate": selected, "eligible_candidates": [row["candidate"] for row in eligible],
            "outer_outcomes": outer_outcomes,
            "outer_seed_mean_error_um": float(np.mean([r["mean_error_um"] for r in outer_outcomes])),
            "fit_wall_sec_diagnostic": time.perf_counter()-started}


def run(workers):
    check()
    _, _, sessions, order, data_hash = data_and_folds()
    result = existing_result(data_hash, sessions, order)
    pending = [fold for fold in OUTER_FOLDS if fold not in result["folds"]]
    print(f"NESTED LOAO: {len(result['folds'])} complete, {len(pending)} pending, workers={workers}", flush=True)
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fit_outer, fold): fold for fold in pending}
        for future in as_completed(futures):
            item = future.result()
            fold = item["outer_fold"]
            result["folds"][fold] = item
            write_result(result)
            print(f"DONE {fold}: selected {item['selected_candidate']['candidate']}, outer seed mean {item['outer_seed_mean_error_um']:.6f} um", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be 1-4")
    check() if args.action == "check" else run(args.workers)
