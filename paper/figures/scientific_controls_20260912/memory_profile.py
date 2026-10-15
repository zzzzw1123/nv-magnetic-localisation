"""Measure Windows process memory for saved CNN and MLP8 models.

Sample working-set RSS and private committed memory around loading and
inference. PrivateUsage differs from resident memory; lifetime peak counters
include imports, model loading and previous stages.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MIB = 1024 ** 2
MANIFEST = ROOT / 'results/study_l31_evidence_manifest.json'
EXPECTED = {
    'cnn': ('checkpoints/study_l31_author_first5.pt', '50f40ae12a7d46aa7db132fccef6a4b6b445c841ed517752a8d65c6b917da508'),
    'mlp8': ('checkpoints/study_l31_selected_compact.pkl', 'bbe58dd0e7518f63511dcbc00449ab01a29e1eefc66fbfcbebda7e18d2826af6'),
}
METRICS = ('rss_bytes', 'private_commit_bytes', 'process_lifetime_peak_rss_bytes',
           'process_lifetime_peak_commit_bytes')


class CountersEx(ctypes.Structure):
    """Windows ``PROCESS_MEMORY_COUNTERS_EX`` layout used by PSAPI."""
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
            'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]


def sha(path):
    """Return the SHA-256 digest of a source, checkpoint, or result file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def counters():
    """Read current and lifetime Windows process counters in bytes."""
    if os.name != 'nt':
        raise RuntimeError('This protocol requires Windows PSAPI counters.')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    fn = psapi.GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(CountersEx), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    value = CountersEx()
    value.cb = ctypes.sizeof(value)
    if not fn(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'rss_bytes': int(value.WorkingSetSize),
            'private_commit_bytes': int(value.PrivateUsage),
            'process_lifetime_peak_rss_bytes': int(value.PeakWorkingSetSize),
            'process_lifetime_peak_commit_bytes': int(value.PeakPagefileUsage),
            'page_fault_count': int(value.PageFaultCount)}


def verify_sources():
    """Check checkpoint and input-file checksums against the stored manifest."""
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert manifest['status'] == 'L31_COMPACT_NONLINEAR_BRIDGE_EVIDENCE_FROZEN'
    entries = {entry['path']: entry for entry in manifest['entries']}
    checks = {}
    needed = [p for p, _ in EXPECTED.values()] + [
        'checkpoints/study_l31_author_first5.json',
        'src/qml_microrobot/study_l31_compact.py',
        'scripts/run_study_l31_compact_nonlinear_bridge.py',
        'results/study_l31_compact_replay_rows.csv',
        'results/study_l31_compact_resource.csv',
        'results/study_l31_compact_nonlinear_bridge_result.json']
    for rel in needed:
        digest = sha(ROOT / rel)
        assert digest == entries[rel]['sha256'], ('Manifest hash mismatch', rel)
        checks[rel] = digest
    for model, (rel, expected) in EXPECTED.items():
        assert checks[rel] == expected, model
    meta = json.loads((ROOT / 'checkpoints/study_l31_author_first5.json').read_text())
    assert meta['seed'] == 20260804 and meta['state_sha256'] == EXPECTED['cnn'][1]
    checks['results/study_l31_evidence_manifest.json'] = sha(MANIFEST)
    return checks


def runtime():
    """Load the pinned scientific runtime and enforce deterministic single-thread use."""
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import torch
    import sklearn
    import psutil
    import pickle
    from qml_microrobot.study_l30_author_cnn import (
        AuthorLocalizationCNN, predict_author_cnn, load_author_data)
    from qml_microrobot.study_l31_compact import predict_compact_mlp
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260912)
    return locals()


def load_model(model, rt):
    """Load a saved estimator after verifying its metadata."""
    # Permit pickle only for the exact manifest-bound L31 compact-model artifact.
    rel, expected = EXPECTED[model]
    assert sha(ROOT / rel) == expected
    if model == 'cnn':
        obj = rt['AuthorLocalizationCNN']()
        state = rt['torch'].load(ROOT / rel, map_location='cpu', weights_only=True)
        obj.load_state_dict(state)
        del state
        obj.eval()
        assert sum(p.numel() for p in obj.parameters()) == 721858
    else:
        with (ROOT / rel).open('rb') as handle:
            obj = rt['pickle'].load(handle)
        assert type(obj).__module__ == 'qml_microrobot.study_l31_compact'
        assert type(obj).__name__ == 'CompactMlpState' and obj.fitted_parameters == 218
        obj.model.eval()
    return obj


def predict(model, obj, values, rt):
    """Return model coordinates with shape ``(n,2)`` in micrometres."""
    func = rt['predict_author_cnn'] if model == 'cnn' else rt['predict_compact_mlp']
    return func(obj, values)


def prepare():
    """Prepare the 256 benchmark inputs and check reference predictions."""
    checks = verify_sources()
    rt = runtime()
    np = rt['np']
    data = rt['load_author_data'](ROOT)
    sessions = list(dict.fromkeys(data.session_ids.tolist()))
    ids = np.flatnonzero(np.isin(data.session_ids, sessions[5:]))
    assert len(ids) == 256 and len(sessions) == 8
    features = np.array(data.features[ids], dtype=np.float64, copy=True)
    assert features.shape == (256, 24) and np.isfinite(features).all()
    truth = data.positions_um[ids]
    replay_sessions = data.session_ids[ids]
    expected_rows = list(csv.DictReader((ROOT / 'results/study_l31_compact_replay_rows.csv').open()))
    prediction_checks = {}
    for model in ('cnn', 'mlp8'):
        obj = load_model(model, rt)
        batch256_prediction = predict(model, obj, features, rt)
        assert batch256_prediction.shape == (256, 2) and np.isfinite(batch256_prediction).all()
        yhat = np.empty_like(batch256_prediction)
        field = 'author_cnn_error_um' if model == 'cnn' else 'compact_error_um'
        observed = []
        for row in expected_rows:
            mask = replay_sessions == row['test_session']
            assert int(mask.sum()) == int(row['test_rows'])
            # Match the replay reference's per-acquisition call sizes (52, 102, 102);
            # retain the float32 batch-256 path as a separate numerical diagnostic.
            yhat[mask] = predict(model, obj, features[mask], rt)
            mean = float(np.linalg.norm(yhat[mask] - truth[mask], axis=1).mean())
            batch_mean = float(np.linalg.norm(batch256_prediction[mask] - truth[mask], axis=1).mean())
            reference = float(row[field])
            difference = abs(mean - reference)
            # Apply the frozen 1e-6 µm tolerance to the original call path.
            assert difference <= 1e-6, (model, row['test_session'], mean, reference, difference)
            observed.append({'session': row['test_session'], 'rows': int(mask.sum()),
                             'observed_mean_error_um': mean, 'frozen_mean_error_um': reference,
                             'absolute_difference_um': difference,
                             'single_batch256_mean_error_um': batch_mean,
                             'single_batch256_minus_frozen_um': batch_mean-reference})
        prediction_checks[model] = {'rows': observed,
            'equal_acquisition_mean_error_um': float(np.mean([r['observed_mean_error_um'] for r in observed])),
            'original_per_acquisition_prediction_sha256': hashlib.sha256(yhat.tobytes()).hexdigest(),
            'prediction_sha256': hashlib.sha256(batch256_prediction.tobytes()).hexdigest(),
            'batch256_vs_original_max_coordinate_difference_um': float(np.max(np.abs(batch256_prediction-yhat))),
            'batch256_numerics':'Batch256 float32 path retained separately; equivalence to frozen means is checked using original per-acquisition calls.'}
        del obj, yhat, batch256_prediction
        gc.collect()
    from qml_microrobot.study_l_nv import _inventory_member
    for name in ('para_list.npy', 'para_std_list.npy', 'x_pos.npy', 'y_pos.npy'):
        p = _inventory_member(ROOT, name)
        checks[str(p.relative_to(ROOT)).replace('\\', '/')] = sha(p)
    for rel in ('results/study_l1_archive_member_structure.csv',
                'src/qml_microrobot/study_l30_author_cnn.py',
                'src/qml_microrobot/study_l_session.py', 'src/qml_microrobot/study_l_nv.py'):
        checks[rel] = sha(ROOT / rel)
    np.save(OUT / 'memory_input_features.npy', features, allow_pickle=False)
    result = {'status': 'FITTED_CHECKPOINTS_REPLAY_VALIDATED', 'source_sha256': checks,
              'features_sha256': sha(OUT / 'memory_input_features.npy'),
              'input_shape': list(features.shape), 'input_rows': ids.tolist(),
              'input_scope': 'All 256 historical A6-A8 replay observations, original log features.',
              'checkpoint_scope': 'L31 seed20260804 first-five-acquisition fitted states; not new matched-control fits.',
              'validation_tolerance_um': 1e-6, 'prediction_checks': prediction_checks,
              'model_training_performed': False, 'future_data_opened': False}
    write_json('memory_input_provenance.json', result)
    print(json.dumps({'status': result['status'], 'models': prediction_checks}), flush=True)


def sample_stage(stage, started, elapsed_operation=0.):
    """Collect five process-memory snapshots for one named stage."""
    gc.collect()
    samples = []
    for i in range(5):
        time.sleep(.04)
        snap = counters()
        snap.update(stage=stage, sample=i + 1, elapsed_wall_sec=time.perf_counter() - started,
                    preceding_operation_wall_sec=elapsed_operation)
        for metric in METRICS:
            snap[metric.replace('_bytes', '_MiB')] = snap[metric] / MIB
        samples.append(snap)
    return samples


def profile_child(model, repeat):
    """Profile one model in one fresh process and write its raw JSON record."""
    started = time.perf_counter()
    source_hashes = verify_sources()
    rt = runtime()
    np, torch, psutil = rt['np'], rt['torch'], rt['psutil']
    provenance = json.loads((OUT / 'memory_input_provenance.json').read_text(encoding='utf-8'))
    assert sha(OUT / 'memory_input_features.npy') == provenance['features_sha256']
    features = np.load(OUT / 'memory_input_features.npy', allow_pickle=False)
    batch_one = features[:1]
    proc = psutil.Process()
    # Use one library-only warm baseline without prewarming either architecture.
    for _ in range(20):
        product = torch.ones((8, 8), dtype=torch.float32) @ torch.ones((8, 8), dtype=torch.float32)
    del product
    for _ in range(10):
        counters()
        proc.memory_info()
    records = sample_stage('runtime_baseline', started)
    baseline_rss = statistics.median(r['rss_bytes'] for r in records)
    baseline_private = statistics.median(r['private_commit_bytes'] for r in records)
    tick = time.perf_counter()
    obj = load_model(model, rt)
    records += sample_stage('model_loaded', started, time.perf_counter() - tick)
    prediction_digests = {}
    for batch, values in ((1, batch_one), (256, features)):
        tick = time.perf_counter()
        for _ in range(50):
            output = predict(model, obj, values, rt)
        for _ in range(100):
            output = predict(model, obj, values, rt)
        assert np.isfinite(output).all()
        prediction_digests[str(batch)] = hashlib.sha256(output.tobytes()).hexdigest()
        del output
        records += sample_stage('post_batch_' + str(batch), started, time.perf_counter() - tick)
    assert prediction_digests['256'] == provenance['prediction_checks'][model]['prediction_sha256']
    for row in records:
        row.update(model=model, repeat=repeat, pid=os.getpid())
        row['rss_increment_from_runtime_baseline_bytes'] = row['rss_bytes'] - baseline_rss
        row['private_commit_increment_from_runtime_baseline_bytes'] = row['private_commit_bytes'] - baseline_private
        row['rss_increment_from_runtime_baseline_MiB'] = row['rss_increment_from_runtime_baseline_bytes'] / MIB
        row['private_commit_increment_from_runtime_baseline_MiB'] = row['private_commit_increment_from_runtime_baseline_bytes'] / MIB
    last_psutil = proc.memory_info()._asdict()
    result = {'model': model, 'repeat': repeat, 'pid': os.getpid(), 'samples': records,
              'prediction_sha256_by_batch': prediction_digests, 'checkpoint_sha256': EXPECTED[model][1],
              'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                  'sklearn': rt['sklearn'].__version__, 'psutil': psutil.__version__,
                  'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
                  'process_threads_at_end': proc.num_threads(), 'platform': platform.platform(),
                  'processor': platform.processor(), 'physical_memory_bytes': psutil.virtual_memory().total},
              'psutil_end_crosscheck': last_psutil,
              'elapsed_total_sec': time.perf_counter() - started}
    assert torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1
    assert verify_sources() == source_hashes
    write_json(f'memory_raw_{repeat:02}_{model}.json', result)
    print(json.dumps({'model': model, 'repeat': repeat, 'pid': os.getpid(),
                      'completed': True, 'elapsed_sec': result['elapsed_total_sec']}), flush=True)


def self_test():
    """Cross-check PSAPI and psutil counters with a touched 8 MiB allocation."""
    import psutil
    initial = counters()
    buffer = bytearray(8 * MIB)
    for i in range(0, len(buffer), 4096):
        buffer[i] = 1
    allocated = counters()
    other = psutil.Process().memory_info()._asdict()
    assert allocated['rss_bytes'] > 0 and allocated['private_commit_bytes'] > 0
    assert allocated['process_lifetime_peak_rss_bytes'] >= allocated['rss_bytes']
    assert allocated['process_lifetime_peak_commit_bytes'] >= allocated['private_commit_bytes']
    assert abs(other['private'] - allocated['private_commit_bytes']) < MIB
    assert abs(other['rss'] - allocated['rss_bytes']) < MIB
    assert allocated['private_commit_bytes'] - initial['private_commit_bytes'] >= 7 * MIB
    result = {'status': 'PSAPI_PSUTIL_SELF_TEST_PASS', 'scope': '8 MiB touched bytearray; not model evidence',
              'baseline': initial, 'allocated': allocated, 'psutil': other,
              'counter_struct_bytes': ctypes.sizeof(CountersEx)}
    write_json('memory_self_test.json', result)
    print(json.dumps(result), flush=True)


def run_child(*args):
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    env['PYTHONHASHSEED'] = '20260912'
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *args],
                            env=env, capture_output=True, text=True, encoding='utf-8',
                            creationflags=subprocess.CREATE_NO_WINDOW, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout + '\n' + result.stderr)
    print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print('CHILD_WARNING ' + result.stderr.strip(), flush=True)


def run():
    """Measure ten isolated processes and summarise their memory samples."""
    started = datetime.now(timezone.utc).isoformat()
    assert not (OUT / 'memory_profiles_summary.json').exists(), 'Refuse silent repeat/overwrite of a completed run.'
    run_child('--self-test')
    run_child('--prepare')
    for repeat in range(1, 6):
        order = ('cnn','mlp8') if repeat % 2 else ('mlp8','cnn')
        for model in order:
            run_child('--child', model, '--repeat', str(repeat))
    raw = [json.loads((OUT / f'memory_raw_{repeat:02}_{model}.json').read_text(encoding='utf-8'))
           for repeat in range(1, 6) for model in ('cnn','mlp8')]
    rows = [row for obj in raw for row in obj['samples']]
    assert len(rows) == 200 and len({obj['pid'] for obj in raw}) == 10
    with (OUT / 'memory_profiles_raw.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    metrics = METRICS + ('rss_increment_from_runtime_baseline_bytes',
                         'private_commit_increment_from_runtime_baseline_bytes')
    for model in ('cnn','mlp8'):
        for stage in ('runtime_baseline','model_loaded','post_batch_1','post_batch_256'):
            values = [r for r in rows if r['model'] == model and r['stage'] == stage]
            item = {'model': model, 'stage': stage, 'independent_process_repeats':5,
                    'snapshots_per_stage_per_process':5}
            for metric in metrics:
                reps = [statistics.median(r[metric] for r in values if r['repeat'] == repeat) for repeat in range(1,6)]
                item[metric] = {'per_repeat_medians':reps, 'median':statistics.median(reps),
                                'min':min(reps), 'max':max(reps)}
                item[metric.replace('_bytes','_MiB')] = {k:([v/MIB for v in value] if isinstance(value,list) else value/MIB)
                                                       for k,value in item[metric].items()}
            summary.append(item)
    result = {'status':'COMPLETED_FIVE_ISOLATED_PROCESSES_PER_FITTED_MODEL',
              'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
              'protocol':{'models':['cnn','mlp8'],'repeats':5,'batch_order':[1,256],
                          'warmup_inferences_per_batch':50,'measured_workload_inferences_per_batch':100,
                          'steady_snapshots':5,'snapshot_interval_sec':.04,
                          'MiB_definition':1048576,'model_training_performed':False,
                          'order':'Odd repetitions CNN then MLP8; even repetitions MLP8 then CNN',
                          'baseline':'Same imports, 256-row feature array and generic 8x8 matrix multiplication warmup',
                          'peak_definition':'OS process-lifetime peak, including runtime/imports/load/prior stages; not model-only peak.',
                          'increments':'Raw differences from within-process runtime baseline medians; signed allocator variation retained.',
                          'batch_dependency':'Batch256 follows batch1 in the same fresh process; phase peaks are cumulative, not independent batch peaks.'},
              'summary':summary,'runtime':raw[0]['runtime'],
              'input_provenance':json.loads((OUT/'memory_input_provenance.json').read_text(encoding='utf-8')),
              'script_sha256':sha(__file__)}
    write_json('memory_profiles_summary.json',result)
    lines=['# Process-memory profile of frozen L31 estimators','',
           'Completed UTC: '+result['completed_utc'],'',
           'Five fresh Windows processes per model; both fitted checkpoints and their source result rows were hash-verified. No training was performed. All ten process records and 200 steady snapshots are retained.','',
           '| Model | Stage | RSS MiB, median [min, max] | Private commit MiB, median [min, max] | RSS increment MiB | Commit increment MiB |',
           '|---|---|---|---|---|---|']
    for item in summary:
        def fmt(key):
            x=item[key];return f"{x['median']:.3f} [{x['min']:.3f}, {x['max']:.3f}]"
        lines.append('| '+item['model']+' | '+item['stage']+' | '+fmt('rss_MiB')+' | '+fmt('private_commit_MiB')+' | '+fmt('rss_increment_from_runtime_baseline_MiB')+' | '+fmt('private_commit_increment_from_runtime_baseline_MiB')+' |')
    lines += ['', '## Interpretation', '',
              'RSS is the process working set, including shared runtime-library pages. Private commit is committed private virtual memory, not resident RAM. Increments include allocator state, Python objects and checkpoint-loading effects; they are not isolated tensor-memory measurements. Negative increments are retained.', '',
              'PeakWorkingSetSize and PeakPagefileUsage are lifetime peaks for the entire process. They include Python/Torch, imports, loading and earlier inference stages. Baseline subtraction is not used to invent a model-only peak. Batch256 follows batch1; peak counters cannot isolate each batch independently.', '',
              'These measurements describe this Windows/Python/PyTorch implementation and the frozen L31 fitted states. They do not replace serialized-state byte counts, establish training-memory costs, describe microcontroller deployment, or imply physical sensor latency/energy gains.', '',
              'Per-stage summaries first take the median of five steady snapshots in each process, then report the median and range across all five process repetitions. No significance claim or selective repetition exclusion is made.', '',
              'Windows API definitions: https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex',
              'Sampling API: https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo']
    (OUT/'memory_profiles_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'completed_utc':result['completed_utc'],'processes':len(raw),'snapshots':len(rows)}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test',action='store_true')
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--child',choices=['cnn','mlp8'])
    parser.add_argument('--repeat',type=int,default=0)
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.prepare:prepare()
    elif args.run:run()
    elif args.child:profile_child(args.child,args.repeat)
