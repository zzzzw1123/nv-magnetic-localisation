"""Benchmark PyTorch and NumPy inference on one CPU core.

Use --check for file and configuration validation. Measurement runs require
--confirm-training-finished and should run without concurrent model training.
"""
from __future__ import annotations

import argparse
import csv
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

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
MIB=1048576
CONDITIONS=[('torch_l31','cnn'),('numpy_only','cnn'),('torch_l31','mlp8'),('numpy_only','mlp8')]
STAGES=['runtime_baseline','model_loaded','post_batch_1','post_batch_256']
MEMORY_METRICS=['rss_bytes','private_commit_bytes','process_lifetime_peak_rss_bytes','process_lifetime_peak_commit_bytes']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def config():
    """Load and validate the fixed benchmark protocol."""
    p=OUT/'portable_benchmark_protocol.json'
    c=json.loads(p.read_text(encoding='utf-8'))
    assert c['numeric_tolerance_um']==.01 and c['logical_processor']==12
    assert c['isolated_process_repeats_per_condition']==5
    assert c['warmup_calls_per_batch']==50 and c['timed_calls_per_batch']==100
    return c


def batch1_indices(repeat,warmup=False):
    """Return the deterministic single-row replay schedule for one process."""
    start=(repeat-1)*100-(50 if warmup else 0)
    return [(start+i)%256 for i in range(50 if warmup else 100)]


def check_sources():
    """Verify checksums for models, inputs and reference predictions."""
    import numpy as np
    from memory_profile import verify_sources
    c=config()
    checkpoint_hashes=verify_sources()
    proof=json.loads((OUT/'portable_equivalence_result.json').read_text(encoding='utf-8'))
    assert proof['status']=='NUMPY_PORTABLE_NUMERICAL_GATE_PASS' and proof['fixed_tolerance_um']==.01
    assert all(case['gate_passed'] for case in proof['cases'])
    export=json.loads((OUT/'portable_source_provenance.json').read_text(encoding='utf-8'))
    for rel,digest in export['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    for model in c['models']:
        p=OUT/f'portable_{model}.npz'
        assert sha(p)==export['models'][model]['export_npz_sha256']
        with np.load(p,allow_pickle=False) as arrays:
            assert all(arrays[key].dtype in (np.dtype('float32'),np.dtype('float64')) for key in arrays.files)
    memory=json.loads((OUT/'memory_input_provenance.json').read_text(encoding='utf-8'))
    p=OUT/c['input_file'];assert sha(p)==memory['features_sha256']
    values=np.load(p,allow_pickle=False)
    assert values.shape==(256,24) and np.isfinite(values).all()
    timed_rows=[i for repeat in range(1,6) for i in batch1_indices(repeat)]
    assert len(timed_rows)==500 and set(timed_rows)==set(range(256))
    return {'protocol_sha256':sha(OUT/'portable_benchmark_protocol.json'),
            'input_sha256':sha(p),'portable_proof_sha256':sha(OUT/'portable_equivalence_result.json'),
            'checkpoint_sha256':checkpoint_hashes,'export_models':export['models']}


def self_check():
    """Write a preflight record without running inference or measurements."""
    result={'status':'PORTABLE_BENCHMARK_READY_NOT_RUN','source_check':check_sources(),
            'benchmark_script_sha256':sha(__file__),
            'validator_script_sha256':sha(OUT/'portable_benchmark_validate.py') if (OUT/'portable_benchmark_validate.py').exists() else None,
            'process_conditions':20,'planned_timed_calls':4000,'planned_memory_snapshots':400,
            'batch1_timed_rows_covered':256,'inference_executed':False,
            'timing_executed':False,'memory_profile_executed':False,
            'affinity_changed':False,'training_executed':False,
            'execution_gate':'Run resource measurements separately from training and other CPU-intensive work.'}
    assert not any(name=='torch' or name.startswith('torch.') or name=='sklearn' or name.startswith('sklearn.') for name in sys.modules)
    write_json('portable_benchmark_preflight.json',result)
    print(json.dumps({key:value for key,value in result.items() if key!='source_check'}),flush=True)


def configure_own_process():
    """Pin the current benchmark process to the protocol's logical CPU."""
    import psutil
    c=config();p=psutil.Process()
    # Affinity and priority apply only to this benchmark process.
    p.cpu_affinity([c['logical_processor']])
    p.nice(psutil.NORMAL_PRIORITY_CLASS)
    assert p.cpu_affinity()==[12] and p.nice()==psutil.NORMAL_PRIORITY_CLASS
    return psutil,p


def dependency_versions():
    return {name:{'loaded':name in sys.modules,
                  'version':str(getattr(sys.modules.get(name),'__version__','unavailable'))}
            for name in ('numpy','torch','sklearn','scipy','pandas','joblib','threadpoolctl')}


def original_runtime():
    from memory_profile import runtime
    return runtime()


def make_reference():
    """Write Torch reference coordinates shaped ``(256,2)`` for both batch scopes."""
    configure_own_process()
    import numpy as np
    from memory_profile import load_model,predict
    checks=check_sources();rt=original_runtime()
    x=np.load(OUT/'memory_input_features.npy',allow_pickle=False)
    arrays={}
    for model in ('cnn','mlp8'):
        state=load_model(model,rt)
        arrays[model+'_batch1']=np.concatenate([predict(model,state,x[i:i+1],rt) for i in range(256)])
        arrays[model+'_batch256']=predict(model,state,x,rt)
        assert arrays[model+'_batch1'].shape==arrays[model+'_batch256'].shape==(256,2)
    np.savez(OUT/'portable_benchmark_reference.npz',**arrays)
    write_json('portable_benchmark_reference.json',{'status':'SAME_BATCH_TORCH_REFERENCE_READY',
        'reference_npz_sha256':sha(OUT/'portable_benchmark_reference.npz'),'source_check':checks,
        'profile_or_latency_timing_performed':False,'training_performed':False})
    print(json.dumps({'status':'SAME_BATCH_TORCH_REFERENCE_READY'}),flush=True)


def environment_sample(psutil,process):
    per_cpu=psutil.cpu_percent(interval=.2,percpu=True)
    return {'observed_utc':datetime.now(timezone.utc).isoformat(),
            'system_mean_logical_CPU_percent':sum(per_cpu)/len(per_cpu),
            'logical12_CPU_percent':per_cpu[12], 'per_logical_CPU_percent':per_cpu,
            'process_affinity':process.cpu_affinity(),'process_priority':int(process.nice()),
            'process_threads':process.num_threads(),
            'available_system_memory_bytes':psutil.virtual_memory().available}


def memory_stage(stage,started):
    """Collect five process-memory snapshots for one named benchmark stage."""
    from memory_profile import counters
    gc.collect();rows=[]
    for i in range(5):
        time.sleep(.04)
        row=counters();row.update(stage=stage,snapshot=i+1,elapsed_wall_sec=time.perf_counter()-started)
        for metric in MEMORY_METRICS:row[metric.replace('_bytes','_MiB')]=row[metric]/MIB
        rows.append(row)
    return rows


def worker(backend,model,repeat):
    """Measure one backend/model repeat and write its raw process record."""
    started=time.perf_counter()
    psutil,process=configure_own_process()
    import numpy as np
    import threadpoolctl
    from memory_profile import counters
    c=config();sources=check_sources()
    reference_meta=json.loads((OUT/'portable_benchmark_reference.json').read_text(encoding='utf-8'))
    assert sha(OUT/'portable_benchmark_reference.npz')==reference_meta['reference_npz_sha256']
    with np.load(OUT/'portable_benchmark_reference.npz',allow_pickle=False) as source:
        expected={batch:source[model+'_batch'+str(batch)] for batch in (1,256)}
    x=np.load(OUT/'memory_input_features.npy',allow_pickle=False)
    # Precreate row views so row selection remains outside the timed call.
    single_rows=[x[i:i+1] for i in range(256)]
    if backend=='torch_l31':
        from memory_profile import load_model,predict
        rt=original_runtime()
        loader=lambda:load_model(model,rt)
        predictor=lambda state,values:predict(model,state,values,rt)
    else:
        from portable_inference import PortableEstimator
        loader=lambda:PortableEstimator(model,OUT/f'portable_{model}.npz')
        predictor=lambda state,values:state.predict(values)
    for _ in range(20):
        warm=np.ones((8,8),dtype=np.float32)@np.ones((8,8),dtype=np.float32)
        if backend=='torch_l31':warm_t=rt['torch'].ones((8,8))@rt['torch'].ones((8,8))
    del warm
    if backend=='torch_l31':del warm_t
    for _ in range(10):counters();process.memory_info()
    before=environment_sample(psutil,process)
    native_pools_before=threadpoolctl.threadpool_info()
    memory=memory_stage('runtime_baseline',started)
    baseline_rss=statistics.median(row['rss_bytes'] for row in memory)
    baseline_private=statistics.median(row['private_commit_bytes'] for row in memory)
    state=loader()
    memory+=memory_stage('model_loaded',started)
    latency=[];max_difference={1:0.,256:0.}
    for batch in (1,256):
        warm_rows=batch1_indices(repeat,True) if batch==1 else [None]*50
        for row_id in warm_rows:
            values=single_rows[row_id] if batch==1 else x
            output=predictor(state,values)
        timed_rows=batch1_indices(repeat) if batch==1 else [None]*100
        for call_index,row_id in enumerate(timed_rows):
            values=single_rows[row_id] if batch==1 else x
            tick=time.perf_counter_ns()
            output=predictor(state,values)
            elapsed=time.perf_counter_ns()-tick
            target=expected[1][row_id:row_id+1] if batch==1 else expected[256]
            difference=float(np.max(np.abs(output-target)))
            max_difference[batch]=max(max_difference[batch],difference)
            latency.append({'backend':backend,'model':model,'repeat':repeat,'pid':os.getpid(),
                'batch_size':batch,'call_index':call_index,'replay_row_index':row_id,
                'latency_ns':elapsed,'latency_us':elapsed/1000.,
                'latency_ns_per_prediction':elapsed/batch,
                'max_abs_coordinate_difference_um':difference,
                'numeric_gate_passed':bool(np.isfinite(output).all() and difference<=c['numeric_tolerance_um'])})
        del output
        memory+=memory_stage('post_batch_'+str(batch),started)
    after=environment_sample(psutil,process)
    native_pools_after=threadpoolctl.threadpool_info()
    assert native_pools_before and native_pools_after
    assert all(pool.get('num_threads')==1 for pool in native_pools_before+native_pools_after)
    deps=dependency_versions()
    if backend=='numpy_only':
        assert not deps['torch']['loaded'] and not deps['sklearn']['loaded']
    else:
        assert deps['torch']['loaded'] and deps['sklearn']['loaded']
        assert rt['torch'].get_num_threads()==rt['torch'].get_num_interop_threads()==1
    for row in memory:
        row.update(backend=backend,model=model,repeat=repeat,pid=os.getpid())
        row['rss_increment_from_runtime_baseline_bytes']=row['rss_bytes']-baseline_rss
        row['private_commit_increment_from_runtime_baseline_bytes']=row['private_commit_bytes']-baseline_private
        for metric in ('rss_increment_from_runtime_baseline','private_commit_increment_from_runtime_baseline'):
            row[metric+'_MiB']=row[metric+'_bytes']/MIB
    result={'backend':backend,'model':model,'repeat':repeat,'pid':os.getpid(),
        'source_check':sources,'dependency_baseline':deps,'environment_before':before,'environment_after':after,
        'platform':platform.platform(),'python':sys.version,'normal_priority_and_P_logical12':True,
        'parallelism':{'torch_intraop_threads':rt['torch'].get_num_threads() if backend=='torch_l31' else None,
            'torch_interop_threads':rt['torch'].get_num_interop_threads() if backend=='torch_l31' else None,
            'thread_environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
            'native_threadpools_before':native_pools_before,'native_threadpools_after':native_pools_after},
        'measurement_dependencies':{'psutil':psutil.__version__,'threadpoolctl':threadpoolctl.__version__},
        'memory_samples':memory,'latency_samples':latency,
        'numeric_gate_passed':all(row['numeric_gate_passed'] for row in latency),
        'max_abs_coordinate_difference_um_by_batch':max_difference,
        'memory_instrumentation':'Steady snapshots include the retained measurement records; both conditions retain identical record counts.',
        'latency_scope':'One predictor API call; no input-row slicing, reference check, memory sampling, imports or loading inside timer.',
        'batch_order':[1,256],'whole_process_peaks_cumulative':True,
        'implementation_effect_including_dependencies':True}
    write_json(f'portable_benchmark_raw_{repeat:02}_{backend}_{model}.json',result)
    print(json.dumps({'backend':backend,'model':model,'repeat':repeat,'pid':os.getpid(),
                      'numeric_gate_passed':result['numeric_gate_passed']}),flush=True)


def launch(script_args):
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    env['PYTHONHASHSEED']='20260912';env['PYTHONIOENCODING']='utf-8'
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),*script_args,'--confirm-training-finished'],
        env=env,capture_output=True,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW,timeout=240)
    print(result.stdout.strip(),flush=True)
    if result.stderr.strip():print(result.stderr.strip(),flush=True)
    if result.returncode:raise RuntimeError('Benchmark child failed with code '+str(result.returncode))


def summarise(started):
    """Write process-level summaries from the 20 benchmark records."""
    raw=[json.loads((OUT/f'portable_benchmark_raw_{repeat:02}_{backend}_{model}.json').read_text(encoding='utf-8'))
         for repeat in range(1,6) for backend,model in CONDITIONS]
    assert len(raw)==20 and len({r['pid'] for r in raw})==20
    memory=[r for item in raw for r in item['memory_samples']]
    latency=[r for item in raw for r in item['latency_samples']]
    assert len(memory)==400 and len(latency)==4000
    for name,rows in (('memory',memory),('latency',latency)):
        with (OUT/f'portable_benchmark_{name}_raw.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    mem_summary=[];lat_summary=[]
    for backend,model in CONDITIONS:
        for stage in STAGES:
            selected=[r for r in memory if r['backend']==backend and r['model']==model and r['stage']==stage]
            entry={'backend':backend,'model':model,'stage':stage}
            for metric in MEMORY_METRICS+['rss_increment_from_runtime_baseline_bytes','private_commit_increment_from_runtime_baseline_bytes']:
                # Lifetime peaks use the stage maximum and are not baseline differences.
                aggregate=max if metric.startswith('process_lifetime_peak') else statistics.median
                values=[aggregate(r[metric] for r in selected if r['repeat']==repeat) for repeat in range(1,6)]
                entry[metric]={'five_process_values':values,'median':statistics.median(values),'min':min(values),'max':max(values)}
                entry[metric.replace('_bytes','_MiB')]={k:[z/MIB for z in v] if isinstance(v,list) else v/MIB for k,v in entry[metric].items()}
            mem_summary.append(entry)
        for batch in (1,256):
            selected=[r for r in latency if r['backend']==backend and r['model']==model and r['batch_size']==batch]
            assert len(selected)==500
            values=[statistics.median(r['latency_ns'] for r in selected if r['repeat']==repeat) for repeat in range(1,6)]
            lat_summary.append({'backend':backend,'model':model,'batch_size':batch,
                'timed_calls_per_process':100,'five_process_median_latency_ns':values,
                'median_latency_us':statistics.median(values)/1000.,
                'range_of_process_medians_us':[min(values)/1000.,max(values)/1000.],
                'median_us_per_prediction':statistics.median(values)/1000./batch,
                'all_raw_call_range_ns':[min(r['latency_ns'] for r in selected),max(r['latency_ns'] for r in selected)],
                'numeric_gate_passed':all(r['numeric_gate_passed'] for r in selected),
                'max_abs_coordinate_difference_um':max(r['max_abs_coordinate_difference_um'] for r in selected)})
    passed=all(r['numeric_gate_passed'] for r in raw)
    result={'status':'COMPLETED_ALL_IMPLEMENTATIONS' if passed else 'COMPLETED_WITH_NUMERICAL_GATE_FAILURE',
        'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'protocol':config(),'protocol_sha256':sha(OUT/'portable_benchmark_protocol.json'),
        'script_sha256':sha(__file__),'memory_summary':mem_summary,'latency_summary':lat_summary,
        'isolated_process_count':20,'memory_snapshots':400,'raw_timed_calls':4000,
        'all_conditions_reported':True,'training_performed':False,
        'original_L31_Torch_dependency_baseline_includes_sklearn':True,
        'interpretation':'Dependency-inclusive implementation comparison, not pure model RAM, PyTorch-only footprint, or MCU deployment.',
        'source_check':check_sources()}
    write_json('portable_benchmark_result.json',result)
    lines=['# Dependency-inclusive inference implementation benchmark','',
        'Completed UTC: '+result['completed_utc'],'',
        'All four implementations and both fixed batch scopes are shown. Latencies are median [range] of five process medians, each based on all 100 recorded calls. No slow repetition is excluded.','',
        '| Backend | Model | Batch | API-call latency µs | µs/prediction | Numeric gate |','|---|---|---|---|---|---|']
    for r in lat_summary:
        lo,hi=r['range_of_process_medians_us']
        lines.append(f"| {r['backend']} | {r['model']} | {r['batch_size']} | {r['median_latency_us']:.3f} [{lo:.3f}, {hi:.3f}] | {r['median_us_per_prediction']:.3f} | {r['numeric_gate_passed']} |")
    lines+=['','## Memory scope','',
        'Full stage-level RSS, private commit, signed increments and whole-process lifetime peaks are retained in portable_benchmark_result.json and the raw CSV. Baselines include actual dependencies: original L31 Torch functions import NumPy, Torch, scikit-learn and transitive dependencies; NumPy-only workers import no Torch or scikit-learn. Both instrumented processes additionally import psutil and threadpoolctl for measurement. The difference is an implementation/dependency effect, not isolated model RAM. RSS includes shared pages; private commit is not physical resident memory. Peaks include imports/load/earlier batch stages and are never converted to model-only peaks by baseline subtraction.','',
        'Each process is Normal priority, pinned to the same P logical processor 12, with one Torch/BLAS thread. System CPU load and affinity are recorded before and after every process. Batch1 rotates input rows; all 256 replay observations are covered across five repetitions. Each batch256 call uses all 256 rows. Batch256 follows batch1, so memory peaks are cumulative.','',
        'The unchanged fitted models and 0.01 µm numerical gate are retained. These are PC software measurements, not a sensor end-to-end benchmark or a microcontroller deployment claim.']
    (OUT/'portable_benchmark_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'completed_utc':result['completed_utc'],'processes':20}),flush=True)


def run():
    """Run the fixed 20-process campaign and refuse to overwrite a completed result."""
    assert not (OUT/'portable_benchmark_result.json').exists(),'Refuse silent rerun/overwrite of a completed benchmark.'
    checks=check_sources();started=datetime.now(timezone.utc).isoformat()
    launch(['--prepare-reference'])
    for repeat in range(1,6):
        shift=(repeat-1)%len(CONDITIONS)
        for backend,model in CONDITIONS[shift:]+CONDITIONS[:shift]:
            launch(['--worker','--backend',backend,'--model',model,'--repeat',str(repeat)])
    assert check_sources()==checks
    summarise(started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true')
    group.add_argument('--prepare-reference',action='store_true');group.add_argument('--worker',action='store_true')
    parser.add_argument('--backend',choices=['torch_l31','numpy_only'])
    parser.add_argument('--model',choices=['cnn','mlp8']);parser.add_argument('--repeat',type=int)
    parser.add_argument('--confirm-training-finished',action='store_true')
    args=parser.parse_args()
    if args.check:self_check()
    else:
        if not args.confirm_training_finished:parser.error('--confirm-training-finished is required; run benchmarks without concurrent training.')
        if args.run:run()
        elif args.prepare_reference:make_reference()
        else:
            if args.backend is None or args.model is None or args.repeat not in range(1,6):parser.error('Worker requires backend, model and repeat 1..5.')
            worker(args.backend,args.model,args.repeat)
