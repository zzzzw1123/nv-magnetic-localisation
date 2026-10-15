"""Independently reconstruct the completed portable benchmark summaries.

The validator reads 20 raw process JSON records, raw CSVs, and hash-bound
inputs, then recomputes the 16 memory and 8 latency summary conditions. It
writes ``portable_benchmark_validation.json`` without importing model or
benchmark implementations and without running inference.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

W=Path(__file__).resolve().parent
ROOT=W.parents[2]
MODELS=('cnn','mlp8')
BACKENDS=('torch_l31','numpy_only')
STAGES=('runtime_baseline','model_loaded','post_batch_1','post_batch_256')
MIB=1048576
MEM_METRICS=('rss_bytes','private_commit_bytes','process_lifetime_peak_rss_bytes','process_lifetime_peak_commit_bytes',
             'rss_increment_from_runtime_baseline_bytes','private_commit_increment_from_runtime_baseline_bytes')
READ_FILES=set()


def sha(path):
    path=Path(path).resolve()
    assert path.is_relative_to(ROOT),str(path)
    READ_FILES.add(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name):
    path=(W/name).resolve();READ_FILES.add(path)
    return json.loads(path.read_text(encoding='utf-8'))


def value_csv(value):
    return '' if value is None else str(value)


def stats(values):
    return {'five_process_values':values,'median':statistics.median(values),'min':min(values),'max':max(values)}


def convert_mib(values):
    return {key:([v/MIB for v in value] if isinstance(value,list) else value/MIB) for key,value in values.items()}


def csv_matches(name,raw_rows,key_fields):
    """Require exact field-wise agreement between raw JSON rows and one CSV."""
    path=W/name;READ_FILES.add(path)
    with path.open(encoding='utf-8',newline='') as handle:
        rows=list(csv.DictReader(handle))
    expected={tuple(value_csv(r[key]) for key in key_fields):{k:value_csv(v) for k,v in r.items()} for r in raw_rows}
    actual={tuple(r[key] for key in key_fields):r for r in rows}
    assert len(expected)==len(raw_rows)==len(actual)==len(rows),(name,'missing or duplicate rows')
    assert expected==actual,(name,'raw JSON/CSV value mismatch')
    return len(rows)


def validate():
    """Recompute process-level statistics and write the verification record."""
    result=read_json('portable_benchmark_result.json')
    protocol=read_json('portable_benchmark_protocol.json')
    assert result['status']=='COMPLETED_ALL_IMPLEMENTATIONS'
    assert result['protocol']==protocol and protocol['numeric_tolerance_um']==.01
    assert protocol['isolated_process_repeats_per_condition']==5
    assert protocol['warmup_calls_per_batch']==50 and protocol['timed_calls_per_batch']==100
    assert protocol['batch_order_in_each_process']==[1,256]
    assert protocol['logical_processor']==12 and protocol['torch_and_blas_threads']==1
    assert protocol['common_measurement_dependencies']==['psutil','threadpoolctl','Python standard library and Windows PSAPI']
    assert result['script_sha256']==sha(W/'portable_benchmark.py')
    assert result['protocol_sha256']==sha(W/'portable_benchmark_protocol.json')
    assert result['all_conditions_reported'] and not result['training_performed']
    assert result['original_L31_Torch_dependency_baseline_includes_sklearn']

    expected={(backend,model,repeat) for backend in BACKENDS for model in MODELS for repeat in range(1,6)}
    files=sorted(W.glob('portable_benchmark_raw_*.json'))
    assert len(files)==20,('raw process file count',len(files))
    raw={}
    for path in files:
        obj=read_json(path.name);key=(obj['backend'],obj['model'],obj['repeat'])
        assert key in expected and key not in raw,key
        assert path.name==f'portable_benchmark_raw_{key[2]:02}_{key[0]}_{key[1]}.json'
        raw[key]=obj
    assert set(raw)==expected
    assert len({obj['pid'] for obj in raw.values()})==20
    all_memory=[];all_latency=[];negative_rss=0;negative_commit=0
    max_errors={}
    for (backend,model,repeat),obj in raw.items():
        assert obj['source_check']==result['source_check']
        assert obj['normal_priority_and_P_logical12']
        assert obj['batch_order']==[1,256] and obj['whole_process_peaks_cumulative']
        assert obj['implementation_effect_including_dependencies']
        deps=obj['dependency_baseline']
        assert deps['numpy']['loaded']
        if backend=='numpy_only':assert not deps['torch']['loaded'] and not deps['sklearn']['loaded']
        else:assert deps['torch']['loaded'] and deps['sklearn']['loaded']
        assert obj['measurement_dependencies']['psutil'] and obj['measurement_dependencies']['threadpoolctl']
        threads=obj['parallelism']
        assert set(threads['thread_environment'])=={'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'}
        assert set(threads['thread_environment'].values())=={'1'}
        if backend=='torch_l31':assert threads['torch_intraop_threads']==threads['torch_interop_threads']==1
        else:assert threads['torch_intraop_threads'] is None and threads['torch_interop_threads'] is None
        for when in ('before','after'):
            pools=threads['native_threadpools_'+when]
            assert pools and all(pool['num_threads']==1 for pool in pools)
            env=obj['environment_'+when]
            assert env['process_affinity']==[12] and env['process_priority']==32
            assert len(env['per_logical_CPU_percent'])>12
            assert env['logical12_CPU_percent']==env['per_logical_CPU_percent'][12]
            assert all(math.isfinite(v) and 0<=v<=100 for v in env['per_logical_CPU_percent'])
        memory=obj['memory_samples'];latency=obj['latency_samples']
        assert len(memory)==20 and len(latency)==200
        assert [r['stage'] for r in memory]==[stage for stage in STAGES for _ in range(5)]
        baseline=[r for r in memory if r['stage']=='runtime_baseline']
        baseline_rss=statistics.median(r['rss_bytes'] for r in baseline)
        baseline_commit=statistics.median(r['private_commit_bytes'] for r in baseline)
        for stage in STAGES:
            group=[r for r in memory if r['stage']==stage]
            assert [r['snapshot'] for r in group]==[1,2,3,4,5]
            for row in group:
                assert (row['backend'],row['model'],row['repeat'],row['pid'])==(backend,model,repeat,obj['pid'])
                assert row['rss_bytes']>0 and row['private_commit_bytes']>0
                assert row['process_lifetime_peak_rss_bytes']>=row['rss_bytes']
                assert row['process_lifetime_peak_commit_bytes']>=row['private_commit_bytes']
                assert row['rss_increment_from_runtime_baseline_bytes']==row['rss_bytes']-baseline_rss
                assert row['private_commit_increment_from_runtime_baseline_bytes']==row['private_commit_bytes']-baseline_commit
                for metric in MEM_METRICS:
                    assert math.isfinite(row[metric])
                    assert row[metric.replace('_bytes','_MiB')]==row[metric]/MIB
                negative_rss+=row['rss_increment_from_runtime_baseline_bytes']<0
                negative_commit+=row['private_commit_increment_from_runtime_baseline_bytes']<0
        for metric in ('process_lifetime_peak_rss_bytes','process_lifetime_peak_commit_bytes'):
            values=[r[metric] for r in memory];assert values==sorted(values)
        for batch in (1,256):
            group=[r for r in latency if r['batch_size']==batch]
            assert [r['call_index'] for r in group]==list(range(100))
            expected_rows=[((repeat-1)*100+i)%256 for i in range(100)] if batch==1 else [None]*100
            assert [r['replay_row_index'] for r in group]==expected_rows
            for row in group:
                assert (row['backend'],row['model'],row['repeat'],row['pid'])==(backend,model,repeat,obj['pid'])
                assert isinstance(row['latency_ns'],int) and row['latency_ns']>0
                assert row['latency_us']==row['latency_ns']/1000.
                assert row['latency_ns_per_prediction']==row['latency_ns']/batch
                error=row['max_abs_coordinate_difference_um']
                assert math.isfinite(error) and error>=0
                assert row['numeric_gate_passed']==(error<=.01) and row['numeric_gate_passed']
            error=max(r['max_abs_coordinate_difference_um'] for r in group)
            assert obj['max_abs_coordinate_difference_um_by_batch'][str(batch)]==error
            max_errors[f'{backend}/{model}/repeat{repeat}/batch{batch}']=error
        assert obj['numeric_gate_passed']==all(r['numeric_gate_passed'] for r in latency)
        all_memory+=memory;all_latency+=latency
    assert len(all_memory)==400 and len(all_latency)==4000
    csv_matches('portable_benchmark_memory_raw.csv',all_memory,('backend','model','repeat','stage','snapshot'))
    csv_matches('portable_benchmark_latency_raw.csv',all_latency,('backend','model','repeat','batch_size','call_index'))
    assert (result['isolated_process_count'],result['memory_snapshots'],result['raw_timed_calls'])==(20,400,4000)

    stored_memory={(r['backend'],r['model'],r['stage']):r for r in result['memory_summary']}
    stored_latency={(r['backend'],r['model'],r['batch_size']):r for r in result['latency_summary']}
    assert len(stored_memory)==16 and len(result['memory_summary'])==16
    assert len(stored_latency)==8 and len(result['latency_summary'])==8
    for backend in BACKENDS:
        for model in MODELS:
            covered={r['replay_row_index'] for r in all_latency if r['backend']==backend and r['model']==model and r['batch_size']==1}
            assert covered==set(range(256))
            for stage in STAGES:
                expected_entry={'backend':backend,'model':model,'stage':stage}
                for metric in MEM_METRICS:
                    values=[]
                    for repeat in range(1,6):
                        observed=[r[metric] for r in raw[(backend,model,repeat)]['memory_samples'] if r['stage']==stage]
                        values.append(max(observed) if metric.startswith('process_lifetime_peak') else statistics.median(observed))
                    expected_entry[metric]=stats(values)
                    expected_entry[metric.replace('_bytes','_MiB')]=convert_mib(stats(values))
                assert stored_memory[(backend,model,stage)]==expected_entry,(backend,model,stage)
            for batch in (1,256):
                observed=[r for r in all_latency if r['backend']==backend and r['model']==model and r['batch_size']==batch]
                values=[statistics.median(r['latency_ns'] for r in observed if r['repeat']==repeat) for repeat in range(1,6)]
                expected_entry={'backend':backend,'model':model,'batch_size':batch,'timed_calls_per_process':100,
                    'five_process_median_latency_ns':values,'median_latency_us':statistics.median(values)/1000.,
                    'range_of_process_medians_us':[min(values)/1000.,max(values)/1000.],
                    'median_us_per_prediction':statistics.median(values)/1000./batch,
                    'all_raw_call_range_ns':[min(r['latency_ns'] for r in observed),max(r['latency_ns'] for r in observed)],
                    'numeric_gate_passed':all(r['numeric_gate_passed'] for r in observed),
                    'max_abs_coordinate_difference_um':max(r['max_abs_coordinate_difference_um'] for r in observed)}
                assert stored_latency[(backend,model,batch)]==expected_entry,(backend,model,batch)

    source=result['source_check']
    assert source['protocol_sha256']==sha(W/'portable_benchmark_protocol.json')
    assert source['input_sha256']==sha(W/'memory_input_features.npy')
    assert source['portable_proof_sha256']==sha(W/'portable_equivalence_result.json')
    export=read_json('portable_source_provenance.json')
    assert source['export_models']==export['models']
    for model in MODELS:assert sha(W/f'portable_{model}.npz')==export['models'][model]['export_npz_sha256']
    for rel,digest in source['checkpoint_sha256'].items():assert sha(ROOT/rel)==digest,rel
    for rel,digest in export['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    reference=read_json('portable_benchmark_reference.json')
    assert reference['source_check']==source
    assert reference['reference_npz_sha256']==sha(W/'portable_benchmark_reference.npz')
    assert not reference['profile_or_latency_timing_performed'] and not reference['training_performed']
    for name in ('portable_benchmark.py','portable_benchmark_validate.py','portable_inference.py',
                 'memory_profile.py','portable_benchmark_report.md','portable_benchmark_preflight.json',
                 'memory_input_provenance.json'):
        sha(W/name)
    # Store W-relative keys; sha() confines resolved inputs to the research project.
    artifact_hashes={os.path.relpath(path,W).replace('\\','/'):sha(path) for path in sorted(READ_FILES)}
    output={'status':'PORTABLE_BENCHMARK_RECORDS_VERIFIED','verified_utc':datetime.now(timezone.utc).isoformat(),
        'measurement_completed_utc':result['completed_utc'],
        'isolated_process_count':20,'raw_json_unique_conditions':20,
        'memory_snapshots':400,'raw_timed_calls':4000,'memory_summary_conditions':16,'latency_summary_conditions':8,
        'JSON_CSV_all_fields_match':True,'all_five_process_medians_ranges_recomputed':True,
        'all_numeric_gates_verified_from_recorded_differences':True,'numeric_tolerance_um':.01,
        'all_workers_same_P_logical12_Normal':True,'all_torch_BLAS_thread_counts_one':True,
        'numpy_inference_workers_no_Torch_or_sklearn':True,
        'common_measurement_dependencies_disclosed':['psutil','threadpoolctl'],
        'signed_increments_recomputed_without_clipping':True,
        'negative_RSS_increment_snapshots':negative_rss,'negative_private_commit_increment_snapshots':negative_commit,
        'all_slow_values_retained_and_raw_ranges_recomputed':True,
        'all_input_source_hashes_match':True,'batch1_unique_replay_rows_per_condition':256,
        'peak_counters_cumulative_not_baseline_subtracted':True,
        'independent_of_benchmark_summarise':True,'models_loaded_or_inferred_by_validator':False,
        'artifact_sha256':artifact_hashes}
    assert 'portable_benchmark' not in sys.modules and 'torch' not in sys.modules and 'sklearn' not in sys.modules
    (W/'portable_benchmark_validation.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='artifact_sha256'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-script',action='store_true',help='Only exercise standard-library helper checks; do not read or validate measurement results.')
    args=parser.parse_args()
    if args.check_script:
        assert value_csv(None)=='' and value_csv(False)=='False'
        assert stats([-2,0,1,3,12])=={'five_process_values':[-2,0,1,3,12],'median':1,'min':-2,'max':12}
        assert convert_mib(stats([-MIB,0,MIB]))['min']==-1
        print('VALIDATOR_SCRIPT_READY; no measurements read, inferred, timed or validated.')
    else:validate()
