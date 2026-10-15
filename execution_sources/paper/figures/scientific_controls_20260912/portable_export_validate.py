"""Export trusted L31 weights and compare isolated NumPy inference to Torch.

Torch/scikit-learn imports are confined to the --export child process.
No runtime-memory or latency benchmark is executed by this script.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def protocol():
    config=json.loads((OUT/'portable_protocol.json').read_text(encoding='utf-8'))
    assert config['absolute_coordinate_tolerance_um'] == .01
    assert config['weights_dtype'] == config['cnn_forward_dtype'] == config['mlp_forward_dtype'] == 'float32'
    return config


def export():
    import numpy as np
    from memory_profile import verify_sources, runtime, load_model, predict, EXPECTED
    from portable_inference import SHAPES, SCALERS
    config=protocol()
    sources=verify_sources()
    rt=runtime()
    data=rt['load_author_data'](ROOT)
    features=np.asarray(data.features,dtype=np.float64)
    assert features.shape==(819,24) and np.isfinite(features).all()
    np.save(OUT/'portable_input_features.npy',features,allow_pickle=False)
    np.save(OUT/'portable_position_labels.npy',data.positions_um,allow_pickle=False)
    source_manifest=json.loads((OUT/'memory_input_provenance.json').read_text(encoding='utf-8'))
    for rel,digest in source_manifest['source_sha256'].items():
        assert sha(ROOT/rel)==digest,rel
        sources[rel]=digest
    models={}; reference={}
    for name in config['models']:
        state=load_model(name,rt)
        network=state if name=='cnn' else state.model
        raw=network.state_dict()
        layer_ids={'conv1':0,'conv2':2,'conv3':4,'dense1':7,'dense2':9,'dense3':11} if name=='cnn' else {'dense1':0,'dense2':2}
        arrays={}
        for layer,index in layer_ids.items():
            for field in ('weight','bias'):
                original=raw[f'network.{index}.{field}'].detach().cpu().numpy()
                assert original.dtype==np.float32
                arrays[f'{layer}_{field}']=np.array(original,copy=True)
        if name=='mlp8':
            for key in sorted(SCALERS):
                original=getattr(state,key)
                assert original.dtype==np.float64
                arrays[key]=np.array(original,copy=True)
        assert set(arrays)==set(SHAPES[name])
        for key,value in arrays.items():
            assert value.shape==SHAPES[name][key]
        path=OUT/f'portable_{name}.npz'
        np.savez(path,**arrays)
        with np.load(path,allow_pickle=False) as loaded:
            assert all(np.array_equal(loaded[key],value) and loaded[key].dtype==value.dtype for key,value in arrays.items())
        parameter_count=sum(value.size for key,value in arrays.items() if key not in SCALERS)
        assert parameter_count=={'cnn':721858,'mlp8':218}[name]
        models[name]={'checkpoint_path':EXPECTED[name][0],'checkpoint_sha256':EXPECTED[name][1],
            'export_npz_sha256':sha(path),'export_npz_bytes':path.stat().st_size,
            'parameter_count':parameter_count,'export_arrays_exact_to_fitted_state':True,
            'arrays':{key:{'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.tobytes()).hexdigest()} for key,value in arrays.items()}}
        for batch in config['validation_batch_sizes']:
            reference[f'{name}_batch{batch}']=np.concatenate([predict(name,state,features[start:start+batch],rt)
                                                            for start in range(0,len(features),batch)])
    np.savez(OUT/'portable_torch_reference.npz',**reference)
    result={'status':'EXACT_NUMERIC_WEIGHTS_EXPORTED_FROM_FROZEN_L31_STATES',
            'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'portable_protocol.json'),
            'source_sha256':sources,'models':models,
            'features_sha256':sha(OUT/'portable_input_features.npy'),
            'position_labels_sha256':sha(OUT/'portable_position_labels.npy'),
            'torch_reference_sha256':sha(OUT/'portable_torch_reference.npz'),
            'torch_version':rt['torch'].__version__,'numpy_version':np.__version__,
            'coordinate_ids':data.coordinate_ids.tolist(),'training_performed':False,
            'performance_benchmark_performed':False}
    write_json('portable_source_provenance.json',result)
    print(json.dumps({'status':result['status'],'models':{k:v['parameter_count'] for k,v in models.items()}}),flush=True)


def validate():
    import numpy as np
    config=protocol()
    meta=json.loads((OUT/'portable_source_provenance.json').read_text(encoding='utf-8'))
    assert meta['protocol_sha256']==sha(OUT/'portable_protocol.json')
    assert meta['torch_reference_sha256']==sha(OUT/'portable_torch_reference.npz')
    assert meta['features_sha256']==sha(OUT/'portable_input_features.npy')
    for rel,digest in meta['source_sha256'].items():assert sha(ROOT/rel)==digest
    truth=np.load(OUT/'portable_position_labels.npy',allow_pickle=False)
    rows=[]; checks=[]
    with np.load(OUT/'portable_torch_reference.npz',allow_pickle=False) as references:
        for model in config['models']:
            assert sha(OUT/f'portable_{model}.npz')==meta['models'][model]['export_npz_sha256']
            for batch in config['validation_batch_sizes']:
                path=OUT/f'portable_predictions_{model}_batch{batch}.npy'
                current=np.load(path,allow_pickle=False)
                worker=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                assert worker['forbidden_imports']==[] and worker['backend']=='numpy_only'
                expected=references[f'{model}_batch{batch}']
                assert current.shape==expected.shape==(819,2) and np.isfinite(current).all()
                difference=current-expected
                abs_difference=np.abs(difference)
                max_error=float(abs_difference.max())
                worst=np.unravel_index(np.argmax(abs_difference),abs_difference.shape)
                passed=max_error<=config['absolute_coordinate_tolerance_um']
                checks.append({'model':model,'batch_size':batch,'observations':819,
                    'actual_batch_shapes':[[min(batch,819-start),24] for start in range(0,819,batch)],
                    'max_abs_coordinate_difference_um':max_error,
                    'x_max_abs_difference_um':float(abs_difference[:,0].max()),
                    'y_max_abs_difference_um':float(abs_difference[:,1].max()),
                    'mean_abs_coordinate_difference_um':float(abs_difference.mean()),
                    'worst_row_index':int(worst[0]),'worst_coordinate':'xy'[worst[1]],
                    'gate_passed':passed,'numpy_worker_forbidden_imports':worker['forbidden_imports']})
                for i in range(819):
                    rows.append({'model':model,'batch_size':batch,'row_index':i,
                        'coordinate_id':meta['coordinate_ids'][i],
                        'recorded_x_um':float(truth[i,0]),'recorded_y_um':float(truth[i,1]),
                        'torch_x_um':float(expected[i,0]),'torch_y_um':float(expected[i,1]),
                        'numpy_x_um':float(current[i,0]),'numpy_y_um':float(current[i,1]),
                        'numpy_minus_torch_x_um':float(difference[i,0]),'numpy_minus_torch_y_um':float(difference[i,1]),
                        'abs_x_difference_um':float(abs_difference[i,0]),'abs_y_difference_um':float(abs_difference[i,1])})
    with (OUT/'portable_prediction_equivalence.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    passed=all(c['gate_passed'] for c in checks)
    result={'status':'NUMPY_PORTABLE_NUMERICAL_GATE_PASS' if passed else 'NUMPY_PORTABLE_NUMERICAL_GATE_FAIL',
            'completed_utc':datetime.now(timezone.utc).isoformat(),'fixed_tolerance_um':.01,
            'protocol_sha256':sha(OUT/'portable_protocol.json'),
            'cases':checks,'paired_prediction_rows':len(rows),'coordinate_comparisons':len(rows)*2,
            'topologies_and_dtypes_changed':False,'training_performed':False,'quantisation_performed':False,
            'numerical_agreement_not_statistical_equivalence':True,
            'formal_memory_or_latency_profile_performed':False,
            'performance_status':'WAITING_FOR_ROOT_TRAINING_FINISHED_NOTICE',
            'artifact_sha256':{p.name:sha(p) for p in sorted(OUT.glob('portable_*')) if p.is_file() and p.name not in ['portable_equivalence_result.json','portable_readme.md']}}
    write_json('portable_equivalence_result.json',result)
    lines=['# NumPy-only fitted-estimator readout','',
        'This implementation retains the original fitted L31 CNN and MLP8 weights, layer order, float32 forward precision and original MLP float64 scaling. It does not train, quantise or prune. Inputs are the original 24 log spectral features; outputs are x/y positions in micrometres.','',
        '| Model | Batch | Maximum absolute coordinate difference (µm) | Fixed 0.01 µm gate |','|---|---|---|---|']
    for c in checks:lines.append(f"| {c['model']} | {c['batch_size']} | {c['max_abs_coordinate_difference_um']:.10g} | {'Pass' if c['gate_passed'] else 'Fail'} |")
    lines += ['', 'All 819 observations were evaluated for every case. Batch256 uses chunks of 256, 256, 256 and 51 rows; Torch and NumPy receive identical chunks. The CSV retains every coordinate prediction and signed/absolute difference. The gate was fixed before execution and was not changed. This is numerical agreement, not a statistical-equivalence claim.', '',
        'Each NumPy worker verified that neither torch nor sklearn was imported. NPZ contains only float32 weight/bias arrays and, for MLP8, the original float64 scale/mean arrays; loading uses allow_pickle=False.', '',
        '## Use', '', '```python', 'from portable_inference import PortableEstimator, predict_in_batches',
        "estimator = PortableEstimator('cnn', 'portable_cnn.npz')",
        'xy_um = predict_in_batches(estimator, log_spectral_features, batch_size=256)', '```','',
        '## Scope', '', 'No formal memory or latency profiling has been run for this backend yet. Such profiling is deferred until concurrent matched training finishes. This is a working NumPy readout on this PC, not evidence of deployment on a microcontroller or physical sensor system.']
    (OUT/'portable_readme.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'cases':checks,'paired_prediction_rows':len(rows)}),flush=True)
    if not passed:raise SystemExit(2)


def child(script,args):
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    env['PYTHONHASHSEED']='20260912';env['PYTHONIOENCODING']='utf-8'
    result=subprocess.run([sys.executable,str(script),*args],env=env,capture_output=True,text=True,
                          encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW,timeout=240)
    print(result.stdout.strip(),flush=True)
    if result.stderr.strip():print(result.stderr.strip(),flush=True)
    if result.returncode:raise RuntimeError(f'Child failed ({result.returncode}): '+str(script))


def run():
    assert not (OUT/'portable_equivalence_result.json').exists(),'Refuse silent overwrite of a completed numerical check.'
    protocol()
    child(Path(__file__).resolve(),['--export'])
    for model in ('cnn','mlp8'):
        for batch in (1,256):
            child(OUT/'portable_inference.py',['--model',model,'--weights',str(OUT/f'portable_{model}.npz'),
                '--input',str(OUT/'portable_input_features.npy'),'--output',str(OUT/f'portable_predictions_{model}_batch{batch}.npy'),
                '--batch-size',str(batch)])
    child(Path(__file__).resolve(),['--validate'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',action='store_true');group.add_argument('--export',action='store_true');group.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    if args.export:export()
    elif args.validate:validate()
    else:run()
