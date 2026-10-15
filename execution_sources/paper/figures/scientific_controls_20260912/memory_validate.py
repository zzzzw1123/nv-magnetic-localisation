"""Read-only verification of completed memory measurements; no model is loaded."""
import csv
import json
import statistics
from memory_profile import OUT, ROOT, MIB, METRICS, EXPECTED, sha, verify_sources, write_json

summary = json.loads((OUT / 'memory_profiles_summary.json').read_text(encoding='utf-8'))
assert summary['script_sha256'] == sha(OUT / 'memory_profile.py')
assert summary['status'] == 'COMPLETED_FIVE_ISOLATED_PROCESSES_PER_FITTED_MODEL'
sources = summary['input_provenance']['source_sha256']
for rel, digest in sources.items():
    assert sha(ROOT / rel) == digest, rel
verify_sources()
assert sha(OUT / 'memory_input_features.npy') == summary['input_provenance']['features_sha256']
for model in ('cnn', 'mlp8'):
    for record in summary['input_provenance']['prediction_checks'][model]['rows']:
        assert record['absolute_difference_um'] == 0
stages = ('runtime_baseline', 'model_loaded', 'post_batch_1', 'post_batch_256')
raw = [json.loads((OUT / f'memory_raw_{repeat:02}_{model}.json').read_text(encoding='utf-8'))
       for repeat in range(1, 6) for model in ('cnn', 'mlp8')]
assert len({obj['pid'] for obj in raw}) == 10
for model in ('cnn', 'mlp8'):
    assert len({obj['prediction_sha256_by_batch']['1'] for obj in raw if obj['model'] == model}) == 1
for obj in raw:
    assert obj['runtime']['torch_threads'] == obj['runtime']['torch_interop_threads'] == 1
    assert obj['checkpoint_sha256'] == EXPECTED[obj['model']][1]
    assert obj['prediction_sha256_by_batch']['256'] == summary['input_provenance']['prediction_checks'][obj['model']]['prediction_sha256']
    assert len(obj['samples']) == 20
    baseline = [r for r in obj['samples'] if r['stage'] == 'runtime_baseline']
    for stage in stages:
        group = [r for r in obj['samples'] if r['stage'] == stage]
        assert len(group) == 5 and [r['sample'] for r in group] == [1, 2, 3, 4, 5]
        for metric in METRICS[2:]:
            # For this observed run, each stage's peak counter is constant;
            # its within-stage median is consequently also the observed maximum.
            assert len({r[metric] for r in group}) == 1, (obj['model'], obj['repeat'], stage, metric)
        for row in group:
            assert row['rss_bytes'] > 0 and row['private_commit_bytes'] > 0
            assert row['process_lifetime_peak_rss_bytes'] >= row['rss_bytes']
            assert row['process_lifetime_peak_commit_bytes'] >= row['private_commit_bytes']
            for metric in METRICS:
                assert row[metric.replace('_bytes', '_MiB')] == row[metric] / MIB
            for point_metric in ('rss', 'private_commit'):
                expected = row[point_metric + '_bytes'] - statistics.median(r[point_metric + '_bytes'] for r in baseline)
                assert row[point_metric + '_increment_from_runtime_baseline_bytes'] == expected
    for metric in METRICS[2:]:
        values = [r[metric] for r in obj['samples']]
        assert values == sorted(values)
rows = [row for obj in raw for row in obj['samples']]
with (OUT / 'memory_profiles_raw.csv').open(encoding='utf-8', newline='') as handle:
    csv_rows = list(csv.DictReader(handle))
assert len(rows) == len(csv_rows) == 200
for expected, actual in zip(rows, csv_rows):
    assert {key: str(value) for key, value in expected.items()} == actual
for item in summary['summary']:
    group = [row for row in rows if row['model'] == item['model'] and row['stage'] == item['stage']]
    for metric in METRICS + ('rss_increment_from_runtime_baseline_bytes', 'private_commit_increment_from_runtime_baseline_bytes'):
        medians = [statistics.median(row[metric] for row in group if row['repeat'] == repeat) for repeat in range(1, 6)]
        assert item[metric] == {'per_repeat_medians': medians, 'median': statistics.median(medians), 'min': min(medians), 'max': max(medians)}
negative = sum(row['private_commit_increment_from_runtime_baseline_bytes'] < 0 for row in rows)
assert negative > 0
result = {'status': 'MEMORY_PROFILE_RECORDS_VERIFIED', 'processes': 10, 'snapshots': 200,
          'source_files_sha256_verified': len(sources), 'CSV_matches_all_raw_JSON_rows': True,
          'all_six_original_replay_means_exact': True, 'all_batch256_prediction_hashes_match': True,
          'all_threads_equal_one': True, 'all_summary_medians_ranges_recomputed': True,
          'within_stage_peak_counters_constant': True, 'lifetime_peak_counters_monotonic': True,
          'negative_private_commit_increment_snapshots_retained': negative,
          'model_training_performed': False, 'validation_loaded_models': False,
          'completed_measurement_utc': summary['completed_utc'],
          'artifact_sha256': {p.name: sha(p) for p in sorted(OUT.glob('memory_*')) if p.is_file() and p.name != 'memory_validation.json'}}
write_json('memory_validation.json', result)
lines = ['# Whole-process lifetime peak counters', '',
         'These peaks include Python/Torch imports, model loading and preceding stages. They are not model-only RAM or independently reset per-batch peaks. Values are median [min, max] across all five fresh processes; MiB = 1,048,576 bytes.', '',
         '| Model | Stage | Lifetime peak RSS (MiB) | Lifetime peak private commit (MiB) |',
         '|---|---|---|---|']
for item in summary['summary']:
    def display(name):
        value = item[name]
        return f"{value['median']:.3f} [{value['min']:.3f}, {value['max']:.3f}]"
    lines.append('| ' + item['model'] + ' | ' + item['stage'] + ' | ' + display('process_lifetime_peak_rss_MiB') + ' | ' + display('process_lifetime_peak_commit_MiB') + ' |')
(OUT / 'memory_peak_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
result['artifact_sha256']['memory_peak_summary.md'] = sha(OUT / 'memory_peak_summary.md')
write_json('memory_validation.json', result)
print(json.dumps({key: value for key, value in result.items() if key != 'artifact_sha256'}))
