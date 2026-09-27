"""Independent saved-output audit; does not call the study's aggregation code.

Recomputes grouped means, extrema, Student-t intervals, paired improvements,
Tables 1--3, scenario matching, FFT-budget endpoints, and recorded source hashes.
This is a data-lineage check, not independent physical validation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from scipy.stats import t


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def condition(row):
    return (int(row['radius']), str(row['calibration']) == 'True',
            str(row['quantized']) == 'True')


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def ci(values):
    values = list(values)
    return None if len(values) < 2 else float(t.ppf(.975, len(values)-1)) * statistics.stdev(values) / math.sqrt(len(values))


def audit(results, assets, code):
    config = json.loads((results/'config.json').read_text())
    raw = read_csv(results/'raw_metrics.csv')
    seed_rows = read_csv(results/'seed_metrics.csv')
    summary = json.loads((results/'summary.json').read_text())
    scenarios = read_csv(results/'scenarios_evaluation.csv')
    groups = defaultdict(list)
    for row in raw:
        groups[(int(row['seed']), row['method'], *condition(row))].append(row)
    seed_map = {(int(r['seed']), r['method'], *condition(r)): r for r in seed_rows}
    issues = []
    if len(seed_map) != len(seed_rows):
        issues.append('Duplicate seed-level records')
    if set(groups) != set(seed_map):
        issues.append('Raw-data group keys differ from seed-level keys')
    metrics = ('nmse','correct_power','wrong_power','background_power')
    grouped = {}
    group_error = 0.0
    for key, rows in groups.items():
        expected = 2 * config['eval_draws'] * (2*key[2]+1)**2
        if len(rows) != expected:
            issues.append(f'Wrong scenario/slot count: {key}')
        if len({(int(r['scenario']), int(r['slot'])) for r in rows}) != expected:
            issues.append(f'Duplicate or missing scenario/slot records: {key}')
        d = {m: mean(float(r[m]) for r in rows) for m in metrics}
        d['max_nmse'] = max(float(r['nmse']) for r in rows)
        d['min_correct_power'] = min(float(r['correct_power']) for r in rows)
        grouped[key] = d
        for metric, value in d.items():
            group_error = max(group_error, abs(value - float(seed_map[key][metric])))
    if group_error > 1e-12:
        issues.append('Seed aggregation differs from independently grouped raw records')

    conditions = defaultdict(list)
    for key, record in grouped.items():
        conditions[key[1:]].append((key[0], record))
    computed = {}
    summary_error = 0.0
    for key, rows in conditions.items():
        if sorted(seed for seed, _ in rows) != sorted(config['seeds']):
            issues.append(f'Seed coverage differs: {key}')
        computed[key] = {m: {'mean': mean(r[m] for _, r in rows),
                            'ci95_halfwidth': ci(r[m] for _, r in rows)}
                         for m in (*metrics, 'max_nmse', 'min_correct_power')}
    for row in summary['summary']:
        key = (row['method'], int(row['radius']), row['calibration'], row['quantized'])
        for metric, values in computed[key].items():
            for stat, value in values.items():
                if value is None:
                    if row[metric][stat] is not None:
                        issues.append(f'CI with one seed is not null: {key}')
                else:
                    summary_error = max(summary_error, abs(value-row[metric][stat]))
    if summary_error > 1e-12:
        issues.append('Summary means or CIs differ from independent aggregation')

    scenario_map = {(int(r['seed']), *condition(r), int(r['scenario'])): r for r in scenarios}
    scenario_error = 0.0
    for row in raw:
        key = (int(row['seed']), *condition(row), int(row['scenario']))
        reference = scenario_map[key]
        for field in ('dx','dy','gamma','b'):
            scenario_error = max(scenario_error, abs(float(row[field])-float(reference[field])))
    if scenario_error > 0:
        issues.append('Raw records do not match archived evaluation scenarios')

    paired_error = 0.0
    for row in summary['paired_changes']:
        method = row['method']
        values = [100*(1-grouped[(seed,method,1,True,True)]['nmse'] /
                         grouped[(seed,'Nominal',1,True,True)]['nmse'])
                  for seed in config['seeds']]
        actual = {'mean_reduction_percent': mean(values), 'ci95_halfwidth': ci(values),
                  'wrong_power_ratio_of_means': computed[(method,1,True,True)]['wrong_power']['mean'] /
                                              computed[('Nominal',1,True,True)]['wrong_power']['mean']}
        for field, value in actual.items():
            if value is not None:
                paired_error = max(paired_error, abs(value-row[field]))
    if paired_error > 1e-10:
        issues.append('Paired percentages or power ratios differ')

    table2_error = 0.0
    for row in read_csv(assets/'table2.csv'):
        stats = computed[(row['method'],1,True,True)]
        for metric in ('nmse','correct_power','wrong_power'):
            table2_error = max(table2_error, abs(float(row[metric+'_mean'])-stats[metric]['mean']))
            if row[metric+'_ci95']:
                table2_error = max(table2_error, abs(float(row[metric+'_ci95'])-stats[metric]['ci95_halfwidth']))
    if table2_error > 1e-12:
        issues.append('Table 2 does not match independently aggregated raw data')
    table3_error = 0.0
    for row in read_csv(assets/'table3.csv'):
        for method in ('Nominal','R-F','R-U','Collapsed'):
            if row[method] == '--':
                continue
            value = computed[(method,int(row['radius']),row['calibration']=='Yes',True)]['nmse']['mean']
            table3_error = max(table3_error, abs(float(row[method])-value))
    if table3_error > 0.5000001e-6:
        issues.append('Table 3 differs beyond six-decimal rounding')
    expected_design = [(0,0,1.15,.08),(-1,-1,1.07,.05),(-1,1,1.23,.05),(1,-1,1.07,.11),(1,1,1.23,.11)]
    for file in (results/'scenarios_design.csv', assets/'table1.csv'):
        actual = [tuple(float(r[k]) for k in ('dx','dy','gamma','b')) for r in read_csv(file)]
        if actual != expected_design:
            issues.append(f'Table 1/design input mismatch: {file.name}')

    histories = defaultdict(list)
    for row in read_csv(results/'histories.csv'):
        histories[(int(row['seed']),row['method'])].append(row)
    budget_ends = {}
    for key, rows in histories.items():
        rows.sort(key=lambda r: int(r['update']))
        method = key[1]
        scenarios_per_update = 5 if method.startswith('R-') else 1
        updates = config['robust_updates'] if method.startswith('R-F') else config['updates']
        endpoint = (int(rows[-1]['update']),int(rows[-1]['fft_pairs_per_slot']))
        budget_ends[method] = list(endpoint)
        if endpoint != (updates,updates*scenarios_per_update):
            issues.append(f'History endpoint budget mismatch: {key}')
        if any(int(r['fft_pairs_per_slot']) != scenarios_per_update*int(r['update']) for r in rows):
            issues.append(f'History budget-axis mismatch: {key}')

    environment = json.loads((results/'environment.json').read_text())
    hashes = {}
    for filename, recorded in environment['code_sha256'].items():
        actual = hashlib.sha256((code/filename).read_bytes()).hexdigest()
        hashes[filename] = {'recorded': recorded, 'current': actual, 'match': recorded == actual}
        if actual != recorded:
            issues.append(f'Recorded run source changed: {filename}')
    manifest = json.loads((assets/'asset_provenance.json').read_text())
    figure_inputs_exist = {}
    for name in ('angular_channels','reconstruction_comparison','robustness_sweep','collapsed_diagnostic','optimization_history'):
        figure_inputs_exist[name] = all((results/f).is_file() for f in manifest[name]['inputs'])
        if not figure_inputs_exist[name] or not (assets/(name+'.pdf')).is_file():
            issues.append(f'Missing figure or declared input: {name}')
    return {'audit_type': 'independent_raw_to_table_aggregation', 'raw_rows': len(raw),
            'seed_rows': len(seed_rows), 'raw_groups': len(groups), 'seed_group_max_abs_error': group_error,
            'summary_max_abs_error': summary_error, 'evaluation_scenario_max_abs_error': scenario_error,
            'paired_comparison_max_abs_error': paired_error,
            'table2_max_abs_error': table2_error, 'table3_max_rounding_error': table3_error,
            'history_endpoints_updates_and_pairs_per_slot': budget_ends,
            'run_source_hashes': hashes, 'figure_inputs_exist': figure_inputs_exist,
            'primary_recomputed': {key[0]: value for key,value in computed.items() if key[1:] == (1,True,True)},
            'issues': issues, 'passed': not issues,
            'scope': 'Independent statistics recomputation without importing run_study aggregation. Does not validate physical optics or recover original missing data.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results_reconstructed'))
    parser.add_argument('--assets', type=Path, default=Path('artifacts_reconstructed'))
    parser.add_argument('--code', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit(args.results,args.assets,args.code)
    path = args.out or args.results/'aggregation_audit.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'primary_recomputed'},indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
