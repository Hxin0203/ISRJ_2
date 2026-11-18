"""Read frozen server results, verify raw evidence, and describe fixed strata."""
import json
import math
from pathlib import Path

from paper2.io import ROOT, now, sha
from tools.durable_io import atomic_json, atomic_write
from tools.idea_probe_20261118.run import (KINDS, METHODS, RUN, describe_parents,
                                         require_server, summarize)


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def check_rows(inference, structure, manifest):
    parents = {seed * 1000 + parent for seed in manifest['seeds']
               for parent in range(manifest['parents_per_seed'])}
    expected = {(parent, kind, method) for parent in parents for kind in KINDS for method in METHODS}
    actual = [(row['parent'], row['kind'], row['method']) for row in inference]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Missing or duplicated inference cells')
    representations = ('slow_time_8x1800', 'raw_full', 'dechirped_full',
                       'dechirped_oracle_target_crop', 'raw_long_lag_64_600')
    expected = {(parent, kind, component, representation) for parent in parents for kind in KINDS
                for component in ('s', 'j', 'e', 'y') for representation in representations}
    actual = [(row['parent'], row['kind'], row['component'], row['representation']) for row in structure]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Missing or duplicated structure cells')
    for row in inference + structure:
        if row['seed'] not in manifest['seeds'] or row['parent'] // 1000 != row['seed']:
            raise ValueError('Invalid seed/parent binding')
    if not finite_tree(inference) or not finite_tree(structure):
        raise ValueError('Nonfinite measurements')
    return dict(parents=len(parents), inference_rows=len(inference), structure_rows=len(structure))


def run():
    require_server()
    base = ROOT / 'outputs' / RUN
    load = lambda name: json.loads((base / name).read_text(encoding='utf-8'))
    manifest, summary, complete = [load(name + '.json') for name in ('MANIFEST', 'SUMMARY', 'COMPLETE')]
    expected = dict(complete['files'], **{'MANIFEST.json': complete['manifest_sha256'],
                                        'SUMMARY.json': complete['summary_sha256']})
    verified = {name: sha(base / name) for name in expected}
    if verified != expected:
        raise ValueError('Raw evidence hash mismatch')
    if any(sha(ROOT / name) != digest for name, digest in manifest['sources'].items()):
        raise ValueError('Frozen source hash mismatch')
    inference, structure = load('INFERENCE_ROWS.json'), load('STRUCTURE_ROWS.json')
    counts = check_rows(inference, structure, manifest)
    recomputed = summarize(inference, structure)
    if any(summary[key] != value for key, value in recomputed.items()):
        raise ValueError('Summary not reproduced from raw rows')
    metrics = ['endpoint', 'baseline_endpoint', 'damage', 'frozen_damage', 'leakage', 'drift',
               'cross', 'target_coverage_hard', 'h0_jammer_protected_fraction']
    strata = []
    for kind in KINDS:
        for jammed in (False, True):
            for method in METHODS:
                rows = [row for row in inference if row['kind'] == kind and row['jammed'] == jammed
                        and row['method'] == method]
                strata.append(dict(kind=kind, jammed=jammed, method=method,
                                   metrics=describe_parents(rows, metrics)))
    ambiguity = []
    for jammed in (False, True):
        rows = [row for row in inference if row['method'] == 'hard' and row['jammed'] == jammed]
        selected = [row for row in rows if row['near_waveform_spread'] > .01
                    and row['near_fixed_damage_spread'] > .001]
        ambiguity.append(dict(jammed=jammed, records=len(rows),
            parents=len({row['parent'] for row in rows}), jointly_flagged_records=len(selected),
            jointly_flagged_parents=len({row['parent'] for row in selected})))
    lag_by_kind = []
    for kind in KINDS:
        for component in ('s', 'j', 'e', 'y'):
            rows = [row for row in structure if row['kind'] == kind and row['component'] == component
                    and row['representation'] == 'raw_long_lag_64_600' and row['pulse_nonzero']]
            lag_by_kind.append(dict(kind=kind, component=component, **describe_parents(rows, ['peak'])))
    receipt = dict(stage='VERIFIED', audited_utc=now(), counts=counts, summary_reproduced=True,
        frozen_sources_match=True, verified_sha256=verified, strata=strata,
        ambiguity_by_jammed=ambiguity, long_lag_by_kind=lag_by_kind,
        audit_source_sha256=sha(Path(__file__)),
        limitations=['Descriptive post-run strata; no new model selection or confirmatory test',
                     'Candidate ambiguity is specific to a coarse relaxed library, not a posterior',
                     'H0 protection metric refers to the hard-selected candidate for all methods'])
    target = base / 'AUDIT.json'
    if target.exists():
        raise FileExistsError(target)
    atomic_json(target, receipt)
    atomic_write(ROOT / 'evidence/outputs' / RUN / 'AUDIT.json',
                 lambda stream: stream.write(target.read_bytes()))
    print(json.dumps(dict(stage='VERIFIED', **counts, audit_sha256=sha(target))), flush=True)


if __name__ == '__main__':
    run()
