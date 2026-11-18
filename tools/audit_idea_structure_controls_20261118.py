"""Post-run controls on frozen parents; no training or parameter selection."""
import json
from pathlib import Path

from tools.idea_probe_20261118.run import KINDS, RUN, describe_parents, require_server

import numpy as np

from paper2.io import ROOT, now, read, sha
from tools.audit_idea_probe_20261118 import finite_tree
from tools.durable_io import atomic_json, atomic_write
from tools.idea_probe_20261118.mechanics import spectrum_metrics

WINDOWS = (32, 64)
LAGS = tuple(range(32, 257, 16))


def local_lag_values(signal, window, lags=LAGS):
    power = np.abs(signal)**2
    cumulative = np.concatenate(([0.], np.cumsum(power)))
    result = []
    floor = max(float(power.sum()) * 1e-12, 1e-24)
    for lag in lags:
        product = signal[:-lag] * signal[lag:].conj()
        sums = np.concatenate(([0j], np.cumsum(product)))
        numerator = np.abs(sums[window:] - sums[:-window])
        left = cumulative[window:len(signal) - lag + 1] - cumulative[:len(signal) - lag - window + 1]
        right = cumulative[window + lag:] - cumulative[lag:len(signal) - window + 1]
        valid = (left > floor) & (right > floor)
        denominator = np.sqrt(np.maximum(left * right, 0.))
        result.extend(np.clip(numerator[valid] / denominator[valid], 0., 1.))
    return np.asarray(result)


def normalize_rows(stack):
    norms = np.linalg.norm(stack, axis=1)
    active = norms > max(float(norms.max()) * 1e-12, 1e-24)
    return stack[active] / norms[active, None], int(active.sum())


def summarize_rows(rows):
    groups = []
    for kind in KINDS:
        for component in ('s', 'j', 'e', 'y'):
            for representation in ('lag_w32', 'lag_w64', 'slow_normalized'):
                selected = [row for row in rows if row['kind'] == kind and row['component'] == component
                            and row['representation'] == representation and row['valid']]
                metrics = ['mean', 'p95', 'max'] if representation.startswith('lag') else [
                    'top1', 'rank90', 'null_top1', 'excess_top1', 'active_pulses']
                groups.append(dict(kind=kind, component=component, representation=representation,
                                   **describe_parents(selected, metrics)))
                if representation == 'slow_normalized' and component == 'j':
                    groups.append(dict(kind=kind, component=component, representation='slow_j_multi',
                        **describe_parents([row for row in selected if row['active_pulses'] >= 2], metrics)))
    return groups


def run():
    admission = require_server()
    base = ROOT / 'outputs' / RUN
    output = base / 'structure_controls'
    output.mkdir(exist_ok=False)
    manifest, complete = read(base / 'MANIFEST.json'), read(base / 'COMPLETE.json')
    if sha(base / 'MANIFEST.json') != complete['manifest_sha256']:
        raise ValueError('Input manifest changed')
    if any(sha(ROOT / name) != digest for name, digest in manifest['sources'].items()):
        raise ValueError('Frozen source changed')
    source = sha(__file__)
    binding = dict(stage='FROZEN', created_utc=now(), admission=admission,
        source_sha256=source, original_manifest_sha256=complete['manifest_sha256'],
        windows=list(WINDOWS), lags=list(LAGS), null_seed=2026111891,
        null='Independent per-sample phase rotations; exact amplitudes and pulse occupancy preserved',
        scope='Post-run descriptive DEV controls on the same 80 parents, not independent confirmation',
        aggregation='Local mean/p95/max within each record, then three types within physical parent')
    atomic_json(output / 'MANIFEST.json', binding)
    rng = np.random.default_rng(binding['null_seed'])
    rows, verified = [], {}
    for seed in manifest['seeds']:
        name = f'components_{seed}.npz'
        verified[name] = sha(base / name)
        if verified[name] != complete['files'][name]:
            raise ValueError('Component data changed')
        with np.load(base / name, allow_pickle=False) as data:
            for parent in range(manifest['parents_per_seed']):
                target, noise = data['s'][parent], data['epsilon'][parent]
                for kind_index, kind in enumerate(KINDS):
                    jammer = data['j'][parent, kind_index]
                    for component, stack in dict(s=target, j=jammer, e=noise,
                                                  y=target + jammer + noise).items():
                        common = dict(seed=seed, parent=seed * 1000 + parent, kind=kind, component=component)
                        for window in WINDOWS:
                            values = local_lag_values(stack[0], window)
                            rows.append(dict(**common, representation=f'lag_w{window}', valid=bool(len(values)),
                                mean=float(values.mean()) if len(values) else 0.,
                                p95=float(np.quantile(values, .95)) if len(values) else 0.,
                                max=float(values.max()) if len(values) else 0.))
                        normalized, active = normalize_rows(stack)
                        actual = spectrum_metrics(normalized)
                        null = spectrum_metrics(normalized * np.exp(
                            2j * np.pi * rng.random(normalized.shape)))
                        rows.append(dict(**common, representation='slow_normalized', valid=active > 0,
                            active_pulses=active, top1=actual['top1'], rank90=actual['rank90'],
                            null_top1=null['top1'], excess_top1=actual['top1'] - null['top1']))
    expected = manifest['parents_per_seed'] * len(manifest['seeds']) * 3 * 4 * 3
    cells = {(row['parent'], row['kind'], row['component'], row['representation']) for row in rows}
    if len(rows) != expected or len(cells) != expected or not finite_tree(rows):
        raise ValueError('Invalid control row grid')
    atomic_json(output / 'ROWS.json', rows)
    reread = read(output / 'ROWS.json')
    groups = summarize_rows(rows)
    if groups != summarize_rows(reread) or sha(__file__) != source:
        raise ValueError('Summary/source verification failed')
    summary = dict(stage='COMPLETE', completed_utc=now(), rows=len(rows),
        summary_reproduced_from_saved_rows=True, source_sha256=source,
        manifest_sha256=sha(output / 'MANIFEST.json'), rows_sha256=sha(output / 'ROWS.json'),
        verified_input_sha256=verified, groups=groups,
        limitations=['Same parents as the first probe; no CFAR/Pfa linkage or causal recovery test',
                     'Phase null is amplitude/occupancy matched, not a realistic jammer generator',
                     'Local lag metrics aggregate overlapping windows; windows are not independent',
                     'Single active pulse has rank one even without cross-pulse coherence'])
    atomic_json(output / 'SUMMARY.json', summary)
    receipt = dict(stage='VERIFIED', completed_utc=now(), summary_sha256=sha(output / 'SUMMARY.json'),
                   manifest_sha256=sha(output / 'MANIFEST.json'), rows_sha256=sha(output / 'ROWS.json'))
    atomic_json(output / 'COMPLETE.json', receipt)
    for name in ('MANIFEST', 'SUMMARY', 'COMPLETE'):
        atomic_write(ROOT / 'evidence/outputs' / RUN / f'STRUCTURE_CONTROLS_{name}.json',
                     lambda stream, name=name: stream.write((output / f'{name}.json').read_bytes()))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    run()
