"""Fresh-parent CPU screening on server B, not confirmation or training."""
import argparse
from dataclasses import asdict
import json
import os
import platform
import shutil
import time
from pathlib import Path

for thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[thread_variable] = '1'

import numpy as np
import psutil

from paper2.data_1103 import _config
from paper2.io import ROOT, now, sha
from paper2.vendor.generate_v2 import _draw_target, _draw_noise, _draw_jammer
from tools.durable_io import ProcessLock, atomic_json, atomic_write
from .mechanics import (DECHIRP, FS, LENGTH, complement, energy, hankel_metrics,
                        infer, long_lag_peak, spectrum_metrics)

RUN = 'idea_probe_20261118_v1'
SEEDS = (2026111881, 2026111882, 2026111883, 2026111884, 2026111885)
KINDS = ('direct', 'repeat', 'cyclic')
METHODS = ('unprotected', 'hard', 'mixture')
DOCUMENTS = {
    'gptpro_innovation': '368ca894f2a575cc53e0032b5fc8cc90496e4584a2ad59fa73512d5f79cc0fd9',
    'model_ideas': 'd0df96cb1994fcf4a48429b98e1c87c3c2fc1126feb2b923678d3add79700d99',
}


def source_hashes():
    paths = [*sorted((ROOT / 'tools/idea_probe_20261118').glob('*.py')),
             ROOT / 'paper2/data_1103.py', ROOT / 'paper2/io.py',
             ROOT / 'paper2/vendor/generate_v2.py', ROOT / 'paper2/vendor/label_builder.py',
             ROOT / 'tools/durable_io.py']
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def require_server():
    if platform.node().upper() != 'DESKTOP-RFFERM3' or ROOT.resolve() != Path('D:/code/Xin/ISRJ_222').resolve():
        raise RuntimeError('This experiment must run in the dedicated server-B tree')
    sample = dict(cpu_percent=psutil.cpu_percent(interval=1),
                  available_ram=psutil.virtual_memory().available,
                  free_disk=shutil.disk_usage(ROOT).free)
    if sample['cpu_percent'] >= 85 or sample['available_ram'] < 5 * 1024**3 or sample['free_disk'] < 8 * 1024**3:
        raise RuntimeError(f'Resource admission refused: {sample}')
    return sample


def make_parent(rng, seed):
    config = _config((-30., 15.), (-12., 45.), seed)
    target, active, meta = _draw_target(config, rng)
    jammed = np.zeros(8, dtype=bool)
    count = int(rng.choice([1, 2, 4, 6, 8]))
    jammed[rng.choice(8, size=count, replace=False)] = True
    targets, noises = [], []
    for pulse in range(8):
        targets.append(target * np.exp(2j * np.pi * meta['doppler_hz'] * LENGTH / FS * pulse))
        noise, _, _ = _draw_noise(meta['signal_power_active'], config, rng)
        noises.append(noise)
    jammers = np.zeros((3, 8, LENGTH), complex)
    for kind_index, kind in enumerate(KINDS):
        for pulse in range(8):
            if jammed[pulse]:
                jammers[kind_index, pulse], _ = _draw_jammer(
                    targets[pulse], active, meta['signal_power_active'], kind, config, rng)
    return np.asarray(targets), jammers, np.asarray(noises), active


def structure_rows(targets, jammers, noises, active):
    rows = []
    for kind_index, kind in enumerate(KINDS):
        components = {'s': targets, 'j': jammers[kind_index], 'e': noises,
                      'y': targets + jammers[kind_index] + noises}
        for component, stack in components.items():
            first = stack[0]
            common = dict(kind=kind, component=component, nonzero=bool(energy(stack) > 1e-20))
            rows.append(dict(**common, representation='slow_time_8x1800', **spectrum_metrics(stack)))
            for representation, signal in [('raw_full', first), ('dechirped_full', first * DECHIRP),
                                           ('dechirped_oracle_target_crop', (first * DECHIRP)[active])]:
                rows.append(dict(**common, representation=representation,
                                 pulse_nonzero=bool(energy(signal) > 1e-20),
                                 **hankel_metrics(signal)))
            rows.append(dict(**common, representation='raw_long_lag_64_600',
                             peak=long_lag_peak(first), pulse_nonzero=bool(energy(first) > 1e-20)))
    return rows


def parent_mean(rows, metric):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['seed'], row['parent']), []).append(row[metric])
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def describe_parents(rows, metrics):
    result = {}
    for metric in metrics:
        values = np.asarray(list(parent_mean(rows, metric).values()))
        if len(values):
            result[metric] = dict(mean=float(values.mean()), median=float(np.median(values)),
                                  p95=float(np.quantile(values, .95)), parents=int(len(values)))
    return result


def summarize(inference, structure):
    metrics = [name for name, value in inference[0].items()
               if isinstance(value, (int, float)) and name not in ('seed', 'parent', 'jammed')]
    groups = {}
    for method in METHODS:
        selected = [row for row in inference if row['method'] == method]
        groups[method] = describe_parents(selected, metrics)
    paired = []
    rng = np.random.default_rng(2026111880)
    for metric in ('endpoint', 'damage', 'noise_change', 'clean_change'):
        hard = parent_mean([row for row in inference if row['method'] == 'hard'], metric)
        mixture = parent_mean([row for row in inference if row['method'] == 'mixture'], metric)
        keys = sorted(hard)
        delta = np.asarray([mixture[key] - hard[key] for key in keys])
        bootstrap = delta[rng.integers(0, len(keys), size=(4000, len(keys)))].mean(axis=1)
        paired.append(dict(metric=metric, mixture_minus_hard=float(delta.mean()),
                           ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                           fraction_improved=float((delta < 0).mean())))
    structures = []
    for component in ('s', 'j', 'e', 'y'):
        for representation in sorted({row['representation'] for row in structure}):
            rows = [row for row in structure if row['component'] == component
                    and row['representation'] == representation
                    and row.get('pulse_nonzero', row['nonzero'])]
            names = ['peak'] if 'long_lag' in representation else ['top1', 'rank90', 'effective_rank']
            structures.append(dict(component=component, representation=representation,
                                   **describe_parents(rows, names)))
    unique = [row for row in inference if row['method'] == 'hard']
    jammed = [row for row in unique if row['jammed']]
    ambiguity = dict(records=len(unique), independent_parents=len({row['parent'] for row in unique}),
        waveform_spread_over_001=sum(row['near_waveform_spread'] > .01 for row in unique),
        fixed_damage_spread_over_0001=sum(row['near_fixed_damage_spread'] > .001 for row in unique),
        jointly_over_threshold=sum(row['near_waveform_spread'] > .01
                                  and row['near_fixed_damage_spread'] > .001 for row in unique),
        best_target_coverage_over_09=sum(row['target_coverage_best'] >= .9 for row in unique),
        hard_target_coverage_over_09=sum(row['target_coverage_hard'] >= .9 for row in unique),
        jammed_first_pulse_records=len(jammed))
    repeats = {str(seed): {method: describe_parents(
        [row for row in inference if row['seed'] == seed and row['method'] == method],
        ['endpoint', 'damage', 'noise_change', 'clean_change']) for method in METHODS} for seed in SEEDS
        if any(row['seed'] == seed for row in inference)}
    return dict(inference=groups, paired=paired, structure=structures, ambiguity=ambiguity,
                repeats=repeats,
                max_decomposition_error=max(row['decomposition_error'] for row in inference),
                limitations=['DEV mechanism screen; no training, CFAR Pd/Pfa or independent confirmation',
                             'First-pulse inference only; full eight-pulse spectra are descriptive',
                             'Short-chirp dictionary relaxes legal shared replay structure',
                             'Score weights use a heuristic noise estimate, not a calibrated posterior',
                             'Oracle crop uses scoring-only true support and is not deployable'])


def save_npz(path, arrays):
    atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))


def run(smoke=False):
    admission = require_server()
    sources = source_hashes()
    base = ROOT / 'outputs' / (RUN + ('_smoke' if smoke else ''))
    base.mkdir(parents=True, exist_ok=False)
    with ProcessLock(base / 'PROCESS.lock'):
        manifest = dict(run_id=RUN, created_utc=now(), hostname=platform.node(), root=str(ROOT),
            scope='CPU_ONLY_MECHANISM_DEV', cuda_used=False, threads=1, resource_admission=admission,
            sources=sources, document_sha256=DOCUMENTS, seeds=list(SEEDS[:1] if smoke else SEEDS),
            parents_per_seed=1 if smoke else 16, generator=asdict(_config((-30., 15.), (-12., 45.), 0)),
            inference='observation only; null plus three global target proposals, six chirplets per proposal',
            ridge=.01, target='S+same E', reference='identity/no subtraction and unprotected chirplet solver',
            comparison='same candidate library: minimum normalized profiled residual versus exp(-relative score)',
            score_noise='median adjacent complex difference energy / (2 log 2)',
            statistical_unit='independent physical parent; three jammer types averaged within parent',
            prespecified_thresholds=dict(near_score_delta=2., waveform_difference=.01,
                                        fixed_damage_difference=.001, target_coverage=.9),
            selection_rule='No parameter selection or model training from these results',
            source_base_commit='34c1489ee3e73a55908793c9e3a42db55b47e051')
        atomic_json(base / 'MANIFEST.json', manifest)
        started = time.perf_counter()
        all_inference, all_structure, files = [], [], {}
        for seed in manifest['seeds']:
            rng = np.random.default_rng(seed)
            saved_targets, saved_jammers, saved_noises = [], [], []
            for parent in range(manifest['parents_per_seed']):
                targets, jammers, noises, active = make_parent(rng, seed)
                saved_targets.append(targets)
                saved_jammers.append(jammers)
                saved_noises.append(noises)
                background_inference = infer(noises[0])
                clean_inference = infer(targets[0] + noises[0])
                records = []
                for kind_index, kind in enumerate(KINDS):
                    records.extend(inference_rows(targets[0], jammers[kind_index, 0], noises[0],
                                                  background_inference, clean_inference, kind))
                spectra = structure_rows(targets, jammers, noises, active)
                for row in [*records, *spectra]:
                    row.update(seed=seed, parent=seed * 1000 + parent)
                all_inference.extend(records)
                all_structure.extend(spectra)
                status = dict(stage='PROBING', seed=seed, saved_parent=parent + 1,
                              elapsed_s=time.perf_counter() - started, updated_utc=now())
                atomic_json(base / 'STATUS.json', status)
                print(json.dumps(status), flush=True)
            data_path = base / f'components_{seed}.npz'
            save_npz(data_path, dict(s=np.asarray(saved_targets), j=np.asarray(saved_jammers),
                                     epsilon=np.asarray(saved_noises)))
            files[data_path.name] = sha(data_path)
        for name, rows in [('INFERENCE_ROWS.json', all_inference), ('STRUCTURE_ROWS.json', all_structure)]:
            atomic_json(base / name, rows)
            files[name] = sha(base / name)
        summary = summarize(all_inference, all_structure)
        summary.update(run_id=RUN, completed_utc=now(), elapsed_s=time.perf_counter() - started,
                       manifest_sha256=sha(base / 'MANIFEST.json'), data_sha256=files, source_sha256=sources)
        if source_hashes() != sources:
            raise RuntimeError('Source changed during diagnostic')
        atomic_json(base / 'SUMMARY.json', summary)
        atomic_json(base / 'COMPLETE.json', dict(stage='COMPLETE', completed_utc=now(),
            manifest_sha256=sha(base / 'MANIFEST.json'), summary_sha256=sha(base / 'SUMMARY.json'), files=files))
        evidence = ROOT / 'evidence/outputs' / base.name
        evidence.mkdir(parents=True, exist_ok=False)
        for name in ('MANIFEST.json', 'SUMMARY.json', 'COMPLETE.json'):
            atomic_write(evidence / name, lambda stream, name=name: stream.write((base / name).read_bytes()))
        print(json.dumps(dict(stage='COMPLETE', output=str(base), summary_sha256=sha(base / 'SUMMARY.json'))), flush=True)


def inference_rows(target, jammer, noise, background_inference, clean_inference, kind):
    background = jammer + noise
    observation = target + background
    started = time.perf_counter()
    before = infer(background) if energy(jammer) > 0 else background_inference
    after = infer(observation) if energy(jammer) > 0 else clean_inference
    elapsed = (time.perf_counter() - started) / 2
    target_energy = energy(target)
    reference_energy = energy(target + noise)
    near = np.flatnonzero(after.scores - after.scores.min() <= 2.)
    waveform_spread = max((energy(after.corrections[left] - after.corrections[right]) / target_energy
                           for left in near for right in near), default=0.)
    fixed_risks = [energy(after.operators[index].apply(target)) / target_energy for index in near]
    coverages = [1 - energy(complement(operator.protected, target)) / target_energy
                 for operator in after.operators]
    chosen = int(after.scores.argmin())
    rows = []
    for method in METHODS:
        correction_before = before.apply(background, method)
        correction_after = after.apply(observation, method)
        leakage = -after.apply(target, method)
        drift = before.apply(background, method) - after.apply(background, method)
        damage = correction_before - correction_after
        rows.append(dict(kind=kind, method=method, jammed=bool(energy(jammer) > 0),
            endpoint=energy(jammer - correction_after) / reference_energy,
            baseline_endpoint=energy(jammer) / reference_energy,
            background_endpoint=energy(jammer - correction_before) / energy(noise),
            damage=energy(damage) / target_energy,
            frozen_damage=energy(before.apply(target, method)) / target_energy,
            leakage=energy(leakage) / target_energy, drift=energy(drift) / target_energy,
            cross=2 * float(np.vdot(leakage, drift).real) / target_energy,
            decomposition_error=energy(damage - leakage - drift) / target_energy,
            noise_change=energy(background_inference.apply(noise, method)) / energy(noise),
            clean_change=energy(clean_inference.apply(target + noise, method)) / reference_energy,
            h0_jammer_protected_fraction=energy(before.operators[int(before.scores.argmin())].protected.conj().T @ jammer)
                / max(energy(jammer), 1e-20),
            target_coverage_hard=coverages[chosen], target_coverage_best=max(coverages),
            near_candidate_count=int(len(near)), near_waveform_spread=waveform_spread,
            near_fixed_damage_spread=max(fixed_risks) - min(fixed_risks),
            mixture_max_weight=float(after.weights.max()),
            proposal_pair_seconds=elapsed))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    arguments = parser.parse_args()
    run(arguments.smoke)
