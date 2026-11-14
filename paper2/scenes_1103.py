"""Streaming common-random-number scenes; no files or legacy assets are read."""
from dataclasses import replace
import hashlib
import json

import numpy as np

from .data_1103 import JAM_TYPES, _config
from .radar_eval import PRI_S
from .vendor.generate_v2 import _draw_target, _draw_noise, _draw_jammer, _raw_jammer

ROLE_CODES = {'engineering': 0, 'dev': 1, 'test': 2}
SUITE_CODES = {'main': 1, 'pd': 2, 'sliceA': 3, 'sliceB': 4, 'retention': 5, 'coherent': 6}


def scene(role, suite, parent, jam_type, snr_db, jsr_db, *, width=None, repeats=None):
    if role not in ROLE_CODES or suite not in SUITE_CODES or jam_type not in JAM_TYPES:
        raise ValueError('unknown scene namespace')
    role_code, suite_code = ROLE_CODES[role], SUITE_CODES[suite]
    key = [2026111401, role_code, suite_code, int(parent)]
    if suite == 'retention':
        # 1103 uses three independently drawn no-jammer scene groups.
        key.append(JAM_TYPES.index(jam_type))
    streams = np.random.SeedSequence(key).spawn(3)
    target_rng, noise_rng, jam_rng = [np.random.default_rng(x) for x in streams]
    cfg = _config((0., 0.), (0., 0.), 2026111401)
    if suite == 'coherent':
        cfg = replace(cfg, target_doppler_min_hz=-14000., target_doppler_max_hz=14000.)
    target, active, tm = _draw_target(cfg, target_rng)
    jammed = np.zeros(8, bool)
    jammed[jam_rng.choice(8, int(jam_rng.choice([1,2,4,6,8])), replace=False)] = True
    s, j, noise0, nuisance = [], [], [], []
    for pulse in range(8):
        sp = target * np.exp(2j*np.pi*tm['doppler_hz']*PRI_S*pulse)
        ep, _, _ = _draw_noise(tm['signal_power_active'], cfg, noise_rng)
        # Separate type-specific stream; timing draw cannot affect other pulses
        # or target/noise streams when a width/M intervention changes replay.
        jrng = np.random.default_rng(np.random.SeedSequence(key+[JAM_TYPES.index(jam_type), pulse, 99]))
        raw, jm = _draw_jammer(sp, active, tm['signal_power_active'], jam_type, cfg, jrng)
        if width is not None:
            params = dict(jm)
            period = 260 if suite == 'sliceA' else (int(repeats)+1)*int(width)+1
            params.update(window_samples=int(width), period_samples=period,
                          repeat_count=int(repeats), sampling_phase_samples=0,
                          forward_delay_samples=1, replay_gap_samples=0)
            raw = _raw_jammer(sp, active, jam_type, params)
            t = np.arange(1800)/60e6
            raw *= np.exp(1j*(jm['channel_phase_rad']+2*np.pi*jm['frequency_offset_hz']*t))
            jm = params
        if not jammed[pulse]:
            raw = np.zeros(1800, complex)
        s.append(sp)
        j.append(raw)
        noise0.append(ep)
        nuisance.append({k: jm[k] for k in ('forward_delay_samples','replay_gap_samples',
                                           'channel_phase_rad','frequency_offset_hz')})
    s, j, noise0 = [np.asarray(x) for x in (s, j, noise0)]
    if width is not None:
        # Width scans explicitly use total pulse-group energy JSR_E.
        j *= np.sqrt(np.sum(abs(s)**2)*10**(jsr_db/10)/np.sum(abs(j)**2))
    else:
        # _draw_jammer already uses active-support power at 0 dB per pulse.
        j *= 10**(jsr_db/20)
    e = noise0*10**(-snr_db/20)
    invariant = hashlib.sha256(s.tobytes()+noise0.tobytes()+jammed.tobytes()+
                               json.dumps(nuisance, sort_keys=True).encode()).hexdigest()
    support = abs(j) > 0
    active_target = abs(s) > 0
    selected = jammed[:, None] & active_target
    qa = dict(total_jsr_db=float(10*np.log10(np.sum(abs(j)**2)/np.sum(abs(s)**2))),
              target_coverage_all=float((support & active_target).sum()/active_target.sum()),
              target_coverage_selected=float((support & selected).sum()/selected.sum()))
    if width is not None:
        qa.update(width_samples=int(width), period_samples=int(period), repeat_count=int(repeats),
                  sampling_phase_samples=0, forward_delay_samples=1, replay_gap_samples=0)
    suffix = f':{jam_type}' if suite == 'retention' else ''
    return s, j, e, dict(parent_id=f'{role}:{suite}:{parent}{suffix}', doppler_hz=tm['doppler_hz'],
                        paired_invariant_sha256=invariant, jammed_pulses=jammed.tolist(),
                        target_delay_samples=tm['delay_samples'], **qa)


def conditions(suite):
    if suite == 'main':
        snrs = [-30,-28,-26,-25,-24,-23,-22,-20,-15,-10,-5,0,5,10]
        return [dict(jam_type=j, snr_db=s, jsr_db=r, count=256)
                for j in JAM_TYPES for s in snrs for r in [0,10,20,35]]
    if suite == 'pd':
        return [dict(jam_type='repeat', snr_db=s, jsr_db=r, count=512)
                for s in range(-30,1) for r in range(-10,36,5)]
    if suite in ('sliceA', 'sliceB'):
        strengths = [(-10,20),(-20,35),(0,10)] if suite == 'sliceA' else [(-10,20),(-20,35)]
        widths = [3,6,18,36,54,72,84] if suite == 'sliceA' else [6,18,36,54,72]
        return [dict(jam_type='repeat', snr_db=s, jsr_db=r, width=w, repeats=m,
                     count=256 if suite=='sliceA' else 128)
                for s,r in strengths for m in ([2] if suite=='sliceA' else [2,4]) for w in widths]
    if suite == 'retention':
        return [dict(jam_type=j, snr_db=s, jsr_db=0, count=512)
                for j in JAM_TYPES for s in list(range(-30,-19))+[-15,-10,0]]
    if suite == 'coherent':
        return [dict(jam_type='repeat', snr_db=s, jsr_db=r, count=16)
                for s,r in [(-20,20),(0,20),(0,35)]]
    raise ValueError(suite)
