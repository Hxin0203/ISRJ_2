"""Target-basis perturbations and independent observation-only operator fits."""
import itertools

import numpy as np

from vp_1103_common import (FAMILIES, WORK, distribution, frame_jnr, parent_roster,
                           stratified, worst_cases, write_cases)
from vp_1103_geometry import FS, SAMPLES, lfm_atom
from vp_1103_p1 import (Projection, complement, observations, oracle_geometry,
                       orthogonal_basis, save_metrics, score)
from paper2.radar_eval import pulse_compress


DELTAS = (-2., -1., -.5, -.25, 0., .25, .5, 1., 2.)
HERTZ_PER_DELTA = 50_000.
MODES = ("zero", "tangent", "neighbors")
EMPTY = np.empty((1800, 0), dtype=np.complex128)


def protection(delay, nu, mode):
    if mode == "zero":
        return lfm_atom(delay, nu, smooth=True)[:, None]
    if mode == "tangent":
        return np.column_stack(lfm_atom(delay, nu, smooth=True, derivative=True))
    return np.column_stack([lfm_atom(delay + step, nu, smooth=True) for step in (-1, 0, 1)])


def energy(x):
    return np.sum(np.abs(x)**2, axis=(-2, -1))


def mismatch(dev, y, grid, private, update):
    jnr = frame_jnr(dev)
    s = dev["s"].reshape(-1, 8, 1800)
    s_energy = energy(s)
    delay = dev["target_delay_samples"]
    nu = dev["doppler_hz"]
    bs = [[b for b in oracle_geometry(i, f, delay[i], nu[i], grid)[1]]
          for i in range(len(delay)) for f in range(3)]
    # Projection with one B column can be expressed without materializing N×N H.
    bank = np.array([[b[:, 0] / np.linalg.norm(b) if b.shape[1] else np.zeros(1800, complex)
                      for b in row] for row in bs]).reshape(256, 3, 8, 1800)
    present = np.isfinite(dev["jsr_db"])
    output = {}
    for axis, mode, delta in itertools.product(("delay", "doppler"), MODES, DELTAS):
        name = f"{axis}_{mode}_{delta:+g}"
        update(experiment=name)
        aq = [orthogonal_basis(protection(delay[i] + (delta if axis == "delay" else 0.),
                                          nu[i] + (delta * HERTZ_PER_DELTA if axis == "doppler" else 0.), mode))
              for i in range(len(delay))]
        qbs = np.stack([complement(q, bank[i].reshape(24, 1800).T).T.reshape(3, 8, 1800)
                        for i, q in enumerate(aq)])
        chi = np.clip(np.einsum("...n,...n->...", qbs.conj(), qbs).real, 0., 1.)
        ridge = .0001 + .001 * (1 - chi)**2 / (chi + .001)
        inverse = np.where(present, 1. / (chi + ridge), 0.)
        qbs = qbs.reshape(-1, 8, 1800)
        amps = np.einsum("...pn,...spn->...sp", qbs.conj(), y[:, 2:]) * inverse.reshape(-1, 8)[:, None]
        restored = y.copy()
        restored[:, 2:] -= bank.reshape(-1, 8, 1800)[:, None] * amps[..., None]
        leak_amps = np.einsum("...n,...n->...", qbs.conj(), s) * inverse.reshape(-1, 8)
        metrics = score(dev, y, restored)
        metrics["target_leakage_l2"] = np.linalg.norm(leak_amps, axis=1).reshape(-1, 3)
        metrics["relative_target_leakage_l2"] = (
            np.linalg.norm(leak_amps, axis=1) / np.sqrt(s_energy)).reshape(-1, 3)
        metrics["removed_noise_gain_frobenius_squared"] = (chi * inverse**2).sum(axis=-1)
        metrics["min_chi"] = np.where(present, chi, 1).min(axis=-1)
        output[name] = {"axis": axis, "mode": mode, "delta": delta,
                        "metrics": stratified(metrics, jnr),
                        "private_metrics_sha256": save_metrics(private, name, metrics)}
    return output


def estimate_protection(y):
    mf, _ = pulse_compress(y)
    power = np.abs(mf)**2
    location = int(np.argmax(power))
    # Single-pulse confidence rule, fixed before inspection.
    peak_to_background = power[location] / max(np.median(power) / np.log(2), 1e-30)
    if peak_to_background < 14. or not 64 <= location <= 1136:
        return EMPTY, -1
    return protection(location, 0, "tangent"), location


def estimate_replay(y):
    """Energy-window proxy with local chirplet atoms and unknown complex gains."""
    power = np.abs(y)**2
    noise = max(np.median(power) / np.log(2), 1e-30)
    available = np.ones(len(y), dtype=bool)
    dechirped = y * np.exp(-1j * np.pi / 1200 * SAMPLES**2)
    candidates = []
    for _ in range(4):
        weighted = np.where(available, power, 0.)
        sums = np.r_[0., np.cumsum(weighted)]
        best = (0., 0, 0)
        for width in (12, 24, 48, 72):
            response = ((sums[width:] - sums[:-width]) / noise - width) / np.sqrt(width)
            start = int(np.argmax(response))
            candidate = (float(response[start]), start, width)
            if candidate[0] > best[0]:
                best = candidate
        confidence, start, width = best
        if confidence < 7.:
            break
        stop = start + width
        t = SAMPLES[start:stop].astype(float) - start
        if not available[start:stop].all():
            # Candidates overlapping an accepted interval cannot contribute a
            # second atom for the same high-energy segment.
            break
        freq = np.fft.fftfreq(128)[np.argmax(np.abs(np.fft.fft(dechirped[start:stop], 128)))]
        atom = np.zeros_like(y)
        atom[start:stop] = np.exp(2j * np.pi * freq * t + 1j * np.pi / 1200 * (t + start)**2)
        candidates.append(atom)
        available[start:stop] = False
    return np.column_stack(candidates) if candidates else EMPTY


def blind_audit(dev, y, private, update):
    s, j = [dev[key].reshape(-1, 8, 1800) for key in ("s", "j")]
    ids = parent_roster(dev, 256)
    jnr = frame_jnr(dev)
    s_energy = energy(s)
    results = {}
    cases = {}
    for method in ("raw", "isolated"):
        update(experiment=f"blind_{method}")
        r = y.copy()
        frozen = y.copy()
        no_protection = y.copy()
        leak, drift = (np.zeros(s.shape, dtype=np.complex128) for _ in range(2))
        gain = np.zeros(len(y))
        ghost = np.zeros((len(y), 8), bool)
        any_protection = np.zeros_like(ghost)
        for row, states in enumerate(y):
            for pulse in range(8):
                fitted = []
                for state in range(4):
                    waveform = states[state, pulse]
                    a, location = estimate_protection(waveform)
                    source = complement(orthogonal_basis(a), waveform) if method == "isolated" else waveform
                    b = estimate_replay(source)
                    operator = Projection(a, b, "adaptive")
                    fitted.append(operator)
                    r[row, state, pulse] -= operator(waveform)
                    # Change A in the fit only. Keep B and observation fixed.
                    no_protection[row, state, pulse] -= Projection(EMPTY, b, "adaptive")(waveform)
                    if state == 2:
                        gain[row] += operator.noise_gain
                        any_protection[row, pulse] = location >= 0
                        if location >= 0 and np.any(j[row, pulse]):
                            mf, _ = pulse_compress(j[row, pulse])
                            ghost[row, pulse] = abs(mf[location]) >= .8 * np.max(np.abs(mf))
                op0, op1 = fitted[2:4]
                target = s[row, pulse]
                background = states[2, pulse]
                frozen[row, 2, pulse] = r[row, 2, pulse]
                frozen[row, 3, pulse] -= op0(states[3, pulse])
                leak[row, pulse] = -op1(target)
                drift[row, pulse] = -(op1(background) - op0(background))
        lhs = r[:, 3] - r[:, 2] - s
        error = np.sqrt(energy(lhs - leak - drift))
        max_error = float(np.max(error / np.maximum(np.sqrt(energy(y[:, 3])), 1e-12)))
        if max_error > 1e-12:
            raise ValueError(f"drift identity mismatch {max_error}")
        metrics = score(dev, y, r)
        frozen_metrics = score(dev, y, frozen)
        control_metrics = score(dev, y, no_protection)
        total = energy(leak) + energy(drift)
        fraction = np.divide(energy(drift), total, out=np.zeros_like(total), where=total > 0.)
        metrics.update(
            leak_over_target_energy=(energy(leak) / s_energy).reshape(-1, 3),
            drift_over_target_energy=(energy(drift) / s_energy).reshape(-1, 3),
            cross_term_over_target_energy=(2 * np.sum((leak.conj() * drift).real, axis=(1, 2)) /
                                           s_energy).reshape(-1, 3),
            drift_fraction_noninterference=fraction.reshape(-1, 3),
            removed_noise_gain_frobenius_squared=gain.reshape(-1, 3),
            frozen_damage=frozen_metrics["damage"],
            frozen_pd=frozen_metrics["pd"],
            frozen_frame_pfa_jammer=frozen_metrics["frame_pfa_jammer"],
            protection_without_target_frame=any_protection.any(axis=1).reshape(-1, 3),
            ghost_protection_frame=ghost.any(axis=1).reshape(-1, 3))
        with_delta = metrics["frame_pfa_jammer"] - control_metrics["frame_pfa_jammer"]
        flagged = metrics["ghost_protection_frame"]
        metrics["protection_frame_pfa_delta_on_ghost_cases"] = np.where(flagged, with_delta, np.nan)
        metrics["protection_frame_pfa_delta_all"] = with_delta
        results[method] = {"relative_algebra_error_max": max_error, "metrics": stratified(metrics, jnr),
                           "private_metrics_sha256": save_metrics(private, f"blind_{method}", metrics)}
        cases[method] = worst_cases(metrics, ids, jnr)
        if method == "raw":
            raw = metrics
    results["isolated_minus_raw"] = stratified({key: metrics[key].astype(float) - raw[key].astype(float)
                                                for key in metrics}, jnr)
    return results, cases


def stress_test():
    a = np.zeros((1800, 1), complex)
    a[0] = 1.
    result = []
    for chi in (1., .1, .01, 1e-3, 1e-4, 1e-6, 1e-8, 0.):
        b = a * np.sqrt(1 - chi)
        b[1] = np.sqrt(chi)
        result.append({"chi": chi, "fixed_1e-6": Projection(a, b, 1e-6).noise_gain,
                       "adaptive": Projection(a, b, "adaptive").noise_gain})
    return result


def audit(dev, grid, private, update):
    y = observations(dev)
    blind, cases = blind_audit(dev, y, private, update)
    local = mismatch(dev, y, grid, private, update)
    singular = stress_test()
    lines = ["# P1-B · 几何失配与算子漂移", "",
             "时延和频偏分别施加扰动。频偏的 1 单位等于 50 kHz。",
             "零阶、切空间和邻域模板采用相同的平滑边缘。仿真原始目标保留硬边缘。",
             "代理仅从当前脉冲独立估计保护空间和干扰候选。噪声及干扰真值只用于评分与分层。",
             "漂移和泄漏的能量之外保留交叉项。冻结对照在两状态共同使用背景算子。", "",
             "| 输入 | 虚警 | Pd | 损伤 | 泄漏能量/目标 | 漂移能量/目标 | 冻结损伤 | 误保护帧 | 恒等式误差 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in ("raw", "isolated"):
        row = blind[mode]
        m = row["metrics"]["global"]
        values = [m[key]["mean"] for key in ("frame_pfa_jammer", "pd", "damage", "leak_over_target_energy",
                                            "drift_over_target_energy", "frozen_damage", "ghost_protection_frame")]
        lines.append(f"| {mode} | " + " | ".join(f"{v:.6g}" for v in values) +
                     f" | {row['relative_algebra_error_max']:.3g} |")
    lines += ["", "## 局部失配曲线", "", "详见 `SUMMARY.json` 的逐扰动曲线数据。", "",
              "| 轴 | 基 | 扰动 | 相对目标泄漏 | 虚警 | 损伤 | 噪声增益 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in local.values():
        m = row["metrics"]["global"]
        v = [m[key]["mean"] for key in ("relative_target_leakage_l2", "frame_pfa_jammer", "damage",
                                       "removed_noise_gain_frobenius_squared")]
        lines.append(f"| {row['axis']} | {row['mode']} | {row['delta']:+g} | " +
                     " | ".join(f"{x:.6g}" for x in v) + " |")
    lines += ["", "## 代理最差 parent", ""]
    for mode, selected in cases.items():
        lines += [f"### {mode}", ""]
        write_cases(lines, selected)
    lines += ["## 门槛 2", "", "判定见总闸门报告。", ""]
    return {"perturbations": local, "blind_proxy": blind, "noise_stress": singular,
            "worst_cases": cases, "training_seeds": [], "simulator_seed": 2026111302}, lines
