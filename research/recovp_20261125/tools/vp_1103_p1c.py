"""DEV structure, lag peaks, profile ambiguity and Hankel overlap."""
import itertools

import numpy as np
from scipy.stats import spearmanr

from vp_1103_common import (BOOTSTRAP_SEED, FAMILIES, distribution, frame_jnr, parent_mean,
                           parent_roster, stratified, strata)
from vp_1103_geometry import FS, SAMPLES, lfm_atom, replay_atom
from vp_1103_p1 import complement, save_metrics
from vp_1103_references import reference_metrics


LAGS = np.arange(32, 261, 4)
WINDOW = 48
HANKEL_COLUMNS = 32


def window_sums(x):
    c = np.concatenate([np.zeros((*x.shape[:-1], 1), x.dtype), np.cumsum(x, axis=-1)], axis=-1)
    return c[..., WINDOW:] - c[..., :-WINDOW]


def lag_peak(x):
    e = window_sums(np.abs(x)**2).clip(0.)
    roof = e.max(axis=-1, keepdims=True)
    raw = np.zeros(len(x))
    valid = np.zeros(len(x))
    for lag in LAGS:
        n = len(x[0]) - lag - WINDOW + 1
        corr = window_sums(x[:, :-lag] * x[:, lag:].conj())
        left, right = e[:, :n], e[:, lag:lag + n]
        normalized = np.abs(corr) / (np.sqrt(left * right) + np.maximum(roof, 1e-30) * 1e-12)
        normalized = np.clip(normalized, 0., 1.)
        raw = np.maximum(raw, normalized.max(axis=-1))
        mask = (left >= .25 * roof) & (right >= .25 * roof)
        valid = np.maximum(valid, np.where(mask, normalized, 0.).max(axis=-1))
    return raw, valid


def hankel_ratio(x):
    rotated = x * np.exp(-1j * np.pi / 1200 * SAMPLES**2)
    rows = np.lib.stride_tricks.sliding_window_view(rotated, HANKEL_COLUMNS).copy()
    gram = rows.conj().T @ rows
    eigen = np.linalg.eigvalsh(gram)
    return float(np.sqrt(max(0., eigen[-2]) / max(eigen[-1], 1e-30)))


def overlap(a, b, mask):
    edges = np.linspace(0., 1., 41)
    original = []
    for values in (a, b):
        rows = []
        for v, keep in zip(values, mask, strict=True):
            if keep.any():
                hist = np.histogram(v[keep], edges)[0]
                rows.append(hist / hist.sum())
        original.append(np.asarray(rows))
    x, y = original
    if not len(x):
        return {"parents": 0, "overlap": None, "parent_bootstrap_ci95": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), 2000) / len(x)
    boot = np.minimum(weights @ x, weights @ y).sum(axis=1)
    return {"parents": len(x), "overlap": float(np.minimum(x.mean(axis=0), y.mean(axis=0)).sum()),
            "parent_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist()}


def correlation(feature, fa, mask):
    any_family = mask.any(axis=1)
    x, y = parent_mean(feature, mask), parent_mean(fa, mask)
    count = int(any_family.sum())
    if len(x) != count or len(y) != count or count < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"parents": count, "rho": None, "parent_bootstrap_ci95": None}
    result = spearmanr(x, y)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for idx in rng.integers(count, size=(1200, count)):
        if np.ptp(x[idx]) and np.ptp(y[idx]):
            values.append(spearmanr(x[idx], y[idx]).statistic)
    return {"parents": count, "rho": float(result.statistic),
            "nominal_two_sided_p": float(result.pvalue),
            "parent_bootstrap_ci95": np.quantile(values, [.025, .975]).tolist()}


def chi_by_period(dev, grid):
    chi = np.full(grid.shape, np.nan)
    periods = chi.copy()
    for i, delay in enumerate(dev["target_delay_samples"]):
        doppler = dev["doppler_hz"][i]
        a = lfm_atom(delay, doppler)
        a /= np.linalg.norm(a)
        for family, pulse in itertools.product(range(3), range(8)):
            geo = grid[i, family, pulse]
            if geo is None:
                continue
            b = replay_atom(delay, doppler, family, geo)
            chi[i, family, pulse] = np.clip(1 - abs(np.vdot(a, b))**2 / np.vdot(b, b).real, 0, 1)
            periods[i, family, pulse] = geo["period_samples"]
    output = {}
    for name, mask in strata(frame_jnr(dev)):
        for label, period_mask in [
            ("all", np.ones_like(periods, bool)), ("36–95", periods < 96),
            ("96–159", (periods >= 96) & (periods < 160)), ("160–260", periods >= 160)
        ]:
            filtered = np.where(mask[..., None] & period_mask, chi, np.nan).reshape(256, -1)
            selected = np.isfinite(filtered).any(axis=1)
            vals = np.nanmin(filtered[selected], axis=1)
            output[f"{name}/{label}"] = {
                "min_chi_per_parent": distribution(vals),
                "fraction_of_parents_below_0.1": distribution((vals < .1).astype(float))}
    conflicts = []
    for parent, family, pulse in zip(*np.where(chi < .1)):
        conflicts.append({"parent_id": int(dev["parent_id"][parent]), "family": FAMILIES[family],
                          "pulse": int(pulse), "period_samples": int(periods[parent, family, pulse]),
                          "chi": float(chi[parent, family, pulse])})
    return output, conflicts, np.min(np.where(np.isnan(chi), 1., chi), axis=-1)


def candidates(geometry):
    bounds = {"window_samples": (6, 72, 2), "period_samples": (36, 260, 4),
              "sampling_phase_samples": (0, 260, 2), "forward_delay_samples": (1, 96, 2),
              "repeat_count": (1, 10, 1), "replay_gap_samples": (0, 64, 3),
              "frequency_offset_hz": (-200_000, 200_000, 25_000)}
    yield geometry
    moves = []
    for name, (lo, hi, step) in bounds.items():
        for k in (-4, -2, -1, 1, 2, 4):
            value = np.clip(geometry[name] + step * k, lo, hi).item()
            moves.append((name, value))
    pairs = [(("forward_delay_samples", a), ("sampling_phase_samples", b))
             for a, b in itertools.product((-8, -4, 4, 8), repeat=2)]
    proposals = [{**geometry, name: val} for name, val in moves]
    proposals += [{**geometry, **{name: geometry[name] + movement for name, movement in pair}}
                  for pair in pairs]
    for proposal in proposals:
        if not all(lo <= proposal[name] <= hi for name, (lo, hi, _) in bounds.items()):
            continue
        if proposal["period_samples"] <= proposal["window_samples"]:
            continue
        if proposal["sampling_phase_samples"] > proposal["period_samples"] - proposal["window_samples"]:
            continue
        yield proposal


def profile(y, delay, doppler, family, geometry):
    a = lfm_atom(delay, doppler)[:, None]
    a /= np.linalg.norm(a)
    projected = complement(a, y)
    # Estimate the noise scale from observation samples. The score is a local
    # complex-Gaussian profile likelihood with one complex amplitude per B.
    noise = max(np.median(np.abs(projected)**2) / np.log(2), 1e-24)
    atoms = []
    for proposal in candidates(geometry):
        atom = replay_atom(delay, doppler, family, proposal)
        norm = np.linalg.norm(atom)
        if norm < 1e-14:
            continue
        atom /= norm
        # Collinear templates represent the same possible fitted waveform.
        if any(abs(np.vdot(old, atom)) >= 1 - 1e-10 for old in atoms):
            continue
        atoms.append(atom)
    b = np.column_stack(atoms)
    qb = complement(a, b)
    denominator = np.sum(np.abs(qb)**2, axis=0)
    beta = qb.conj().T @ projected / denominator.clip(1e-20)
    likelihood = np.abs(beta)**2 * denominator / noise
    k = int(np.argmax(likelihood))
    loss = likelihood[k] - likelihood
    waveforms = b * beta
    relative = np.linalg.norm(waveforms - waveforms[:, k, None], axis=0) / max(abs(beta[k]), 1e-20)
    possible = (loss <= 2.) & (relative > .1)
    return {"distinct_templates": len(atoms), "best_loglikelihood_gain": float(likelihood[k]),
            "close_distinct_waveforms": int(possible.sum()), "resolved_signal": bool(likelihood[k] >= 10.),
            "multimodal": bool(likelihood[k] >= 10. and possible.any())}


def profile_subset(dev, grid, y, selected_pulses, jnr):
    # Sample 32 parents from each tertile of peak frame JNR using one fixed RNG.
    rng = np.random.default_rng(20261123)
    roster = np.sort(np.concatenate([
        rng.choice(group, size=32, replace=False)
        for group in np.array_split(np.argsort(jnr.max(axis=1)), 3)]))
    rows = {}
    columns = {name: np.full((256, 3), np.nan) for name in (
        "multimodal", "resolved_signal", "close_distinct_waveforms", "best_loglikelihood_gain")}
    for parent in roster:
        parent_rows = {}
        for family, name in enumerate(FAMILIES):
            pulse = selected_pulses[parent, family]
            row = profile(y[parent, family], dev["target_delay_samples"][parent], dev["doppler_hz"][parent],
                          family, grid[parent, family, pulse])
            for key, values in columns.items():
                values[parent, family] = row[key]
            parent_rows[name] = row
        rows[str(dev["parent_id"][parent])] = parent_rows
    hi_cases = jnr[roster] >= 30
    resolved = columns["resolved_signal"][roster] == 1.
    active = (hi_cases & resolved).any(axis=1)
    conflicts = ((columns["multimodal"][roster] == 1.) & hi_cases & resolved).any(axis=1)
    return {"parent_count": len(roster), "sample_seed": 20261123,
            "strata": stratified(columns, jnr), "parent_results": rows,
            "high_risk_parent_conflict_rate": distribution(conflicts[active].astype(float)),
            "high_risk_parent_power": distribution(active[hi_cases.any(axis=1)].astype(float)),
            "limit": "Local, truth-anchored finite grid within the known replay family. This cannot rule out remote modes."}


def audit(dev, grid, private, update):
    ids = parent_roster(dev, 256)
    jnr = frame_jnr(dev)
    update(experiment="principal_angles")
    chi, conflicts, chi_min = chi_by_period(dev, grid)
    pulse_jnr = dev["jsr_db"] + dev["snr_db"][:, None]
    selected = np.nanargmax(pulse_jnr, axis=-1)
    gather = selected[..., None, None]
    s, j, e = [np.take_along_axis(dev[name], gather, axis=2).squeeze(2).astype(np.complex128)
               for name in ("s", "j", "epsilon")]
    y = s + j + e
    update(experiment="lag_similarity")
    results = {"min_chi_jammed_pulse": chi_min}
    for key, x in [("S", s), ("J", j), ("E", e), ("Y", y)]:
        raw, active = lag_peak(x.reshape(-1, 1800))
        results[f"peak_{key}_raw"] = raw.reshape(-1, 3)
        results[f"peak_{key}_active_windows"] = active.reshape(-1, 3)
    for mode in ("raw", "active_windows"):
        delta = results[f"peak_J_{mode}"] - results[f"peak_S_{mode}"]
        results[f"lag_J_minus_S_{mode}"] = delta
        results[f"lag_J_exceeds_S_{mode}"] = delta > 0
    references, binding = reference_metrics("unet_tail", ids)
    fa = np.mean([row["frame_pfa_jammer"] for row in references.values()], axis=0)
    assoc = {}
    for name, mask in strata(jnr):
        assoc[name] = {}
        for key in ("peak_Y_active_windows", "peak_J_active_windows", "peak_Y_raw"):
            assoc[name][key] = correlation(results[key], fa, mask)
    update(experiment="profile_likelihood")
    likelihood = profile_subset(dev, grid, y, selected, jnr)
    update(experiment="hankel")
    for key, x in [("S", s), ("J", j), ("Y", y)]:
        ratio = np.array([hankel_ratio(row) for row in x.reshape(-1, 1800)]).reshape(-1, 3)
        results[f"hankel_{key}"] = ratio
    overlaps = {name: overlap(results["hankel_S"], results["hankel_J"], mask)
                for name, mask in strata(jnr)}
    stats = stratified(results, jnr)
    lines = ["# P1-C · 重复、歧义和谱结构", "",
             "每个 parent 与干扰类型用 JNR 最高的受扰脉冲做波形特征审计。",
             "子空间主角覆盖所有受扰脉冲。每个分层先取 parent 内最小 χ。",
             "长程相关扫描 32 至 260 点，步长 4，滑窗 48 点。主分析要求两个窗各自达到该波形最大窗能量的 25%。",
             "后验歧义审计抽取 96 个 parent。扫描使用真值附近的有限粗网格，合并共线模板。",
             "Hankel 采用 32 列嵌入和 40 个等宽直方图分箱。", "",
             "| 干扰 | J−S 滞后峰值 | 观测峰与虚警 Spearman ρ | min χ 的 P5/P10/P50/P90 | Hankel 重叠 |",
             "|---|---:|---:|---:|---:|"]
    for family in FAMILIES:
        key = f"{family}/all"
        diff = stats[key]["lag_J_minus_S_active_windows"]["mean"]
        rho = assoc[key]["peak_Y_active_windows"]["rho"]
        qs = "/".join(f"{i:.4f}" for i in chi[f"{key}/all"]["min_chi_per_parent"]["p05_p10_p50_p90_p95"][:4])
        lines.append(f"| {family} | {diff:.4f} | {rho:.4f} | {qs} | {overlaps[key]['overlap']:.4f} |")
    competition = likelihood["high_risk_parent_conflict_rate"]
    lines += ["", f"- χ < 0.1 的脉冲数为 {len(conflicts)}。",
              f"- 通过强干扰和可辨识筛选的 parent 共 {competition['parents']} 个，多峰比例 {competition['mean']:.2%}。",
              "- 虚警与峰值的相关性按 parent 计算，包含三个历史 U-Net+Tail seed。",
              "- 分层区间、96 个 parent 的受控扫描结果和小主角案例见 `SUMMARY.json`。",
              "", "## 门槛 3 与 4", "", "判定见总闸门报告。", ""]
    return {"stats": stats, "chi_by_period": chi, "chi_conflicts": conflicts,
            "correlations": assoc, "profile_likelihood": likelihood, "hankel_overlap": overlaps,
            "unet_tail_score_binding": binding, "private_metrics_sha256": save_metrics(private, "features", results),
            "training_seeds": [], "simulator_seed": 2026111302}, lines
