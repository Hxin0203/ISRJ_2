"""Replay dictionary sensitivity and observation-only drift controls."""
import itertools

import numpy as np
from scipy.special import expit, softmax

from vp_1103_common import WORK, stratified, worst_cases, write_cases
from vp_1103_geometry import FREQUENCY_FIELD, SAMPLES, replay_atom
from vp_1103_p1 import Projection, complement, observations, orthogonal_basis, save_metrics, score
from vp_1103_p1b import estimate_protection, estimate_replay, protection
from vp_1103_p1d import (FIELDS, SEED_GEOMETRY, SMOOTH, apply_operator, fit_operator,
                          noise_variance, precision)
from paper2.radar_eval import pulse_compress

WIDTHS = (1, 4, 8, 16, 32)
RIDGES = (1e-6, 1e-4, 1e-2)
DELTAS = (-4, -2, -1, 0, 1, 2, 4)
SLOTS, SLOT_WIDTH = 16, 112
EMPTY = np.empty((1800, 0), np.complex128)
PRIMARY = "replay_smooth"
KEYS = ("frame_pfa_jammer", "pd", "damage", "u_nmse", "no_jammer_pooled_nmse",
        "removed_noise_gain_frobenius_squared")


def energy(x):
    return np.sum(np.abs(x)**2, axis=(-2, -1))


def full_geometry(grid):
    # The original archive has no replay metadata for clean pulses. Complete
    # those slots with independently drawn geometry of the same simulator family.
    # The fit always sees one populated geometry grid and never receives a mask.
    rng = np.random.default_rng(SEED_GEOMETRY)
    result = []
    for row in grid.reshape(-1, 8):
        entries = []
        for saved in row:
            width = int(rng.integers(6, 73))
            period = int(rng.integers(max(36, width + 1), 261))
            default = {"window_samples": width, "period_samples": period,
                       "sampling_phase_samples": int(rng.integers(0, period - width + 1)),
                       "forward_delay_samples": int(rng.integers(1, 97)),
                       "repeat_count": int(rng.integers(1, 11)),
                       "replay_gap_samples": int(rng.integers(0, 65)),
                       FREQUENCY_FIELD: float(rng.uniform(-2e5, 2e5))}
            entries.append(default if saved is None else saved)
        result.append(entries)
    return result


def shift_geometry(item, offsets):
    other = {**item, **{key: item[key] + delta for key, delta in zip(FIELDS, offsets, strict=True)}}
    other["window_samples"] = max(2, other["window_samples"])
    other["period_samples"] = max(other["period_samples"], other["window_samples"] + 1)
    other["forward_delay_samples"] = max(0, other["forward_delay_samples"])
    return other


def offset_bank(size):
    starts = [(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
              (0, 0, -1), (0, 0, 1), (-1, -1, 0)]
    rest = sorted(itertools.product(range(-2, 3), repeat=3),
                  key=lambda x: (sum(abs(i) for i in x), x))
    starts.extend(item for item in rest if item not in starts)
    return starts[:size]


def anchored_bank(delay, fd, family, geometry, center, offsets):
    atoms = []
    for row in geometry:
        columns = [replay_atom(delay, fd, family, shift_geometry(row, tuple(map(sum, zip(center, local)))))
                   for local in offsets]
        b = np.column_stack(columns).astype(np.complex128, copy=False)
        norms = np.linalg.norm(b, axis=0)
        if np.any(norms <= 1e-14):
            raise ValueError("A completed pulse dictionary had an empty column")
        atoms.append(b / norms)
    return np.stack(atoms)


def operator_scores(dev, y, getter, diagnostics=False):
    restored = np.empty_like(y, dtype=np.complex64)
    noise_gain = np.zeros((len(y), 4, 8))
    s, j = [dev[key].reshape(-1, 8, 1800).astype(np.complex128) for key in ("s", "j")]
    terms = np.zeros((len(y), 3))
    error = 0.
    ghosts = np.zeros(len(y))
    control_restored = np.empty_like(y, dtype=np.complex64) if diagnostics else None
    chunk_size = 8
    for lo in range(0, len(y), chunk_size):
        pick = slice(lo, lo + chunk_size)
        batch = y[pick]
        correction = np.empty_like(batch)
        observation_operators = []
        for state in range(4):
            op, diagnostic = getter(pick, state, batch[:, state])
            if isinstance(op, tuple):
                correction[:, state] = apply_operator(op, batch[:, state])
                noise_gain[pick, state] = op[2]
            else:
                correction[:, state] = np.array([[op[i][p](x) for p, x in enumerate(group)]
                                                 for i, group in enumerate(batch[:, state])])
                noise_gain[pick, state] = [[h.noise_gain for h in group] for group in op]
            if state in (2, 3):
                observation_operators.append(op)
            if diagnostics and diagnostic is not None:
                base, locations, confidence = diagnostic
                control_restored[pick, state] = batch[:, state] - base
                if state == 2:
                    mf, _ = pulse_compress(j[pick])
                    for index in range(len(batch)):
                        for pulse in range(8):
                            loc = int(np.rint(locations[index, pulse]))
                            if loc < 0 or confidence[index, pulse] < .5:
                                continue
                            peak = np.max(np.abs(mf[index, pulse]))
                            if peak > 0 and abs(mf[index, pulse, loc]) >= .8 * peak:
                                ghosts[lo + index] = 1
            elif diagnostics:
                control_restored[pick, state] = batch[:, state] - correction[:, state]
        restored[pick] = batch - correction
        op0, op1 = observation_operators
        if isinstance(op1, tuple):
            leak = -apply_operator(op1, s[pick])
            drift = apply_operator(op0, batch[:, 2]) - apply_operator(op1, batch[:, 2])
        else:
            leak = -np.array([[op1[i][p](x) for p, x in enumerate(group)] for i, group in enumerate(s[pick])])
            drift = np.array([[op0[i][p](x) - op1[i][p](x) for p, x in enumerate(group)]
                              for i, group in enumerate(batch[:, 2])])
        delta = (batch[:, 3] - correction[:, 3]) - (batch[:, 2] - correction[:, 2]) - s[pick]
        error = max(error, float(np.max(np.sqrt(energy(delta - leak - drift)) /
                                        np.sqrt(energy(batch[:, 3])).clip(1e-24))))
        terms[pick, 0] = energy(leak)
        terms[pick, 1] = energy(drift)
        terms[pick, 2] = 2 * np.sum((leak.conj() * drift).real, axis=(-2, -1))
    if error > 1e-12:
        raise ValueError(f"Complex128 drift identity error {error:g}")
    metrics = score(dev, y, restored)
    metrics["removed_noise_gain_frobenius_squared"] = noise_gain[:, 2].sum(axis=1).reshape(256, 3)
    metrics["clean_removed_noise_gain_frobenius_squared"] = noise_gain[:, :2].sum(axis=2).mean(axis=1).reshape(256, 3)
    for column, key in enumerate(("leak_over_target_energy", "drift_over_target_energy", "cross_term_over_target_energy")):
        metrics[key] = (terms[:, column] / energy(s)).reshape(256, 3)
    metrics["ghost_protection_frame"] = ghosts.reshape(256, 3)
    control = score(dev, y, control_restored) if diagnostics else metrics
    metrics["protection_frame_pfa_delta_all"] = metrics["frame_pfa_jammer"] - control["frame_pfa_jammer"]
    return metrics, error


def dictionary_sweep(dev, y, geom, center, size, rule):
    offsets = offset_bank(size)
    delay, doppler = (dev[key] for key in ("target_delay_samples", "doppler_hz"))
    chunk = None

    def getter(pick, state, waveforms):
        nonlocal chunk
        if state == 0:
            chunk = np.stack([anchored_bank(delay[g // 3], doppler[g // 3], g % 3,
                                            geom[g], center, offsets) for g in range(pick.start, pick.stop)])
        return fit_operator(chunk, precision(chunk, waveforms, rule)), None
    return operator_scores(dev, y, getter)


def result(private, name, metrics, jnr, ids, error, info):
    return {**info, "metrics": stratified(metrics, jnr), "identity_max_relative_error": error,
            "private_metrics_sha256": save_metrics(private, name, metrics),
            "worst_cases": worst_cases(metrics, ids, jnr)}


def compare_lines(rows):
    labels = ("frame_pfa_jammer", "pd", "damage", "u_nmse", "no_jammer_pooled_nmse",
              "removed_noise_gain_frobenius_squared", "drift_over_target_energy")
    text = ["| 方法 | 虚警 | ≥30 dB 虚警 | ≥35 dB 虚警 | Pd | 损伤 | NMSE | 保持 NMSE | 噪声增益 | 漂移比 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in rows.items():
        groups = item["metrics"]
        all_values = [groups["global"][key]["mean"] for key in labels]
        all_values[1:1] = [groups[f"jnr/>={j}"]["frame_pfa_jammer"]["mean"] for j in (30, 35)]
        text.append(f"| {name} | " + " | ".join(f"{value:.6g}" for value in all_values) + " |")
    return text


def d1(dev, ids, jnr, grid, private, update):
    y = observations(dev)
    geom = full_geometry(grid)
    rows = {}
    specs = [("nominal", (0, 0, 0), 1, ridge) for ridge in (*RIDGES, "smooth")]
    for field, delta, ridge in itertools.product(range(3), DELTAS, RIDGES):
        if delta:
            vector = tuple(delta if i == field else 0 for i in range(3))
            specs.append((FIELDS[field], vector, 1, ridge))
    for width, rule in itertools.product(WIDTHS, (1e-4, "smooth")):
        specs.append(("neighborhood", (1, 1, 0), width, rule))
    for axis, center, width, rule in specs:
        name = f"{axis}_{'_'.join(str(x) for x in center)}_k{width}_{rule}"
        update(method=name, finished=len(rows), scheduled=len(specs))
        metrics, error = dictionary_sweep(dev, y, geom, center, width, rule)
        nonzero = sum(x != 0 for x in center)
        eligible = ((width == 1 and nonzero == 1 and max(map(abs, center)) <= 1) or
                    (4 <= width <= 16 and rule == "smooth"))
        rows[name] = result(private, name, metrics, jnr, ids, error,
                            {"axis": axis, "center": list(center), "k": width, "rule": rule,
                             "gate_eligible": eligible})
        # Durable cell results allow inspection without interpreting interim metrics as a gate.
        from vp_1103_common import atomic_json
        atomic_json(private / "CELLS.json", rows)
    lines = ["# P1-D1 · 无保护重放字典敏感度", "",
             "每个脉冲和四种状态都执行求解。带重放的几何用于受控误差实验，干净脉冲由同分布非空几何补齐，求解器无受扰开关。",
             "字典包含仿真重放源的时频参考，属于条件定位敏感度试验。求解不使用目标保护空间或分离波形。",
             "起始延迟定义为切片转发起点的延迟。邻域扫描以延迟和宽度各偏移 +1 点为中心；名义零误差配置只做对照。",
             "线性代数使用 complex128，同噪声写回转为 complex64。", "", *compare_lines(rows), ""]
    example = next(item for item in rows.values() if item["rule"] == "smooth" and item["k"] == 8)
    lines += ["## 平滑 K=8 对照的最差 parent", "",
              "虚警与损伤横向对照三种干扰和 JNR，具体排名保留在各配置收据。", ""]
    write_cases(lines, example["worst_cases"])
    lines += ["## Gate D1", "", "待总控脚本判定。"]
    return {"methods": rows, "precision_rule": SMOOTH, "clean_geometry_fill_seed": SEED_GEOMETRY}, lines


def slot_bank(y, hard=False):
    """Localized chirplet proxy in fixed slots, with a smooth carrier mixture."""
    idx = np.arange(SLOTS * SLOT_WIDTH).reshape(SLOTS, SLOT_WIDTH)
    dechirp = np.exp(-1j * np.pi / 1200 * idx**2)
    centered = np.arange(SLOT_WIDTH) - SLOT_WIDTH // 2
    spectrum = np.fft.fft(np.roll(y[..., idx] * dechirp, -SLOT_WIDTH // 2, axis=-1), axis=-1)
    matched = np.abs(spectrum)**2 / (SLOT_WIDTH * noise_variance(y)[..., None, None])
    if hard:
        weight = np.eye(SLOT_WIDTH)[matched.argmax(axis=-1)]
    else:
        # Logarithmic score compression avoids a numerically hard switch.
        weight = softmax(2.5 * np.log1p(matched), axis=-1)
    carriers = np.fft.fftshift(np.fft.ifft(weight, axis=-1), axes=-1) * SLOT_WIDTH
    window = .1 + .9 * np.cos(np.pi * centered / SLOT_WIDTH)**2
    waves = carriers * window / dechirp
    waves /= np.linalg.norm(waves, axis=-1, keepdims=True)
    b = np.zeros((*y.shape, SLOTS), np.complex128)
    b[..., idx, np.arange(SLOTS)[:, None]] = waves
    return b


def hard_target_first(pick, state, waveform):
    operators, locations, confidence = [], [], []
    counterfactual = np.empty_like(waveform)
    for i, group in enumerate(waveform):
        row, spots, confidences = [], [], []
        for p, x in enumerate(group):
            a, loc = estimate_protection(x)
            b = estimate_replay(complement(orthogonal_basis(a), x))
            row.append(Projection(a, b, "adaptive"))
            counterfactual[i, p] = Projection(EMPTY, b, "adaptive")(x)
            spots.append(loc)
            confidences.append(float(loc >= 0))
        operators.append(row)
        locations.append(spots)
        confidence.append(confidences)
    return operators, (counterfactual, np.asarray(locations), np.asarray(confidence))


def soft_protection(residual):
    mf, _ = pulse_compress(residual)
    power = np.abs(mf[..., 64:1137])**2
    ratio = power / (power.mean(axis=-1, keepdims=True) + 1e-24)
    weights = softmax(3. * np.log1p(ratio), axis=-1)
    location = weights @ np.arange(64, 1137)
    confidence = .98 * expit(.5 * (np.sum(weights * ratio, axis=-1) - 14.))
    q = np.stack([orthogonal_basis(protection(value, 0., "tangent"))
                  for value in location.flat]).reshape(*location.shape, 1800, 3)
    return q, location, confidence


def replay_first(method):
    def get(pick, state, y):
        b = slot_bank(y, hard=method == "replay_hard")
        rule = "hard" if method in ("replay_hard", "smooth_bank_hard_precision") else "smooth"
        ridge = precision(b, y, rule)
        first = fit_operator(b, ridge)
        if method != "replay_guard":
            return first, None
        base = apply_operator(first, y)
        q, location, confidence = soft_protection(y - base)
        # Slot atoms have disjoint support and unit norms, so B itself is an
        # orthonormal basis for the exact principal-angle test.
        qhb = np.swapaxes(q.conj(), -1, -2) @ b
        chi = np.clip(1 - np.linalg.svd(qhb, compute_uv=False)[..., 0]**2, 0., 1.)
        adaptive = 1e-4 + .001 * (1 - chi)**2 / (chi + .001)
        return fit_operator(b, ridge + adaptive[..., None], q, confidence), (base, location, confidence)
    return get


def d2(dev, ids, jnr, grid, private, update):
    y = observations(dev)
    methods = ("target_first_hard", "replay_hard", "smooth_bank_hard_precision", PRIMARY, "replay_guard")
    rows = {}
    for name in methods:
        update(method=name)
        get = hard_target_first if name == "target_first_hard" else replay_first(name)
        metrics, error = operator_scores(dev, y, get, diagnostics=name in ("target_first_hard", "replay_guard"))
        if name in (PRIMARY, "replay_guard"):
            # Reuse the smooth B design to measure its low-χ support independently
            # of the truth target. Both models share this background geometry.
            chi = []
            for lo in range(0, len(y), 8):
                x = y[lo:lo + 8, 2]
                b = slot_bank(x)
                op = fit_operator(b, precision(b, x, "smooth"))
                q, _, _ = soft_protection(x - apply_operator(op, x))
                sigma = np.linalg.svd(np.swapaxes(q.conj(), -1, -2) @ b, compute_uv=False)[..., 0]
                chi.append((1 - sigma**2).clip(0., 1.).min(axis=1))
            metrics["min_blind_chi_background"] = np.concatenate(chi).reshape(256, 3)
        rows[name] = result(private, name, metrics, jnr, ids, error, {})
    geom = full_geometry(grid)
    for rule in ("hard", "smooth"):
        name = f"controlled_neighbor_k8_{rule}"
        update(method=name)
        metrics, error = dictionary_sweep(dev, y, geom, (1, 1, 0), 8, rule)
        rows[name] = result(private, name, metrics, jnr, ids, error, {"conditional_geometry": True})
    lines = ["# P1-D2 · 重放优先与连续精度代理", "",
             "前五组各自从观测重估。连续组使用 16 个固定时窗和连续谱软混合。额外硬精度组共享连续字典，用于隔离精度切换。",
             "精修组在粗残差上用连续加权位置和置信度形成切空间。两组受控邻域对照共享几何锚点，只改变精度切换方式。",
             "恒等式在 complex128 写回前独立核验。计分用 complex64 恢复波形。", "",
             *compare_lines(rows), "", "## 泄漏、漂移与伪峰", "",
             "| 方法 | 泄漏能量比 | 漂移能量比 | 交叉项 | 误保护帧 | 保护引起的虚警差 | 恒等式误差 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in rows.items():
        values = [row["metrics"]["global"][key]["mean"] for key in
                  ("leak_over_target_energy", "drift_over_target_energy", "cross_term_over_target_energy",
                   "ghost_protection_frame", "protection_frame_pfa_delta_all")]
        lines.append(f"| {name} | " + " | ".join(f"{x:.6g}" for x in values) +
                     f" | {row['identity_max_relative_error']:.3g} |")
    lines += ["", "## 纯平滑重放的最差 parent", ""]
    write_cases(lines, rows[PRIMARY]["worst_cases"])
    lines += ["## Gate D2 与 D3", "", "待总控脚本判定。"]
    return {"methods": rows, "primary": PRIMARY, "precision_rule": SMOOTH,
            "slot_proxy": {"windows": SLOTS, "width": SLOT_WIDTH}}, lines
