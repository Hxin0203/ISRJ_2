"""Zero-training DEV audits for the complementary projection hypothesis."""
import argparse
import importlib
import os
import traceback
from pathlib import Path

import numpy as np

from vp_1103_common import (DATA, DEV_HASH, WORK, atomic_json, frame_jnr, now,
                           parent_roster, provenance, read, sha, stratified, validate_role,
                           worst_cases, write_cases)
from vp_1103_geometry import (FREQUENCY_FIELD, INTEGER_FIELDS, lfm_atom, recover_geometry,
                             replay_atom)
from paper2.evaluation_1103_v4 import group_metrics, radar_metrics, states_from_components
from paper2.radar_eval import pulse_compress
from tools.durable_io import ProcessLock, atomic_write

NAMES = {"a": "p1a_oracle_vp_audit", "b": "p1b_mismatch_drift_audit",
         "c": "p1c_structure_ambiguity_audit"}
FIXED = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
ADAPTIVE = (1e-4, 1e-3, 1e-3)


def regularize(chi, choice):
    if choice == "adaptive":
        base, slope, eps = ADAPTIVE
        return base + slope * (1 - chi)**2 / (chi + eps)
    return float(choice)


def unit_columns(x):
    norms = np.linalg.norm(x, axis=0)
    return x[:, norms > 1e-14] / norms[norms > 1e-14]


def orthogonal_basis(x):
    x = unit_columns(x)
    if not x.shape[1]:
        return x
    q, singular, _ = np.linalg.svd(x, full_matrices=False)
    return q[:, singular > singular[0] * 1e-10]


def complement(q, y):
    return y - q @ (q.conj().T @ y)


class Projection:
    """Low-rank removed-component operator. Input is observation only."""

    def __init__(self, a, b, choice):
        self.b = b = unit_columns(b)
        self.chi = None
        self.noise_gain = 0.
        if b.shape[1] == 0:
            self.right = b.conj().T
            return
        q = orthogonal_basis(a)
        qb = complement(q, b)
        ub = orthogonal_basis(b)
        self.chi = float(np.clip(np.linalg.eigvalsh(ub.conj().T @ complement(q, ub))[0], 0., 1.))
        ridge = regularize(self.chi, choice)
        gram = qb.conj().T @ qb
        normal = gram + ridge * np.eye(b.shape[1])
        self.right = np.linalg.solve(normal, qb.conj().T)
        self.noise_gain = float(np.trace((b.conj().T @ b) @ (self.right @ self.right.conj().T)).real)

    def __call__(self, y):
        return self.b @ (self.right @ y)


def geometry_grid(path, dev):
    binding = DATA / "geometry_binding.json"
    grid_path = DATA / "geometry.npz"
    fields = (*INTEGER_FIELDS, FREQUENCY_FIELD)
    if not grid_path.exists():
        grid = recover_geometry(path, dev)
        fit = np.full((*grid.shape, len(fields)), np.nan)
        for index in zip(*np.where(np.isfinite(dev["jsr_db"]))):
            fit[index] = [grid[index][field] for field in fields]
        atomic_write(grid_path, lambda stream: np.savez_compressed(stream, geometry=fit))
        atomic_json(binding, {**read(binding), "geometry_file_sha256": sha(grid_path)})
        return grid
    identity = read(binding)
    if identity["archive_sha256"] != DEV_HASH or identity["geometry_file_sha256"] != sha(grid_path):
        raise ValueError("local geometry cache identity mismatch")
    with np.load(grid_path, allow_pickle=False) as archive:
        geometry = archive["geometry"]
    grid = np.full(geometry.shape[:-1], None, dtype=object)
    for index in zip(*np.where(np.isfinite(dev["jsr_db"]))):
        values = geometry[index]
        grid[index] = {**dict(zip(INTEGER_FIELDS, map(int, values[:-1]))),
                       FREQUENCY_FIELD: float(values[-1])}
    return grid


def oracle_geometry(parent, family, delay, doppler, geometry):
    a = lfm_atom(delay, doppler)[:, None]
    bs = [np.empty((1800, 0), complex) if row is None else
          replay_atom(delay, doppler, family, row)[:, None] for row in geometry[parent, family]]
    return a, bs


def score(dev, y, restored):
    s, j, e = (dev[k].reshape(-1, 8, 1800) for k in ("s", "j", "epsilon"))
    results = {}
    for lo in range(0, len(s), 24):
        sl = slice(lo, lo + 24)
        out = restored[sl]
        values = group_metrics(s[sl], j[sl], e[sl], out)
        values.update(radar_metrics(s[sl], y[sl], out, np.repeat(dev["doppler_hz"], 3)[sl]))
        compressed, _ = pulse_compress(out[:, 2] - e[sl])
        noise, _ = pulse_compress(e[sl])
        values["mf_residual_peak_over_noise_rms"] = (
            np.max(np.abs(compressed), axis=(1, 2)) /
            np.sqrt(np.mean(np.abs(noise)**2, axis=(1, 2))))
        for name, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite score {name}")
            results.setdefault(name, []).append(value)
    return {key: np.concatenate(values).reshape(256, 3) for key, values in results.items()}


def observations(dev):
    return states_from_components(*(dev[name].reshape(-1, 8, 1800).astype(np.complex128)
                                    for name in ("s", "j", "epsilon")))


def save_metrics(private, method, metrics):
    target = private / f"{method}.npz"
    atomic_write(target, lambda stream: np.savez_compressed(stream, **metrics))
    return sha(target)


def audit_a(dev, grid, private, update):
    # Distinct input boundary: these are the only fields the solver receives
    # from DEV. y includes no separated components and no complex coefficients.
    y = observations(dev)
    delay, doppler = (dev[key] for key in ("target_delay_samples", "doppler_hz"))
    ids, jnr = parent_roster(dev, 256), frame_jnr(dev)
    banks = [oracle_geometry(i, f, delay[i], doppler[i], grid)
             for i in range(len(ids)) for f in range(3)]
    results = {}
    cases = {}
    contrasts = {}
    for choice in (*FIXED, "adaptive"):
        for protect in (True, False):
            name = f"{'protect' if protect else 'unprotected'}_{choice}"
            update(method=name)
            restored = y.copy()
            gains = np.zeros(len(banks))
            minimum = np.ones(len(banks))
            for index, (a, bs) in enumerate(banks):
                template = a if protect else a[:, :0]
                for pulse, b in enumerate(bs):
                    if b.shape[1]:
                        op = Projection(template, b, choice)
                        restored[index, 2:, pulse] -= op(y[index, 2:, pulse].T).T
                        gains[index] += op.noise_gain
                        minimum[index] = min(minimum[index], op.chi)
            metrics = score(dev, y, restored)
            metrics["removed_noise_gain_frobenius_squared"] = gains.reshape(-1, 3)
            metrics["min_chi_jammed_pulse"] = minimum.reshape(-1, 3)
            results[name] = {"metrics": stratified(metrics, jnr),
                             "private_metrics_sha256": save_metrics(private, name, metrics)}
            cases[name] = worst_cases(metrics, ids, jnr)
            if protect:
                protected = metrics
            else:
                contrasts[str(choice)] = stratified(
                    {key: protected[key] - metrics[key] for key in metrics}, jnr)
    metrics = score(dev, y, y)
    results["identity"] = {"metrics": stratified(metrics, jnr),
                           "private_metrics_sha256": save_metrics(private, "identity", metrics)}
    summary = {"training_seeds": [], "simulator_seed": 2026111302,
               "methods": results, "target_protection_minus_unprotected": contrasts,
               "worst_cases": cases, "adaptive_rule": dict(zip(("lambda0", "lambda1", "epsilon"), ADAPTIVE)),
               "primary_method": "protect_adaptive",
               "operator_notes": [
                   "One atom per jammed pulse because the frozen simulator couples replay segments through one complex gain.",
                   "The same oracle supports are used for J+E and S+J+E. E and S+E have empty B.",
                   "The fitter receives y and geometry only. The independently computed scoring stage uses references.",
                   "Frobenius gain is summed over pulse blocks of H for the eight-pulse group.",
               ]}
    lines = ["# P1-A · 已知几何的变量投影", "",
             "每组 8 脉冲和同一 parent 的三种干扰在统计前聚合。主规格为自适应正则的保护投影。",
             "求解阶段输入观测与几何。仿真波形分量只用于档案一致性核验及评分。", "",
             "| 方法 | 全局虚警 | ≥30 dB 虚警 | ≥35 dB 虚警 | Pd | 损伤 | 端点 NMSE | 残峰/噪声 RMS | 噪声增益 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in results.items():
        m = item["metrics"]
        v = [m[group][key]["mean"] for group, key in (
            ("global", "frame_pfa_jammer"), ("jnr/>=30", "frame_pfa_jammer"),
            ("jnr/>=35", "frame_pfa_jammer"), ("global", "pd"), ("global", "damage"), ("global", "u_nmse"),
            ("global", "mf_residual_peak_over_noise_rms"), ("global", "removed_noise_gain_frobenius_squared"))
             if key in m[group]]
        lines.append(f"| {name} | " + " | ".join(f"{x:.6g}" if x is not None else "NA" for x in v) + " |")
    lines.extend(["", "## 主规格最差 parent", "",
                  "虚警按命中干扰类型数和虚警单元密度排序。损伤按同背景配对误差排序。", ""])
    write_cases(lines, cases["protect_adaptive"])
    lines += ["## 门槛 1", "", "同批 U-Net+Tail 分层对照和判定见总闸门报告。", ""]
    return summary, lines


def run(stage):
    name = NAMES[stage]
    out = WORK / "evidence" / "outputs" / name
    private = WORK / "outputs" / name
    with ProcessLock(private / "PROCESS.lock"):
        status = {"stage": "STARTED", "started_utc": now(), "ended_utc": None, "pid": os.getpid(),
                  "seeds": {"simulator": 2026111302, "bootstrap": 20261122, "training": []},
                  "micro_batch": None}

        def update(**fields):
            status.update(fields, updated_utc=now())
            atomic_json(out / "STATUS.json", status)

        try:
            update()
            path, receipt = validate_role("dev")
            modules = [Path(__file__), Path(__file__).with_name("vp_1103_geometry.py")]
            entry = audit_a if stage == "a" else importlib.import_module(f"vp_1103_p1{stage}").audit
            if stage != "a":
                modules.append(Path(__file__).with_name(f"vp_1103_p1{stage}.py"))
            update(**provenance(*modules), data={"dev": receipt})
            with np.load(path, allow_pickle=False) as archive:
                dev = {name: archive[name] for name in archive.files}
            parent_roster(dev, 256)
            grid = geometry_grid(path, dev)
            update(geometry_sha256=sha(DATA / "geometry.npz"), stage="AUDIT_RUNNING")
            summary, lines = entry(dev, grid, private, update)
            summary.update(data_sha256=DEV_HASH,
                           frame_jnr_rule="10 log10 of mean linear active-support JNR over jammed pulses")
            atomic_json(out / "SUMMARY.json", summary)
            atomic_write(out / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
            update(stage="COMPLETE", ended_utc=now())
        except Exception as exc:
            atomic_json(private / f"error_{os.getpid()}.json", {"Traceback": traceback.format_exc()})
            update(stage="FAILED", ended_utc=now(), error=repr(exc))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=NAMES)
    run(parser.parse_args().stage)
