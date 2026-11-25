"""DEV-only neural VP scoring, paired baselines, drift and private failures."""
import numpy as np
import torch

from vp_1103_common import (WORK, atomic_json, frame_jnr, parent_mean, parent_roster, read, sha,
                           stratified, worst_cases)
from vp_1103_p0 import evaluate
from vp_1103_p2_smoothvp import noise_gain, operator_apply
from vp_1103_references import reference_metrics
from tools.durable_io import atomic_write

OUT = WORK / "evidence/outputs/p2_smoothvp_screen"
PRIVATE = WORK / "outputs/p2_smoothvp_screen"
CORE = ("frame_pfa_jammer", "pd", "damage", "sjrif_db", "u_nmse", "no_jammer_pooled_nmse")


def iq_tensor(waveforms, device):
    return torch.from_numpy(np.stack((waveforms.real, waveforms.imag), axis=1).astype(np.float32)).to(device)


def m2_metrics(seed, coordinates):
    if coordinates == "DECHIRP":
        rows, evidence = reference_metrics("correction_tail", current_ids())
        return rows[seed], evidence[str(seed)]
    p0 = read(WORK / "evidence/outputs/p0_dechirp_tail_2x2/SUMMARY.json")
    row = p0["new_arms"]["P0-NoDechirp-Tail"][str(seed)]
    path = WORK / "outputs/p0_dechirp_tail_2x2/P0-NoDechirp-Tail" / str(seed) / "dev_parent_metrics.npz"
    if sha(path) != row["parent_metrics_sha256"]:
        raise ValueError("M2 DEV metrics hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z["parent_id"], current_ids()):
            raise ValueError("M2 DEV roster mismatch")
        metrics = {k: z[k] for k in CORE}
    return metrics, {"private_metrics_sha256": sha(path), "checkpoint_sha256": row["checkpoint_sha256"]}


def current_ids():
    from vp_1103_common import DATA
    with np.load(DATA / "dev.npz", allow_pickle=False) as z:
        return parent_roster(z, 256)


def drift(model, data):
    device = next(model.parameters()).device
    result = {key: np.empty((256, 3)) for key in (
        "leak_over_target_energy", "drift_over_target_energy", "cross_term_over_target_energy",
        "removed_noise_gain_frobenius_squared", "float32_correction_relative_error")}
    maximum_error = 0.
    with torch.inference_mode():
        for parent in range(256):
            for family in range(3):
                s, j, e = [data[k][parent, family].astype(np.complex128) for k in ("s", "j", "epsilon")]
                y0, y1 = j + e, s + j + e
                p0, p1 = [model.factor(iq_tensor(y, device)) for y in (y0, y1)]
                tensors = [torch.from_numpy(v).to(device) for v in (s, y0, y1)]
                st, x0, x1 = tensors
                leak = -operator_apply(p1, st)
                c0, c10, c1 = operator_apply(p0, x0), operator_apply(p1, x0), operator_apply(p1, x1)
                drift_term = c0 - c10
                lhs = (x1 - c1) - (x0 - c0) - st
                relative_error = float(torch.linalg.vector_norm(lhs - leak - drift_term) /
                                       torch.linalg.vector_norm(x1).clamp_min(1e-24))
                maximum_error = max(maximum_error, relative_error)
                es = st.abs().square().sum().clamp_min(1e-24)
                for key, value in (
                    ("leak_over_target_energy", leak.abs().square().sum() / es),
                    ("drift_over_target_energy", drift_term.abs().square().sum() / es),
                    ("cross_term_over_target_energy", 2 * (leak.conj() * drift_term).real.sum() / es),
                    ("removed_noise_gain_frobenius_squared", noise_gain(p1).sum()),
                    ("float32_correction_relative_error", torch.linalg.vector_norm(p1["correction"] - c1) /
                     torch.linalg.vector_norm(x1).clamp_min(1e-24))):
                    result[key][parent, family] = float(value)
    if maximum_error > 1e-12:
        raise ValueError(f"Neural operator decomposition inconsistent: {maximum_error}")
    return result, maximum_error


def failure_details(model, dev, cases, folder):
    ids = dev["parent_id"]
    selected = sorted({row["parent_id"] for rows in cases.values() for row in rows})
    path = folder / "failure_atoms.npz"
    payload = {"parent_id": np.asarray(selected, np.int64)}
    device = next(model.parameters()).device
    with torch.inference_mode():
        for parent_id in selected:
            i = int(np.searchsorted(ids, parent_id))
            s, j, e = [dev[key][i].astype(np.complex128) for key in ("s", "j", "epsilon")]
            for state, wave in (("background", j + e), ("target_present", s + j + e)):
                parts = model.factor(iq_tensor(wave.reshape(-1, 1800), device))
                for key in ("b", "delay", "precision", "rho", "mask"):
                    payload[f"{parent_id}_{state}_{key}"] = parts[key].cpu().numpy()
    atomic_write(path, lambda stream: np.savez_compressed(stream, **payload))
    return {"path": path.relative_to(WORK).as_posix(), "sha256": sha(path),
            "parent_ids": selected, "scope": "Private local diagnosis, arrays excluded from Git"}


def evaluate_seed(model, dev, seed, arm, coordinates, checkpoint):
    folder = PRIVATE / arm / str(seed)
    metrics = evaluate(model.eval(), dev)
    extra, error = drift(model, dev)
    metrics.update(extra)
    ids, jnr = parent_roster(dev, 256), frame_jnr(dev)
    target = folder / "dev_metrics.npz"
    atomic_write(target, lambda stream: np.savez_compressed(stream, parent_id=ids, **metrics))
    references, binding = reference_metrics("unet_tail", ids)
    m2, m2_binding = m2_metrics(seed, coordinates)
    comparisons = {}
    for label, baseline in (("unet_tail", references[seed]), ("m2", m2)):
        comparisons[label] = stratified({key: metrics[key] - baseline[key] for key in CORE}, jnr)
    baseline_pfa = references[seed]["frame_pfa_jammer"].mean()
    fa, damage, keep = [metrics[key].mean() for key in
                        ("frame_pfa_jammer", "damage", "no_jammer_pooled_nmse")]
    high = jnr >= 35.
    hi_diff = parent_mean(metrics["frame_pfa_jammer"] - references[seed]["frame_pfa_jammer"], high).mean()
    signal = ((fa <= baseline_pfa - .01 and damage <= .055) or
              (fa - baseline_pfa <= .005 and (damage <= .035 or hi_diff <= -.02)))
    breaker = fa > .14 or keep > 5e-4
    cases = worst_cases(metrics, ids, jnr)
    diagnosis = failure_details(model, dev, cases, folder)
    row = {"seed": seed, "epoch": 256, "arm": arm, "coordinates": coordinates,
           "checkpoint_sha256": sha(checkpoint), "private_metrics_sha256": sha(target),
           "metrics": stratified(metrics, jnr), "paired_comparisons": comparisons,
           "unet_tail_binding": binding[str(seed)], "m2_binding": m2_binding,
           "operator_identity_max_relative_error": error, "worst_cases": cases, "diagnosis": diagnosis,
           "continue_other_seeds": bool(signal and not breaker), "circuit_breaker": bool(breaker),
           "gate35_parent_averaged_delta": float(hi_diff),
           "scope": "Fixed epoch-256 DEV screen"}
    atomic_json(OUT / arm / f"seed_{seed}.json", row)
    return row
