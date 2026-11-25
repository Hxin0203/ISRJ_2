"""TRAIN/DEV mechanism measurements with variable K and parent grouping."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from vp_1103_p2_evaluate import iq_tensor
from vp_1103_p2_smoothvp import operator_apply, noise_gain

def mechanism_metrics(model, small):
    size = len(small["parent_id"])
    metrics = {key: np.empty((size, 3)) for key in (
        "blind_j_relative_residual", "learned_b_oracle_beta_relative_residual",
        "leak_over_target_energy", "drift_over_target_energy", "cross_term_over_target_energy",
        "removed_noise_gain_frobenius_squared", "matched_delay_change_samples",
        "raw_channel_delay_change_samples", "J_energy_outside_strong_atom_regions",
        "effective_rank_J_weighted", "anchor_fallback_count")}
    max_error, permutation_error = 0., 0.
    with torch.inference_mode():
        for parent in range(size):
            for family in range(3):
                s, j, e = [small[key][parent, family].astype(np.complex128) for key in ("s", "j", "epsilon")]
                y0, y1 = j + e, s + j + e
                x = iq_tensor(y0, "cpu")
                p0, p1 = model.factor(x), model.factor(iq_tensor(y1, "cpu"))
                # Fixed independent permutations are generated without using data.
                perm = torch.tensor(np.random.default_rng(20261127 + parent * 3 + family).permutation(model.k))
                switched = model.factor(x, anchors_override=p0["anchors"][:, perm])
                permutation_error = max(permutation_error,
                                        float((p0["correction"] - switched["correction"]).abs().max()))
                st, jt, x0, x1 = [torch.from_numpy(v) for v in (s, j, y0, y1)]
                c0 = operator_apply(p0, x0)
                c1 = operator_apply(p1, x1)
                leak = -operator_apply(p1, st)
                drift = c0 - operator_apply(p1, x0)
                lhs = x1 - c1 - (x0 - c0) - st
                max_error = max(max_error, float(torch.linalg.vector_norm(lhs - leak - drift) /
                                                 torch.linalg.vector_norm(x1)))
                normal = p0["gram"] + 1e-6 * torch.eye(model.k, dtype=torch.complex128)[None]
                beta = torch.linalg.solve(normal, p0["b"].mH @ jt[..., None])
                oracle = (p0["b"] @ beta)[..., 0]
                es, ej = st.abs().square().sum(), jt.abs().square().sum()
                weights = jt.abs().square().sum(-1) / ej
                costs = []
                for pulse in range(8):
                    distances = (p0["delay"][pulse, :, None] - p1["delay"][pulse, None]).abs().numpy()
                    a, b = linear_sum_assignment(distances)
                    costs.append(float(distances[a, b].mean()))
                density = p0["b"].abs().square()
                support = (density / density.amax(1, keepdim=True).clamp_min(1e-24) >= .1).any(-1)
                rank = float(model.k**2) / p0["gram"].abs().square().sum((-2, -1))
                values = {
                    "blind_j_relative_residual": (jt - c0).abs().square().sum() / ej,
                    "learned_b_oracle_beta_relative_residual": (jt - oracle).abs().square().sum() / ej,
                    "leak_over_target_energy": leak.abs().square().sum() / es,
                    "drift_over_target_energy": drift.abs().square().sum() / es,
                    "cross_term_over_target_energy": 2 * (leak.conj() * drift).real.sum() / es,
                    "removed_noise_gain_frobenius_squared": noise_gain(p0).sum(),
                    "matched_delay_change_samples": (torch.tensor(costs) * weights).sum(),
                    "raw_channel_delay_change_samples": ((p0["delay"] - p1["delay"]).abs().mean(-1) * weights).sum(),
                    "J_energy_outside_strong_atom_regions": (jt.abs().square() * ~support).sum() / ej,
                    "effective_rank_J_weighted": (rank * weights).sum(),
                    "anchor_fallback_count": p0["anchor_fallback_count"].float().mean()}
                for key, value in values.items():
                    metrics[key][parent, family] = float(value)
    if max_error > 1e-12:
        raise ValueError("Paired VP algebra identity failed")
    return metrics, max_error, permutation_error

