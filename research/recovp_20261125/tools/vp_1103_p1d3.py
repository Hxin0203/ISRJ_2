"""Does the fixed Hankel spectrum separate the four noisy DEV states?"""
import numpy as np

from vp_1103_common import BOOTSTRAP_SEED, FAMILIES, stratified, strata
from vp_1103_p1 import save_metrics
from vp_1103_p1c import hankel_ratio, overlap

PAIRS = {
    "jammed_vs_unjammed": (("J+E", "S+J+E"), ("E", "S+E")),
    "jammer_vs_single_target": (("J+E",), ("S+E",)),
    "jammed_target_vs_single_target": (("S+J+E",), ("S+E",)),
    "jammer_vs_noise": (("J+E",), ("E",)),
}


def feature(x):
    # All eight pulses contribute; the low quartile lets a minority of
    # contaminated pulses contribute without using their simulator flags.
    return float(np.quantile([hankel_ratio(pulse) for pulse in x], .25))


def auc_clustered(positive, negative, mask, reps=2000):
    active = mask.any(axis=1)
    count = int(active.sum())
    if not count:
        return {"parents": 0, "auc": None, "parent_bootstrap_ci95": None}
    mask = mask[active]
    scores, parents, weights, labels = [], [], [], []
    for group, label in ((positive, True), (negative, False)):
        expand = np.repeat(mask, group.shape[1] // mask.shape[1], axis=1)
        ordinal, _ = np.where(expand)
        # Lower spectrum ratio is declared the positive direction.
        scores.extend(-group[active][expand])
        parents.extend(ordinal)
        weights.extend(1. / expand.sum(axis=1)[ordinal])
        labels.extend([label] * len(ordinal))
    order = np.argsort(scores)
    scores, parents, weights, labels = (np.asarray(field)[order] for field in (scores, parents, weights, labels))
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]

    def rank_sum(cluster_weights):
        mass = cluster_weights[..., parents] * weights
        p = np.add.reduceat(mass * labels, starts, axis=-1)
        n = np.add.reduceat(mass * ~labels, starts, axis=-1)
        return np.sum(p * (n.cumsum(axis=-1) - .5 * n), axis=-1) / (p.sum(axis=-1) * n.sum(axis=-1))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.multinomial(count, np.full(count, 1. / count), reps)
    boot = np.concatenate([rank_sum(block) for block in np.array_split(draws, 10)])
    value = float(rank_sum(np.ones(count)))
    lo, hi = np.quantile(boot, [.025, .975])
    return {"parents": count, "auc": value, "parent_bootstrap_ci95": [float(lo), float(hi)],
            "reverse_direction_auc": 1. - value}


def collect(dev, update):
    s, j, e = (dev[key] for key in ("s", "j", "epsilon"))
    values = {key: np.empty((256, 3)) for key in ("E", "S+E", "J+E", "S+J+E")}
    for i in range(256):
        if i % 16 == 0:
            update(parents_scored=i)
        sp, ep = (tensor[i, 0].astype(np.complex128) for tensor in (s, e))
        values["E"][i] = feature(ep)
        values["S+E"][i] = feature(sp + ep)
        for family in range(3):
            observed = j[i, family].astype(np.complex128) + ep
            values["J+E"][i, family] = feature(observed)
            values["S+J+E"][i, family] = feature(sp + observed)
    return values


def audit(dev, ids, jnr, private, update):
    values = collect(dev, update)
    update(parents_scored=256, method="cluster_bootstrap")
    stats = stratified(values, jnr)
    comparisons, worst = {}, {}
    for label, (positive, negative) in PAIRS.items():
        p = np.stack([values[key] for key in positive], axis=-1).reshape(256, -1)
        n = np.stack([values[key] for key in negative], axis=-1).reshape(256, -1)
        rows = {}
        for key, mask in strata(jnr):
            rows[key] = {"roc": auc_clustered(p, n, mask),
                         "histogram_overlap": overlap(p, n, np.repeat(mask, p.shape[1] // 3, axis=1))}
        comparisons[label] = rows
        margin = n.mean(axis=1) - p.mean(axis=1)
        order = np.argsort(margin, kind="stable")[:5]
        worst[label] = [
            {"parent_id": int(ids[i]), "positive_ordering_margin": float(margin[i]),
             "jnr_by_type": dict(zip(FAMILIES, map(float, jnr[i]))),
             "ratios": {name: dict(zip(FAMILIES, map(float, array[i]))) for name, array in values.items()}}
            for i in order]
    primary = comparisons["jammed_vs_unjammed"]
    useful = all(primary[key]["roc"]["parent_bootstrap_ci95"][0] >= .70
                 for key in ("global", "jnr/<=0"))
    lines = ["# P1-D3 · 含噪 Hankel 谱", "",
             "每帧 8 个脉冲都参与 32 列 Hankel 谱测量，取脉冲谱比的 25 分位数作为帧分数。",
             "ROC 以较低谱比作为含干扰方向。重叠采用 40 个等宽箱，区间按 parent 配对重采样。", "",
             "| 比较 | 分层 | ROC-AUC | 95% 区间 | 重叠 | parent 数 |",
             "|---|---|---:|---:|---:|---:|"]
    for label, rows in comparisons.items():
        for key in ("global", "jnr/<=0", "jnr/5-25", "jnr/>=30", "jnr/>=35"):
            roc, hist = rows[key]["roc"], rows[key]["histogram_overlap"]
            lo, hi = roc["parent_bootstrap_ci95"]
            lines.append(f"| {label} | {key} | {roc['auc']:.4f} | {lo:.4f}–{hi:.4f} | "
                         f"{hist['overlap']:.4f} | {roc['parents']} |")
    lines += ["", f"辅助特征筛查 **{'KEEP_AS_CANDIDATE' if useful else 'BACKGROUND_EVIDENCE_ONLY'}**。",
              "筛查要求整体及 ≤0 dB 的 ROC-AUC 区间下界均达到 0.70。", "",
              "## 含干扰排序最差的 5 个 parent", ""]
    for row in worst["jammed_vs_unjammed"]:
        ratios = ", ".join(f"{name} {np.mean(list(items.values())):.3f}"
                           for name, items in row["ratios"].items())
        lines.append(f"- `{row['parent_id']}`，无扰减受扰谱比 {row['positive_ordering_margin']:+.4f}，{ratios}。")
    return {"state_distributions": stats, "comparisons": comparisons, "worst_cases": worst,
            "auxiliary_feature": "KEEP_AS_CANDIDATE" if useful else "BACKGROUND_EVIDENCE_ONLY",
            "private_metrics_sha256": save_metrics(private, "hankel", values)}, lines
