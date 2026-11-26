"""E1/E2: paired DEV256 statistics of frozen baselines against ReCoVP-Hybrid seed 17.

Read-only reuse of existing per-parent arrays (no inference, no training, no
CONFIRM/TEST). Every array is checked for parent-id alignment with the Hybrid
array before use. Unit of resampling is the physical parent (three jammer
variants already averaged inside each parent by the original evaluators).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = r"D:\code\Xin\ISRJ_222\outputs"
HYB = r"recovp_hybrid_seed17\seed_17\dev_epoch_256_parent_values.npz"
HYB_SHA = "b81774bf19c0ab8f833fc916637450ed43fbba42aa29466ccd2c5dba6b537bda"
SEEDS = (17, 29, 41)
METRICS = ("u_nmse", "damage", "no_jammer_pooled_nmse", "background_nmse", "amplitude",
           "sjrif_db", "false_peak_suppression_db", "pd", "frame_pfa_jammer",
           "cell_pfa_jammer", "frame_pfa_noise", "cell_pfa_noise")
LOWER_BETTER = {"u_nmse", "damage", "no_jammer_pooled_nmse", "background_nmse",
                "frame_pfa_jammer", "cell_pfa_jammer", "frame_pfa_noise", "cell_pfa_noise"}
# name -> (relative path template, key prefix or "" for bare keys)
FAMILIES = {
    "CorrectionNet": (r"paper2_256_epoch_8seed_20261114\seed_{s}\dev_epoch_256_v4.npz", "model"),
    "CorrectionNet-Ctrl": (r"correction_original_control_20261117_v1\arm\seed_{s}\dev_paired_epoch_256.npz", "study"),
    "CorrectionNet-Tail": (r"correction_mf_tail_20261117_v1\arm\seed_{s}\dev_paired_epoch_256.npz", "study"),
    "CN-NoDechirp": (r"unet_ablation_3seed_256_20261115\no_dechirp\seed_{s}\dev_epoch_256.npz", ""),
    "CN-NoRMS": (r"unet_ablation_3seed_256_20261115\no_rms\seed_{s}\dev_epoch_256.npz", ""),
    "CN-NoPC": (r"unet_ablation_3seed_256_20261115\no_pc\seed_{s}\dev_epoch_256.npz", ""),
    "CN-NoDamage": (r"unet_ablation_3seed_256_20261115\no_damage\seed_{s}\dev_epoch_256.npz", ""),
    "UNet": (r"unet_ablation_3seed_256_20261115\unet\seed_{s}\dev_epoch_256.npz", ""),
    "UNet-Tail": (r"unet_mf_tail_screen_20261117_v2\tail\seed_{s}\dev_paired_epoch_256.npz", "tail"),
    "UNet-NoiseNorm": (r"unet_noise_norm_screen_20261117_v1\norm\seed_{s}\dev_paired_epoch_256.npz", None),
    "UNet-Freq": (r"unet_frequency_screen_20261117_v1\factor\seed_{s}\dev_paired_epoch_256.npz", None),
    "UNet-Dilation": (r"unet_dilation_screen_20261117_v1\factor\seed_{s}\dev_paired_epoch_256.npz", None),
}
REPS = 4000


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pick(z, prefix, metric):
    if prefix is None:  # factor screens: the non-baseline, non-delta prefix
        prefix = sorted({k.split("__")[0] for k in z.files if "__" in k} - {"baseline", "delta"})[0]
    key = metric if prefix == "" else f"{prefix}__{metric}"
    return np.asarray(z[key], dtype=np.float64), prefix


def boot(diff, rng_seed):
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(len(diff), size=(REPS, len(diff)))
    lo, hi = np.quantile(diff[draws].mean(axis=1), [.025, .975])
    return float(lo), float(hi)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    hp = os.path.join(ROOT, HYB)
    if sha(hp) != HYB_SHA:
        raise RuntimeError("Hybrid parent array identity mismatch")
    h = np.load(hp)
    pid = h["parent_id"]
    hyb = {m: np.asarray(h[f"hybrid__{m}"], np.float64) for m in METRICS}
    st1 = {m: np.asarray(h[f"stage1__{m}"], np.float64) for m in METRICS}
    inputs = {HYB: HYB_SHA}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "parents": int(len(pid)),
              "reference": "ReCoVP-Hybrid seed17 epoch256 Final", "bootstrap_reps": REPS,
              "difference": "reference minus comparator, per parent; positive favours reference only for higher-is-better metrics",
              "families": {}}
    raw = {}
    for fi, (name, (tmpl, prefix)) in enumerate(FAMILIES.items()):
        per_seed, used_prefix = {}, None
        for s in SEEDS:
            p = os.path.join(ROOT, tmpl.format(s=s))
            z = np.load(p)
            if not np.array_equal(z["parent_id"], pid):
                raise RuntimeError(f"parent order mismatch: {p}")
            inputs[tmpl.format(s=s)] = sha(p)
            per_seed[s] = {}
            for m in METRICS:
                per_seed[s][m], used_prefix = pick(z, prefix, m)
        fam = {"prefix": used_prefix, "seed_mean": {}, "seed_sd": {}, "paired_vs_seed_average": {}, "paired_by_seed": {}}
        for mi, m in enumerate(METRICS):
            seed_means = np.array([per_seed[s][m].mean() for s in SEEDS])
            fam["seed_mean"][m] = float(seed_means.mean())
            fam["seed_sd"][m] = float(seed_means.std(ddof=1))
            avg = np.mean([per_seed[s][m] for s in SEEDS], axis=0)
            raw[f"{name}__{m}__seedavg"] = avg
            d = hyb[m] - avg
            better = (d < 0) if m in LOWER_BETTER else (d > 0)
            fam["paired_vs_seed_average"][m] = {
                "reference_mean": float(hyb[m].mean()), "comparator_mean": float(avg.mean()),
                "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                "ci95": boot(d, 340000 + 100 * fi + mi),
                "fraction_parents_reference_better": float(better.mean()),
                "fraction_ties": float((d == 0).mean())}
            fam["paired_by_seed"][m] = {}
            for s in SEEDS:
                ds = hyb[m] - per_seed[s][m]
                fam["paired_by_seed"][m][str(s)] = {"mean_diff": float(ds.mean()),
                                                     "ci95": boot(ds, 350000 + 1000 * s + 100 * fi + mi)}
        result["families"][name] = fam
    result["stage1_reference"] = {m: float(st1[m].mean()) for m in METRICS}
    result["input_sha256"] = inputs
    with open(os.path.join(out_dir, "PAIRED_DEV_STATS.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, "SEEDAVG_PARENT_VALUES.npz"), parent_id=pid, **raw)
    print("ok", len(result["families"]))


if __name__ == "__main__":
    main(sys.argv[1])
