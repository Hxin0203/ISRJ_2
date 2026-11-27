"""WP8: zero-training structural diagnostics of frozen ReCoVP-Hybrid checkpoints on DEV256.

No training, no new data. From one Stage-1 forward pass the K=16 dictionary, Gram
matrix and learned per-atom precision are taken and the coefficient solve is
repeated with
  * the first k atoms only (k = 4, 8, 12, 16; anchors are ordered by greedy
    matched-filter peak selection, so these are the k strongest candidates), and
  * learned precision vs. near-zero precision (1e-6, i.e. plain projection).
Stage 2 is re-run on each modified Stage-1 residual. Also: gate forced to 1.

Variants
  stage1_K{k}_{learned,noreg}   Stage-1 output only (gate alpha = 0)
  final_K{k}_learned            full model with k atoms
  final_K16_noreg               full model, plain projection in Stage 1
  final_K16_alpha1              full model, scalar gate forced to 1
For stage1 variants, ``noise_nmse`` equals the fraction of energy removed from a
noise-only input, to be compared with the fixed-dictionary bound K/T.
Sanity: the stage1_K16_learned correction must reproduce the model's own one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("D:/code/Xin/ISRJ_222")
OUT = ROOT / "outputs" / "paper2_baseline_suite_20261126" / "WP8"
DEV = ROOT / "data" / "paper2_1103" / "dev.npz"
DEV_SHA = "0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1"
SEEDS_ALL = (17, 29, 41, 53, 67, 79, 97, 109)
KS = (4, 8, 12, 16)
T = 1800
MICRO = 192
CHUNK = 8
REPS = 4000
REPORT = ("no_jammer_pooled_nmse", "noise_nmse", "u_nmse", "damage", "amplitude", "sjrif_db",
          "false_peak_suppression_db", "pd", "frame_pfa_jammer", "frame_pfa_noise")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def variant_names() -> list[str]:
    names = [f"stage1_K{k}_{tag}" for k in KS for tag in ("learned", "noreg")]
    return names + [f"final_K{k}_learned" for k in KS] + ["final_K16_noreg", "final_K16_alpha1"]


def restore_variants(model, states: np.ndarray):
    import torch
    shape = states.shape
    flat = states.reshape(-1, T)
    outs = {n: np.empty(flat.shape, np.complex64) for n in variant_names()}
    alpha, check = [], []
    with torch.inference_mode():
        for lo in range(0, len(flat), MICRO):
            hi = min(lo + MICRO, len(flat))
            part = flat[lo:hi]
            iq = torch.from_numpy(np.stack((part.real, part.imag), 1).astype(np.float32)).cuda()
            y = torch.complex(iq[:, 0], iq[:, 1])
            one = model.stage1(iq)
            basis, gram, prec = one["dictionary"], one["gram"], one["precision"]
            x = one["normalized_input"].to(torch.complex128)
            scale = one["scale"]
            bx = basis.mH @ x[..., None]
            for k in KS:
                for reg in (True, False):
                    lam = prec[..., :k] if reg else torch.full_like(prec[..., :k], 1e-6)
                    g = gram[..., :k, :k]
                    normal = .5 * (g + g.mH) + torch.diag_embed(lam).to(torch.complex128)
                    coef = torch.linalg.solve(normal, bx[:, :k])
                    corr = (basis[..., :k] @ coef)[..., 0].to(torch.complex64) * scale
                    u = y - corr
                    tag = "learned" if reg else "noreg"
                    outs[f"stage1_K{k}_{tag}"][lo:hi] = u.cpu().numpy()
                    if reg or k == 16:
                        ref = model.stage2(u, corr, y, scale, model.stage1.matched_spectrum)
                        outs[f"final_K{k}_{tag}"][lo:hi] = (u - ref["delta"]).cpu().numpy()
                        if reg and k == 16:
                            raw = ref["delta"] / ref["alpha"]
                            outs["final_K16_alpha1"][lo:hi] = (u - raw).cpu().numpy()
                            alpha.append(ref["alpha"].cpu().numpy().ravel())
                            own = one["correction"]
                            check.append(float(((corr - own).abs().max() /
                                                own.abs().max().clamp_min(1e-30)).cpu()))
    return {n: v.reshape(shape) for n, v in outs.items()}, np.concatenate(alpha), max(check)


def run_seed(seed: int) -> None:
    folder = OUT / f"seed_{seed}"
    if (folder / "SUMMARY.json").is_file():
        print("skip", seed, flush=True)
        return
    from tools.paper2_baseline_suite import equal_pfa_suite as E
    from tools.paper2_baseline_suite.classical_suite import dev_metrics
    from paper2.evaluation_1103_v4 import states_from_components
    admission = E.require_server()
    E.configure_precision()
    import torch
    torch.set_num_threads(4)
    if sha(DEV) != DEV_SHA:
        raise RuntimeError("DEV identity changed")
    model, binding = E.load("ReCoVP-Hybrid", seed)
    params = model.parameter_report()
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(DEV, allow_pickle=False) as z:
        s, j, e = (z[k] for k in ("s", "j", "epsilon"))
        pid, fd = z["parent_id"].copy(), z["doppler_hz"].copy()
    t0 = time.perf_counter()
    worst = 0.0
    for lo in range(0, len(pid), CHUNK):
        part = folder / f"chunk_{lo:03d}.npz"
        if part.is_file():
            continue
        hi = min(lo + CHUNK, len(pid))
        states = states_from_components(s[lo:hi], j[lo:hi], e[lo:hi])
        outs, alpha, check = restore_variants(model, states)
        worst = max(worst, check)
        vals = {}
        for name, u in outs.items():
            for m, v in dev_metrics(s[lo:hi], j[lo:hi], e[lo:hi], u, fd[lo:hi]).items():
                vals[f"{name}__{m}"] = v
        tmp = folder / f"chunk_{lo:03d}.tmp.npz"
        np.savez(tmp, parent_id=pid[lo:hi], alpha=alpha, check=np.array([check]), **vals)
        os.replace(tmp, part)
        print(seed, "chunk", lo, round(time.perf_counter() - t0, 1), flush=True)
    chunks = [np.load(folder / f"chunk_{lo:03d}.npz") for lo in range(0, len(pid), CHUNK)]
    keys = [k for k in chunks[0].files if "__" in k]
    merged = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
    worst = max(float(c["check"][0]) for c in chunks)
    np.savez_compressed(folder / "PARENT_VALUES.npz", parent_id=pid, **merged)
    stored = ROOT / f"outputs/recovp_hybrid_seed{seed}/seed_{seed}/dev_epoch_256_parent_values.npz"
    cross = None
    if stored.is_file():
        with np.load(stored) as z:
            if np.array_equal(z["parent_id"], pid):
                cross = {m: {"stored": float(z[f"hybrid__{m}"].mean()),
                             "recomputed": float(merged[f"final_K16_learned__{m}"].mean())}
                         for m in ("u_nmse", "damage", "pd", "frame_pfa_jammer", "sjrif_db")}
    write_json(folder / "SUMMARY.json", {
        "seed": seed, "finished_utc": now(), "server": platform.node(), "gpu_admission": admission,
        **binding, "parameters": params, "dev_sha256": DEV_SHA, "parents": int(len(pid)),
        "seconds": time.perf_counter() - t0,
        "sanity_max_rel_diff_stage1_K16_vs_model_correction": worst,
        "cross_check_vs_stored_dev": cross,
        "means": {k: float(v.mean()) for k, v in merged.items()}})
    for c in folder.glob("chunk_*.npz"):
        c.unlink()


def aggregate() -> None:
    done = [s for s in SEEDS_ALL if (OUT / f"seed_{s}" / "SUMMARY.json").is_file()]
    if not done:
        return
    vals = [np.load(OUT / f"seed_{s}" / "PARENT_VALUES.npz") for s in done]
    ref = "final_K16_learned"
    table = {}
    for i, name in enumerate(variant_names()):
        row = {}
        for j, m in enumerate(REPORT):
            per_seed = np.array([v[f"{name}__{m}"].mean() for v in vals])
            row[m] = {"mean": float(per_seed.mean()),
                      "seed_sd": float(per_seed.std(ddof=1)) if len(done) > 1 else None}
            if name != ref:
                d = np.mean([v[f"{name}__{m}"] - v[f"{ref}__{m}"] for v in vals], axis=0)
                rng = np.random.default_rng(5100000 + 100 * i + j)
                idx = rng.integers(len(d), size=(REPS, len(d)))
                row[m]["diff_vs_full"] = float(d.mean())
                row[m]["diff_ci95"] = np.quantile(d[idx].mean(1), [.025, .975]).tolist()
        table[name] = row
    summaries = [json.loads((OUT / f"seed_{s}" / "SUMMARY.json").read_text(encoding="utf-8")) for s in done]
    write_json(OUT / "ZERO_TRAIN_ABLATION.json", {
        "created_utc": now(), "seeds": done, "reference": ref, "K_over_T": {str(k): k / T for k in KS},
        "variants": table, "parameters": summaries[0]["parameters"],
        "sanity_max_rel_diff": max(x["sanity_max_rel_diff_stage1_K16_vs_model_correction"] for x in summaries),
        "cross_checks": {str(x["seed"]): x["cross_check_vs_stored_dev"] for x in summaries}})
    lines = [f"# WP8 零训练结构诊断（DEV256，种子 {done}）", "",
             "| 变体 | E0（无干扰合并） | 噪声能量移除（noise_nmse） | K/T | E_U | D_S | 幅度 | SJRIF (dB) | Pd (%) | 受扰帧Pfa (%) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in table.items():
        k = int(name.split("_K")[1].split("_")[0])
        f = lambda m, p=False: (f"{100 * row[m]['mean']:.2f}" if p else f"{row[m]['mean']:.3g}")
        lines.append(f"| {name} | {f('no_jammer_pooled_nmse')} | {f('noise_nmse')} | {k / T:.4f} | {f('u_nmse')} | "
                     f"{f('damage')} | {f('amplitude')} | {row['sjrif_db']['mean']:.2f} | {f('pd', True)} | "
                     f"{f('frame_pfa_jammer', True)} |")
    (OUT / "ZERO_TRAIN_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def queue() -> None:
    """Run every completed seed, then poll for new ones (30 min) up to 96 h."""
    from tools.server_gpu_admission import NoGpuAvailable, choose_gpu
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        complete = [s for s in SEEDS_ALL
                    if (ROOT / f"outputs/recovp_hybrid_seed{s}/seed_{s}/TRAINING_COMPLETE.json").is_file()]
        todo = [s for s in complete if not (OUT / f"seed_{s}" / "SUMMARY.json").is_file()]
        write_json(OUT / "QUEUE_STATUS.json", {"updated_utc": now(), "complete": complete, "todo": todo})
        for s in todo:
            while True:
                try:
                    gpu = choose_gpu()["physical_gpu_index"]
                except NoGpuAvailable:
                    time.sleep(60)
                    continue
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                       "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
                log = OUT / f"seed_{s}.log"
                with open(log, "a", encoding="utf-8") as f:
                    code = subprocess.call([sys.executable, "-m", "tools.paper2_baseline_suite.zero_train_ablation",
                                            "one", "--seed", str(s)], cwd=str(ROOT), env=env,
                                           stdout=f, stderr=subprocess.STDOUT)
                if code == 3:
                    time.sleep(60)
                    continue
                if code != 0:
                    write_json(OUT / f"seed_{s}" / "FAILED.json", {"exit_code": code, "utc": now(), "log": str(log)})
                break
            aggregate()
        if len(complete) == len(SEEDS_ALL) or time.time() - start > 96 * 3600:
            write_json(OUT / "QUEUE_STATUS.json", {"updated_utc": now(), "stage": "QUEUE_DONE", "complete": complete})
            return
        time.sleep(1800)


def main() -> None:
    if platform.node().casefold() != "desktop-rfferm3":
        raise RuntimeError("server B only")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("one", "queue", "aggregate"))
    parser.add_argument("--seed", type=int)
    a = parser.parse_args()
    if a.action == "queue":
        queue()
    elif a.action == "aggregate":
        aggregate()
    else:
        from tools.server_gpu_admission import NoGpuAvailable
        try:
            run_seed(a.seed)
        except NoGpuAvailable as exc:
            print("gpu busy", exc, flush=True)
            sys.exit(3)


if __name__ == "__main__":
    main()
