"""Active SNR/JSR reporting contract; historical receipts remain immutable."""
import numpy as np
from vp_1103_common import WORK, DATA, FAMILIES, atomic_json, distribution, parent_mean, sha

CONFIG = WORK / "evidence/outputs/snr_jsr_protocol/CONFIG.json"
DEFINITION = ("SNR/JSR 优先使用规范 snr_db/jsr_db；生成器定义为有效目标支撑平均功率/噪声方差、"
              "有效干扰支撑平均功率/有效目标支撑平均功率的 10log10。帧值先对有效脉冲线性比取均值再转 dB；"
              "SNR 使用全部八脉冲，JSR 仅使用受扰脉冲。整段能量比因占空比不同不替代该口径。")
BINS = {"snr": [-float("inf"), 0, 10, float("inf")],
        "jsr": [-float("inf"), 0, 15, 30, float("inf")]}
LABELS = {"snr": ("<0", "0-10", ">=10"), "jsr": ("<0", "0-15", "15-30", ">=30")}

def frame_axes(data):
    snr = 10*np.log10(np.mean(10**(data["snr_db"].astype(float)/10), axis=-1))
    pulse = data["jsr_db"].astype(float)
    jsr = 10*np.log10(np.nansum(10**(pulse/10), axis=-1)/np.isfinite(pulse).sum(-1))
    return {"snr": np.broadcast_to(snr[:, None], jsr.shape), "jsr": jsr}

def masks(axes):
    all_cells = np.ones_like(axes["jsr"], bool)
    yield "global", all_cells
    for index, name in enumerate(FAMILIES):
        yield f"{name}/all", all_cells & (np.arange(3)[None] == index)
    for axis in BINS:
        for lo, hi, label in zip(BINS[axis][:-1], BINS[axis][1:], LABELS[axis]):
            band = (axes[axis] >= lo) & (axes[axis] < hi)
            yield f"{axis}/{label}", band
            for index, name in enumerate(FAMILIES):
                yield f"{name}/{axis}/{label}", band & (np.arange(3)[None] == index)
    threat = axes["jsr"] >= 15
    yield "jsr/>=15", threat
    for index, name in enumerate(FAMILIES):
        yield f"{name}/jsr/>=15", threat & (np.arange(3)[None] == index)

def summarize(metrics, axes):
    return {label: {key: distribution(parent_mean(values, mask)) for key, values in metrics.items()}
            for label, mask in masks(axes)}

def cases(metrics, ids, axes):
    result = {}
    for criterion in ("frame_pfa_jammer", "damage"):
        score = metrics[criterion].mean(1)
        order = np.lexsort((ids, -score))[:5]
        result[criterion] = [{"parent_id": int(ids[i]), "score": float(score[i]),
            "families": {f: {"score": float(metrics[criterion][i,j]),
                             "snr_db": float(axes["snr"][i,j]), "jsr_db": float(axes["jsr"][i,j])}
                         for j,f in enumerate(FAMILIES)}} for i in order]
    return result

def protocol():
    ranges = {}
    for role in ("train", "dev"):
        with np.load(DATA/f"{role}.npz", allow_pickle=False) as z:
            ranges[role] = {key: {"minimum": float(np.nanmin(z[key])), "maximum": float(np.nanmax(z[key]))}
                            for key in ("snr_db", "jsr_db")}
    payload = {"definition": DEFINITION, "ranges_metadata_only": ranges,
               "snr_edges": [None, 0, 10, None], "jsr_edges": [None, 0, 15, 30, None],
               "intervals": "left closed, right open; outer bins unbounded",
               "pooling_unit": "parent; families/pulses/states are not independent samples",
               "extractor_selection": "TRAIN only; DEV metadata range does not select extractors"}
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(CONFIG, payload)
    return {"path": CONFIG.relative_to(WORK).as_posix(), "sha256": sha(CONFIG)}

def crosscheck(data):
    s,j,e = (data[key] for key in ("s","j","epsilon"))
    ps = np.sum(abs(s)**2, -1)/np.count_nonzero(s, axis=-1)
    pj = np.sum(abs(j)**2, -1)/np.maximum(np.count_nonzero(j, axis=-1),1)
    valid = np.isfinite(data["jsr_db"])
    error = abs(10*np.log10(np.maximum(pj[valid],1e-30)/ps[valid])-data["jsr_db"][valid])
    # A realized finite Gaussian-noise record need not equal its generating variance.
    actual = 10*np.log10(ps/np.mean(abs(e)**2, -1))
    noise_error = (actual-data["snr_db"][:,None]).ravel()
    if error.max() > 1e-4 or abs(noise_error.mean()) > .03:
        raise ValueError("Metadata power convention cross-check failed")
    return {"jsr_active_power_max_abs_db_error": float(error.max()),
            "snr_noise_realization_error_mean_db": float(noise_error.mean()),
            "snr_noise_realization_error_p01_p99_db": np.quantile(noise_error,[.01,.99]).tolist()}
