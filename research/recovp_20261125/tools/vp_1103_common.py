"""Shared data and parent-level statistics for the 5723605 DEV study."""
import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
SNAPSHOT = WORK / "outputs" / "vp_1103_baseline"
BASELINE = "5723605"
HISTORICAL = Path("evidence/outputs/correction_mf_tail_20261117_v1/FOUR_CELL_DEV_SUMMARY.json")
DATA = WORK / "data" / "vp_1103"
DEV_HASH = "0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1"
TRAIN_HASH = "6148d097fecb0381e56227eb80cf1379e5f2750816ba65ce12f5457ee678f0fd"
SEEDS = (17, 29, 41)
FAMILIES = ("direct", "repeat", "cyclic")
BOOTSTRAP_SEED = 20261122


def prepare_snapshot():
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    tar = subprocess.check_output(
        ["git", "archive", BASELINE, "paper2", "tools", HISTORICAL.as_posix()], cwd=WORK)
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        archive.extractall(SNAPSHOT, filter="data")


if __name__ == "__main__":
    prepare_snapshot()
    raise SystemExit

sys.path.insert(0, str(SNAPSHOT))
from paper2.io import now, read, sha
from tools.durable_io import atomic_copy, atomic_json


def validate_role(role, source=None):
    if role not in ("train", "dev"):
        raise ValueError("The study accepts TRAIN and DEV roles only")
    expected, count = (TRAIN_HASH, 512) if role == "train" else (DEV_HASH, 256)
    path, receipt = (DATA / f"{role}{suffix}" for suffix in (".npz", ".receipt.json"))
    if source is not None:
        for destination in (path, receipt):
            if not destination.exists():
                atomic_copy(source / destination.name, destination)
    metadata = read(receipt)
    if (metadata["role"], metadata["count"], metadata["sha256"], sha(path)) != (
            role, count, expected, expected):
        raise ValueError(f"{role} archive or receipt differs from the historical study")
    return path, {"archive": str(path.relative_to(WORK)), "sha256": expected,
                  "receipt_sha256": sha(receipt), "parents": count}


def provenance(*scripts):
    paths = [*sorted((SNAPSHOT / "paper2").rglob("*.py")),
             *sorted((SNAPSHOT / "tools").rglob("*.py")), Path(__file__),
             *(Path(script) for script in scripts)]
    return {
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", BASELINE], cwd=WORK, text=True).strip(),
        "worktree_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WORK, text=True).strip(),
        "source_sha256": {path.relative_to(WORK).as_posix(): sha(path) for path in paths},
    }


def parent_roster(archive, count):
    ids = archive["parent_id"]
    expected = np.arange((2026111302 if count == 256 else 2026111301) * 10**6,
                         (2026111302 if count == 256 else 2026111301) * 10**6 + count)
    if not np.array_equal(ids, expected) or tuple(archive["jam_types"]) != FAMILIES:
        raise ValueError("unexpected parent roster")
    return ids


def frame_jnr(archive):
    # Historical active-support convention. Pool linear ratios over jammed
    # pulses before taking logs, because detection uses one eight-pulse frame.
    jsr = archive["jsr_db"]
    ratio = np.where(np.isfinite(jsr), 10 ** ((archive["snr_db"][:, None] + jsr) / 10), 0)
    return 10 * np.log10(ratio.sum(axis=-1) / np.isfinite(jsr).sum(axis=-1))


def parent_mean(values, mask):
    masked = np.where(mask & np.isfinite(values), values, np.nan)
    valid = np.isfinite(masked).sum(axis=1)
    means = np.divide(np.nansum(masked, axis=1), valid, where=valid > 0,
                      out=np.full(len(valid), np.nan))
    return means[np.isfinite(means)]


def distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"parents": 0, "mean": None, "parent_bootstrap_ci95": None}
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(values), size=(4000, len(values)))
    return {"parents": len(values), "mean": float(values.mean()),
            "parent_bootstrap_ci95": np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist(),
            "p05_p10_p50_p90_p95": np.quantile(values, [.05, .10, .50, .90, .95]).tolist()}


def strata(jnr):
    bands = {"all": np.ones_like(jnr, dtype=bool), "<=0": jnr <= 0,
             "5-25": (jnr >= 5) & (jnr <= 25), ">=30": jnr >= 30, ">=35": jnr >= 35,
             "0-5_exclusive": (jnr > 0) & (jnr < 5), "25-30_exclusive": (jnr > 25) & (jnr < 30)}
    yield "global", bands["all"]
    for label, band in bands.items():
        if label != "all":
            yield f"jnr/{label}", band
        for index, family in enumerate(FAMILIES):
            family_mask = np.arange(len(FAMILIES))[None, :] == index
            yield f"{family}/{label}", band & family_mask


def stratified(metrics, jnr):
    return {key: {name: distribution(parent_mean(values, mask))
                  for name, values in metrics.items()} for key, mask in strata(jnr)}


def worst_cases(metrics, ids, jnr):
    result = {}
    for name in ("frame_pfa_jammer", "damage"):
        values = metrics[name]
        secondary = metrics.get("cell_pfa_jammer", values).mean(axis=1)
        order = np.lexsort((ids, -secondary, -values.mean(axis=1)))[:5]
        result[name] = [
            {"parent_id": int(ids[i]), "score": float(values[i].mean()),
             "families": {family: {"score": float(values[i, j]), "jnr_db": float(jnr[i, j])}
                          for j, family in enumerate(FAMILIES)}} for i in order]
    return result


def write_cases(stream, cases):
    for name, rows in cases.items():
        stream.extend([f"### {name}", ""])
        for row in rows:
            details = ", ".join(
                f"{family} {item['score']:.6g} (JNR {item['jnr_db']:.2f} dB)"
                for family, item in row["families"].items())
            stream.append(f"- `{row['parent_id']}` {row['score']:.6g} · {details}")
        stream.append("")
