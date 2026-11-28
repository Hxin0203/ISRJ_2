"""Incrementally reconcile R3's paired mechanism results from verified chunks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import traceback
from pathlib import Path

import numpy as np

from tools.durable_io import atomic_json, atomic_write
from tools.stable_hash import sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/original_paper_replication_256_v1"
OUT = BASE / "R3_by_mechanism/postprocessing"
DATA = ROOT / "data/original_paper_replication_256_v1/R3_main"
MASTER = ROOT / "data/hybrid_condition_master_20261127"
SEEDS = (67, 97, 53, 41, 109)
SNRS = (-30, -25, -20, -15, -10, -5, 0)
JSRS = (0, 10, 20, 30, 35)
VARIANTS = ("ISDRJ", "ISRRJ", "ISCRJ")
PARENTS = 256
CHUNK = 16


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate_values(values, expected_ids, binding_sha):
    if str(values["binding_sha256"]) != binding_sha:
        raise RuntimeError("mechanism chunk binding mismatch")
    if not np.array_equal(values["parent_id"], expected_ids):
        raise RuntimeError("mechanism chunk parent mismatch")
    metrics = {key: values[key].copy() for key in values.files
               if key not in ("parent_id", "binding_sha256")}
    for name, value in metrics.items():
        if value.shape != (len(expected_ids), 3) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid per-mechanism metric {name}: {value.shape}")
    for alias, canonical in (("E_U", "u_nmse"), ("D_S", "damage"),
                             ("e_S", "damage"), ("E_0", "no_jammer_pooled_nmse"),
                             ("Hits", "pd")):
        if not np.array_equal(metrics[alias], metrics[canonical]):
            raise RuntimeError(f"mechanism metric alias mismatch {alias}")
    return metrics


def sources():
    return [("ReCoVP-Hybrid", seed, BASE / "R5_detection" / f"seed_{seed}") for seed in SEEDS] + [
        (method, None, BASE / "R3_by_mechanism" / method) for method in ("Max-TF", "GMM-OMP")]


def validate_data_cell(name, manifest, master, manifest_sha):
    audit_path = OUT / "data_audits" / f"{name}.json"
    if audit_path.exists():
        if read(audit_path)["manifest_sha256"] != manifest_sha:
            raise RuntimeError("R3 data manifest changed after reconciliation")
        return
    entry = manifest["cells"][name]
    path = Path(entry["path"])
    if sha(path) != entry["sha256"]:
        raise RuntimeError(f"R3 cell SHA mismatch {name}")
    with np.load(path, allow_pickle=False) as z:
        snr, jsr = entry["snr_db"], entry["jsr_db"]
        expected = {"s": master["s"][:PARENTS],
                    "j": master["j"][:PARENTS] * np.float32(10 ** (jsr / 20)),
                    "epsilon": master["epsilon"][:PARENTS] * np.float32(10 ** (-snr / 20)),
                    "parent_id": master["parent_id"][:PARENTS],
                    "doppler_hz": master["doppler_hz"][:PARENTS]}
        if any(not np.array_equal(z[key], value) for key, value in expected.items()):
            raise RuntimeError(f"R3 and R5 physical components differ {name}")
    atomic_json(audit_path, {"manifest_sha256": manifest_sha, "cell_sha256": entry["sha256"],
                             "r5_master_transform_identical": True, "verified_utc": now()})


def reconcile_once(context=None):
    from tools.server_gpu_admission import require_server_identity
    require_server_identity()
    if context is None:
        component = MASTER / "components/test_0_0.npz"
        manifest = read(DATA / "DATA_MANIFEST.json")
        registration = read(ROOT / "data/original_paper_replication_256_v1/R5_detection/REGISTRATION.json")
        digest = sha(component)
        if digest != manifest["master_component_sha256"] or digest != registration["source_master_sha256"]:
            raise RuntimeError("R3 and R5 master SHA mismatch")
        selection = read(ROOT / "reports/Original_paper_replication_256/SELECTED_SEEDS.json")
        if tuple(selection["selected_seeds"]) != SEEDS or selection["epoch"] != 256:
            raise RuntimeError("R3 frozen seed selection differs")
        with np.load(component, allow_pickle=False) as z:
            master = {key: z[key].copy() for key in ("s", "j", "epsilon", "parent_id", "doppler_hz")}
        context = manifest, master, sha(DATA / "DATA_MANIFEST.json"), selection
    manifest, master, manifest_sha, selection = context
    parent_ids = master["parent_id"][:PARENTS]
    records, coverage = [], {}
    for method, seed, folder in sources():
        task_id = f"{method}_{seed}" if seed is not None else method
        coverage[task_id] = 0
        binding_path = folder / "BINDING.json"
        if not binding_path.exists():
            continue
        binding, binding_sha = read(binding_path), sha(binding_path)
        if seed is not None:
            ranking = next(row for row in selection["ranking_rows"] if row["binding"]["seed"] == seed)
            if binding["checkpoint"]["checkpoint_sha256"] != ranking["binding"]["checkpoint_sha256"]:
                raise RuntimeError(f"R3 frozen endpoint mismatch {seed}")
        for snr in SNRS:
            for jsr in JSRS:
                name = f"snr_{snr}_jsr_{jsr}"
                row_path = OUT / "cells" / task_id / f"{name}.json"
                if row_path.exists():
                    row = read(row_path)
                    if row["binding_sha256"] != binding_sha or row["manifest_sha256"] != manifest_sha:
                        raise RuntimeError(f"reconciled row identity differs {row_path}")
                else:
                    paths = [folder / name / f"parents_{lo:04d}_{lo + CHUNK:04d}.npz"
                             for lo in range(0, PARENTS, CHUNK)]
                    if not all(path.exists() for path in paths):
                        continue
                    validate_data_cell(name, manifest, master, manifest_sha)
                    parts, chunks = [], []
                    for lo, path in zip(range(0, PARENTS, CHUNK), paths):
                        with np.load(path, allow_pickle=False) as z:
                            parts.append(validate_values(z, parent_ids[lo:lo + CHUNK], binding_sha))
                        chunks.append({"path": str(path), "sha256": sha(path)})
                    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
                    row = {"method": method, "seed": seed, "snr_db": snr, "jsr_db": jsr,
                           "physical_parents": PARENTS, "mechanisms": VARIANTS,
                           "parent_id_sha256": hashlib.sha256(parent_ids.astype("<i8").tobytes()).hexdigest(),
                           "binding_sha256": binding_sha, "manifest_sha256": manifest_sha,
                           "chunks": chunks, "verified_utc": now(),
                           "mean_by_mechanism": {key: value.mean(axis=0).tolist() for key, value in merged.items()},
                           "hits": merged["Hits"].sum(axis=0).astype(int).tolist(),
                           "false_cells": merged["FA_cells"].sum(axis=0).astype(int).tolist(),
                           "false_cell_denominator_per_mechanism": PARENTS * 9032,
                           "seed109_preflight_limitation": seed == 109}
                    atomic_json(row_path, row)
                coverage[task_id] += 1
                for index, mechanism in enumerate(VARIANTS):
                    for metric, values in row["mean_by_mechanism"].items():
                        records.append({"method": method, "seed": seed, "mechanism": mechanism,
                                        "snr_db": snr, "jsr_db": jsr,
                                        "snr_group": "extreme" if snr < -20 else "ordinary",
                                        "metric": metric, "mean": values[index], "physical_parents": PARENTS})
    stream = io.StringIO(newline="")
    fields = ("method", "seed", "mechanism", "snr_db", "jsr_db", "snr_group", "metric", "mean", "physical_parents")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    atomic_write(OUT / "R3_PER_SEED_BY_MECHANISM.csv", lambda handle: handle.write(stream.getvalue().encode("utf-8")))
    status = {"state": "RECONCILING", "updated_utc": now(), "coverage": coverage,
              "cells_per_instance": 35, "physical_parents": PARENTS,
              "hybrid_source": "R5 first 256 parent IDs, component equality audited against R3",
              "classical_source": "R3 independently recomputed per-mechanism chunks",
              "formal_cved256": "PENDING_SAME_DATA_TRAINING_AND_EVALUATION",
              "full_r3_completed": False, "seed109_preflight_limitation": "original bytes missing; recovery bridge retained"}
    if all(count == 35 for count in coverage.values()):
        status["state"] = "HYBRID_CLASSICAL_READY_FORMAL_CVED_PENDING"
    atomic_json(OUT / "STATUS.json", status)
    return context, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("once", "watch"))
    args = parser.parse_args()
    context = None
    while True:
        try:
            context, status = reconcile_once(context)
            if args.action == "once" or status["state"] == "HYBRID_CLASSICAL_READY_FORMAL_CVED_PENDING":
                return
        except Exception:
            atomic_json(OUT / "LAST_ERROR.json", {"at_utc": now(), "error": traceback.format_exc()})
            if args.action == "once":
                raise
            context = None
        time.sleep(60)


if __name__ == "__main__":
    main()
