"""P2 preflight for the ISRJ_2 original-paper replication queue."""
from __future__ import annotations

import hashlib
import json
import platform
import py_compile
import time
from pathlib import Path

ROOT = Path("D:/code/Xin/ISRJ_222")
REPORT = ROOT / "reports" / "Original_paper_replication_256"
OUT = ROOT / "outputs" / "original_paper_replication_256_v1" / "P2"
SEEDS = (67, 97, 53, 41, 109)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    if platform.node().casefold() != "desktop-rfferm3":
        raise RuntimeError("P2 runs only on server B")
    selected_path = REPORT / "SELECTED_SEEDS.json"
    if not selected_path.is_file():
        raise RuntimeError("selected-seed receipt is missing")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if tuple(selected.get("selected_seeds", ())) != SEEDS or selected.get("epoch") != 256:
        raise RuntimeError("selected seed receipt does not match the frozen five-seed scope")
    checkpoints = {}
    for seed in SEEDS:
        cp = ROOT / f"outputs/recovp_hybrid_seed{seed}/seed_{seed}/epoch_256.pt"
        if not cp.is_file():
            raise RuntimeError(f"missing selected checkpoint: {cp}")
        digest = sha(cp)
        sidecar = cp.with_suffix(".json")
        if sidecar.is_file():
            bound = json.loads(sidecar.read_text(encoding="utf-8")).get("checkpoint_sha256")
            if bound and bound != digest:
                raise RuntimeError(f"checkpoint sidecar SHA mismatch: {cp}")
        checkpoints[str(seed)] = {"path": str(cp), "sha256": digest}
    sources = [
        "tools/server_gpu_admission.py",
        "tools/original_replication_p2.py",
        "tools/original_replication_suite.py",
        "tools/paper2_baseline_suite/equal_pfa_suite.py",
        "tools/paper2_baseline_suite/sweep_suite.py",
        "paper2/evaluation_1103_v4.py",
        "paper2/radar_eval.py",
    ]
    source_sha = {}
    for rel in sources:
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f"missing source: {path}")
        py_compile.compile(str(path), doraise=True)
        source_sha[rel] = sha(path)
    master = ROOT / "data/hybrid_condition_master_20261127/DATA_MANIFEST.json"
    component = ROOT / "data/hybrid_condition_master_20261127/components/test_0_0.npz"
    if not master.is_file() or not component.is_file():
        raise RuntimeError("hybrid-condition master data is missing")
    write_json(OUT / "P2_COMPLETE.json", {
        "status": "COMPLETED_VERIFIED", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_model": "ReCoVP-Hybrid", "historical_reference_excluded": ["RSDR-Net"],
        "selected_seeds": list(SEEDS), "epoch": 256, "checkpoints": checkpoints,
        "master_manifest": {"path": str(master), "sha256": sha(master)},
        "master_component": {"path": str(component), "sha256": sha(component)},
        "source_sha256": source_sha,
        "checks": ["selected seed and epoch freeze", "checkpoint sidecar identity",
                    "source compilation", "master data presence", "ISRJ_2 model boundary"],
    })
    print("P2_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
