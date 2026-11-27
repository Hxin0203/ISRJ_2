"""Resume a user-paused ReCoVP-Hybrid seed using its frozen run identity.

This entry point is deliberately separate from the registered training
sources. It verifies the original config, preflight, data, sources, and all
committed recovery checkpoints, then calls the frozen trainer without
regenerating or rewriting the preflight/config receipts.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

from paper2.io import ROOT, read, sha
from tools import server_gpu_admission


SERVER_HOST = "DESKTOP-RFFERM3"
SERVER_ROOT = Path("D:/code/Xin/ISRJ_222")
EXPECTED = {
    79: {
        "epoch": 75,
        "sha256": "1c215e815c5cc114cfe33e68f2e09fde29b99c988e45ebc9d73ce6d1eae067f8",
        "gpu": 1,
    },
    97: {
        "epoch": 139,
        "sha256": "9c71cc2b1ecf1b9121d493ed85beb452713643ff58eefff3472c0dae66ce11f9",
        "gpu": 0,
    },
}


def _verify(seed: int, gpu_index: int) -> tuple[dict, dict, dict]:
    if platform.node().casefold() != SERVER_HOST.casefold():
        raise RuntimeError(f"training is restricted to server B / {SERVER_HOST}")
    if ROOT.resolve() != SERVER_ROOT.resolve():
        raise RuntimeError(f"training root must be {SERVER_ROOT}; got {ROOT}")
    if gpu_index != EXPECTED[seed]["gpu"]:
        raise RuntimeError("resume must preserve the frozen physical GPU binding")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly the frozen GPU")

    base = ROOT / "outputs" / f"recovp_hybrid_seed{seed}"
    evidence = ROOT / "evidence" / "outputs" / f"recovp_hybrid_seed{seed}"
    folder = base / f"seed_{seed}"
    config_path = evidence / "CONFIG.json"
    preflight_path = evidence / "PREFLIGHT.json"
    config = read(config_path)
    preflight = read(preflight_path)
    if config.get("seed") != seed or config.get("epochs") != 256:
        raise RuntimeError("frozen config seed/epoch identity mismatch")
    if preflight.get("stage") != "PREFLIGHT_PASSED" or preflight.get("status") != "PASS":
        raise RuntimeError("frozen preflight is not a passing receipt")
    if config.get("preflight_sha256") != sha(preflight_path):
        raise RuntimeError("frozen preflight SHA differs from CONFIG.json")
    if preflight.get("seed") != seed or preflight.get("scope") != "TRAIN_AND_DEV_ONLY":
        raise RuntimeError("frozen preflight scope/seed mismatch")

    for rel, expected_hash in config.get("source_sha256", {}).items():
        if sha(ROOT / rel) != expected_hash:
            raise RuntimeError(f"frozen source SHA mismatch: {rel}")
    for rel, expected_hash in config.get("data_sha256", {}).items():
        if sha(ROOT / "data" / "paper2_1103" / rel) != expected_hash:
            raise RuntimeError(f"frozen data SHA mismatch: {rel}")
    recorded_gpu = config.get("execution", {})
    physical = int(recorded_gpu.get("physical_gpu_index", -1))
    uuid = server_gpu_admission.GPU_UUIDS.get(physical)
    if physical != gpu_index or recorded_gpu.get("gpu_uuid") != uuid:
        raise RuntimeError("frozen physical GPU/UUID binding mismatch")

    latest_epoch = EXPECTED[seed]["epoch"]
    latest = folder / "recovery" / f"epoch_{latest_epoch:03d}.pt"
    if sha(latest) != EXPECTED[seed]["sha256"]:
        raise RuntimeError("latest recovery checkpoint differs from the pause receipt")
    expected_epochs = set(range(latest_epoch + 1))
    paths = sorted((folder / "recovery").glob("epoch_*.pt"))
    found_epochs = {int(p.stem.split("_")[-1]) for p in paths}
    if found_epochs != expected_epochs:
        raise RuntimeError("recovery checkpoint sequence has a gap or unexpected epoch")
    for path in paths:
        receipt = read(path.with_suffix(".json"))
        if receipt.get("checkpoint_sha256") != sha(path):
            raise RuntimeError(f"recovery checkpoint SHA/sidecar mismatch: {path.name}")
        if int(receipt.get("epoch", -1)) != int(path.stem.split("_")[-1]):
            raise RuntimeError(f"recovery epoch/sidecar mismatch: {path.name}")

    sample = server_gpu_admission.query_gpus()[gpu_index]
    if sample["utilization_percent"] >= server_gpu_admission.UTILIZATION_LIMIT:
        raise server_gpu_admission.NoGpuAvailable(f"frozen GPU is not below 85%: {sample}")
    if sample["gpu_uuid"] != uuid:
        raise RuntimeError("current GPU UUID differs from frozen binding")
    return config, preflight, {"base": base, "evidence": evidence, "folder": folder,
                               "gpu": sample}


def run(seed: int, gpu_index: int, verify_only: bool = False) -> dict:
    if seed not in EXPECTED:
        raise ValueError("only paused seeds 79 and 97 may use this recovery entry")
    if verify_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    config, _preflight, paths = _verify(seed, gpu_index)
    if verify_only:
        return {"seed": seed, "stage": "RESUME_PREFLIGHT_PASSED", "epochs": 256,
                "saved_epoch": EXPECTED[seed]["epoch"], "gpu_index": gpu_index,
                "gpu_uuid": config["execution"]["gpu_uuid"],
                "config_sha256": sha(paths["evidence"] / "CONFIG.json"),
                "checkpoint_sha256": EXPECTED[seed]["sha256"]}

    import tools.vp_1103_recovp_hybrid_extended as extended
    import tools.vp_1103_recovp_hybrid_preflight as preflight_module
    import tools.vp_1103_recovp_hybrid_train as trainer

    run_id = f"recovp_hybrid_seed{seed}"
    gpu_uuid = server_gpu_admission.GPU_UUIDS[gpu_index]
    bindings = dict(config["source_sha256"])

    preflight_module.RUN = run_id
    preflight_module.BASE = paths["base"]
    preflight_module.EVIDENCE = paths["evidence"]
    preflight_module.GPU_INDEX = str(gpu_index)
    preflight_module.GPU_UUID = gpu_uuid
    preflight_module.source_bindings = lambda: dict(bindings)

    trainer.RUN = run_id
    trainer.SEED = seed
    trainer.SERVER_HOST = SERVER_HOST
    trainer.SERVER_ROOT = SERVER_ROOT
    trainer.GPU_INDEX = str(gpu_index)
    trainer.GPU_UUID = gpu_uuid
    trainer.BASE = paths["base"]
    trainer.EVIDENCE = paths["evidence"]
    trainer.PREFLIGHT = paths["evidence"] / "PREFLIGHT.json"
    trainer.FORMAT = f"recovp-hybrid-seed{seed}-epoch-v1"
    trainer.source_hashes = lambda: dict(bindings)
    trainer.require_server_and_gpu = lambda: extended._gpu_identity(gpu_index)
    # Preserve the frozen config byte-for-byte; its execution section records
    # the original admitted GPU snapshot and is part of every checkpoint ID.
    trainer.freeze_config = lambda *_args, **_kwargs: config

    result = trainer.train(seed)
    if result.get("stage") != "TRAINING_COMPLETE" or result.get("epochs") != 256:
        raise RuntimeError(f"trainer returned without full completion: {result}")
    return {"seed": seed, "stage": result["stage"], "epochs": result["epochs"],
            "gpu_index": gpu_index, "gpu_uuid": gpu_uuid,
            "saved_before_resume": EXPECTED[seed]["epoch"],
            "config_sha256": sha(paths["evidence"] / "CONFIG.json"),
            "training_complete": str(paths["folder"] / "TRAINING_COMPLETE.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=tuple(EXPECTED), required=True)
    parser.add_argument("--gpu-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.gpu_index, args.verify_only), indent=2), flush=True)


if __name__ == "__main__":
    main()
