"""Read frozen epoch-256 DEV references on the historical server via stdin."""
import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from paper2.evaluation_1103_v4 import group_metrics, radar_metrics, restore_model, states_from_components
from paper2.io import sha
from tools.unet_ablation.models import make_model
from tools.unet_ablation.runtime import configure_precision

ROOT = Path.cwd()
DEV = ROOT / "data/paper2_1103/dev.npz"
DEV_SHA = "0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1"
ARMS = {
    "unet_original": ("unet", "unet_ablation_3seed_256_20261115/unet",
                      "unet_mf_tail_screen_20261117_v2/tail", "baseline_checkpoint_sha256"),
    "unet_tail": ("unet", "unet_mf_tail_screen_20261117_v2/tail",
                  "unet_mf_tail_screen_20261117_v2/tail", "tail_checkpoint_sha256"),
    "correction_original": ("full", "correction_original_control_20261117_v1/arm",
                            "correction_mf_tail_20261117_v1/arm", "baseline_checkpoint_sha256"),
    "correction_tail": ("full", "correction_mf_tail_20261117_v1/arm",
                        "correction_mf_tail_20261117_v1/arm", "study_checkpoint_sha256"),
}
METRICS = ("frame_pfa_jammer", "pd", "damage", "sjrif_db", "u_nmse", "no_jammer_pooled_nmse")


def run(arm, seed):
    if seed not in (17, 29, 41) or sha(DEV) != DEV_SHA:
        raise RuntimeError("invalid historical DEV or seed")
    variant, source, reference, hash_key = ARMS[arm]
    checkpoint = ROOT / "outputs" / source / f"seed_{seed}" / "epoch_256.pt"
    receipt = ROOT / "outputs" / reference / f"seed_{seed}" / "dev_paired_epoch_256.json"
    evidence = json.loads(receipt.read_text(encoding="utf8"))
    digest = sha(checkpoint)
    if digest != evidence[hash_key] or evidence["dev_data_sha256"] != DEV_SHA:
        raise RuntimeError("checkpoint does not match the archived four-cell evidence")
    precision = configure_precision()
    torch.set_num_threads(2)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["seed"] != seed or payload["epoch"] != 256:
        raise RuntimeError("unexpected checkpoint identity")
    device = torch.device("cuda")
    model = make_model(variant).to(device).eval()
    model.load_state_dict(payload["model"])
    with np.load(DEV, allow_pickle=False) as archive:
        s, j, e = [archive[name] for name in ("s", "j", "epsilon")]
        doppler = archive["doppler_hz"]
        ids = archive["parent_id"]
    result = {key: [] for key in METRICS}
    for lo in range(0, len(ids), 8):
        hi = lo + 8
        ss, jj, ee = [array[lo:hi].astype(np.complex128).reshape(-1, 8, 1800)
                      for array in (s, j, e)]
        y = states_from_components(ss, jj, ee)
        restored = restore_model(model, y, device)
        metrics = group_metrics(ss, jj, ee, restored)
        metrics.update(radar_metrics(ss, y, restored, np.repeat(doppler[lo:hi], 3)))
        for key in METRICS:
            result[key].append(metrics[key].reshape(-1, 3))
    result = {key: np.concatenate(blocks) for key, blocks in result.items()}
    raw = io.BytesIO()
    np.savez_compressed(raw, parent_id=ids, **result)
    data = raw.getvalue()
    print(json.dumps({"dev_sha256": DEV_SHA, "arm": arm, "seed": seed, "epoch": 256,
                      "checkpoint_sha256": digest, "historical_receipt_sha256": sha(receipt),
                      "metrics_sha256": hashlib.sha256(data).hexdigest(),
                      "precision": precision, "torch": torch.__version__,
                      "metrics_base64": base64.b64encode(data).decode()}))


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
