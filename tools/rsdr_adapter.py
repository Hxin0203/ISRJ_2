"""Observed-only adapter for the frozen first-paper complete F-R7 RSDR-Net.

Vendor files are byte-preserved. D1's training-only logit hook and interpolation
backward wrappers are omitted; native inference parity is checked separately.
No old dataset or Gate array is opened by this module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from paper2.io import ROOT, sha
from tools.rsdr_vendor.m11_model import M11Config, M11RGSSR
from tools.rsdr_vendor.m12_model import ExpertRouter, mix_actions
from tools.rsdr_vendor.gate_core import BoundedGate, features

ASSETS = ROOT / "outputs/rsdr_unified_20261115/assets"
EXPECTED = {
    "u1_c32.pt": "7ae9f62434d058a57b08cc0f8c549bcad3e3c86533e87cebbfee9871195040d7",
    "u2_32.pt": "28086570dbe946870a491b5ae92931096b6dedb6993d7e6e383e919b4956fa38",
    "gate_e32_step16384.pt": "3a8f170f07d52149d08b2b21001e06760547a4f7d8f6e40eb907060a0624d3d1",
    "FIT.json": "85c07a6d7ca73c6d68b8fa93b68f7694ede50821692c7cccb63efef1d1a6218b",
    "router1.pt": "e8b70c4ca60bb6748bc93e69e8fa0d201a753407b7b5c7c477f71c1c6dc13fea",
    "router2.pt": "9f1b502d8435a0191db5cc588c1e97109fa7ca1b4b5b62dc0a554d8dcda2b139",
    "router3.pt": "11f2a26e2e7a6f50079671f3b1d2f15e2322aa301b6c2bc2aeb0e0a684ad1c26",
}


class DecoupledU1(M11RGSSR):
    def forward(self, y):
        out = super().forward(y)
        out["action"] = out["action_gate"]
        out["u_hat"] = y + out["action"] * out["delta_z"]
        out["j_hat"] = y - out["u_hat"]
        out["z_out"] = out["u_hat"]
        return out


def configure():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


class RSDRNet:
    def __init__(self, device="cuda:0", batch_pulses=256, verify_vendor=True):
        configure()
        self.device, self.batch_pulses = device, batch_pulses
        for name, expected in EXPECTED.items():
            if sha(ASSETS / name) != expected:
                raise ValueError(f"RSDR asset identity changed: {name}")
        if verify_vendor:
            provenance = json.loads((ROOT / "tools/rsdr_vendor/PROVENANCE.json").read_text())
            for name, row in provenance.items():
                assert sha(ROOT / "tools/rsdr_vendor" / name) == row["sha256"], name
        self.experts = [
            DecoupledU1(M11Config(name="control_default_noaug", mechanism="output_gate_control")),
            M11RGSSR(M11Config(name="aug_default", mechanism="internal_gated_conv")),
        ]
        for model, name, spec in zip(self.experts, ("u1_c32.pt", "u2_32.pt"), (
            "118d5c123c9d532d13a9f2223a548b6039b32ac51ccb905bd90e9d3061d7a126",
            "071d49dbdeb18e8db0cedeb695c7a4dfda73955c6b885096e54818f523e240fb")):
            cp = torch.load(ASSETS / name, map_location="cpu", weights_only=False)
            assert (cp["seed"], cp["steps"], cp["spec_sha"]) == (40312102, 16384, spec)
            model.load_state_dict(cp["model"], strict=True)
            model.to(device).eval().requires_grad_(False)
        self.routers = []
        for i in (1, 2, 3):
            cp = torch.load(ASSETS / f"router{i}.pt", map_location="cpu", weights_only=False)
            assert (cp["index"], cp["seed"], cp["epoch"], cp["steps"]) == (i, 40312102, 32, 16384)
            model = ExpertRouter(2)
            model.load_state_dict(cp["model"], strict=True)
            self.routers.append(model.to(device).eval().requires_grad_(False))
        cp = torch.load(ASSETS / "gate_e32_step16384.pt", map_location="cpu", weights_only=False)
        assert cp["steps"] == 16384 and len(cp["model"]) == 4
        self.gate = BoundedGate().to(device).eval().requires_grad_(False)
        self.gate.load_state_dict(cp["model"], strict=True)
        fit = json.loads((ASSETS / "FIT.json").read_text())
        self.weights = fit["anchor"]["weights"]
        self.mean, self.std = [torch.tensor(fit["normalization"][k], device=device) for k in ("mean", "std")]

    @torch.inference_mode()
    def restore(self, states, diagnostics=False, state_indexes=None):
        """[parent,4,8,1800] complex, new order E,S+E,J+E,S+J+E.

        State order is mapped to the historical E,J+E,S+E,S+J+E before the
        unchanged statewise group feature extraction, then mapped back.
        """
        x = np.asarray(states)
        assert x.ndim == 4 and x.shape[1:] == (4, 8, 1800) and np.isfinite(x).all()
        order = [0, 2, 1, 3]
        iq = np.stack((x[:, order].real, x[:, order].imag), axis=3).astype(np.float32)
        y = torch.from_numpy(iq).to(self.device)
        # Each gate row depends only on its own eight observed pulses. Selecting
        # rows avoids re-running identical no-jammer inputs across grid cells.
        selected = [0,1,2,3] if state_indexes is None else [order.index(i) for i in state_indexes]
        selected_y = y[:, selected]
        flat = selected_y.reshape(-1, 2, 1800)
        outputs, reliabilities = [], []
        for model in self.experts:
            chunks, rel = [], []
            for lo in range(0, len(flat), self.batch_pulses):
                result = model(flat[lo:lo + self.batch_pulses])
                chunks.append(result["u_hat"])
                rel.append(result["reliability"])
            outputs.append(torch.cat(chunks))
            reliabilities.append(torch.cat(rel))
        a, b = outputs
        heads = []
        for router in self.routers:
            p = torch.softmax(router(flat, a, b, *reliabilities) / .25, 1)
            heads.append(mix_actions(p, flat, a, b))
        m_selected = torch.stack(heads).mean(0).reshape_as(selected_y)
        a_selected, b_selected = [v.reshape_as(selected_y) for v in outputs]
        a, b, m = [torch.zeros_like(y) for _ in range(3)]
        a[:, selected], b[:, selected], m[:, selected] = a_selected, b_selected, m_selected
        w = self.weights
        anchor = w[0] * y + w[1] * a + w[2] * b
        features_x = features(y, anchor, m)
        rho = self.gate((features_x - self.mean) / self.std)
        f = anchor + rho[..., None, None, None] * (m - anchor)
        assert torch.isfinite(f).all() and torch.isfinite(rho).all()
        f = f.cpu().numpy()[:, order]
        restored = f[..., 0, :] + 1j * f[..., 1, :]
        if state_indexes is not None:
            restored = restored[:, state_indexes]
        if diagnostics:
            return restored, {"rho": rho.cpu().numpy()[:, order]}
        return restored
