"""Server-only engineering checks for the differentiable tail loss."""
from __future__ import annotations

import platform
import torch

from paper2.train_1103 import batches, load_arrays, make_batch
from paper2.train_256_8seed import seed_rng
from tools.unet_ablation.models import PlainUNet, objective as baseline_objective
from tools.unet_ablation.protocol import DATA
from .objective import matched_filter_tail_loss, TAIL_WEIGHT
from tools.unet_ablation.runtime import require_server, configure_precision


def main() -> None:
    require_server()
    precision = configure_precision()
    torch.set_num_threads(4)
    seed_rng(17)
    arrays = load_arrays(DATA / "train.npz")
    ids = next(batches(len(arrays["s"]), 1, 17))[:8]
    batch = make_batch(arrays, ids, "cuda")
    model = PlainUNet().cuda().train()
    corr = model(batch["y"]).reshape(-1, 4, 2, 1800)
    baseline = baseline_objective(corr, batch["s"], batch["j"], batch["e"], "unet")
    tail, details = matched_filter_tail_loss(corr.reshape(8, 8, 4, 2, 1800), batch["j"], batch["e"])
    total = baseline["total"] + TAIL_WEIGHT * tail
    if not torch.isfinite(total):
        raise RuntimeError("non-finite combined training objective")
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("combined objective did not produce finite model gradients")
    torch.cuda.synchronize()
    result = {"stage": "ENGINEERING_SMOKE_PASSED", "host": platform.node(),
              "device": torch.cuda.get_device_name(), "precision": precision,
              "baseline_loss": float(baseline["total"].detach()),
              "tail_loss": float(tail.detach()),
              "weighted_tail_to_baseline": float((TAIL_WEIGHT * tail.detach() / baseline["total"].detach().clamp_min(1e-8))),
              "diagnostics": {k: float(v) for k, v in details.items()},
              "finite_gradient": bool(all(torch.isfinite(g).all() for g in gradients)),
              "gradient_l2": float(torch.sqrt(sum(g.detach().float().square().sum() for g in gradients)))}
    print(result, flush=True)


if __name__ == "__main__":
    main()
