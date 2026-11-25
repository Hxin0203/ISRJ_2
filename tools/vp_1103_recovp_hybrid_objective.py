"""Shared ReCoVP-Hybrid joint objective for training and the GPU preflight."""
from __future__ import annotations

import torch

from paper2.network import training_loss
from tools.unet_tail_loss.objective import matched_filter_tail_loss
from tools.vp_1103_recovp_hybrid import ReCoVPHybrid


def hybrid_objective(model: ReCoVPHybrid, batch: dict, *, need_details: bool = False,
                     verify_stage1_identity: bool = False):
    """Compute L_new(final) + 0.25 L_new(Stage1) on the same B4 records.

    Both B4 and matched-filter tail terms take predicted jammer corrections C.
    ``factor()['restored']`` and ``factor()['stage1_restored']`` are desired
    signal-plus-noise waveforms, so their corrections are y - restored.
    """
    iq = batch["y"]
    observed = torch.complex(iq[:, 0], iq[:, 1])
    factor = model.factor(iq)
    record_count = len(batch["s"]) // 8
    if len(iq) != record_count * 8 * 4:
        raise ValueError("Hybrid objective expects record × 8 pulses × 4 states")

    def correction_iq(restored: torch.Tensor) -> torch.Tensor:
        correction = observed - restored
        return torch.stack((correction.real, correction.imag), dim=1).reshape(
            record_count * 8, 4, 2, iq.shape[-1])

    final_correction = correction_iq(factor["restored"])
    stage1_correction = correction_iq(factor["stage1_restored"])
    # Keep the Stage-1 correction semantics explicit and guard model refactors.
    if verify_stage1_identity:
        stage1_direct = torch.stack((factor["correction"].real,
                                     factor["correction"].imag), dim=1).reshape_as(stage1_correction)
        # ``restored = observed - correction`` is rounded in float32, so
        # subtracting it from ``observed`` incurs cancellation error. Bound
        # that error by the signal scale rather than using a fixed absolute
        # tolerance that changes meaning across dataset amplitude conventions.
        reference_scale = torch.maximum(observed.abs(), factor["stage1_restored"].abs()).amax()
        roundoff_bound = 8 * torch.finfo(torch.float32).eps * reference_scale.clamp_min(1e-30)
        identity_error = (stage1_correction - stage1_direct).abs().amax()
        if not torch.isfinite(identity_error) or identity_error > roundoff_bound:
            raise RuntimeError(
                "y - Stage1 restoration differs from its physical correction "
                f"beyond float32 roundoff (max_error={float(identity_error.detach()):.6g}, "
                f"bound={float(roundoff_bound.detach()):.6g})")

    def one_stage(correction: torch.Tensor) -> tuple[dict, torch.Tensor, dict]:
        base = training_loss(correction, batch["s"], batch["j"], batch["e"], "B4")
        grouped = correction.reshape(record_count, 8, 4, 2, iq.shape[-1])
        tail, tail_details = matched_filter_tail_loss(grouped, batch["j"], batch["e"])
        return base, tail, tail_details

    final_base, final_tail, final_diag = one_stage(final_correction)
    stage1_base, stage1_tail, stage1_diag = one_stage(stage1_correction)
    losses = {
        "total": final_base["total"] + 0.05 * final_tail
                 + 0.25 * (stage1_base["total"] + 0.05 * stage1_tail),
        "endpoint": final_base["endpoint"] + 0.25 * stage1_base["endpoint"],
        "pc": final_base["pc"] + 0.25 * stage1_base["pc"],
        "damage": final_base["damage"] + 0.25 * stage1_base["damage"],
        "tail": final_tail + 0.25 * stage1_tail,
        "final_total": final_base["total"] + 0.05 * final_tail,
        "stage1_total": stage1_base["total"] + 0.05 * stage1_tail,
        "final_tail": final_tail,
        "stage1_tail": stage1_tail,
    }
    diagnostics = {
        "final_tail_peak_db": final_diag["tail_peak_db_mean"],
        "stage1_tail_peak_db": stage1_diag["tail_peak_db_mean"],
        "tail_peak_db_mean": final_diag["tail_peak_db_mean"],
        "stage2_delta_rms": factor["stage2_delta"].abs().square().mean().sqrt(),
    }
    if need_details:
        return losses, diagnostics
    return losses
