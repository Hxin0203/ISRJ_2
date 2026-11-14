"""DEV-only evaluation for frozen 1103-aligned checkpoints.

Reads only ``data/paper2_1103/dev.npz`` and writes compact JSON summaries.
It deliberately refuses to fall back to TRAIN or TEST data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from .data_1103 import JAM_TYPES
from .io import ROOT, now, read, sha, write
from .network import CorrectionNet
from .radar_eval import (alias_doppler_frequency, ca_cfar, doppler_process,
                         mainlobe_guard, pulse_compress, sjrif, target_peak_range)
from .science import metrics, parent_mean


def _iq(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack((x.real, x.imag), axis=-2).astype(np.float32))


def _ci(values: np.ndarray, seed: int = 20261113, reps: int = 4000) -> list[float]:
    """Percentile bootstrap over physical parents, never over pulses/variants."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(x), size=(reps, len(x)))
    return [float(v) for v in np.quantile(x[draw].mean(axis=1), [.025, .975])]


def evaluate(seed: int, epoch: int, *, batch_records: int = 48) -> dict:
    base = ROOT / "outputs" / "paper2_1103_retry_20261114"
    data_path = ROOT / "data" / "paper2_1103" / "dev.npz"
    checkpoint = base / f"seed_{seed}" / f"epoch_{epoch:03d}.pt"
    if not data_path.is_file():
        raise FileNotFoundError(f"DEV role is required; not substituting another role: {data_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    data_sha, checkpoint_sha = sha(data_path), sha(checkpoint)
    with np.load(data_path, allow_pickle=False) as z:
        s, j, e = [z[k].astype(np.complex64) for k in ("s", "j", "epsilon")]
        parent_id = z["parent_id"].copy()
        delay = z["target_delay_samples"].copy()
        doppler = z["doppler_hz"].copy()
    n, n_jam, pulses, length = s.shape
    if n_jam != len(JAM_TYPES) or pulses != 8:
        raise ValueError(f"unexpected DEV tensor shape {s.shape}")

    # One row is one parent × jammer type × pulse. Four states retain the
    # shared parent/pulse pairing used by the B4 response-difference metric.
    sr, jr, er = (v.reshape(n * n_jam * pulses, length) for v in (s, j, e))
    states = np.stack((er, sr + er, jr + er, sr + jr + er), axis=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    model = CorrectionNet().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("seed") != seed or payload.get("epoch") != epoch:
        raise ValueError("checkpoint identity does not match requested seed/epoch")
    model.load_state_dict(payload["state"])
    model.eval()
    correction = np.empty((len(states), 4, length), dtype=np.complex64)
    with torch.inference_mode():
        for lo in range(0, len(states), batch_records):
            hi = min(lo + batch_records, len(states))
            x = _iq(states[lo:hi]).reshape(-1, 2, length).to(device)
            c = model(x).reshape(hi - lo, 4, 2, length).cpu().numpy()
            correction[lo:hi] = c[:, :, 0] + 1j * c[:, :, 1]

    flat_parent = np.repeat(np.repeat(parent_id, n_jam), pulses)
    cont = metrics(sr, jr, er, correction)
    continuous = {}
    for name in ("u_nmse", "damage", "background_nmse", "h0_nmse", "noise_nmse", "amplitude", "phase_deg"):
        per_parent = parent_mean(cont[name], flat_parent)
        continuous[name] = dict(mean=float(per_parent.mean()), parent_median=float(np.median(per_parent)),
                                parent_p95=float(np.quantile(per_parent, .95)),
                                parent_bootstrap_ci95=_ci(per_parent, seed=seed * 100 + epoch),
                                parents=int(len(per_parent)))

    restored = states[:, 3] - correction[:, 3]
    noise_restored = states[:, 0] - correction[:, 0]
    before_pc, range_lags = pulse_compress(states[:, 3].reshape(n * n_jam, pulses, length))
    after_pc, _ = pulse_compress(restored.reshape(n * n_jam, pulses, length))
    noise_before_pc, _ = pulse_compress(states[:, 0].reshape(n * n_jam, pulses, length))
    noise_after_pc, _ = pulse_compress(noise_restored.reshape(n * n_jam, pulses, length))
    # Use the no-jammer target-only matched-filter peak. Fast-time Doppler
    # causes LFM range-Doppler coupling, so the physical start delay is not
    # necessarily the peak bin used by SJRIF and detection scoring.
    target_range = target_peak_range(s[:, 0])
    guard = mainlobe_guard()
    sjrif_rows = []
    pd_before, pd_after, cell_fa_before, cell_fa_after, frame_fa_before, frame_fa_after = [], [], [], [], [], []
    valid_cells_before = valid_cells_after = 0
    for i in range(n):
        for jt in range(n_jam):
            frame = i * n_jam + jt
            tr = int(target_range[i])
            if tr not in range_lags:
                continue
            m = sjrif(before_pc[frame], after_pc[frame], tr, mainlobe_guard=guard)
            sjrif_rows.append((i, m))
            for compressed, pd_out in (
                (before_pc[frame], pd_before),
                (after_pc[frame], pd_after),
            ):
                rd, fd_bins = doppler_process(compressed)
                detections, _, valid, _, _ = ca_cfar(np.abs(rd) ** 2)
                folded_doppler = alias_doppler_frequency(float(doppler[i]))
                target_doppler_bin = int(np.argmin(np.abs(fd_bins - folded_doppler)))
                hit = bool(detections[target_doppler_bin:target_doppler_bin+1,
                                      max(0, tr-1):min(detections.shape[1], tr+2)].any())
                pd_out.append(hit)
            # E is shared across the three jammer variants. Count it once per
            # physical parent so the no-target false-alarm estimate is not
            # triplicated as if those were independent samples.
            if jt == 0:
                for noise_pc, cfa_out, ffa_out, which in (
                    (noise_before_pc[frame], cell_fa_before, frame_fa_before, "before"),
                    (noise_after_pc[frame], cell_fa_after, frame_fa_after, "after"),
                ):
                    noise_rd, _ = doppler_process(noise_pc)
                    noise_detections, _, noise_valid, _, _ = ca_cfar(np.abs(noise_rd) ** 2)
                    false = noise_detections & noise_valid
                    cfa_out.append(int(false.sum()))
                    ffa_out.append(bool(false.any()))
                    if which == "before":
                        valid_cells_before += int(noise_valid.sum())
                    else:
                        valid_cells_after += int(noise_valid.sum())
    if not sjrif_rows:
        raise RuntimeError("no DEV targets were inside the evaluated range")
    sjrif_parent = np.asarray([v[1]["sjrif_db"] for v in sjrif_rows])
    false_peak_parent = np.asarray([v[1]["false_peak_suppression_db"] for v in sjrif_rows])
    sjrif_by_parent = sjrif_parent.reshape(n, n_jam).mean(axis=1)
    false_peak_by_parent = false_peak_parent.reshape(n, n_jam).mean(axis=1)
    pd_keep_by_parent = np.asarray(pd_before, dtype=np.float64).reshape(n, n_jam).mean(axis=1)
    pd_restored_by_parent = np.asarray(pd_after, dtype=np.float64).reshape(n, n_jam).mean(axis=1)
    pd_delta_by_parent = pd_restored_by_parent - pd_keep_by_parent
    frame_false_before = np.asarray(frame_fa_before, dtype=np.float64)
    frame_false_after = np.asarray(frame_fa_after, dtype=np.float64)
    result = dict(
        stage="DEV_ONLY", created_utc=now(), seed=seed, epoch=epoch, parents=n,
        device=str(device), data_sha256=data_sha, checkpoint_sha256=checkpoint_sha,
        source_sha256=payload.get("source_sha256"),
        evaluator_source_sha256={
            "paper2/evaluate_1103.py": sha(Path(__file__)),
            "paper2/radar_eval.py": sha(Path(__file__).with_name("radar_eval.py")),
        },
        continuous=continuous,
        radar=dict(mainlobe_guard_samples=guard,
                   sjrif_db_mean=float(sjrif_by_parent.mean()),
                   sjrif_db_parent_median=float(np.median(sjrif_by_parent)),
                   sjrif_db_parent_bootstrap_ci95=_ci(sjrif_by_parent, seed=seed * 100 + epoch + 1),
                   false_peak_suppression_db_mean=float(false_peak_by_parent.mean()),
                   false_peak_suppression_db_parent_bootstrap_ci95=_ci(false_peak_by_parent, seed=seed * 100 + epoch + 2),
                   pd_keep=float(pd_keep_by_parent.mean()), pd_restored=float(pd_restored_by_parent.mean()),
                   pd_delta=float(pd_delta_by_parent.mean()),
                   pd_delta_parent_bootstrap_ci95=_ci(pd_delta_by_parent, seed=seed * 100 + epoch + 3),
                   cell_pfa_keep=float(sum(cell_fa_before) / max(valid_cells_before, 1)),
                   cell_pfa_restored=float(sum(cell_fa_after) / max(valid_cells_after, 1)),
                   frame_pfa_keep=float(frame_false_before.mean()),
                   frame_pfa_restored=float(frame_false_after.mean()),
                   frame_pfa_keep_parent_bootstrap_ci95=_ci(frame_false_before, seed=seed * 100 + epoch + 4),
                   frame_pfa_restored_parent_bootstrap_ci95=_ci(frame_false_after, seed=seed * 100 + epoch + 5),
                   frames_per_method=int(len(pd_before))),
        selection_note="DEV describes checkpoint performance only; it is not an independent TEST result.",
    )
    out = base / f"seed_{seed}" / f"dev_epoch_{epoch:03d}_bootstrap_v3.json"
    write(out, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True, choices=(17, 29, 41))
    p.add_argument("--epoch", type=int, required=True, choices=(32, 64, 96, 128))
    p.add_argument("--batch-records", type=int, default=48)
    args = p.parse_args()
    print(evaluate(args.seed, args.epoch, batch_records=args.batch_records))


if __name__ == "__main__":
    main()
