"""Plot the frozen model's first-stage dictionary for the complete receiver record."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOCAL = ROOT / "outputs/jest_signal_insets/receiver_view"
SOURCE = ROOT / "tools/vp_1103_recovp_hybrid.py"
CHECKPOINT = ROOT / "outputs/recovp_hybrid_seed17/seed_17/epoch_256.pt"
DB_FLOOR = -48.0


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_heatmap(basis):
    magnitude = np.abs(basis).T
    peak = magnitude.max()
    relative = np.maximum(magnitude / peak, 10 ** (DB_FLOOR / 20))
    levels = np.rint(255 * (20 * np.log10(relative) - DB_FLOOR) / -DB_FLOOR).astype(np.uint8)
    stops = np.array([[18, 49, 108], [88, 155, 198], [197, 213, 151],
                      [248, 215, 68], [255, 239, 98]])
    palette = np.rint(np.stack([
        np.interp(np.arange(256), np.linspace(0, 255, len(stops)), channel)
        for channel in stops.T
    ], axis=-1)).astype(np.uint8)
    rows, width = levels.shape
    row_height = 24
    height = rows * row_height
    png = HERE / "dictionary_full_db.png"
    svg = HERE / "dictionary_full_db.svg"
    Image.fromarray(palette[levels]).resize((width, height), Image.Resampling.NEAREST).save(png)
    paths = [[] for _ in palette]
    for row, pixels in enumerate(levels):
        boundaries = np.r_[0, np.flatnonzero(np.diff(pixels)) + 1, width]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            paths[pixels[left]].append(f"M{left},{row * row_height}h{right-left}v{row_height}H{left}z")
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="45mm" height="{45*height/width:g}mm" viewBox="0 0 {width} {height}" shape-rendering="crispEdges">',
                '<title>Full receiver record: dictionary magnitude, -48 to 0 dB relative to the matrix peak</title>']
    for colour, runs in zip(palette, paths):
        if runs:
            elements.append(f'<path fill="#{bytes(colour).hex()}" d="{"".join(runs)}"/>')
    svg.write_text("\n".join(elements + ["</svg>"]) + "\n", encoding="utf-8")
    return {path.name: digest(path) for path in (png, svg)}


def main():
    receipt_path = CHECKPOINT.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    source_sha = digest(SOURCE)
    checkpoint_sha = digest(CHECKPOINT)
    if checkpoint_sha != receipt["checkpoint_sha256"] or source_sha != receipt["identity"]["source_sha256"][SOURCE.relative_to(ROOT).as_posix()]:
        raise ValueError("Checkpoint or model source differs from the frozen receipt")
    observation = LOCAL / "observations.npz"
    with np.load(observation, allow_pickle=False) as data:
        y = data["y"]
        fields = ("parent_id", "parent_index", "pulse_index", "snr_db", "jsr_db", "jam_type", "source_sha256")
        metadata = {key: data[key].item() for key in fields}
    master = ROOT / "data/hybrid_condition_master_20261127/DATA_MANIFEST.json"
    if metadata["source_sha256"] != json.loads(master.read_text())["component_sha256"]:
        raise ValueError("Observation and registered component identities differ")
    if y.dtype != np.complex64 or y.shape != (1800,) or not np.isfinite(y).all():
        raise ValueError("Expected one complete finite complex64 receiver record")
    spec = importlib.util.spec_from_file_location("frozen_receiver_model", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if state["epoch"] != 256 or receipt["identity"]["seed"] != 17:
        raise ValueError("Unexpected checkpoint identity")
    model = module.ReCoVPHybrid().eval()
    model.load_state_dict(state["model"], strict=True)
    with torch.inference_mode():
        result = model.stage1(torch.from_numpy(y)[None])
    basis = result["dictionary"][0].numpy()
    norm_error = float(np.max(abs(np.linalg.norm(basis, axis=0) - 1)))
    gram_error = float(np.max(abs(basis.conj().T @ basis - result["gram"][0].numpy())))
    if basis.shape != (1800, 16) or not np.isfinite(basis).all() or norm_error > 1e-8 or gram_error > 1e-8:
        raise ValueError("Dictionary failed its shape, finiteness, normalization, or Gram check")
    array_path = LOCAL / "dictionary.npz"
    np.savez_compressed(array_path, B=basis)
    report = {
        "model": "ReCoVP-Hybrid Stage 1", "seed": 17, "epoch": 256, "device": "cpu",
        "observation": metadata, "observations_npz_sha256": digest(observation),
        "input_y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(), "checkpoint_sha256": checkpoint_sha,
        "model_source": SOURCE.relative_to(ROOT).as_posix(), "source_sha256": source_sha,
        "matrix": array_path.relative_to(ROOT).as_posix(), "matrix_key": "B",
        "matrix_file_sha256": digest(array_path), "matrix_shape": list(basis.shape),
        "matrix_dtype": str(basis.dtype), "max_unit_norm_error": norm_error, "max_gram_error": gram_error,
        "display": "abs(B).T, atoms 1 to 16 top to bottom, input samples 0 to 1799 left to right",
        "limits_db": [DB_FLOOR, 0.0], "reference_amplitude": float(np.abs(basis).max()),
        "figure_sha256": export_heatmap(basis),
    }
    (HERE / "dictionary_provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
