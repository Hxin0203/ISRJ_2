"""Export the actual first-stage dictionary for the waveform inset's observation."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL = ROOT / "outputs/jest_signal_insets"
SOURCE = ROOT / "tools/vp_1103_recovp_hybrid.py"
CHECKPOINT = ROOT / "outputs/recovp_hybrid_seed17/seed_17/epoch_256.pt"
NAME = "03_dictionary_magnitude"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render(magnitude, name, vmax):
    # Original time bins and atom order; uniform linear scale across all atoms.
    levels = np.rint(255 * magnitude / vmax).astype(np.uint8)
    stops = np.array([[18, 49, 108], [88, 155, 198], [197, 213, 151],
                      [248, 215, 68], [255, 239, 98]])
    palette = np.rint(np.stack([
        np.interp(np.arange(256), np.linspace(0, 255, len(stops)), channel)
        for channel in stops.T
    ], axis=-1)).astype(np.uint8)
    height, width = levels.shape
    row_height = max(1, round(width * 11 / 35 / height))
    image = Image.fromarray(palette[levels]).resize((width, height * row_height), Image.Resampling.NEAREST)
    image.save(HERE / f"{name}.png")

    # Group same-colour horizontal runs into vector paths for editable SVG.
    runs = [[] for _ in palette]
    for row, pixels in enumerate(levels):
        boundaries = np.r_[0, np.flatnonzero(np.diff(pixels)) + 1, width]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            runs[pixels[start]].append(f"M{start},{row_height*row}h{end-start}v{row_height}H{start}z")
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="35mm" height="{35*row_height*height/width:.6f}mm" viewBox="0 0 {width} {row_height*height}" shape-rendering="crispEdges">',
           '<title>Magnitude of the first-stage dictionary; atom 1 at top, time increasing rightward</title>']
    for colour, paths in zip(palette, runs):
        if paths:
            rgb = bytes(colour).hex()
            svg.append(f'<path fill="#{rgb}" d="{"".join(paths)}"/>')
    (HERE / f"{name}.svg").write_text("\n".join(svg + ["</svg>"]), encoding="utf-8")


def main():
    identity_path = CHECKPOINT.with_suffix(".json")
    record = json.loads(identity_path.read_text())
    model_sha, checkpoint_sha = sha(SOURCE), sha(CHECKPOINT)
    if model_sha != record["identity"]["source_sha256"][SOURCE.relative_to(ROOT).as_posix()]:
        raise ValueError("Registered model source digest mismatch")
    if checkpoint_sha != record["checkpoint_sha256"]:
        raise ValueError("Registered checkpoint digest mismatch")
    source_spec = importlib.util.spec_from_file_location("frozen_hybrid", SOURCE)
    module = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(module)
    model = module.ReCoVPHybrid().eval()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if state["epoch"] != 256 or record["identity"]["seed"] != 17:
        raise ValueError("Unexpected seed or epoch")
    model.load_state_dict(state["model"], strict=True)

    observation = LOCAL / "observations.npz"
    with np.load(observation, allow_pickle=False) as data:
        y = data["y"]
        metadata = {key: data[key].item() for key in
                    ("parent_id", "parent_index", "pulse_index", "snr_db", "jsr_db", "jam_type", "source_sha256")}
    manifest = json.loads((ROOT / "data/hybrid_condition_master_20261127/DATA_MANIFEST.json").read_text())
    if metadata["source_sha256"] != manifest["component_sha256"] or y.shape != (1800,) or not np.isfinite(y).all():
        raise ValueError("Unexpected observation or archive identity")
    with torch.inference_mode():
        result = model.stage1(torch.from_numpy(y)[None])
    basis = result["dictionary"][0].numpy()
    norm_error = float(np.max(np.abs(np.linalg.norm(basis, axis=0) - 1)))
    gram_error = float(np.max(np.abs(basis.conj().T @ basis - result["gram"][0].numpy())))
    if basis.shape != (1800, 16) or not np.isfinite(basis).all() or norm_error > 1e-8 or gram_error > 1e-8:
        raise ValueError("Frozen model dictionary failed numerical checks")
    matrix_file = LOCAL / "dictionary.npz"
    np.savez_compressed(matrix_file, B=basis)
    magnitude = np.abs(basis).T
    vmax = float(magnitude.max())
    render(magnitude, NAME, vmax)
    active = np.flatnonzero(np.max(magnitude, axis=0) >= .1 * vmax)
    start = max(0, int(active[0]) - 16)
    end = min(magnitude.shape[1], int(active[-1]) + 17)
    render(magnitude[:, start:end], NAME + "_detail", vmax)
    # Same time span as the observation inset, referenced to the full matrix peak.
    with np.load(observation, allow_pickle=False) as data:
        j_abs = np.abs(data["j"])
    support = np.flatnonzero(j_abs > j_abs.max() / 100)
    left = max(0, int(support[0]) - 15)
    right = min(magnitude.shape[1], int(support[-1]) + 16)
    floor_db = -48.
    db = 20 * np.log10(np.maximum(magnitude[:, left:right] / vmax, 10 ** (floor_db / 20)))
    render(db - floor_db, NAME + "_detail_db", -floor_db)
    receipt = {
        "model": "ReCoVP-Hybrid", "stage": 1, "seed": 17, "epoch": 256,
        "model_source": SOURCE.relative_to(ROOT).as_posix(), "model_source_sha256": model_sha,
        "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(), "checkpoint_sha256": checkpoint_sha,
        "identity_sha256": sha(identity_path), "observation": metadata,
        "observations_npz_sha256": sha(observation), "input_key": "y",
        "input_complex64_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "matrix": matrix_file.relative_to(ROOT).as_posix(), "matrix_key": "B", "matrix_file_sha256": sha(matrix_file),
        "matrix_shape_time_atom": list(basis.shape), "dtype": str(basis.dtype),
        "max_unit_norm_error": norm_error, "max_gram_error": gram_error,
        "display": "abs(B).T, original atom order, time increases left to right, atom 1 at top",
        "scale": "global linear, 256 colours, zero to global maximum", "vmin": 0., "vmax": vmax,
        "detail_time_range_zero_based_half_open": [start, end],
        "detail_db": {"time_range_zero_based_half_open": [left, right], "limits_db": [floor_db, 0],
                      "reference": "global peak of abs(B), preserved atom order"},
        "figure_sha256": {p.name: sha(p) for p in sorted(HERE.glob(f"{NAME}*")) if p.suffix in (".png", ".svg")},
    }
    (HERE / "dictionary_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
