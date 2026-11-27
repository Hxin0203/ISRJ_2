"""Recover structural labels by exact deterministic replay of the fixed master."""
import gzip
import json

import numpy as np

from tools.hybrid_condition_benchmark import DATA, REPORT, SEED, MASTER_PARENTS, sha, write_json


def main():
    import paper2.data_1103 as generator
    destination = DATA / "STRUCTURAL_METADATA.json.gz"
    if destination.exists():
        return
    targets, jammers = [], []
    original_target, original_jammer = generator._draw_target, generator._draw_jammer

    def target(*args, **kwargs):
        waveform, active, meta = original_target(*args, **kwargs)
        targets.append(meta)
        return waveform, active, meta

    def jammer(*args, **kwargs):
        waveform, meta = original_jammer(*args, **kwargs)
        jammers.append(dict(parent_index=len(targets) - 1, **meta))
        return waveform, meta

    replay_folder = DATA / "metadata_replay"
    generator._draw_target, generator._draw_jammer = target, jammer
    try:
        replay, receipt = generator.generate("test", count=MASTER_PARENTS, seed=SEED, snr_db=0, jsr_db=0, out_dir=replay_folder)
    finally:
        generator._draw_target, generator._draw_jammer = original_target, original_jammer
    original = DATA / "components/test_0_0.npz"
    with np.load(original, allow_pickle=False) as saved, np.load(replay, allow_pickle=False) as reproduced:
        for key in saved.files:
            equal = np.array_equal(saved[key], reproduced[key], equal_nan=True) if saved[key].dtype.kind in "fc" else np.array_equal(saved[key], reproduced[key])
            if not equal:
                raise RuntimeError(f"deterministic replay mismatch {key}")
        cursor = 0
        for p in range(MASTER_PARENTS):
            for jt, label in enumerate(("direct", "repeat", "cyclic")):
                for pulse in np.flatnonzero(np.any(saved["j"][p, jt] != 0, axis=-1)):
                    record = jammers[cursor]
                    if record["parent_index"] != p or record["jam_type"] != label:
                        raise RuntimeError("structural label ordering mismatch")
                    record["pulse_index"] = int(pulse)
                    cursor += 1
        if cursor != len(jammers):
            raise RuntimeError("structural label count mismatch")
    with gzip.open(destination, "wt", encoding="utf-8") as f:
        json.dump({"targets": targets, "jammed_pulses": jammers}, f, ensure_ascii=False)
    summary = {"path": str(destination), "sha256": sha(destination), "component_sha256": sha(original),
               "exact_replay_all_arrays": True, "parents": len(targets), "jammed_pulse_records": len(jammers),
               "jammer_parameter_ranges": {k: [float(min(r[k] for r in jammers)), float(max(r[k] for r in jammers))]
                   for k in ("window_samples", "period_samples", "repeat_count", "sampling_phase_samples", "forward_delay_samples", "replay_gap_samples", "frequency_offset_hz")}}
    write_json(DATA / "STRUCTURAL_METADATA_AUDIT.json", summary)
    write_json(REPORT / "STRUCTURAL_METADATA_AUDIT.json", summary)
    # These two duplicate arrays were created solely for replay validation.
    replay.unlink()
    receipt.unlink()
    replay_folder.rmdir()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
