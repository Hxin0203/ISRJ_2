"""Bind re-scored historical DEV parent metrics to the original four cells."""
import numpy as np

from vp_1103_common import DEV_HASH, HISTORICAL, SEEDS, SNAPSHOT, WORK, read, sha

REF = WORK / "outputs" / "vp_1103_references"
HISTORY = {
    "unet_original": "unet_mf_tail_screen_20261117_v2/tail",
    "unet_tail": "unet_mf_tail_screen_20261117_v2/tail",
    "correction_original": "correction_mf_tail_20261117_v1/arm",
    "correction_tail": "correction_mf_tail_20261117_v1/arm",
}


def reference_metrics(arm, ids):
    original = read(SNAPSHOT / HISTORICAL)
    scores, evidence = {}, {}
    for seed, historical_row in zip(SEEDS, original["rows"], strict=True):
        if historical_row["seed"] != seed:
            raise ValueError("historical seed order changed")
        stem = REF / f"{arm}_{seed}"
        receipt = read(stem.with_suffix(".json"))
        path = stem.with_suffix(".npz")
        key = f"outputs/{HISTORY[arm]}/seed_{seed}/dev_paired_epoch_256.json"
        if (receipt["arm"], receipt["seed"], receipt["epoch"],
            receipt["dev_sha256"], receipt["metrics_sha256"], receipt["historical_receipt_sha256"]) != (
                arm, seed, 256, DEV_HASH, sha(path), original["source_evidence_sha256"][key]):
            raise ValueError(f"reference identity mismatch for {arm} seed {seed}")
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["parent_id"], ids):
                raise ValueError("reference DEV parent roster mismatch")
            rows = {name: archive[name] for name in archive.files if name != "parent_id"}
        differences = {}
        for name, array in rows.items():
            published = historical_row[arm][name]["mean"]
            error = abs(float(array.mean()) - published)
            atol = 0. if name in ("frame_pfa_jammer", "pd") else 2e-4
            if error > max(atol, 1e-15):
                raise ValueError(f"nonmatching historical mean {arm}/{seed}/{name}: {error}")
            differences[name] = error
        scores[seed] = rows
        evidence[str(seed)] = {**receipt, "absolute_mean_reproduction_error": differences}
    return scores, evidence
