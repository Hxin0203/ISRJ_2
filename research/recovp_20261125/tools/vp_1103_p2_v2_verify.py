"""Verify source, fixed budgets, private artifact hashes and the retained pause."""
import ast

import numpy as np
import torch

from vp_1103_common import WORK, atomic_json, now, provenance, read, sha
from vp_1103_p2_smoothvp import real_parameters
from vp_1103_p2_smoothvp_v2 import SmoothVPNetV2

OUT = WORK / "evidence/outputs/p2_smoothvp_v2_preflight"
PRIVATE = WORK / "outputs/p2_smoothvp_v2_preflight"


def source_binding(receipt):
    bindings = receipt["source_sha256"]
    for name, digest in bindings.items():
        if sha(WORK / name) != digest:
            raise ValueError(f"Source changed: {name}")
    return len(bindings)


def run():
    torch.set_num_threads(2)
    started = now()
    status = read(OUT / "STATUS.json")
    summary = read(OUT / "SUMMARY.json")
    final = read(OUT / "FINAL_SEAL.json")
    review = read(OUT / "REVIEW_SEAL.json")
    head = read(OUT / "HEAD_REVIEW.json")
    anchors = read(OUT / "ANCHOR_RECALL.json")
    assert status["stage"] == "COMPLETE"
    assert set(summary["coordinates"]) == {"RAW_IQ", "DECHIRP"}
    assert anchors["complete_both_coordinates"]
    for receipt, key in ((status, "summary_sha256"), (final, "final_summary_sha256"),
                         (review, "summary_sha256"), (head, "summary_sha256")):
        assert receipt[key] == sha(OUT / "SUMMARY.json")
    for filename, key in (("REPORT.md", "report_sha256"), ("HEAD_REVIEW.json", "head_review_sha256"),
                          ("ANCHOR_RECALL.json", "anchor_recall_sha256")):
        assert review[key] == sha(OUT / filename)
    counts = {label: source_binding(receipt) for label, receipt in
              (("cpu_run", status), ("final_gate", final), ("anchor_review", anchors), ("head_review", head))}
    roster = np.asarray(summary["parent_ids"])
    assert len(roster) == len(np.unique(roster)) == 16
    assert status["parent_ids"] == summary["parent_ids"]
    assert summary["seed"] == 17
    models = {}
    for coord, row in summary["coordinates"].items():
        path = PRIVATE / coord / "step_3840.pt"
        assert sha(path) == row["checkpoint_sha256"]
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["steps"] == 3840 and state["coordinates"] == coord
        assert np.array_equal(state["parent_ids"], roster)
        model = SmoothVPNetV2(coord).cpu().eval()
        model.load_state_dict(state["model"], strict=True)
        assert 56728 <= real_parameters(model) <= 62700
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        for filename, digest in (("metrics.npz", row["private_metrics_sha256"]),
                                 ("failure_atoms.npz", row["failure_atoms"]["sha256"]),
                                 ("anchor_recall.npz", anchors["coordinates"][coord]["private_metrics_sha256"])):
            assert sha(PRIVATE / coord / filename) == digest
        with np.load(PRIVATE / coord / "metrics.npz", allow_pickle=False) as z:
            assert np.array_equal(z["parent_id"], roster)
            for name in z.files:
                if name != "parent_id":
                    assert z[name].shape == (16, 3) and np.isfinite(z[name]).all()
        with np.load(PRIVATE / coord / "failure_atoms.npz", allow_pickle=False) as z:
            name = next(key for key in z.files if key.endswith("_background_b"))
            assert z[name].dtype == np.complex128 and z[name].shape[-2:] == (1800, 8)
        assert row["initial_health"]["all_gradients_finite_and_connected"]
        assert row["initial_health"]["all_parameter_tensors_have_nonzero_gradient"]
        assert row["postfit_permutation_error"] < 1e-5 and row["algebra_error_max"] < 1e-12
        trace_path = PRIVATE / coord / "TRAJECTORY.json"
        assert sha(trace_path) == head["coordinates"][coord]["trajectory_statistics"]["trajectory_sha256"]
        assert len(read(trace_path)) == 3840
        models[coord] = {"effective_real_parameters": real_parameters(model), "steps": state["steps"],
                         "checkpoint_sha256": sha(path), "all_private_metrics_finite": True}
    p0 = read(WORK / "evidence/outputs/p0_dechirp_tail_2x2/SUMMARY.json")
    assert p0["six_cell_status"] == "COMPLETE" and p0["coordinates"] == summary["p0_coordinates"]
    assert sha(WORK / "evidence/outputs/p0_dechirp_tail_2x2/SUMMARY.json") == summary["p0_summary_sha256"]
    p0_runs = []
    for arm, seeds in p0["new_arms"].items():
        assert set(seeds) == {"17", "29", "41"}
        for seed, row in seeds.items():
            folder = WORK / "outputs/p0_dechirp_tail_2x2" / arm / seed
            assert row["status"] == "COMPLETE" and row["epoch"] == 256
            assert sha(folder / "epoch_256.pt") == row["checkpoint_sha256"]
            assert sha(folder / "dev_parent_metrics.npz") == row["parent_metrics_sha256"]
            p0_runs.append(f"{arm}/{seed}")
    assert len(p0_runs) == 6
    admission = WORK / summary["substitution_admission_snapshot"]
    assert sha(admission) == summary["substitution_summary_sha256"]
    assert sha(WORK / "evidence/outputs/p2_head_substitution_audit/SUMMARY.json") == summary["substitution_refinement_sha256"]
    pause = WORK / "evidence/outputs/p2_smoothvp_screen/USER_PAUSE.json"
    assert pause.exists() and (WORK / "outputs/p2_smoothvp_screen/PAUSE").exists()
    assert read(pause)["p2_automatic_start_authorized"] is False
    assert not list((WORK / "outputs/p2_smoothvp_screen").rglob("*.pt"))
    for name in ("vp_1103_p2_head_ablation.py", "vp_1103_p2_delay_only.py", "vp_1103_replay_truth.py",
                 "vp_1103_p2_smoothvp_v2.py", "vp_1103_p2_v2_preflight.py", "vp_1103_gate_preflight_v2.py",
                 "vp_1103_p2_v2_anchor_audit.py", "vp_1103_p2_v2_review.py"):
        ast.parse((WORK / "tools" / name).read_text(encoding="utf8"))
    result = {**provenance(__file__), "started_utc": started, "ended_utc": now(), "status": "PASS",
              "scope": "Local source/receipt/checkpoint verification; no optimizer or confirmation evaluation",
              "source_binding_counts": counts, "coordinates": models, "p0_final_runs_verified": p0_runs,
              "p0_coordinates": p0["coordinates"], "review_route": summary["route"],
              "formal_gpu_training_paused": True, "formal_gpu_checkpoints_found": 0,
              "summary_sha256": sha(OUT / "SUMMARY.json"), "report_sha256": sha(OUT / "REPORT.md")}
    atomic_json(OUT / "VERIFICATION.json", result)
    print({"verification": "PASS", "route": summary["route"], "coordinates": list(models),
           "p0_runs": len(p0_runs), "formal_gpu_training_paused": True})


if __name__ == "__main__":
    run()
