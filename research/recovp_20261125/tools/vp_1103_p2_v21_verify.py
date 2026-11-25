"""Source/data receipt and fixed-budget verification; never releases training."""
import argparse
import numpy as np
import torch
from vp_1103_common import WORK, atomic_json, now, provenance, read, sha
from vp_1103_p2_preflight import load
from vp_1103_p2_smoothvp_v21 import SmoothVPNetV21
from vp_1103_p2_smoothvp import real_parameters
from vp_1103_v21_extractors import extract_numpy, extract_torch
from vp_1103_snr_jsr import CONFIG

def bindings(receipt):
    for path,digest in receipt["source_sha256"].items():
        if sha(WORK/path)!=digest:raise ValueError(f"Source changed: {path}")
    return len(receipt["source_sha256"])

def run(anchors_only=False):
    torch.set_num_threads(2)
    started=now()
    out=WORK/"evidence/outputs/p2_v21_anchor_audit"
    state,summary=read(out/"STATUS.json"),read(out/"SUMMARY.json")
    assert state["stage"]=="COMPLETE" and sha(out/"SUMMARY.json")==state["summary_sha256"]
    count=bindings(state)
    assert len(summary["methods"])==21 and summary["training_updates"]==0
    assert summary["protocol"]["sha256"]==sha(CONFIG)
    for name,row in summary["methods"].items():
        assert sha(WORK/"outputs/p2_v21_anchor_audit"/f"{name}.npz")==row["private_metrics_sha256"]
    data,binding=load("train")
    model=SmoothVPNetV21(summary["locked_config"]).cpu().eval()
    tested=0
    # Independent deterministic coverage across the entire roster, not a hand-picked waveform.
    for parent in np.arange(0,512,8):
        for family in range(3):
            wave=data["j"][parent,family].astype(np.complex128)+data["epsilon"][parent,family].astype(np.complex128)
            expected=extract_numpy(wave,summary["locked_config"])
            actual=extract_torch(torch.from_numpy(wave),summary["locked_config"],model.pulse_spectrum).numpy()
            assert np.array_equal(expected,actual)
            tested+=len(wave)
    result={**provenance(__file__),"started_utc":started,"ended_utc":now(),"status":"PASS",
            "selection_source_bindings":count,"selection_summary_sha256":sha(out/"SUMMARY.json"),
            "train_data":binding,"selected_numpy_torch_exact_pulses":tested,
            "selected_parity_scope":"Every eighth TRAIN parent, all three families and eight background pulses; 64 parents",
            "BEST_EXTRACTOR":summary["BEST_EXTRACTOR"],"K_LOCKED":summary["K_LOCKED"]}
    atomic_json(out/"VERIFICATION.json",result)
    if anchors_only:
        print({"anchor_verification":"PASS","parity_pulses":tested});return
    out=WORK/"evidence/outputs/p2_smoothvp_v21_preflight"
    private=WORK/"outputs/p2_smoothvp_v21_preflight"
    state,cpu=read(out/"STATUS.json"),read(out/"SUMMARY.json")
    assert state["stage"]=="COMPLETE" and sha(out/"SUMMARY.json")==state["summary_sha256"]
    result["cpu_source_bindings"]=bindings(state)
    cp=private/"step_3840.pt"
    assert sha(cp)==cpu["checkpoint_sha256"]
    checkpoint=torch.load(cp,map_location="cpu",weights_only=False)
    assert checkpoint["steps"]==3840 and checkpoint["coordinates"]=="RAW_IQ"
    assert len(set(checkpoint["parent_ids"]))==16 and checkpoint["parent_ids"]==state["parent_ids"]
    assert checkpoint["extractor_config"]==cpu["extractor"]==summary["locked_config"]
    model.load_state_dict(checkpoint["model"],strict=True)
    assert 56728<=real_parameters(model)<=62700
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert len(read(private/"TRAJECTORY.json"))==3840
    assert sha(private/"metrics.npz")==cpu["private_metrics_sha256"]
    with np.load(private/"metrics.npz",allow_pickle=False) as z:
        for key in z.files:
            if key!="parent_id":assert z[key].shape==(16,3) and np.isfinite(z[key]).all()
    assert sha(WORK/cpu["failure_atoms"]["path"])==cpu["failure_atoms"]["sha256"]
    assert cpu["permutation_error"]<1e-5 and cpu["operator_identity_error"]<1e-10
    assert cpu["initial_health"]["all_gradients_finite_and_connected"]
    assert cpu["initial_health"]["all_parameter_tensors_have_nonzero_gradient"]
    controller=read(WORK/"evidence/outputs/p2_smoothvp_v21_screen/STATUS.json")
    if cpu["route"]=="NO_GO_REVIEW":
        assert controller["stage"]=="PAUSED_PREFLIGHT_FAIL"
        assert (WORK/"outputs/p2_smoothvp_v21_screen/PAUSE").exists()
        assert not list((WORK/"outputs/p2_smoothvp_v21_screen").rglob("*.pt"))
    assert cpu["protocol_sha256"]==sha(CONFIG)
    for directory in ("snr_jsr_protocol","snr_jsr_baselines","p2_v21_anchor_audit","p2_smoothvp_v21_preflight"):
        for path in (WORK/"evidence/outputs"/directory).glob("*.md"):
            if "jnr" in path.read_text(encoding="utf8").lower():raise ValueError(f"Obsolete reporting axis: {path}")
    result.update(ended_utc=now(),cpu_summary_sha256=sha(out/"SUMMARY.json"),cpu_report_sha256=sha(out/"REPORT.md"),
                  effective_real_parameters=real_parameters(model),cpu_route=cpu["route"],controller_stage=controller["stage"],
                  new_reports_use_snr_jsr_only=True,formal_checkpoints_found=len(list((WORK/"outputs/p2_smoothvp_v21_screen").rglob("*.pt"))))
    atomic_json(out/"VERIFICATION.json",result)
    print({"verification":"PASS","route":cpu["route"],"controller":controller["stage"]})

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--anchors-only",action="store_true")
    run(parser.parse_args().anchors_only)
