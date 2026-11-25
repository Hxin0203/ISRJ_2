"""Final CPU-stage source/data/math receipts, preserving research failures."""
import ast
import numpy as np
import torch
from vp_1103_common import WORK,atomic_json,now,provenance,read,sha
from vp_1103_p2_preflight import load,subset
from vp_1103_recovp import ReCoVPNet
from vp_1103_recovp_units import run as random_math
from vp_1103_recovp_failures import failure_heads
from vp_1103_p2_smoothvp import real_parameters
from vp_1103_snr_jsr import CONFIG
from vp_1103_snr_jsr import frame_axes,summarize
from vp_1103_recovp_drift import three_layers

def source_bindings(receipt):
    for name,digest in receipt["source_sha256"].items():
        if sha(WORK/name)!=digest:raise ValueError(f"Source changed: {name}")
    return len(receipt["source_sha256"])

def run():
    started=now();torch.set_num_threads(2)
    out=WORK/"evidence/outputs/recovp_gate_summary"
    counts={}
    for label,folder in (("Exp0","exp0_anchor_audit"),("Exp1","exp1_recovp_base_preflight"),("Exp2a","exp2a_softshare_audit"),("references","p0_snr_jsr_reference")):
        path=WORK/"evidence/outputs"/folder;status=read(path/"STATUS.json")
        assert status["stage"]=="COMPLETE" and status["summary_sha256"]==sha(path/"SUMMARY.json")
        counts[label]=source_bindings(status)
    selection=read(WORK/"evidence/outputs/exp0_anchor_audit/SUMMARY.json")
    assert len(selection["methods"])==21 and selection["all_complex_scaling_checks_pass"]
    assert all(row["complex_scale_anchor_disagreements"]==0 for row in selection["methods"].values())
    assert selection["protocol"]["sha256"]==sha(CONFIG)
    for name,row in selection["methods"].items():
        assert sha(WORK/"outputs/exp0_anchor_audit"/f"{name}.npz")==row["private_metrics_sha256"]
    cpu_path=WORK/"evidence/outputs/exp1_recovp_base_preflight"
    cpu,state=read(cpu_path/"SUMMARY.json"),read(cpu_path/"STATUS.json")
    private=WORK/"outputs/exp1_recovp_base_preflight";checkpoint=private/"step_3840.pt"
    assert sha(checkpoint)==cpu["checkpoint_sha256"]
    cp=torch.load(checkpoint,map_location="cpu",weights_only=False)
    assert cp["steps"]==3840 and cp["parent_ids"]==state["parent_ids"] and len(set(cp["parent_ids"]))==16
    assert cp["extractor_config"]==cpu["extractor"]==selection["locked_config"]
    assert len(read(private/"TRAJECTORY.json"))==3840
    assert sha(private/"metrics.npz")==cpu["private_metrics_sha256"]
    assert cpu["initial_math"]["status"]==cpu["final_math"]["status"]=="PASS"
    assert cpu["initial_health"]["all_gradients_finite_and_connected"] and cpu["initial_health"]["all_parameter_tensors_have_nonzero_gradient"]
    assert cpu["operator_identity_error"]<1e-10
    with np.load(private/"metrics.npz",allow_pickle=False) as z:
        assert np.array_equal(z["parent_id"],cp["parent_ids"])
        for key in z.files:
            if key!="parent_id":assert z[key].shape==(16,3) and np.isfinite(z[key]).all()
    model=ReCoVPNet(selection["locked_config"]).cpu().eval();model.load_state_dict(cp["model"])
    assert 56728<=real_parameters(model)<=62700 and all(not p.is_complex() for p in model.parameters())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    random_result=random_math(selection["locked_config"],checkpoint)
    assert random_result["status"]=="PASS"
    data,binding=load("train");ids=np.sort(np.array(cp["parent_ids"]))
    small=subset(data,np.searchsorted(data["parent_id"],ids))
    heads=failure_heads(model,small,cpu["worst_cases"],private)
    atomic_json(cpu_path/"ESTIMATED_HEADS.json",heads)
    layer_metrics,layer_errors=three_layers(model,small)
    atomic_json(cpu_path/"THREE_LAYER_TRAIN16.json",{"scope":"Same TRAIN16 final checkpoint; not a DEV result",
        "checkpoint_sha256":sha(checkpoint),"training_updates":0,
        "layers":{name:summarize(values,frame_axes(small)) for name,values in layer_metrics.items()},"identity_errors":layer_errors})
    share=read(WORK/"evidence/outputs/exp2a_softshare_audit/SUMMARY.json")
    assert share["training_updates"]==0 and share["mathematical_checks"]["status"]=="PASS"
    for name,row in share["methods"].items():
        assert sha(WORK/"outputs/exp2a_softshare_audit"/f"{name}.npz")==row["private_metrics_sha256"]
    boundary=read(out/"BOUNDARY_NORM_TEST.json")
    assert boundary["status"]=="PASS"
    assert all(row[key]<1e-12 for row in boundary["curves"] for key in ("row_l2","column_l2","active_block_frobenius"))
    assert read(out/"CVAR_UNIT_TEST.json")["status"]=="PASS"
    assert read(out/"NOEQUIV_CONTROL_TEST.json")["status"]=="PASS"
    assert read(out/"THREE_LAYER_ALGEBRA_TEST.json")["status"]=="PASS"
    native=read(out/"NATIVE_POWER_CONVENTION.json")
    assert native["metadata_check"]["jsr_active_power_max_abs_db_error"]<1e-4
    assert max(abs(np.array(native["whole_window_minus_active_snr_db_min_max"])-10*np.log10(600/1800)))<1e-11
    controller=read(out/"STATUS.json")
    if cpu["route"]=="NO_GO_REVIEW":
        assert controller["stage"]=="COMPLETE_NO_GO"
        assert not list((WORK/"outputs/recovp_formal").rglob("*.pt"))
        assert (WORK/"outputs/recovp_formal/PAUSE").exists()
    counts["controller"]=source_bindings(controller)
    for folder in ("exp0_anchor_audit","exp1_recovp_base_preflight","exp2a_softshare_audit","p0_snr_jsr_reference","recovp_gate_summary"):
        for path in (WORK/"evidence/outputs"/folder).glob("*.md"):
            assert "jnr" not in path.read_text(encoding="utf8").lower(),path
    for pattern in ("*recovp*.py","vp_1103_exp0*.py","vp_1103_exp1*.py","vp_1103_exp2a*.py"):
        for path in (WORK/"tools").glob(pattern):ast.parse(path.read_text(encoding="utf8"))
    result={**provenance(__file__),"started_utc":started,"ended_utc":now(),"integrity_status":"PASS",
            "research_route":cpu["route"],"source_binding_counts":counts,"train_data":binding,
            "effective_real_parameters":real_parameters(model),"random_math":random_result["status"],
            "formal_status":controller["stage"],"formal_checkpoints":len(list((WORK/"outputs/recovp_formal").rglob("*.pt"))),
            "independent_confirmation_unlocked":False,"cpu_summary_sha256":sha(cpu_path/"SUMMARY.json"),
            "cpu_report_sha256":sha(cpu_path/"REPORT.md"),"gate_summary_sha256":sha(out/"SUMMARY.json") if (out/"SUMMARY.json").exists() else None,
            "estimated_heads":heads}
    atomic_json(out/"VERIFICATION.json",result)
    print({"integrity":"PASS","research_route":cpu["route"],"formal_status":controller["stage"]})

if __name__=="__main__":run()
