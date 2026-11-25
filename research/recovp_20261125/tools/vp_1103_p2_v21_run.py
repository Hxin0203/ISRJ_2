"""Release only the v2.1 GPU queue after all four new CPU gates pass."""
import os
import time
import traceback
import numpy as np
from vp_1103_common import WORK, atomic_json, now, provenance, read, sha, validate_role
from tools.durable_io import ProcessLock, atomic_write
from vp_1103_snr_jsr import DEFINITION

OUT=WORK/"evidence/outputs/p2_smoothvp_v21_screen"
PRIVATE=WORK/"outputs/p2_smoothvp_v21_screen"
PREFLIGHT=WORK/"evidence/outputs/p2_smoothvp_v21_preflight"

def failure_diagnosis(seed,arm,config,data,error):
    failure={"seed":seed,"arm":arm,"utc":now(),"error":repr(error),"traceback":traceback.format_exc(),
             "scope":"Failure retained; any diagnostic scores below use the last valid epoch, not epoch256"}
    try:
        import torch
        from vp_1103_p2_smoothvp_v21 import SmoothVPNetV21
        from vp_1103_p0 import evaluate
        from vp_1103_snr_jsr import frame_axes,cases,summarize
        from vp_1103_p2_evaluate import failure_details
        from vp_1103_v21_mechanisms import mechanism_metrics
        folder=PRIVATE/arm/str(seed)
        cp=folder/"latest.pt"
        if cp.exists():
            if sha(cp)!=read(cp.with_suffix(".json"))["checkpoint_sha256"]:raise ValueError("Last checkpoint hash mismatch")
            state=torch.load(cp,map_location="cpu",weights_only=False)
            model=SmoothVPNetV21(config["extractor"],fixed_lambda=arm=="M3-FixedLambda").cpu().eval()
            model.load_state_dict(state["model"])
            metrics=evaluate(model,data)
            extra,algebra,permutation=mechanism_metrics(model,data);metrics.update(extra)
            axes=frame_axes(data);worst=cases(metrics,data["parent_id"],axes)
            failure.update(last_valid_epoch=state["epoch"],checkpoint_sha256=sha(cp),metrics=summarize(metrics,axes),
                worst_cases=worst,atoms=failure_details(model,data,worst,folder),algebra_error=algebra,permutation_error=permutation)
    except Exception as diagnosis_error:
        failure["diagnosis_error"]=repr(diagnosis_error)
    atomic_json(OUT/f"FAILED_{arm}_{seed}.json",failure)
    return failure

def run():
    started=now()
    status={**provenance(__file__,*[WORK/"tools"/name for name in ("vp_1103_p2_v21_train.py","vp_1103_v21_score.py",
            "vp_1103_p2_smoothvp_v21.py","vp_1103_v21_extractors.py","vp_1103_snr_jsr.py",
            "vp_1103_v21_mechanisms.py","vp_1103_replay_truth.py","benchmark_latency_unified.py")]),
            "stage":"WAITING_CPU_GATE","started_utc":started,"ended_utc":None,"pid":os.getpid(),
            "seeds":[17,29,41],"micro_batch":8,"epochs":256,"confirmation_unlocked":False}
    def update(**fields):
        status.update(fields,updated_utc=now());atomic_json(OUT/"STATUS.json",status)
    update()
    atomic_json(OUT/"AUTHORIZATION.json",{"recorded_utc":now(),"automatic_start_authorized":True,
         "condition":"All G-v21-1..4 PASS on fixed TRAIN16 RAW_IQ CPU preflight",
         "scope":"seed17 fixed256 DEV screen; conditional29/41 and fixed-precision ablation; no independent confirmation",
         "historical_v2_pause_retained":True})
    while True:
        try:
            state=read(PREFLIGHT/"STATUS.json") if (PREFLIGHT/"STATUS.json").exists() else {}
        except PermissionError:
            time.sleep(1);continue
        if state.get("stage")=="FAILED":
            update(stage="BLOCKED_CPU_FAILURE",ended_utc=now());return
        if state.get("stage")=="COMPLETE":break
        if (PRIVATE/"STOP").exists():update(stage="STOPPED_BY_USER",ended_utc=now());return
        time.sleep(30)
    gate=read(PREFLIGHT/"SUMMARY.json")
    if sha(PREFLIGHT/"SUMMARY.json")!=state["summary_sha256"]:raise ValueError("CPU receipt changed")
    if gate["route"]!="GO_FORMAL_GPU_TRAIN" or any(v!="PASS" for v in gate["checks"].values()):
        atomic_write(PRIVATE/"PAUSE",lambda f:f.write(b"CPU admission failed; formal GPU training remains paused.\n"))
        atomic_json(OUT/"SUMMARY.json",{"route":"NO_GO_PREFLIGHT","preflight_checks":gate["checks"],
                    "formal_training_started":False,"confirmation_unlocked":False})
        atomic_write(OUT/"REPORT.md",lambda f:f.write(("# v2.1 正式训练接续\n\n"+DEFINITION+"\n\n"+
                    "CPU 预检未全部通过；正式 GPU 训练未启动，暂停保留。完整 SNR/JSR 指标见 CPU 预检报告。\n").encode("utf8")))
        update(stage="PAUSED_PREFLIGHT_FAIL",ended_utc=now(),preflight_summary_sha256=state["summary_sha256"]);return
    selection=WORK/"evidence/outputs/p2_v21_anchor_audit/SUMMARY.json"
    if sha(selection)!=gate["selection_summary_sha256"]:raise ValueError("Selected extractor receipt changed")
    for name,digest in state["source_sha256"].items():
        if sha(WORK/name)!=digest:raise ValueError(f"CPU source binding changed: {name}")
    p0=read(WORK/"evidence/outputs/p0_dechirp_tail_2x2/SUMMARY.json")
    if p0["coordinates"]!="RAW_IQ" or p0["six_cell_status"]!="COMPLETE":raise ValueError("P0 coordinate not sealed")
    train_path,train_binding=validate_role("train")
    dev_path,dev_binding=validate_role("dev")
    import torch
    from paper2.train_1103 import load_arrays
    from vp_1103_p2_v21_train import train
    from vp_1103_v21_score import report
    if not torch.cuda.is_available():raise RuntimeError("GPU unavailable; no CPU training fallback")
    pause=PRIVATE/"PAUSE"
    if pause.exists():pause.unlink()
    atomic_json(OUT/"RELEASE.json",{"utc":now(),"preflight_summary_sha256":state["summary_sha256"],
         "checks":gate["checks"],"independent_confirmation_authorized":False,"historical_v2_pause_retained":True})
    config={"extractor":gate["extractor"],"coordinates":"RAW_IQ","seeds":[17,29,41],"epochs":256,"micro_batch":8,
            "train":train_binding,"dev":dev_binding,"preflight_sha256":state["summary_sha256"],
            "source_sha256":status["source_sha256"]}
    path=PRIVATE/"CONFIG.json"
    if path.exists():
        if read(path)!=config:raise ValueError("Refusing to alter an existing formal training protocol")
    else:atomic_json(path,config)
    with np.load(dev_path,allow_pickle=False) as archive:
        dev={key:archive[key] for key in archive.files}
    rows={"M3-SmoothVP-v21":{},"M3-FixedLambda":{}}
    def execute(seed,arm):
        try:
            return train(seed,"RAW_IQ",train_path,dev,config,update,arm=arm)
        except Exception as error:
            failure=failure_diagnosis(seed,arm,config,dev,error)
            report(rows,"STOPPED_EXCEPTION",{"failed_seed":failure})
            update(stage="STOPPED_EXCEPTION",ended_utc=now(),failed_seed=seed,failed_arm=arm,error=repr(error))
            return None
    for seed in (17,29,41):
        row=execute(seed,"M3-SmoothVP-v21")
        if row is None:return
        rows["M3-SmoothVP-v21"][str(seed)]=row
        report(rows,"DEV_SCREEN_IN_PROGRESS")
        if seed==17 and not row["continue_other_seeds"]:
            report(rows,"STOPPED_SEED17",{"circuit_breaker":row["circuit_breaker"]})
            update(stage="STOPPED_SEED17",ended_utc=now());return
    fixed=execute(17,"M3-FixedLambda")
    if fixed is None:return
    rows["M3-FixedLambda"]["17"]=fixed
    smooth=rows["M3-SmoothVP-v21"]["17"]["metrics"]["global"]
    f=fixed["metrics"]["global"]
    clear=(f["no_jammer_pooled_nmse"]["mean"]>=2*smooth["no_jammer_pooled_nmse"]["mean"] or
           f["drift_over_target_energy"]["mean"]>=2*smooth["drift_over_target_energy"]["mean"])
    if clear:
        for seed in (29,41):
            row=execute(seed,"M3-FixedLambda")
            if row is None:return
            rows["M3-FixedLambda"][str(seed)]=row
    update(stage="LATENCY_BENCHMARK")
    from benchmark_latency_unified import run_v21
    run_v21()
    report(rows,"AWAIT_GATE_P2_REVIEW",{"latency":"evidence/outputs/p2_smoothvp_v21_latency/SUMMARY.json"})
    update(stage="COMPLETE",ended_utc=now(),route="AWAIT_GATE_P2_REVIEW",summary_sha256=sha(OUT/"SUMMARY.json"))

if __name__=="__main__":
    with ProcessLock(PRIVATE/"CONTROLLER.lock"):
        try:run()
        except Exception as exc:
            atomic_json(OUT/"FAILURE.json",{"utc":now(),"error":repr(exc),"traceback":traceback.format_exc()})
            old=read(OUT/"STATUS.json") if (OUT/"STATUS.json").exists() else {}
            atomic_json(OUT/"STATUS.json",{**old,"stage":"FAILED","ended_utc":now()})
            raise
