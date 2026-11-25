"""Conditional ReCoVP queue; hard stop at CPU failure or Gate-ReCoVP."""
import os
import time
import traceback
import numpy as np
from vp_1103_common import WORK,atomic_json,now,provenance,read,sha,validate_role
from vp_1103_snr_jsr import DEFINITION
from tools.durable_io import atomic_write,ProcessLock

OUT=WORK/"evidence/outputs/recovp_gate_summary"
PRIVATE=WORK/"outputs/recovp_formal"
PREFLIGHT=WORK/"evidence/outputs/exp1_recovp_base_preflight"

def success_summary(route):
    paths={"formal":"recovp_formal/SUMMARY.json","risk":"recovp_cvar_2x2/SUMMARY.json","latency":"recovp_latency/SUMMARY.json",
           "complex_scaling":"recovp_formal/COMPLEX_SCALING.json","piecewise_stress":"recovp_formal/PIECEWISE_STRESS.json"}
    artifacts={name:read(WORK/"evidence/outputs"/path) for name,path in paths.items() if (WORK/"evidence/outputs"/path).exists()}
    result={"route":route,"artifacts":artifacts,"bindings":{name:sha(WORK/"evidence/outputs"/path) for name,path in paths.items() if (WORK/"evidence/outputs"/path).exists()},
            "confirmation_unlocked":False,"scope":"Only TRAIN/DEV; missing conditional cells remain explicitly untrained"}
    atomic_json(OUT/"SUMMARY.json",result)
    report=WORK/"evidence/outputs/recovp_formal/REPORT.md"
    body=report.read_text(encoding="utf8") if report.exists() else "# Gate-ReCoVP\n\n"+DEFINITION+"\n"
    body += f"\n## 总闸门\n\n路由 **{route}**，独立确认未解锁。\n"
    for name,path in paths.items():
        body+=f"\n- {name}: {'COMPLETE' if name in artifacts else 'NOT_RUN_GATE'} — {path}\n"
    atomic_write(OUT/"REPORT.md",lambda f:f.write(body.encode("utf8")))

def failure_diagnosis(seed,arm,config,data,error):
    failure={"seed":seed,"arm":arm,"utc":now(),"error":repr(error),"traceback":traceback.format_exc(),
             "scope":"Last valid checkpoint diagnosis; never substituted as a fixed epoch256 result"}
    try:
        import torch
        from vp_1103_recovp import ReCoVPNet
        from vp_1103_recovp_ablation import make_recovp
        from vp_1103_p0 import evaluate
        from vp_1103_snr_jsr import frame_axes,cases,summarize
        from vp_1103_p2_evaluate import failure_details
        cp=PRIVATE/arm/str(seed)/"latest.pt"
        if cp.exists():
            if sha(cp)!=read(cp.with_suffix(".json"))["checkpoint_sha256"]:raise ValueError("Last checkpoint changed")
            state=torch.load(cp,map_location="cpu",weights_only=False)
            model=make_recovp(config["extractor"],arm).cpu().eval()
            model.load_state_dict(state["model"])
            metrics=evaluate(model,data);axes=frame_axes(data);worst=cases(metrics,data["parent_id"],axes)
            failure.update(last_valid_epoch=state["epoch"],checkpoint_sha256=sha(cp),metrics=summarize(metrics,axes),worst_cases=worst,
                atoms=failure_details(model,data,worst,PRIVATE/arm/str(seed)))
    except Exception as diagnosis_error:failure["diagnosis_error"]=repr(diagnosis_error)
    atomic_json(OUT/f"FAILED_{arm}_{seed}.json",failure)
    return failure

def blocked_summary(reason,state=None):
    baseline=read(WORK/"evidence/outputs/p0_snr_jsr_reference/SUMMARY.json")
    cpu=read(PREFLIGHT/"SUMMARY.json") if (PREFLIGHT/"SUMMARY.json").exists() else {}
    audit=read(WORK/"evidence/outputs/exp0_anchor_audit/SUMMARY.json")
    share_path=WORK/"evidence/outputs/exp2a_softshare_audit/SUMMARY.json"
    share=read(share_path) if share_path.exists() else {}
    result={"route":reason,"coordinates":"RAW_IQ","independent_confirmation_unlocked":False,
        "references_dev":baseline["methods"],"train16_preflight":cpu,"exp0_selection":{"BEST_EXTRACTOR":audit["BEST_EXTRACTOR"],"K_LOCKED":audit["K_LOCKED"]},
        "exp2a_route":share.get("route","PENDING"),"mathematical_softshare_checks":share.get("mathematical_checks"),
        "phase_status":{"Exp0":"COMPLETE","Exp1_CPU":state.get("stage") if state else "UNKNOWN","Exp1_GPU":"NOT_STARTED_CPU_GATE",
            "Exp2a":"COMPLETE" if share else "PENDING","Exp2b":"BLOCKED_EXP1_GATE","Exp3":"BLOCKED_EXP1_GATE","latency":"NOT_RUN_NO_FORMAL_CHECKPOINT"},
        "formal_seed_status":{arm:{str(seed):"NOT_STARTED_CPU_GATE" for seed in (17,29,41)} for arm in ("ReCoVP-Base","ReCoVP-SoftShare")},
        "no_equiv_seed17":"NOT_STARTED_CPU_GATE","formal_checkpoints_found":len(list(PRIVATE.rglob("*.pt"))),
        "bindings":{"Exp0":sha(WORK/"evidence/outputs/exp0_anchor_audit/SUMMARY.json"),"Exp1":sha(PREFLIGHT/"SUMMARY.json") if cpu else None,
                    "Exp2a":sha(share_path) if share else None,"reference":sha(WORK/"evidence/outputs/p0_snr_jsr_reference/SUMMARY.json")}}
    atomic_json(OUT/"SUMMARY.json",result)
    lines=["# Gate-ReCoVP","",DEFINITION,"",f"路由 **{reason}**。本轮没有新神经模型的 DEV 成绩，不以 TRAIN16 替代 DEV。",
           "正式 GPU、NoEquiv、SoftShare、CVaR 与完整神经模型延迟均未启动。独立确认始终锁定。","",
           "## 已封存 DEV 参照：三种子均值","","| 模型 | 虚警 | Pd | 损伤 |","|---|---:|---:|---:|"]
    for name,row in baseline["methods"].items():
        m=row["seed_average"]["global"]
        lines.append(f"| {name} | {m['frame_pfa_jammer']['mean']:.3%} | {m['pd']['mean']:.3%} | {m['damage']['mean']:.6g} |")
    if cpu:
        lines += ["","## TRAIN16：固定 seed17、3840 步","","| 条件 | 状态 |","|---|---|"]
        lines += [f"| {name} | {value} |" for name,value in cpu["checks"].items()]
        lines += ["","| 分层 | 虚警 | Oracle-β 残差 | 盲残差 | 漂移 | 泄漏 | 无扰 NMSE |","|---|---:|---:|---:|---:|---:|---:|"]
        for label in ("global","jsr/>=15","repeat/jsr/>=15","direct/all","cyclic/all"):
            m=cpu["levels"][label]
            keys=("frame_pfa_jammer","learned_b_oracle_beta_relative_residual","blind_j_relative_residual","drift_over_target_energy","leak_over_target_energy","no_jammer_pooled_nmse")
            lines.append(f"| {label} | "+" | ".join(f"{m[key]['mean']:.7g}" for key in keys)+" |")
    lines += ["",f"Exp2a 路由 **{result['exp2a_route']}**，其 Oracle 字典与未训练节点头的结果不解锁绕过 Exp1 的 GPU 作业。",
        "","完整分层、同 parent 区间、Top-5 与数学收据分别见：",
        "- [Exp0](../exp0_anchor_audit/REPORT.md)","- [Exp1 CPU](../exp1_recovp_base_preflight/REPORT.md)",
        "- [Exp2a](../exp2a_softshare_audit/REPORT.md)","- [三基线参照](../p0_snr_jsr_reference/REPORT.md)"]
    atomic_write(OUT/"REPORT.md",lambda f:f.write(("\n".join(lines)+"\n").encode("utf8")))

def run():
    started=now()
    status={**provenance(__file__,*[WORK/"tools"/name for name in ("vp_1103_recovp.py","vp_1103_recovp_train.py","vp_1103_recovp_score.py",
          "vp_1103_recovp_drift.py","vp_1103_recovp_robustness.py","vp_1103_recovp_extractors.py",
          "vp_1103_recovp_units.py","vp_1103_recovp_ablation.py","vp_1103_recovp_failures.py","vp_1103_recovp_cvar.py",
          "vp_1103_recovp_exp3.py","vp_1103_recovp_stress.py","vp_1103_snr_jsr.py","vp_1103_p2_smoothvp.py","vp_1103_p2_smoothvp_v2.py",
          "vp_1103_p0.py","vp_1103_p2_evaluate.py","vp_1103_p2_preflight.py","vp_1103_replay_truth.py","vp_1103_geometry.py",
          "vp_1103_references.py","benchmark_latency_unified.py")]),
          "stage":"WAITING_EXP1_CPU_GATE","started_utc":started,"ended_utc":None,"pid":os.getpid(),"coordinates":"RAW_IQ",
          "training_seeds":[17,29,41],"micro_batch":8,"confirmation_unlocked":False}
    def update(**fields):
        status.update(fields,updated_utc=now());atomic_json(OUT/"STATUS.json",status)
    update()
    atomic_json(OUT/"AUTHORIZATION.json",{"utc":now(),"automatic_exp1_gpu_after_all_cpu_checks":True,
        "automatic_seed_extensions":"Only the specified seed17 DEV gates", "independent_confirmation":False})
    while True:
        try:state=read(PREFLIGHT/"STATUS.json") if (PREFLIGHT/"STATUS.json").exists() else {}
        except PermissionError:time.sleep(1);continue
        if state.get("stage")=="FAILED":
            blocked_summary("BLOCKED_EXP1_CPU_EXECUTION",state);update(stage="BLOCKED",ended_utc=now());return
        if state.get("stage")=="COMPLETE":break
        if (PRIVATE/"STOP").exists():update(stage="STOPPED_BY_USER",ended_utc=now());return
        time.sleep(30)
    gate=read(PREFLIGHT/"SUMMARY.json")
    if sha(PREFLIGHT/"SUMMARY.json")!=state["summary_sha256"]:raise ValueError("CPU summary changed")
    if gate["route"]!="GO_EXP1_GPU" or any(value!="PASS" for value in gate["checks"].values()):
        atomic_write(PRIVATE/"PAUSE",lambda f:f.write(b"Exp1 CPU preflight gate failed.\n"))
        blocked_summary("NO_GO_EXP1_PREFLIGHT",state)
        update(stage="COMPLETE_NO_GO",ended_utc=now(),route="NO_GO_EXP1_PREFLIGHT",summary_sha256=sha(OUT/"SUMMARY.json"));return
    for name,digest in state["source_sha256"].items():
        if sha(WORK/name)!=digest:raise ValueError(f"CPU source changed: {name}")
    from vp_1103_recovp_units import run as random_unit_check
    random_math=random_unit_check(gate["extractor"],WORK/"outputs/exp1_recovp_base_preflight/step_3840.pt")
    if random_math["status"]!="PASS":
        atomic_write(PRIVATE/"PAUSE",lambda f:f.write(b"Strict random-input mathematical test failed.\n"))
        blocked_summary("NO_GO_RANDOM_MATH",state);update(stage="COMPLETE_NO_GO",ended_utc=now());return
    if read(WORK/"evidence/outputs/p0_dechirp_tail_2x2/SUMMARY.json")["coordinates"]!="RAW_IQ":raise ValueError("Coordinate mismatch")
    train_path,training_binding=validate_role("train");dev_path,dev_binding=validate_role("dev")
    import torch
    from vp_1103_recovp_train import train
    from vp_1103_recovp_score import report
    if not torch.cuda.is_available():raise RuntimeError("GPU unavailable; no CPU fallback")
    pause=PRIVATE/"PAUSE"
    if pause.exists():pause.unlink()
    config={"extractor":gate["extractor"],"train":training_binding,"dev":dev_binding,"coordinates":"RAW_IQ","epochs":256,
            "seeds":[17,29,41],"micro_batch":8,"preflight_sha256":state["summary_sha256"],"source_sha256":status["source_sha256"]}
    path=PRIVATE/"CONFIG.json"
    if path.exists():
        if read(path)!=config:raise ValueError("Existing protocol differs")
    else:atomic_json(path,config)
    atomic_json(OUT/"RELEASE.json",{"utc":now(),"checks":gate["checks"],"preflight_sha256":state["summary_sha256"],"confirmation_unlocked":False})
    with np.load(dev_path,allow_pickle=False) as z:dev={key:z[key] for key in z.files}
    rows={"ReCoVP-Base":{},"ReCoVP-NoEquiv":{},"ReCoVP-SoftShare":{}}
    def execute(seed,arm):
        try:return train(seed,"RAW_IQ",train_path,dev,config,update,arm=arm)
        except Exception as error:
            failure=failure_diagnosis(seed,arm,config,dev,error)
            report(rows,"STOPPED_EXCEPTION",{"failed_seed":failure});success_summary("STOPPED_EXCEPTION")
            update(stage="STOPPED_EXCEPTION",ended_utc=now(),failed_seed=seed,failed_arm=arm);return None
    for seed in (17,29,41):
        row=execute(seed,"ReCoVP-Base")
        if row is None:return
        rows["ReCoVP-Base"][str(seed)]=row
        report(rows,"EXP1_DEV_SCREEN")
        if seed==17 and not row["continue_other_seeds"]:
            report(rows,"STOPPED_BASE_SEED17");success_summary("STOPPED_BASE_SEED17");update(stage="STOPPED_BASE_SEED17",ended_utc=now());return
    row=execute(17,"ReCoVP-NoEquiv")
    if row is None:return
    rows["ReCoVP-NoEquiv"]["17"]=row
    from vp_1103_recovp_robustness import scaling_audit
    scaling_audit(config,dev)
    share_status=read(WORK/"evidence/outputs/exp2a_softshare_audit/SUMMARY.json")
    if share_status["route"]=="GO_EXP2B_CONDITIONAL_ON_EXP1":
        row=execute(17,"ReCoVP-SoftShare")
        if row is None:return
        rows["ReCoVP-SoftShare"]["17"]=row
        if row["continue_other_seeds"]:
            for seed in (29,41):
                row=execute(seed,"ReCoVP-SoftShare")
                if row is None:return
                rows["ReCoVP-SoftShare"][str(seed)]=row
    report(rows,"EXP1_EXP2_COMPLETE")
    from vp_1103_recovp_exp3 import run as risk_and_latency
    risk_and_latency(config,dev,rows,update)
    success_summary("AWAIT_GATE_RECOVP_REVIEW")
    update(stage="COMPLETE",ended_utc=now(),route="AWAIT_GATE_RECOVP_REVIEW",summary_sha256=sha(OUT/"SUMMARY.json"))

if __name__=="__main__":
    with ProcessLock(PRIVATE/"CONTROLLER.lock"):
        try:run()
        except Exception as error:
            atomic_json(OUT/"FAILURE.json",{"utc":now(),"error":repr(error),"traceback":traceback.format_exc()})
            old=read(OUT/"STATUS.json") if (OUT/"STATUS.json").exists() else {}
            atomic_json(OUT/"STATUS.json",{**old,"stage":"FAILED","ended_utc":now()})
            raise
