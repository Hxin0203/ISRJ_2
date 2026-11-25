"""Fixed RAW_IQ TRAIN16 CPU admission, with the user-authorized new gates."""
import os
import time
import traceback
import numpy as np
import torch
from vp_1103_common import WORK, atomic_json, now, provenance, read, sha
from vp_1103_p2_preflight import load, subset, health
from vp_1103_p2_smoothvp_v21 import SmoothVPNetV21
from vp_1103_v21_mechanisms import mechanism_metrics
from vp_1103_snr_jsr import DEFINITION, frame_axes, summarize, cases, CONFIG
from vp_1103_p0 import evaluate
from vp_1103_p2_evaluate import failure_details
from paper2.train_1103 import make_batch
from paper2.train_256_8seed import seed_rng, atomic_torch_save
from tools.correction_tail_study import objective
from tools.durable_io import atomic_write, ProcessLock

OUT=WORK/"evidence/outputs/p2_smoothvp_v21_preflight"
PRIVATE=WORK/"outputs/p2_smoothvp_v21_preflight"
AUDIT=WORK/"evidence/outputs/p2_v21_anchor_audit"
STAGES=((768,1e-3,20261125),(3072,3e-4,20261126))

def gates(metrics,old,axes,engineering,permutation,algebra):
    levels=summarize(metrics,axes)
    delta=summarize({key:metrics[key]-old[key] for key in metrics if key in old},axes)
    g=levels["global"]
    m=levels["jsr/>=15"]
    repeat_significant=delta["repeat/jsr/>=15"]["blind_j_relative_residual"]["parent_bootstrap_ci95"]
    high_repeat=levels["repeat/jsr/>=30"]["blind_j_relative_residual"]["mean"]
    g1=(56728<=engineering["effective_real_parameters"]<=62700 and engineering["all_gradients_finite_and_connected"]
        and engineering["all_parameter_tensors_have_nonzero_gradient"] and permutation<1e-5 and algebra<1e-10
        and g["no_jammer_pooled_nmse"]["mean"]<=1e-4 and g["leak_over_target_energy"]["mean"]<=.015)
    g2=(m["learned_b_oracle_beta_relative_residual"]["mean"] is not None
        and m["learned_b_oracle_beta_relative_residual"]["mean"]<=.20
        and m["blind_j_relative_residual"]["mean"]<=.22
        and ((repeat_significant is not None and repeat_significant[1]<0) or (high_repeat is not None and high_repeat<=.20)))
    g3=g["frame_pfa_jammer"]["mean"]<=.05 and g["drift_over_target_energy"]["mean"]<=.035
    regression={}
    for family in ("direct","cyclic"):
        for scope in ("all","jsr/>=15"):
            key=f"{family}/{scope}"
            for metric in ("frame_pfa_jammer","blind_j_relative_residual","learned_b_oracle_beta_relative_residual"):
                regression[f"{key}/{metric}"]=delta[key][metric]["mean"]
    g4=all(value is not None and value<=1e-12 for value in regression.values())
    checks={f"G-v21-{i}":"PASS" if value else "FAIL" for i,value in enumerate((g1,g2,g3,g4),1)}
    return {"checks":checks,"route":"GO_FORMAL_GPU_TRAIN" if all((g1,g2,g3,g4)) else "NO_GO_REVIEW",
            "paired_v21_minus_v2":delta,"no_regression_point_differences":regression,
            "no_regression_scope":"direct/cyclic all and effective-threat JSR>=15; Pfa, oracle and blind residual point means",
            "levels":levels}

def run():
    torch.set_num_threads(2)
    while True:
        path=AUDIT/"STATUS.json"
        state=read(path) if path.exists() else {}
        if state.get("stage")=="COMPLETE":break
        if state.get("stage")=="FAILED":raise RuntimeError("Candidate audit failed")
        time.sleep(30)
    selection=read(AUDIT/"SUMMARY.json")
    data,binding=load("train")
    original=torch.load(WORK/"outputs/p2_smoothvp_preflight/RAW_IQ/extended_overfit.pt",map_location="cpu",weights_only=False)
    small=subset(data,np.searchsorted(data["parent_id"],original["parent_ids"]))
    status={**provenance(__file__,*[WORK/"tools"/name for name in ("vp_1103_p2_smoothvp_v21.py","vp_1103_p2_smoothvp_v2.py",
            "vp_1103_v21_extractors.py","vp_1103_v21_mechanisms.py","vp_1103_snr_jsr.py")]),
            "stage":"RUNNING","started_utc":now(),"ended_utc":None,"pid":os.getpid(),"device":"cpu","seed":17,
            "coordinates":"RAW_IQ","data":{"train":binding},"parent_ids":small["parent_id"].tolist(),
            "micro_batch_records":1,"budget":3840,"selection_summary_sha256":sha(AUDIT/"SUMMARY.json")}
    def update(**fields):
        status.update(fields,updated_utc=now());atomic_json(OUT/"STATUS.json",status)
    update(phase="HEALTH")
    seed_rng(17)
    model=SmoothVPNetV21(selection["locked_config"]).cpu()
    initial=health(model,small,np.arange(16))
    objective.ARM="tail"
    total=0;trace=[]
    for steps,lr,seed in STAGES:
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        rng=np.random.default_rng(seed)
        schedule=np.concatenate([rng.permutation(48) for _ in range(steps//48)])
        for record in schedule:
            optimizer.zero_grad(set_to_none=True)
            batch=make_batch(small,np.array([record]),"cpu")
            correction=model(batch["y"]).reshape(-1,4,2,1800)
            terms=objective.study_objective(correction,batch,1)
            if not all(torch.isfinite(v).all() for v in terms.values()):raise FloatingPointError("Nonfinite loss")
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            optimizer.step();total+=1
            trace.append({key:float(value.detach()) for key,value in terms.items()})
            if total%32==0:
                atomic_json(PRIVATE/"TRAJECTORY.json",trace)
                update(phase="CPU_OVERFIT",step=total,latest_loss=trace[-1]["total"])
        atomic_torch_save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"steps":total,
            "coordinates":"RAW_IQ","parent_ids":small["parent_id"].tolist(),"extractor_config":selection["locked_config"]},PRIVATE/f"step_{total:04d}.pt")
    update(phase="POSTFIT",step=3840)
    model.eval()
    metrics=evaluate(model,small)
    extra,algebra,permutation=mechanism_metrics(model,small);metrics.update(extra)
    axes=frame_axes(small)
    raw=PRIVATE/"metrics.npz"
    atomic_write(raw,lambda f:np.savez_compressed(f,parent_id=small["parent_id"],**metrics))
    old_path=WORK/"outputs/p2_smoothvp_v2_preflight/RAW_IQ/metrics.npz"
    with np.load(old_path,allow_pickle=False) as z:
        index={int(value):i for i,value in enumerate(z["parent_id"])}
        order=[index[int(value)] for value in small["parent_id"]]
        old={key:z[key][order] for key in z.files if key!="parent_id"}
    result=gates(metrics,old,axes,initial,permutation,algebra)
    failures=cases(metrics,small["parent_id"],axes)
    ordering=np.argsort(small["parent_id"])
    atoms=failure_details(model,subset(small,ordering),failures,PRIVATE)
    summary={**result,"initial_health":initial,"permutation_error":permutation,"operator_identity_error":algebra,
             "seed":17,"steps":3840,"K_LOCKED":model.k,"extractor":selection["locked_config"],
             "worst_cases":failures,"failure_atoms":atoms,"private_metrics_sha256":sha(raw),
             "checkpoint_sha256":sha(PRIVATE/"step_3840.pt"),"v2_metrics_sha256":sha(old_path),
             "protocol_sha256":sha(CONFIG),"selection_summary_sha256":sha(AUDIT/"SUMMARY.json"),
             "formal_gpu_training_authorized_by_user_if_pass":True,"confirmation_unlocked":False,
             "optimization":{"stages":STAGES,"loss":"endpoint + .1 pc + damage + .05 tail","additional_losses":[]}}
    atomic_json(OUT/"SUMMARY.json",summary)
    lines=["# SmoothVP-Net v2.1 RAW_IQ CPU 预检","",DEFINITION,"",f"路由 **{result['route']}**。同一 TRAIN16、seed 17，固定第 3840 步。",
           "","| 闸门 | 结果 |","|---|---|"]+[f"| {key} | {value} |" for key,value in result["checks"].items()]
    lines += ["","| 分层 | parent | 虚警 | Pd | 损伤 | Oracle-β 残差 | 盲残差 | 漂移比 | 泄漏比 | 保持 NMSE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fields=("frame_pfa_jammer","pd","damage","learned_b_oracle_beta_relative_residual","blind_j_relative_residual",
            "drift_over_target_energy","leak_over_target_energy","no_jammer_pooled_nmse")
    for label,row in result["levels"].items():
        values=["NA" if row[key]["mean"] is None else f"{row[key]['mean']:.6g}" for key in fields]
        lines.append(f"| {label} | {row['pd']['parents']} | "+" | ".join(values)+" |")
    lines += ["","全部同 parent 配对区间、噪声增益与类型×SNR/JSR 明细见 SUMMARY.json。",
              "G-v21-4 按 direct/cyclic 的全局及有效威胁区间虚警、两项残差的点估计检查无回退，不新增容忍带。",
              "仅候选提取离散且无训练参数；其后的微调、软窗与求解对网络参数可微。"]
    for criterion,rows in failures.items():
        lines += ["",f"## Top-5 {criterion}",""]
        for row in rows:
            description="; ".join(f"{name}: SNR {v['snr_db']:.2f} dB, JSR {v['jsr_db']:.2f} dB, score {v['score']:.5g}" for name,v in row["families"].items())
            lines.append(f"- {row['parent_id']}: {description}")
    atomic_write(OUT/"REPORT.md",lambda f:f.write(("\n".join(lines)+"\n").encode("utf8")))
    update(stage="COMPLETE",ended_utc=now(),route=result["route"],summary_sha256=sha(OUT/"SUMMARY.json"))

if __name__=="__main__":
    with ProcessLock(PRIVATE/"PROCESS.lock"):
        try:run()
        except Exception as exc:
            atomic_json(OUT/"FAILURE.json",{"error":repr(exc),"traceback":traceback.format_exc()})
            previous=read(OUT/"STATUS.json") if (OUT/"STATUS.json").exists() else {}
            atomic_json(OUT/"STATUS.json",{**previous,"stage":"FAILED","ended_utc":now()})
            raise
