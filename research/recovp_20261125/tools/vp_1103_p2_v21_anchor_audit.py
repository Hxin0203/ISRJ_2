"""TRAIN-only, zero-training window/CLEAN/K candidate screening."""
import os
import time
import traceback
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from vp_1103_common import WORK, atomic_json, now, provenance, read, sha
from vp_1103_p2_preflight import load, subset
from vp_1103_replay_truth import recover_selected, replay_components
from vp_1103_snr_jsr import DEFINITION, frame_axes, summarize, protocol, crosscheck
from vp_1103_v21_extractors import configurations, extract_numpy, extract_torch
from tools.durable_io import ProcessLock, atomic_write

OUT = WORK/"evidence/outputs/p2_v21_anchor_audit"
PRIVATE = WORK/"outputs/p2_v21_anchor_audit"

def run():
    torch.set_num_threads(2)
    started=now()
    data,binding=load("train")
    cp=torch.load(WORK/"outputs/p2_smoothvp_preflight/RAW_IQ/extended_overfit.pt",map_location="cpu",weights_only=False)
    sixteen=np.searchsorted(data["parent_id"],np.sort(cp["parent_ids"]))
    status={**provenance(__file__,WORK/"tools/vp_1103_v21_extractors.py",WORK/"tools/vp_1103_snr_jsr.py",WORK/"tools/vp_1103_replay_truth.py"),
            "stage":"RUNNING","started_utc":started,"ended_utc":None,"pid":os.getpid(),"device":"cpu",
            "training_updates":0,"data":{"train":binding},"selection_uses_dev":False}
    atomic_json(OUT/"STATUS.json",status)
    config_binding=protocol()
    check=crosscheck(data)
    geometry=recover_selected(data,PRIVATE/"geometry",binding)
    truth={}
    for parent in range(512):
        for family in range(3):
            truth[parent,family]=[replay_components(float(data["target_delay_samples"][parent]),
                float(data["doppler_hz"][parent]),family,geometry[parent,family,pulse]) for pulse in range(8)]
    rows={}
    for name,config in configurations().items():
        metric={key:np.zeros((512,3)) for key in ("anchor_energy_recall_2_samples","weighted_nearest_error_samples",
             "candidate_membership_change_after_target","true_jammer_hit_retention","joint_hit_energy_recall")}
        for parent in range(512):
            for family in range(3):
                s,j,e=[data[key][parent,family].astype(np.complex128) for key in ("s","j","epsilon")]
                anchors=extract_numpy(np.concatenate((j+e,s+j+e),axis=0),config)
                sums=dict.fromkeys(metric,0.)
                energy_total,energy_hit=0.,0.
                for pulse,components in enumerate(truth[parent,family]):
                    if not components:
                        continue
                    tau=np.array([c["phase_delay"] for c in components])
                    energy=np.array([c["energy"] for c in components])
                    a,b=anchors[pulse],anchors[pulse+8]
                    error=abs(tau[:,None]-a).min(-1)
                    hit=error<=2.
                    retained=(abs(tau[:,None]-b).min(-1)<=2.)&hit
                    dist=abs(a[:,None]-b)
                    left,right=linear_sum_assignment(dist)
                    sums["anchor_energy_recall_2_samples"]+=float(energy@hit)
                    sums["weighted_nearest_error_samples"]+=float(energy@error)
                    sums["candidate_membership_change_after_target"]+=float((dist[left,right]>2.).mean()*energy.sum())
                    sums["true_jammer_hit_retention"]+=float(energy@retained)
                    sums["joint_hit_energy_recall"]+=float(energy@retained)
                    energy_hit+=float(energy@hit)
                    energy_total+=float(energy.sum())
                for key in metric:
                    denominator=energy_hit if key=="true_jammer_hit_retention" else energy_total
                    metric[key][parent,family]=sums[key]/denominator if denominator else np.nan
            if parent%64==0:
                atomic_json(OUT/"STATUS.json",{**status,"configuration":name,"parents_done":parent+1,"updated_utc":now()})
        raw=PRIVATE/f"{name}.npz"
        atomic_write(raw,lambda f:np.savez_compressed(f,parent_id=data["parent_id"],**metric))
        # Torch timing includes all FFT/peeling, excluding only network features.
        wave=torch.from_numpy((data["j"][0,1]+data["epsilon"][0,1]).astype(np.complex64))
        from vp_1103_p2_smoothvp import SmoothVPNet
        spec=SmoothVPNet("RAW_IQ").pulse_spectrum
        reference=extract_numpy(wave.numpy(),config)
        actual=extract_torch(wave,config,spec).numpy()
        if not np.array_equal(reference,actual):
            raise ValueError(f"NumPy/Torch extractor disagreement: {name}")
        timings={}
        for count in (1,8):
            for _ in range(5): extract_torch(wave[:count],config,spec)
            times=[]
            for _ in range(32):
                tick=time.perf_counter_ns(); extract_torch(wave[:count],config,spec)
                times.append((time.perf_counter_ns()-tick)/1e6)
            timings[str(count)]={"p50_ms":float(np.median(times)),"p95_ms":float(np.quantile(times,.95))}
        rows[name]={"config":config,"train512":summarize(metric,frame_axes(data)),
                    "train16":summarize({k:v[sixteen] for k,v in metric.items()},frame_axes(subset(data,sixteen))),
                    "timings_cpu_torch":timings,"numpy_torch_anchor_agreement":True,"private_metrics_sha256":sha(raw)}
        atomic_json(OUT/"PARTIAL.json",{"methods":rows,"training_updates":0})
    # Before observing results: equal weight repeat/cyclic in the highest JSR bin.
    for name,row in rows.items():
        m=row["train512"]
        recall=np.mean([m[f"{family}/jsr/>=30"]["anchor_energy_recall_2_samples"]["mean"] for family in ("repeat","cyclic")])
        retained=np.mean([m[f"{family}/jsr/>=30"]["true_jammer_hit_retention"]["mean"] for family in ("repeat","cyclic")])
        row["selection_score"]=float(2*recall*retained/max(recall+retained,1e-12))
    cap=max(10.,5*rows["E0_rect_k8"]["timings_cpu_torch"]["8"]["p50_ms"])
    eligible=[name for name,row in rows.items() if row["timings_cpu_torch"]["8"]["p50_ms"]<=cap]
    highest=max(rows[name]["selection_score"] for name in eligible)
    near=[name for name in eligible if rows[name]["selection_score"]>=highest-.005]
    best=min(near,key=lambda n:(rows[n]["timings_cpu_torch"]["8"]["p50_ms"],rows[n]["config"]["k"],n))
    summary={"methods":rows,"BEST_EXTRACTOR":best,"K_LOCKED":rows[best]["config"]["k"],
             "locked_config":rows[best]["config"],"training_updates":0,"metadata_crosscheck_train":check,
             "protocol":config_binding,"selection_rule":{"score":"harmonic mean of highest-JSR repeat/cyclic recall and conditional true-hit retention",
             "latency_cap_8_pulse_p50_ms":cap,"tie_score_tolerance":.005,"tie_break":"fastest then smallest K"},
             "scope":"TRAIN512 selection; same fixed TRAIN16 also reported; no DEV waveforms read",
             "retention_zero_initial_hit":"undefined, omitted from conditional retention; joint recall remains zero"}
    atomic_json(OUT/"SUMMARY.json",summary)
    lines=["# v2.1 无参数候选提取审计","",DEFINITION,"",f"锁定 **{best}**，K={summary['K_LOCKED']}。仅 TRAIN512 选型，TRAIN16 同步描述。",
       "","| 方法 | 高 JSR repeat/cyclic 召回 | 留存 | 选择分数 | 8 脉冲 P50 ms |","|---|---:|---:|---:|---:|"]
    for name,row in rows.items():
        m=row["train512"]
        recall=np.mean([m[f"{family}/jsr/>=30"]["anchor_energy_recall_2_samples"]["mean"] for family in ("repeat","cyclic")])
        retained=np.mean([m[f"{family}/jsr/>=30"]["true_jammer_hit_retention"]["mean"] for family in ("repeat","cyclic")])
        lines.append(f"| {name} | {recall:.3%} | {retained:.3%} | {row['selection_score']:.5f} | {row['timings_cpu_torch']['8']['p50_ms']:.4f} |")
    lines += ["","加窗不会保证短切片的旁瓣达到完整 LFM 的理论水平；本轮收益以实际召回为准。",
              "CLEAN 仅更新候选提取的临时残差，网络继续读取原观测；离散候选不参与梯度。",
              "每个配置的两套样本、全部 SNR/JSR×类型分层及 parent bootstrap 区间均见 SUMMARY.json。"]
    atomic_write(OUT/"REPORT.md",lambda f:f.write(("\n".join(lines)+"\n").encode("utf8")))
    atomic_json(OUT/"STATUS.json",{**status,"stage":"COMPLETE","ended_utc":now(),"summary_sha256":sha(OUT/"SUMMARY.json")})

if __name__=="__main__":
    with ProcessLock(PRIVATE/"PROCESS.lock"):
        try:run()
        except Exception as exc:
            atomic_json(OUT/"FAILURE.json",{"error":repr(exc),"traceback":traceback.format_exc()})
            previous=read(OUT/"STATUS.json") if (OUT/"STATUS.json").exists() else {}
            atomic_json(OUT/"STATUS.json",{**previous,"stage":"FAILED","ended_utc":now()})
            raise
