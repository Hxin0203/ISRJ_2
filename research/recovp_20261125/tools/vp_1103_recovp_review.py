"""Receipt-only selected-candidate comparisons and fixed-step failure review."""
import argparse
import numpy as np
from vp_1103_common import WORK, atomic_json, now, provenance, read, sha, parent_mean, distribution
from vp_1103_p2_preflight import load, subset
from vp_1103_snr_jsr import DEFINITION, frame_axes, summarize
from tools.durable_io import atomic_write

def interactions(metrics,axes):
    result={}
    for snr_lo,snr_hi,snr_label in ((-np.inf,0,"<0"),(0,10,"0-10"),(10,np.inf,">=10")):
        for jsr_lo,jsr_hi,jsr_label in ((-np.inf,0,"<0"),(0,15,"0-15"),(15,30,"15-30"),(30,np.inf,">=30")):
            mask=(axes["snr"]>=snr_lo)&(axes["snr"]<snr_hi)&(axes["jsr"]>=jsr_lo)&(axes["jsr"]<jsr_hi)
            for family,label in ((None,"all"),(0,"direct"),(1,"repeat"),(2,"cyclic")):
                valid=mask if family is None else mask&(np.arange(3)[None]==family)
                result[f"{label}/snr/{snr_label}/jsr/{jsr_label}"]={key:distribution(parent_mean(value,valid)) for key,value in metrics.items()}
    return result

def archive(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key] for key in z.files}

def run(anchor_only=False):
    started=now()
    out=WORK/"evidence/outputs/exp0_anchor_audit"
    private=WORK/"outputs/exp0_anchor_audit"
    summary=read(out/"SUMMARY.json")
    data,binding=load("train");axes=frame_axes(data)
    selected=summary["BEST_EXTRACTOR"]
    raw=archive(private/f"{selected}.npz")
    if sha(private/f"{selected}.npz")!=summary["methods"][selected]["private_metrics_sha256"]:raise ValueError("Candidate metrics changed")
    baseline=archive(private/"E0_rect_k8.npz")
    fields=[key for key in raw if key!="parent_id"]
    pairs={"selected_minus_E0_k8":summarize({key:raw[key]-baseline[key] for key in fields},axes)}
    for name,row in summary["methods"].items():
        config=row["config"];control=archive(private/f"E0_rect_k{config['k']}.npz")
        actual=archive(private/f"{name}.npz")
        pairs[f"{name}_minus_same_K_E0"]=summarize({key:actual[key]-control[key] for key in fields},axes)
    joint=interactions({key:raw[key] for key in fields},axes)
    details={**provenance(__file__),"started_utc":started,"ended_utc":now(),"training_updates":0,
             "selection_summary_sha256":sha(out/"SUMMARY.json"),"data":binding,"paired_differences":pairs,
             "selected_joint_snr_jsr":joint,
             "constants":{"nms_minimum_distance":6,"local_peak_window":13,"template_samples":600,"fft_length":4096,
                          "CLEAN_loop_gain":.8,"CLEAN_coherent_average_samples":11}}
    atomic_json(out/"ANCHOR_DETAILS.json",details)
    lines=["## 锁定方案的完整分层与配对复核","",DEFINITION,"",
           "延迟召回比较含频偏的等效 LFM 延迟：真实重放延迟 − (目标频偏+干扰频偏)/50000，单位为采样点。几何只用于审计评分，不作为提峰输入。",
           "本轮粗剥离固定使用 0.8 环路增益与 11 点相干均值；负结果不能外推至所有残差剥离设定。",
           "所有区间为训练内点态 parent bootstrap 区间，未对多配置选型做校正，不能当作 DEV 或独立确认的证据。",""]
    for cohort in ("train512","train16"):
        lines += ["",f"### {cohort}","",
                  "| 分层 | 方案 | parent | 召回 | 最近误差/点 | 命中留存 | 集合变化 |","|---|---|---:|---:|---:|---:|---:|"]
        for stratum in summary["methods"][selected][cohort]:
            for name in ("E0_rect_k8",selected):
                m=summary["methods"][name][cohort][stratum]
                keys=("anchor_energy_recall_2_samples","weighted_nearest_error_samples","true_jammer_hit_retention","candidate_membership_change_after_target")
                values=["NA" if m[key]["mean"] is None else f"{m[key]['mean']:.6g}" for key in keys]
                lines.append(f"| {stratum} | {name} | {m[keys[0]]['parents']} | "+" | ".join(values)+" |")
    lines += ["","### TRAIN512：SNR 与 JSR 交互","",
              "| 分层 | parent | 召回 | 95% CI | 命中留存 |","|---|---:|---:|---|---:|"]
    for label,row in joint.items():
        if not label.startswith("all/"):continue
        show=lambda value:"NA" if value is None else f"{value:.6g}"
        lines.append(f"| {label} | {row['anchor_energy_recall_2_samples']['parents']} | "
                     f"{show(row['anchor_energy_recall_2_samples']['mean'])} | {row['anchor_energy_recall_2_samples']['parent_bootstrap_ci95']} | "
                     f"{show(row['true_jammer_hit_retention']['mean'])} |")
    lines += ["","### 高 JSR repeat/cyclic 的 Top-5 漏峰 parent","",
              "| parent | 类型 | SNR dB | JSR dB | 召回 | 命中留存 |","|---|---|---:|---:|---:|---:|"]
    valid=(axes["jsr"]>=30)&(np.arange(3)[None]>0)
    score=np.divide(np.where(valid,raw["anchor_energy_recall_2_samples"],0).sum(1),valid.sum(1),
                    out=np.full(512,np.inf),where=valid.sum(1)>0)
    for parent in np.argsort(score)[:5]:
        family=int(np.argmin(np.where(valid[parent],raw["anchor_energy_recall_2_samples"][parent],np.inf)))
        lines.append(f"| {data['parent_id'][parent]} | {('direct','repeat','cyclic')[family]} | {axes['snr'][parent,family]:.3f} | "
                     f"{axes['jsr'][parent,family]:.3f} | {raw['anchor_energy_recall_2_samples'][parent,family]:.4f} | "
                     f"{raw['true_jammer_hit_retention'][parent,family]:.4f} |")
    path=out/"REPORT.md"
    report=path.read_text(encoding="utf8").split("\n## 锁定方案的完整分层与配对复核")[0]
    report=report.replace("# v2.1 无参数候选提取审计","# Exp 0 齐次候选提取审计")
    atomic_write(path,lambda f:f.write((report.rstrip()+"\n\n"+"\n".join(lines)+"\n").encode("utf8")))
    if anchor_only:return
    out=WORK/"evidence/outputs/exp1_recovp_base_preflight"
    private=WORK/"outputs/exp1_recovp_base_preflight"
    state=read(out/"STATUS.json");cpu=read(out/"SUMMARY.json")
    if state["stage"]!="COMPLETE":raise ValueError("CPU preflight has not finished")
    metrics=archive(private/"metrics.npz")
    if sha(private/"metrics.npz")!=cpu["private_metrics_sha256"]:raise ValueError("CPU metrics changed")
    index={int(value):i for i,value in enumerate(data["parent_id"])}
    order=np.array([index[int(value)] for value in metrics["parent_id"]])
    local_axes=frame_axes(subset(data,order))
    old_path=WORK/"outputs/p2_smoothvp_v2_preflight/RAW_IQ/metrics.npz"
    old=archive(old_path)
    old_index={int(value):i for i,value in enumerate(old["parent_id"])}
    old_order=[old_index[int(value)] for value in metrics["parent_id"]]
    old_levels=summarize({key:old[key][old_order] for key in old if key!="parent_id"},local_axes)
    compare_fields=("frame_pfa_jammer","pd","damage","learned_b_oracle_beta_relative_residual","blind_j_relative_residual",
                    "drift_over_target_energy","leak_over_target_energy","no_jammer_pooled_nmse","removed_noise_gain_frobenius_squared")
    comparison=["# ReCoVP-Base 与 v2 全量配对比较","",DEFINITION,"",
                "同一 TRAIN16、同 seed 和固定末步，差值为 ReCoVP-Base−v2。区间按 parent 配对重采样。", "",
                "| 分层 | 指标 | parent | v2 | ReCoVP-Base | 差值 | 差值 95% CI |", "|---|---|---:|---:|---:|---:|---|"]
    for label,row in cpu["levels"].items():
        for key in compare_fields:
            delta=cpu["paired_recovp_minus_v2"][label][key]
            mean=lambda value:"NA" if value is None else f"{value:.7g}"
            comparison.append(f"| {label} | {key} | {row[key]['parents']} | {mean(old_levels[label][key]['mean'])} | "
                              f"{mean(row[key]['mean'])} | {mean(delta['mean'])} | {delta['parent_bootstrap_ci95']} |")
    atomic_write(out/"PAIRED_COMPARISON.md",lambda f:f.write(("\n".join(comparison)+"\n").encode("utf8")))
    trace=read(private/"TRAJECTORY.json")
    if len(trace)!=3840:raise ValueError("CPU budget changed")
    loss=np.array([row["total"] for row in trace]).reshape(80,48).mean(1)
    review={"started_utc":started,"ended_utc":now(),"training_updates":0,"route":cpu["route"],
            "cpu_summary_sha256":sha(out/"SUMMARY.json"),"last_10_full_pass_losses":loss[-10:].tolist(),
            "extractor_mechanism_train16":summarize({key:raw[key][order] for key in fields},local_axes),
            "baseline_v2":old_levels,"baseline_v2_metrics_sha256":sha(old_path),
            "joint_snr_jsr":interactions({key:metrics[key] for key in compare_fields},local_axes),
            "notes":"Fixed final checkpoint; no epoch selection or budget extension"}
    lines=["## 固定末步失败归因补充","", "完整分层九指标同 parent 配对表见 [PAIRED_COMPARISON.md](PAIRED_COMPARISON.md)。", "",
           "最后十个完整 TRAIN16 遍历的更新前平均损失："+", ".join(f"{value:.6g}" for value in loss[-10:])+"。",
           "", "每一类 Top-5 均列出该 parent 指标最差的干扰类型；排序可能包含零虚警占位项，不代表每行都发生虚警。数组、软窗、精度和延迟保存在私有 failure_atoms.npz。",
           "", "| 排序项 | parent | 类型 | SNR dB | JSR dB | 虚警 | 损伤 | Oracle-β 残差 | 盲残差 | 漂移 | 锚点召回 | 支撑外能量 |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for criterion,rows in cpu["worst_cases"].items():
        for row in rows:
            parent=int(np.flatnonzero(metrics["parent_id"]==row["parent_id"])[0])
            family=int(metrics[criterion][parent].argmax())
            values=[metrics[key][parent,family] for key in ("frame_pfa_jammer","damage","learned_b_oracle_beta_relative_residual",
                    "blind_j_relative_residual","drift_over_target_energy")]
            values += [raw["anchor_energy_recall_2_samples"][order[parent],family],metrics["J_energy_outside_strong_atom_regions"][parent,family]]
            lines.append(f"| {criterion} | {row['parent_id']} | {('direct','repeat','cyclic')[family]} | "
                f"{local_axes['snr'][parent,family]:.2f} | {local_axes['jsr'][parent,family]:.2f} | "+" | ".join(f"{v:.5g}" for v in values)+" |")
    atomic_json(out/"MECHANISM_REVIEW.json",review)
    path=out/"REPORT.md"
    report=path.read_text(encoding="utf8").split("\n## 固定末步失败归因补充")[0]
    report=report.replace("# SmoothVP-Net v2.1 RAW_IQ CPU 预检","# ReCoVP-Base RAW_IQ CPU 预检")
    report=report.replace("G-v21-4 按 direct/cyclic 的全局及有效威胁区间虚警、两项残差的点估计检查无回退，不新增容忍带。",
                          "本轮 repeat 明显改善要求：JSR≥15 档位的盲残差与漂移相对 v2 的同 parent 配对 95% CI 上界均低于零。")
    atomic_write(path,lambda f:f.write((report.rstrip()+"\n\n"+"\n".join(lines)+"\n").encode("utf8")))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--anchor-only",action="store_true")
    run(parser.parse_args().anchor_only)
