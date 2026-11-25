"""TRAIN16 oracle-dictionary soft-sharing audit; no optimizer or DEV reads."""
import os
import math
import numpy as np
import torch
from vp_1103_common import WORK, atomic_json, now, provenance, read, sha
from vp_1103_p2_preflight import load, subset
from vp_1103_recovp import ReCoVPNet, soft_subspace
from vp_1103_recovp_math import math_checks, selected_waves
from vp_1103_replay_truth import recover_selected, replay_components
from vp_1103_snr_jsr import DEFINITION, frame_axes, summarize
from tools.durable_io import atomic_write, ProcessLock

OUT=WORK/"evidence/outputs/exp2a_softshare_audit"
PRIVATE=WORK/"outputs/exp2a_softshare_audit"

def q_phase(c):
    magnitude=abs(c)
    return np.divide(c,magnitude,out=np.ones_like(c),where=magnitude>0)

def right_for(b,ell,eta):
    gram=b.conj().T@b;k=b.shape[1];lam=1e-4
    if eta=="hard":
        wave=b@ell
        return ell[:,None]*wave.conj()[None]/(np.vdot(wave,wave).real+lam*np.vdot(ell,ell).real)
    gamma=soft_subspace(torch.ones((1,k),dtype=torch.float64),torch.from_numpy(ell[None]),torch.tensor([eta],dtype=torch.float64))[0].numpy()
    return np.linalg.solve(gram+lam*np.eye(k)+gamma,b.conj().T)

def run():
    torch.set_num_threads(2);started=now()
    data,binding=load("train")
    cp=torch.load(WORK/"outputs/p2_smoothvp_preflight/RAW_IQ/extended_overfit.pt",map_location="cpu",weights_only=False)
    small=subset(data,np.searchsorted(data["parent_id"],np.sort(cp["parent_ids"])))
    selection=read(WORK/"evidence/outputs/exp0_anchor_audit/SUMMARY.json")
    kmax=selection["K_LOCKED"]
    status={**provenance(__file__,WORK/"tools/vp_1103_recovp.py",WORK/"tools/vp_1103_recovp_math.py",WORK/"tools/vp_1103_replay_truth.py"),
            "stage":"RUNNING","started_utc":started,"ended_utc":None,"pid":os.getpid(),"training_updates":0,
            "data":{"train":binding},"parent_ids":small["parent_id"].tolist(),"dev_access":False,"device":"cpu"}
    atomic_json(OUT/"STATUS.json",status)
    geometry=recover_selected(small,PRIVATE/"geometry",binding)
    torch.manual_seed(17)
    model=ReCoVPNet(selection["locked_config"],share_mode="soft_subspace").cpu().eval()
    math_receipt=math_checks(model,selected_waves(small))
    # This context head is deliberately untrained and is never described as a fitted estimator.
    fixtures={}
    pulse=np.fft.ifft(model.unit_spectrum.numpy())
    freq=model.frequency.numpy()
    for parent in range(16):
        for family in range(3):
            s,j,e=[small[key][parent,family].astype(np.complex128) for key in ("s","j","epsilon")]
            anchor=np.zeros((8,kmax))
            atoms=[]
            for p in range(8):
                comp=replay_components(float(small["target_delay_samples"][parent]),float(small["doppler_hz"][parent]),family,geometry[parent,family,p])[:kmax]
                if not comp:atoms.append(None);continue
                tau=np.array([c["phase_delay"] for c in comp]);anchor[p]=np.pad(tau,(0,kmax-len(tau)),mode="edge")
                template=np.fft.ifft(np.fft.fft(pulse)[None]*np.exp(-2j*np.pi*freq[None]*tau[:,None]),axis=-1)[...,:1800]
                masks=np.stack([c["mask"] for c in comp]);masks=masks/np.maximum(masks.max(-1,keepdims=True),.05)
                atom=masks*template;norm=np.linalg.norm(atom,axis=-1)+1e-12;b=(atom/norm[:,None]).T
                waves=np.stack([c["wave"] for c in comp])
                # Relative physical weights are derived from unit-amplitude geometry, never observed c phases.
                coefficients=np.array([np.vdot(a,w)/np.vdot(a,a) for a,w in zip(atom,waves)])
                q=coefficients/np.sqrt(np.mean(abs(coefficients)**2))
                summed=waves.sum(0)
                alpha=np.vdot(summed,j[p])/np.vdot(summed,summed)  # Fixture generation only, not estimator input.
                atoms.append((b,norm,q,waves,alpha))
            with torch.inference_mode():
                node=model.factor(torch.from_numpy(np.concatenate((j+e,s+j+e,e))),
                                  anchors_override=torch.from_numpy(np.tile(anchor,(3,1))))["q"].numpy()
            fixtures[parent,family]=(s,j,e,atoms,node)
    etas=(0.,.01,.1,1.,10.,"hard")
    perturbations=((0.,0.),(.05,5.),(.15,15.))
    rows={};raw_metrics={}
    for amplitude,angle in perturbations:
        for policy in ("physical_oracle","node_untrained_oracle_anchors","observed_phase_NEGATIVE_CONTROL"):
            for eta in etas:
                label=f"amp{amplitude:g}_phase{angle:g}_{policy}_eta{eta}"
                metrics={key:np.zeros((16,3)) for key in ("u_nmse","damage","removed_noise_gain_frobenius_squared","fitted_noise_energy_ratio")}
                for parent in range(16):
                    for family in range(3):
                        s,j,e,atoms,node=fixtures[parent,family]
                        sums=dict.fromkeys(metrics,0.)
                        for p,item in enumerate(atoms):
                            if item is None:continue
                            b,d,q,waves,alpha=item;count=len(d)
                            rng=np.random.default_rng(20261124+parent*24+family*8+p)
                            signs=rng.choice([-1.,1.],(2,count))
                            multiplier=(1+amplitude*signs[0])*np.exp(1j*np.deg2rad(angle)*signs[1])
                            jam=alpha*(multiplier[:,None]*waves).sum(0)
                            observations=(jam+e[p],s[p]+jam+e[p],e[p])
                            estimates=[];gain=0.
                            for state,y in enumerate(observations):
                                if policy=="physical_oracle":direction=q
                                elif policy=="node_untrained_oracle_anchors":direction=node[state*8+p,:count]
                                else:direction=q_phase(b.conj().T@y)
                                ell=d*direction
                                right=right_for(b,ell,eta)
                                estimates.append(b@(right@y))
                                if state==0:gain=float(np.trace((b.conj().T@b)@(right@right.conj().T)).real)
                            c0,c1,ce=estimates
                            sums["u_nmse"]+=float(np.sum(abs(jam-c1)**2))
                            sums["damage"]+=float(np.sum(abs(c1-c0)**2))
                            sums["removed_noise_gain_frobenius_squared"]+=gain
                            sums["fitted_noise_energy_ratio"]+=float(np.sum(abs(ce)**2))
                        denominator={"u_nmse":float(np.sum(abs(s+e)**2)),"damage":float(np.sum(abs(s)**2)),
                                     "removed_noise_gain_frobenius_squared":1.,"fitted_noise_energy_ratio":float(np.sum(abs(e)**2))}
                        for key in metrics:metrics[key][parent,family]=sums[key]/denominator[key]
                raw=PRIVATE/f"{label}.npz"
                atomic_write(raw,lambda f:np.savez_compressed(f,parent_id=small["parent_id"],**metrics))
                rows[label]={"amplitude_mismatch":amplitude,"phase_mismatch_degrees":angle,"phase_policy":policy,"eta":eta,
                             "metrics":summarize(metrics,frame_axes(small)),"private_metrics_sha256":sha(raw)}
                raw_metrics[label]=metrics
                atomic_json(OUT/"STATUS.json",{**status,"current_case":label,"updated_utc":now()})
    # Pure-noise rank-one calibration: independent synthetic draws, distinct from parent bootstraps.
    rng=np.random.default_rng(20261124)
    c=(rng.normal(size=(16384,8))+1j*rng.normal(size=(16384,8)))/np.sqrt(2)
    valid=abs(c.sum(-1)/np.sqrt(8))**2
    tracking=(abs(c).sum(-1)/np.sqrt(8))**2
    calibration={"independent_synthetic_draws":len(c),"columns":8,"valid_fixed_phase_mean_energy":float(valid.mean()),
                 "observed_phase_mean_energy":float(tracking.mean()),"ratio":float(tracking.mean()/valid.mean()),
                 "conditional_rank_one_frobenius_squared":1.,"unit":"noise variance; not a physical-parent confidence interval"}
    decisions={}
    for policy in ("physical_oracle","node_untrained_oracle_anchors"):
        candidates=[]
        for eta in (.01,.1,1.,10.):
            name=f"amp0_phase0_{policy}_eta{eta}";baseline=raw_metrics[f"amp0_phase0_{policy}_eta0.0"]
            delta=summarize({key:raw_metrics[name][key]-baseline[key] for key in baseline},frame_axes(small))
            good=True
            for family in ("repeat","cyclic"):
                m=delta[f"{family}/jsr/>=15"]
                ci=lambda key:m[key]["parent_bootstrap_ci95"]
                benefit=any(ci(key) is not None and ci(key)[1]<0 for key in ("u_nmse","removed_noise_gain_frobenius_squared"))
                safe=ci("damage") is not None and ci("damage")[1]<=0
                good &= benefit and safe
            candidates.append({"eta":eta,"pass":bool(good),"paired_delta":delta})
        decisions[policy]=candidates
    blind_support=any(item["pass"] for item in decisions["node_untrained_oracle_anchors"])
    oracle_support=any(item["pass"] for item in decisions["physical_oracle"])
    route="GO_EXP2B_CONDITIONAL_ON_EXP1" if (blind_support and math_receipt["status"]=="PASS") else "ORACLE_ONLY_NO_BLIND_UNLOCK" if oracle_support else "NO_GO_SOFTSHARE"
    summary={"methods":rows,"phase_tracking_calibration":calibration,"mathematical_checks":math_receipt,"decisions":decisions,"route":route,
        "training_updates":0,"scope":"TRAIN16 oracle support/dictionary; never a DEV blind score",
        "node_phase_weights":"seed17 untrained; oracle anchors supplied only for this control",
        "fixture_amplitude":"Common coefficient fitted from J only to synthesize controlled fixtures, never supplied to the solvers",
        "activity_control":"Unit activities and fixed lambda=1e-4 isolate sharing; deployed learned damping is not evaluated by these curves",
        "strata":"Original native SNR/JSR metadata, before artificial mismatch; no energy-based relabeling",
        "exp2b_requires_exp1_gpu_admission":True}
    atomic_json(OUT/"SUMMARY.json",summary)
    lines=["# Exp 2a 软子空间共享零训练审计","",DEFINITION,"",f"路由 **{route}**。仅 TRAIN16；全部方向、Oracle 优势和未训练节点头性质在 SUMMARY.json 明确标记。",
           "",f"纯噪声反面校准：固定相位能量 {valid.mean():.4f}，复制观测相位 {tracking.mean():.4f}，相对放大 {calibration['ratio']:.3f} 倍。",
           "条件 rank-one 算子的 Frobenius 平方为 1，不能替代数据自适应相位方向的实测噪声能量。",
           "", "| 幅度失配 | 相位失配 | 相位估计 | eta | 端点 NMSE | 损伤 | 噪声增益 | 拟合噪声能量比 |",
           "|---|---|---|---|---:|---:|---:|---:|"]
    for row in rows.values():
        m=row["metrics"]["global"]
        lines.append(f"| {row['amplitude_mismatch']} | {row['phase_mismatch_degrees']} | {row['phase_policy']} | {row['eta']} | "+
                     " | ".join(f"{m[key]['mean']:.6g}" for key in ("u_nmse","damage","removed_noise_gain_frobenius_squared","fitted_noise_energy_ratio"))+" |")
    lines += ["","共享方向未使用磁图。观测相位复制只存在于明确隔离的负对照，不进入 ReCoVP 模型。",
              "所有曲线采用固定低阻尼与单位活跃度隔离共享效应，不等于模型纯噪声保持能力；该能力由 Exp1/Exp2b 的模型预检与 DEV 收据约束。",
              "当前节点头未经训练；其结果不能证明训练后的盲相位估计可靠。Oracle-only 收益不自动解锁盲训。",
              "数学边界、Hermitian PSD、严格等变和置换结果见 SUMMARY.json。"]
    atomic_write(OUT/"REPORT.md",lambda f:f.write(("\n".join(lines)+"\n").encode("utf8")))
    atomic_json(OUT/"STATUS.json",{**status,"stage":"COMPLETE","ended_utc":now(),"summary_sha256":sha(OUT/"SUMMARY.json")})

if __name__=="__main__":
    with ProcessLock(PRIVATE/"PROCESS.lock"):
        try:run()
        except Exception as error:
            import traceback
            atomic_json(OUT/"FAILURE.json",{"error":repr(error),"traceback":traceback.format_exc()})
            previous=read(OUT/"STATUS.json") if (OUT/"STATUS.json").exists() else {}
            atomic_json(OUT/"STATUS.json",{**previous,"stage":"FAILED","ended_utc":now()})
            raise
