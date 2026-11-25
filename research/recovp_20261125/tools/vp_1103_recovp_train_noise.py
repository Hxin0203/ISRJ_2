"""Noise-phase copying check on native TRAIN noise, clustered by parent."""
import numpy as np
from vp_1103_common import WORK,atomic_json,distribution,now,provenance
from vp_1103_p2_preflight import load
from tools.durable_io import atomic_write

def run():
    data,binding=load("train")
    e=data["epsilon"][:,0].astype(np.complex128);s=data["s"][:,0]
    power=np.sum(abs(s)**2,-1)/np.count_nonzero(s,axis=-1)
    sigma=power/10**(data["snr_db"].astype(float)/10)
    n=np.arange(1800)
    basis=np.exp(2j*np.pi*n[:,None]*np.arange(8)[None]/1800)/np.sqrt(1800)
    c=(e@basis.conj())/np.sqrt(sigma)[...,None]
    fixed=abs(c.sum(-1)/np.sqrt(8))**2
    tracking=(abs(c).sum(-1)/np.sqrt(8))**2
    values={"fixed_direction_noise_energy":fixed.mean(1),"copied_phase_noise_energy":tracking.mean(1),
            "paired_noise_energy_increase":(tracking-fixed).mean(1)}
    result={**provenance(__file__),"utc":now(),"data":{"train":binding},"training_updates":0,
            "independent_parents":512,"pulses_averaged_within_parent":8,"family_copies_used":1,
            "metrics":{key:distribution(value) for key,value in values.items()},
            "mean_energy_ratio":float(tracking.mean()/fixed.mean()),
            "scope":"Fixed orthonormal K=8 direction versus observed-phase copying; native TRAIN E only, not a replay-model DEV score"}
    out=WORK/"evidence/outputs/exp2a_softshare_audit"
    atomic_json(out/"TRAIN_NOISE_TRACKING.json",result)
    lines=["## 原生 TRAIN512 纯噪声补检","",
           "仅使用 TRAIN 的 E，同 parent 的八脉冲先平均；三类型重复噪声仅取一份。噪声单位采用原生 SNR 标定的生成方差，不用实测全窗能量重标元数据。",
           "",f"固定方向均值 {fixed.mean():.6g}，复制相位均值 {tracking.mean():.6g}，比值 {result['mean_energy_ratio']:.4f}。",
           "两项 parent 聚类区间及配对差见 TRAIN_NOISE_TRACKING.json。前述独立高斯 Monte Carlo 仅是数学诊断，不用于提峰选型或场景收益判定。"]
    path=out/"REPORT.md"
    text=path.read_text(encoding="utf8").split("\n## 原生 TRAIN512 纯噪声补检")[0]
    atomic_write(path,lambda f:f.write((text.rstrip()+"\n\n"+"\n".join(lines)+"\n").encode("utf8")))
    print({"parents":512,"mean_energy_ratio":result["mean_energy_ratio"]})

if __name__=="__main__":run()
