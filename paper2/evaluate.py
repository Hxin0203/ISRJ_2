"""DEV-only gain frontiers, paired parent statistics, and honest pilot report."""
import csv
import os
import numpy as np
import torch
from .data import load,tensors
from .network import CorrectionNet
from .science import metrics,optimal_gain,parent_mean,bootstrap_difference,energy
from .io import ROOT,read,write,sha,now
from .run import setup

OUT=ROOT/'outputs'
EVIDENCE=ROOT/'evidence'


def summary(m,ids):
    result={k:float(parent_mean(v,ids).mean()) for k,v in m.items()}
    d=parent_mean(m['damage'],ids)
    result['damage_parent_p95']=float(np.quantile(d,.95))
    result['severe_parent_rate']=float(np.mean([np.any(m['damage'][ids==p]>.25) for p in np.unique(ids)]))
    result['parents']=len(d)
    return result


def infer(tag,a):
    folder=OUT/tag
    receipt=read(folder/'COMPLETE.json')
    assert receipt['checkpoint_sha256']==sha(folder/'model.pt')
    path=folder/'dev_predictions.npz'
    cache=folder/'DEV_PREDICTIONS.json'
    if path.exists():
        saved=read(cache)
        assert saved['sha256']==sha(path) and saved['checkpoint_sha256']==receipt['checkpoint_sha256']
        assert saved['dev_sha256']==sha(ROOT/'data'/'dev.npz')
        with np.load(path) as raw:
            return raw['correction']
    device=setup(receipt['seed'])
    net=CorrectionNet().to(device)
    payload=torch.load(folder/'model.pt',map_location=device,weights_only=False)
    net.load_state_dict(payload['state'])
    net.eval()
    y=tensors(a)['y']
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(y),16):
            b=y[start:start+16].to(device)
            c=net(b.flatten(0,1)).reshape_as(b).cpu().numpy()
            chunks.append(c[:,:,0]+1j*c[:,:,1])
    result=np.concatenate(chunks).astype(np.complex64)
    np.savez_compressed(path,correction=result)
    write(cache,dict(sha256=sha(path),checkpoint_sha256=receipt['checkpoint_sha256'],
                     dev_sha256=sha(ROOT/'data'/'dev.npz'),created_utc=now()))
    return result


def evaluate():
    assert (OUT/'TRAINING_COMPLETE.json').exists()
    cfg=read(ROOT/'PILOT_FROZEN_CONFIG.json')
    a=load('dev')
    s,j,e=[a[k].astype(np.complex128) for k in ('s','j','epsilon')]
    ids=a['scene_id']
    keep=summary(metrics(s,j,e,np.zeros((len(s),4,s.shape[-1]),np.complex64)),ids)
    rows=[]
    curves={}
    evaluated={}
    EVIDENCE.mkdir(exist_ok=True)
    for seed in cfg['seeds']:
        for method in cfg['methods']:
            tag=f'{method}_seed{seed}'
            c=infer(tag,a)
            m=metrics(s,j,e,c)
            # Exact quadratic frontier statistics, avoiding recomputing all waveforms.
            den=energy(s+e).clip(1e-24)
            aa=float(np.mean(energy(c[:,3].astype(np.complex128))/den))
            qq=float(np.mean((c[:,3].conj()*j).sum(-1).real/den))
            damage=float(m['damage'].mean())
            curves[tag]=[dict(alpha=float(g),u_nmse=keep['u_nmse']-2*g*qq+g*g*aa,damage=g*g*damage) for g in cfg['alpha_grid']]
            evaluated[tag]={}
            for budget in cfg['budget']:
                evaluated[tag][str(budget)]={}
                for family,complex_control in [('real',False),('complex',True)]:
                    gain=optimal_gain(s,j,e,c,budget,complex_control=complex_control)
                    values=metrics(s,j,e,c,gain)
                    aggregate=summary(values,ids)
                    assert aggregate['damage']<=budget+1e-10
                    assert np.max(np.abs(values['u_nmse']-values['r0_u']-values['rs_u']-values['cross_u']))<1e-9
                    row=dict(tag=tag,method=method,seed=seed,budget=budget,family=family,
                             gain_real=float(np.real(gain)),gain_imag=float(np.imag(gain)),**aggregate)
                    rows.append(row)
                    evaluated[tag][str(budget)][family]=(row,parent_mean(values['u_nmse'],ids))
            write(EVIDENCE/f'{tag}_alpha1.json',dict(tag=tag,**summary(m,ids)))
            print('DEV evaluated',tag,flush=True)
    comparisons=[]
    for budget in cfg['budget']:
        for seed in cfg['seeds']:
            options=[evaluated[f'{method}_seed{seed}'][str(budget)][family]
                     for method in ('B1','B2') for family in ('real','complex')]
            control,control_parent=min(options,key=lambda pair:pair[0]['u_nmse'])
            candidate,candidate_parent=min(evaluated[f'B4_seed{seed}'][str(budget)].values(),key=lambda pair:pair[0]['u_nmse'])
            improvement=1-candidate['u_nmse']/control['u_nmse']
            suppression=1-candidate['u_nmse']/keep['u_nmse']
            ci=bootstrap_difference(candidate_parent,control_parent)
            comparisons.append(dict(seed=seed,budget=budget,control=control['tag'],control_family=control['family'],
                                    control_u_nmse=control['u_nmse'],candidate_u_nmse=candidate['u_nmse'],
                                    candidate_family=candidate['family'],improvement_fraction=improvement,
                                    suppression_fraction=suppression,severe_parent_rate=candidate['severe_parent_rate'],
                                    paired_parent_bootstrap=ci,
                                    signal=bool(improvement>=.05 and suppression>=.1 and candidate['severe_parent_rate']<=.1)))
    passed=[b for b in cfg['budget'] if all(r['signal'] for r in comparisons if r['budget']==b)]
    decision='CONTINUE_EXPLORATORY' if passed else 'REVISE'
    result=dict(created_utc=now(),stage='G2 exploratory DEV only',decision=decision,qualifying_budgets=passed,
                keep=keep,rows=rows,comparisons=comparisons,config_sha256=sha(ROOT/'PILOT_FROZEN_CONFIG.json'),
                test_generated=False,test_opened=False,formal_novelty_gate='UNRESOLVED_FULL_TEXT',
                uncertainty='DEV-selected gain/model comparisons; bootstrap is descriptive and not selection-adjusted')
    write(EVIDENCE/'PILOT_RESULTS.json',result)
    write(EVIDENCE/'GAIN_FRONTIERS.json',curves)
    with (EVIDENCE/'MATCHED_BUDGET.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report(result)
    plot(curves)
    write(OUT/'PILOT_COMPLETE.json',dict(completed_utc=now(),decision=decision,results_sha256=sha(EVIDENCE/'PILOT_RESULTS.json'),runs=6))
    write(OUT/'STATUS.json',dict(stage='PILOT_COMPLETE',decision=decision,updated_utc=now(),pid=os.getpid()))


def report(result):
    cfg=read(ROOT/'PILOT_FROZEN_CONFIG.json')
    text=['# Paper 2 首轮实际实验结果','',f"状态：{result['decision']}；仅探索 DEV，未生成或读取最终 TEST。",
          '',f"新 TRAIN 512 个物理 parent，DEV 192 个；每个 parent 三类干扰、每类四状态。六次从头训练，每次 {cfg['steps']} 更新。",
          '',f"KEEP 完整 U-NMSE：{result['keep']['u_nmse']:.6f}。",
          '', '| seed | 损伤预算 | 最强对照 | 对照 U-NMSE | B4 U-NMSE | 相对改善 | 严重损伤 parent 比例 |',
          '|---|---|---|---:|---:|---:|---:|']
    for r in result['comparisons']:
        text.append(f"| {r['seed']} | {r['budget']} | {r['control']}/{r['control_family']} | {r['control_u_nmse']:.6f} | {r['candidate_u_nmse']:.6f} | {100*r['improvement_fraction']:.2f}% | {100*r['severe_parent_rate']:.2f}% |")
    text+=['','损伤为目标增量误差能量/目标能量；严重损伤 parent 指任一干扰变体 d>0.25。所有均值先按 parent 聚合；没有把四状态或三种干扰当作额外独立样本。',
           '', '对照允许实数 0≤α≤2、预算内解析最优实增益及 DEV 拟合的全局复增益（模≤2）。B4 具有完全相同控制权限。没有依据 TEST 选择权重、预算或工作点。',
           '', '逐模型 alpha=1 诊断在 evidence/*_alpha1.json，全部工作点在 MATCHED_BUDGET.csv，parent bootstrap 在 PILOT_RESULTS.json；置信区间仅描述 DEV 数据，不纠正工作点选择偏差。',
           '', '限制：受限 SNR/JSR、单目标、紧凑骨干与短训练；未复现 CVEDNet，不能将 B2 称为 CVEDNet；未取得两篇最近邻完整方法。不能据此宣布新颖性、目标无损、风险保证或发表级优势。']
    (ROOT/'PAPER2_PILOT_RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    decision=['# Paper 2 GO / NO-GO','',f"结论：**{result['decision']}**。",'',
              '当前判断只针对预注册的六次小试，不是对整条研究方向的最终裁决。',
              f"同时通过两个 seed 预定数值条件的预算：{result['qualifying_budgets']}。",'',
              '若无合格预算：停止扩建当前约束候选，不增加 backbone、专家或风险头；先检查可辨识性、目标/干扰冲突与已完成数据中的互补信息。若有合格预算：只进入有限的独立机制复核，仍不开正式 TEST。',
              '', '立项必需条件仍包括 ISRJ-FNet/MCA-ADMM-Net 全文差异、发表基线公平重训、多目标弱目标评价、至少三个正式训练 seed 和冻结后的独立 TEST。']
    (ROOT/'PAPER2_GO_NO_GO.md').write_text('\n'.join(decision)+'\n',encoding='utf8')


def plot(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    colors=dict(B1='#4267ac',B2='#b17e25',B4='#ae4165')
    for ax,seed in zip(axes,[17,29]):
        for method in ('B1','B2','B4'):
            c=curves[f'{method}_seed{seed}']
            ax.plot([v['damage'] for v in c],[v['u_nmse'] for v in c],label=method,color=colors[method])
        ax.set_xlim(0,.15);ax.set_title(f'Training seed {seed}')
        ax.set_xlabel('Mean target-response damage');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Complete U-NMSE (lower is better)')
    fig.suptitle('Exploratory DEV: real global correction gain 0 to 2')
    fig.tight_layout()
    fig.savefig(EVIDENCE/'DAMAGE_RECOVERY_FRONTIER.png',dpi=180)
    fig.savefig(EVIDENCE/'DAMAGE_RECOVERY_FRONTIER.pdf')
    plt.close(fig)


if __name__=='__main__':
    evaluate()
