"""Render a portable five-seed report from lightweight, verified summaries.

Usage: python tools/build_hybrid_five_seed_report.py --package reports/Hybrid_5seed_20261127
The copied script is also runnable inside the delivered package (no server access).
"""
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#444444']
CORE=['ReCoVP-Hybrid','ReCoVP-Stage1','CorrectionNet-Tail','UNet-Tail','RSDR-Net','CVEDNet']
METRICS=[('u_nmse','E_U / u_nmse (lower better)'),('damage','E_S / target damage (lower better)'),
         ('pd','Detection probability'),('frame_pfa_jammer','Jammer frame false-alarm rate'),
         ('sjrif_db','SJRI-F (dB)'),('amplitude','Target amplitude ratio')]


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def fmt(x):
    if x is None:return '—'
    if abs(x)<.001 and x!=0:return f'{x:.3e}'
    return f'{x:.5f}'


def ms(s,scale=1):
    def number(x):
        x=x*scale
        return f'{x:.3e}' if 0<abs(x)<.001 else f'{x:.4f}'
    return number(s['mean'])+' ± '+number(s['sd']) if s['sd'] is not None else number(s['mean'])


def build(p):
    p=Path(p);p.mkdir(parents=True,exist_ok=True)
    a=json.loads((p/'ANALYSIS.json').read_text(encoding='utf-8'))
    training=json.loads((p/'TRAINING.json').read_text(encoding='utf-8'))
    figures=p/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':180,'pdf.fonttype':42})
    figlist=[]
    def save(fig,name,caption):
        fig.savefig(figures/(name+'.png'));fig.savefig(figures/(name+'.pdf'));plt.close(fig)
        figlist.append((name,caption))
    # Equal-PFA is threshold calibration, not evaluated-PFA interpolation.
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,m in enumerate(CORE):
        e=a['equal_pfa'][m];keys=list(e['curves']);x=np.array([float(k) for k in keys])*100
        y=np.array([e['curves'][k]['pd']['mean'] for k in keys])*100
        sd=np.array([e['curves'][k]['pd']['sd'] or 0 for k in keys])*100
        label=f"{m} (n={len(e['seeds'])})"
        axs[0].plot(x,y,marker=['o','s','^','D','v','P'][i],ms=4,color=COLORS[i],label=label)
        if len(e['seeds'])>1:axs[0].fill_between(x,y-sd,y+sd,color=COLORS[i],alpha=.12)
        actual=np.array([e['curves'][k]['achieved_pfa']['mean'] for k in keys])*100
        axs[1].plot(actual,y,marker=['o','s','^','D','v','P'][i],ms=4,color=COLORS[i],label=label)
    for ax in axs:ax.set(xscale='log',ylabel='Detection probability (%)',ylim=(0,101));ax.grid(alpha=.2)
    axs[0].set_xlabel('Calibration target frame Pfa (%)');axs[1].set_xlabel('Achieved evaluation frame Pfa (%)')
    axs[0].legend(fontsize=8,loc='lower right')
    save(fig,'01_equal_pfa','等虚警阈值策略下的检测曲线。左图横轴为校准目标，阴影为种子 SD；右图横轴为实际评测虚警率。连线仅引导阅读，不代表在评测集上重新选阈值。')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    h=a['equal_pfa']['ReCoVP-Hybrid']
    for ax,target in zip(axs,['0.01','0.05','0.1']):
        y=h['curves'][target]['pd']['values'];ax.plot(h['seeds'],np.array(y)*100,'o-',color=COLORS[0])
        for m,c in [('UNet-Tail',COLORS[3]),('CorrectionNet-Tail',COLORS[2]),('RSDR-Net',COLORS[4])]:
            ax.axhline(a['equal_pfa'][m]['curves'][target]['pd']['mean']*100,color=c,ls='--',label=m)
        ax.set(title=f'Calibration Pfa {float(target)*100:g}%',xlabel='Training seed',ylabel='Detection (%)');ax.grid(alpha=.2)
    axs[2].legend(fontsize=8)
    save(fig,'02_seed_detection','五种子逐点检测结果及基线均值。不同方法的同号 seed 不视作统计配对；seed 17 在 1% 点最低。')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for i,(seed,epochs) in enumerate(training['curves'].items()):
        for ax,k in zip(axs,['total','final_total','stage1_total']):
            ax.plot([x['epoch'] for x in epochs],[x[k] for x in epochs],color=COLORS[i],lw=1,label=seed)
    for ax,k in zip(axs,['Total training objective','Final-stage objective','Stage-1 objective']):
        ax.set(xlabel='Completed epoch',ylabel=k);ax.grid(alpha=.2)
    axs[-1].legend(title='Seed',fontsize=8)
    save(fig,'03_training','各 seed 的完整 256 epoch 训练目标；是训练损失而非独立验证性能，未据此挑选最佳检查点。')
    snrs=sorted({v['snr_db'] for v in a['sweep']['ReCoVP-Hybrid']['cells'].values()})
    jsrs=sorted({v['jsr_db'] for v in a['sweep']['ReCoVP-Hybrid']['cells'].values()})
    for idx,(key,label) in enumerate(METRICS,4):
        fig,axs=plt.subplots(1,2,figsize=(12,4.2),layout='constrained',sharex=True,sharey=True)
        for ax,m in zip(axs,['ReCoVP-Hybrid','RSDR-Net']):
            for j,jsr in enumerate(jsrs):
                cells=sorted([c for c in a['sweep'][m]['cells'].values() if c['jsr_db']==jsr],key=lambda v:v['snr_db'])
                y=np.array([c['metrics'][key]['mean'] for c in cells]);sd=np.array([c['metrics'][key]['sd'] or 0 for c in cells])
                ax.plot(snrs,y,color=COLORS[j],marker=['o','s','^','D'][j],label=f'JSR {jsr} dB')
                if m=='ReCoVP-Hybrid':ax.fill_between(snrs,y-sd,y+sd,color=COLORS[j],alpha=.12)
            ax.set(title=m+(' (5 seeds)' if m=='ReCoVP-Hybrid' else ' (frozen first-paper checkpoint)'),xlabel='SNR (dB)',ylabel=label)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
            if key in ['u_nmse','damage']:ax.set_yscale('log')
            if key in ['pd','frame_pfa_jammer']:ax.set_ylim(-.02,1.02)
        save(fig,f'{idx:02d}_snr_{key}',f'SNR×JSR 受控扫描：{label}。每格 64 个独立物理 parent、3 类干扰；Hybrid 阴影为 5 个 seed 均值的 SD，RSDR 只有一个冻结检查点。此处检测使用固定 cell-CFAR，未做逐格等整帧虚警校准。')
    fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,(k,label) in zip(axs.flat,METRICS):
        delta=np.array([[next(v['metrics'][k]['mean'] for v in a['sweep']['ReCoVP-Hybrid']['cells'].values() if v['snr_db']==snr and v['jsr_db']==jsr)-next(v['metrics'][k]['mean'] for v in a['sweep']['RSDR-Net']['cells'].values() if v['snr_db']==snr and v['jsr_db']==jsr) for snr in snrs] for jsr in jsrs])
        lim=max(abs(delta.min()),abs(delta.max()),1e-9)
        im=ax.imshow(delta,aspect='auto',origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest')
        ax.set(xticks=range(len(snrs)),xticklabels=snrs,yticks=range(len(jsrs)),yticklabels=jsrs,title=label,xlabel='SNR (dB)',ylabel='JSR (dB)')
        fig.colorbar(im,ax=ax,shrink=.8,label='Hybrid - RSDR')
    save(fig,'10_delta_heatmap','所有 32 格的 Hybrid−RSDR 差值。蓝色表示数值更低，是否更好取决于指标；各面板独立色标，未作平滑。')
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for i,m in enumerate(CORE):
        d=a['equal_pfa'][m]
        cc=d['residual_ccdf'];x=np.array(cc['x']);y=np.array(cc['survival']);valid=y>0
        axs[0].plot(x[valid],y[valid],color=COLORS[i],marker=['o','s','^','D','v','P'][i],markevery=12,ms=3,label=m)
        e=d['diagnostics']['e0_modification']['seed_means'];xs=e['values'];ys=d['curves']['0.01']['pd']['values']
        axs[1].scatter(xs,ys,color=COLORS[i],s=35,label=m)
    axs[0].set(xscale='log',yscale='log',xlabel='Residual peak / reference noise power',ylabel='Exceedance probability');axs[0].legend(fontsize=7)
    axs[1].set(xscale='log',xlabel='Noise-only modification E0',ylabel='Detection at calibration Pfa 1%')
    for ax in axs:ax.grid(alpha=.2)
    save(fig,'11_tail_and_preservation','残余峰值的超越概率与背景改动/检测折衷。曲线是 seed×场景混合分布的描述，不作为独立重复；概率为零的尾部不在对数轴显示。')
    fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,(k,label) in zip(axs.flat,METRICS):
        for i,m in enumerate(['ReCoVP-Hybrid','CorrectionNet','UNet','RSDR-Net','CVEDNet','Max-TF','GMM-OMP']):
            cells=a['sweep'][m]['cells'];n=len(a['sweep'][m]['runs'])
            vals=[np.mean([c['metrics'][k]['values'][s] for c in cells.values()]) for s in range(n)]
            ax.scatter([i]*len(vals),vals,color=COLORS[i],s=23)
        ax.set(xticks=range(7),xticklabels=['Hybrid','CorrNet','U-Net','RSDR','CVED','MaxTF','GMMOMP'],ylabel=label);ax.tick_params(axis='x',rotation=40);ax.grid(axis='y',alpha=.2)
    save(fig,'12_grid_all_methods','32 格等权宏平均，各点为一个训练 seed（经典方法/冻结旧模型各一个点）。不同方法的训练数据、预算和种子数不同。')
    # DEV five-seed and operational/increment Doppler distinctions.
    if all(str(s) in a['dev'] for s in (17,29,41,53,67)) and 'rsdr' in a['dev']:
        fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        keys=['u_nmse','damage','amplitude','operational_phase_rmse_deg','operational_coherent_power_db','operational_peak_error_hz']
        for ax,key in zip(axs.flat,keys):
            for i,output in enumerate(['main','stage1']):
                vals=[a['dev'][str(s)]['means'][output+'__'+key] for s in (17,29,41,53,67)]
                ax.scatter([i]*5,vals,color=COLORS[i],s=30)
            ax.scatter([2],[a['dev']['rsdr']['means']['main__'+key]],color=COLORS[4],s=40,marker='D')
            ax.set(xticks=[0,1,2],xticklabels=['Hybrid','Stage 1','RSDR'],ylabel=key);ax.grid(axis='y',alpha=.2)
        save(fig,'13_dev_and_doppler','固定 DEV256 上五种子、同次前向 Stage 1 与 RSDR 的比较。下排采用实际含残余干扰和噪声的输出；相干功率是相对纯目标的有符号 dB，多普勒峰值来自相同 4096 点插值 FFT。')
    if a.get('dev_convolutional_baselines'):
        fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
        controls=a['dev_convolutional_baselines'];names=list(controls)
        for ax,(k,label) in zip(axs.flat,METRICS):
            means=[a['dev_aggregate']['main__'+k]['seed_means']['mean']]+[controls[m]['mean'][k] for m in names]
            sd=[a['dev_aggregate']['main__'+k]['seed_means']['sd']]+[controls[m]['seed_sd'][k] for m in names]
            ax.errorbar(range(13),means,yerr=sd,fmt='o',ms=4,color=COLORS[0],capsize=2)
            ax.set(xticks=range(13),xticklabels=['Hybrid']+names,ylabel=label)
            ax.tick_params(axis='x',rotation=85,labelsize=7);ax.grid(axis='y',alpha=.2)
        save(fig,'14_dev_convolutional_controls','固定 DEV 上全部 12 类卷积对照。点为种子均值，误差线为种子 SD；Hybrid n=5，对照各 n=3。相同编号不表示训练种子配对。')
    if a['wp8']:
        interventions=['stage1_K4_learned','stage1_K8_learned','stage1_K12_learned','stage1_K16_learned','stage1_K16_noreg','final_K16_learned','final_K16_noreg','final_K16_alpha1']
        fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
        for ax,k in zip(axs.flat,['u_nmse','damage','noise_nmse','pd','frame_pfa_jammer','sjrif_db']):
            for i,v in enumerate(interventions):
                xs=[row['means'][v+'__'+k] for row in a['wp8'].values()]
                ax.scatter([i]*len(xs),xs,color=COLORS[0],s=18)
            ax.set(xticks=range(8),xticklabels=['S1 K4','S1 K8','S1 K12','S1 K16','S1 no-reg','Full','Full no-reg','Full alpha=1'],ylabel=k)
            ax.tick_params(axis='x',rotation=65,labelsize=8);ax.grid(axis='y',alpha=.2)
            if k in ['u_nmse','damage','noise_nmse']:ax.set_yscale('log')
        save(fig,'15_frozen_interventions',f'冻结模型的推理期诊断，每点一个 seed，n={len(a["wp8"])}。前三项采用对数轴；这些干预不是重新训练的因果消融。低虚警必须与目标损伤和检测率合看。')
    # Portable CSV source tables.
    with (p/'equal_pfa.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['method','n_seeds','calibration_target_pfa','pd_mean','pd_seed_sd','achieved_pfa_mean','achieved_pfa_seed_sd','parent_ci_low','parent_ci_high'])
        for m,e in a['equal_pfa'].items():
            for target,c in e['curves'].items():w.writerow([m,len(e['seeds']),target,c['pd']['mean'],c['pd']['sd'],c['achieved_pfa']['mean'],c['achieved_pfa']['sd'],*c['pd_parent_ci']['parent_bootstrap_ci95']])
    with (p/'snr_jsr.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['method','n_seeds','snr_db','jsr_db','metric','mean','seed_sd'])
        for m,e in a['sweep'].items():
            for c in e['cells'].values():
                for k,s in c['metrics'].items():w.writerow([m,len(e['runs']),c['snr_db'],c['jsr_db'],k,s['mean'],s['sd']])
    # Report body deliberately distinguishes checkpoint/seed/scene uncertainty.
    lines=['# ReCoVP-Hybrid：五种子综合实验报告','',f"结果汇总时间：{a['created_utc']}。纳入 seed 17、29、41、53、67，全部使用 epoch 256；是 8 种子计划中的阶段性结果。",'',
    '## 1. 结论先行','',
    '五种子证据支持 Hybrid 在恢复误差、目标保真和虚警控制之间形成较好的综合折衷，但不支持“所有指标全面超过第一篇 RSDR-Net”。相对于卷积 Tail 基线，Hybrid 在 1%、5%、10% 校准虚警目标上的检测优势均存在；相对于 RSDR，5% 点略低、10% 点基本持平，1% 严格低虚警点明显落后。',
    '', '受控 SNR×JSR 扫描中，Hybrid 的网格平均恢复误差和目标损伤优于 RSDR，检测率相当；RSDR 的平均 SJRI-F 略高。DEV 与网格扫描分布不同，不能用网格平均代替 DEV，或据单个漂亮数值宣告最终模型冻结。',
    '', '此外，seed 67 的无目标纯噪声改动明显高于其余四个 seed，虽然绝对值仍小，却说明背景透明性存在种子敏感性。剩余 3 个 seed、独立确认集和同机计时仍是后续任务，本报告没有新增训练或打开独立确认集。',
    '', '## 2. 证据范围与统计口径','',
    table(['层次','场景/方法','统计解释'],[
      ['训练','5 seed × 256 epoch；每 seed 12,288 updates','固定末轮，不选 best seed / best checkpoint'],
      ['Equal-PFA','校准 1,024 parent；评测 2,048 parent；每 parent 3 类干扰','各方法/seed 独立校准；评测实测虚警另报'],
      ['固定 DEV','256 parent；direct/repeat/cyclic；4 状态','本次补齐五种子主输出、Stage 1、脉压相干/多普勒'],
      ['受控扫描','8 SNR × 4 JSR × 64 parent = 2,048 parent','32 格等权；检测沿用 cell-CFAR，非 Equal-PFA'],
      ['基线','卷积网络通常 3 seed；RSDR/CVED 各 1 个冻结检查点','不是 5 对 5；训练数据及预算不同'],
      ['不确定性','seed SD 与 parent bootstrap 分开','先平均种子，再按物理 parent 配对重抽样 4,000 次']]),
    '', 'parent 置信区间以已训练检查点和已校准阈值为条件，不包括重新训练或重抽校准集的不确定性。seed 均值的 t 区间另见 SEED_STATISTICS.csv（n=5 时 df=4），不等于场景泛化区间。对 32 格和多个指标的比较为描述性分析，未作多重比较校正；不根据区间筛选指标。',
    '', 'E_U = ||u(S+J+E)−(S+E)||² / ||S+E||²，对应 `u_nmse`；E_S = ||[u(S+J+E)−u(J+E)]−S||² / ||S||²，对应 `damage`。E0 为纯噪声输出相对输入的改动能量比。目标增量 D=u(S+J+E)−u(J+E)，幅度比为 |<S,D>|/||S||²。SJRI-F 按项目冻结脉压/雷达评估实现计算，报告的是场景 dB 的平均，不能与平均线性比值后转 dB 混用。',
    '', '## 3. 五种子的训练完成情况','']
    lines += [table(['seed','epoch / updates','训练总目标','最终分支目标','完成时间 UTC','checkpoint SHA（前 12 位）'],[
      [s,t['epochs_updates'],fmt(t['final']['total']),fmt(t['final']['final_total']),t['completed_utc'],t['checkpoint_sha256'][:12]] for s,t in training['seeds'].items()]),'',
      '五个 seed 共同绑定的 9 个模型/训练核心源文件及训练集 SHA 一致；后续 seed 的清单额外登记多种子调度与评估包装器，不宣称全部文件清单完全相同。各自配置 SHA 随 seed 变化。完整 checkpoint SHA、32 epoch 里程碑摘要和每轮曲线保存在 TRAINING.json，权重留在服务器。训练损失接近不能代替测试稳定性，尤其 seed 17 的 1% 检测率并不代表五种子平均。', '', '![训练收敛](figures/03_training.png)', '',
      '## 4. 等虚警检测：明确优势与失败边界','',
      table(['方法（seed数）','Pd @1%','实测Pfa @1%','Pd @5%','实测Pfa @5%','Pd @10%','实测Pfa @10%'],[
       [f"{m} ({len(a['equal_pfa'][m]['seeds'])})"]+[ms(a['equal_pfa'][m]['curves'][t][k],100)+'%' for t,k in [('0.01','pd'),('0.01','achieved_pfa'),('0.05','pd'),('0.05','achieved_pfa'),('0.1','pd'),('0.1','achieved_pfa')]] for m in CORE]),'',
      '“@5%”等标记表示校准目标而非强制实测值相等。基线在评测集上的虚警通常更低，因此比较是同一预定校准策略的比较，不声称精确相同实测 Pfa 下的支配关系；本次未在评测集上重调阈值或插值选工作点。', '',
      table(['Hybrid − 对照','ΔPd @1% (百分点)','ΔPd @5% (百分点)','ΔPd @10% (百分点)'],[
       [m]+[f"{a['equal_pfa_paired_differences'][m][t]['mean']*100:+.3f} [{a['equal_pfa_paired_differences'][m][t]['parent_bootstrap_ci95'][0]*100:+.3f}, {a['equal_pfa_paired_differences'][m][t]['parent_bootstrap_ci95'][1]*100:+.3f}]" for t in ['0.01','0.05','0.1']] for m in ['CorrectionNet-Tail','UNet-Tail','ReCoVP-Stage1','RSDR-Net']]),'',
      '方括号是配对 parent 95% bootstrap CI。Hybrid 对两种 Tail 网络的检测优势明确；对 RSDR 的 1% 点落后约 17.12 个百分点，是当前最重要的检测短板。5% 点约低 0.41 个百分点；10% 点均值相同，区间包含零。不能用 5%/10% 成绩掩盖严格低虚警端不足。', '',
      '![等虚警曲线](figures/01_equal_pfa.png)', '',
      table(['seed','Pd @1%','Pd @5%','Pd @10%','E0'],[
       [s]+[f"{h['curves'][t]['pd']['values'][i]*100:.3f}%" for t in ['0.01','0.05','0.1']]+[fmt(h['diagnostics']['e0_modification']['seed_means']['values'][i])] for i,s in enumerate(h['seeds'])]),'',
      '![五种子检测](figures/02_seed_detection.png)', '',
      '种子稳定性随工作点变化：Pd 的 seed SD 在 1%、5%、10% 校准点分别为 2.234、0.571、0.225 个百分点。较宽松工作点比较稳定，但 1% 点的种子敏感性不能忽略。这里的 SD 来自五次训练，不能把 6,144 帧当成 6,144 个训练重复。', '',
      table(['干扰类型','Pd @1%','Pd @5%','Pd @10%'],[[v]+[f"{h['curves'][t]['pd_by_variant'][v]*100:.3f}%" for t in ['0.01','0.05','0.1']] for v in ['direct','repeat','cyclic']]),'',
      'repeat 仍是三类中最难的，但在 5% 校准目标上平均达到 85.72%，10% 达到 89.05%；1% 仅 52.29%。这些是独立 Equal-PFA 评测集的结果，不能与旧 DEV 的 73.83% 直接相减归因。', '',
      '## 5. 背景透明性、残余长尾与阶段作用','',
      table(['方法','E0 mean ± seed SD','残余峰值 p99','残余峰值 p99.9'],[[m,ms(a['equal_pfa'][m]['diagnostics']['e0_modification']['seed_means']),fmt(a['equal_pfa'][m]['diagnostics']['resid_peak_noise_units']['pooled_quantiles']['p99']),fmt(a['equal_pfa'][m]['diagnostics']['resid_peak_noise_units']['pooled_quantiles']['p999'])] for m in CORE]),'',
      'Hybrid 的 E0 五种子平均约 1.53e-5，但 seed 67 为 7.23e-5，其余四个仅约 4.44e-7–1.34e-6。不能把 seed 17 的近透明背景推广为每个 seed 的同等保证。CVEDNet 的 E0 接近 1，说明其迁移输出基本改变了全部纯噪声；其高 Equal-PFA Pd 需要与保真损失共同解释。此处不据迁移实验贬低其在原协议上的性能。', '',
      'Hybrid 残余峰值仍有长尾（p99≈76.12、p99.9≈884.63 个参考噪声功率单位），因此“完全消灭残余假峰”不成立。Stage 1 的低维自适应修正不自动保证噪声不变；Stage 2 也不受同一子空间约束。不能套用固定且与噪声独立字典的 K/N 投影期望上界。', '', '![长尾与保真](figures/11_tail_and_preservation.png)', '',
      '经典方法的 Equal-PFA 成绩也保存在 equal_pfa.csv；BCS-oracle 读取真值支持，只是诊断参照，不是可部署对照，不进入主图或主要结论。Max-TF/GMM-OMP 的迁移扫描结果属于当前实现和参数的描述，并不代替第一篇协议的复现验收；旧 DEV 调参失败记录也不会因本报告被改写为通过。', '',
      '## 6. SNR×JSR：与第一篇及其他方法的同场景比较','',
      'SNR = −25、−20、−15、−10、−5、0、5、10 dB；JSR = 0、10、20、30 dB。每格 64 个物理 parent，三种干扰先在 parent 内平均。所有方法使用同一注册扫描场景；没有为本报告生成新数据或使用确认集。这是选定网格的描述性宏平均，不是实际工况频率加权平均。', '',
      table(['方法（seed数）','E_U↓','E_S↓','Pd↑','干扰帧Pfa↓','SJRI-F dB↑','目标幅度比≈1'],[
       [f"{m} ({len(a['sweep'][m]['runs'])})"]+[fmt(a['sweep'][m]['uniform_grid_mean'][k]) for k,_ in METRICS] for m in ['ReCoVP-Hybrid','CorrectionNet','UNet','RSDR-Net','CVEDNet','Max-TF','GMM-OMP']]),'',
      'Hybrid 相对 RSDR 的网格平均 E_U 约降低 36.53%、E_S 约降低 53.11%，目标幅度比由 0.89248 提升到 0.93196；Pd 为 91.305% 对 91.357%，差异很小。干扰帧虚警约 8.73% 对 9.05%，SJRI-F 为 4.221 对 4.467 dB。这里既有改善也有代价，不是全面支配。', '',
      '逐格看，Hybrid 的 E_S 在 32/32 格均低于 RSDR；E_U 在 15/32 格低于 RSDR，说明较好的宏平均不能解释为所有条件都更好。SJRI-F 在 29/32 格低于 RSDR；检测差异的 32 个逐格 parent 区间均包含零。详见完整 snr_jsr.csv 与 ANALYSIS.json 中逐格配对区间；小样本、未校正多重比较，仅作结构化诊断。', '',
      '![差值热图](figures/10_delta_heatmap.png)', '',
      '六类 SNR 曲线分别如下，颜色/标记表示不同 JSR，左右面板分别是 Hybrid 与 RSDR；阴影仅表示 Hybrid 种子 SD：', '']
    for i,(k,label) in enumerate(METRICS,4):lines+= [f'![{label}](figures/{i:02d}_snr_{k}.png)','']
    lines+=['![所有方法网格宏平均](figures/12_grid_all_methods.png)','', '## 7. 固定 DEV、脉压与多普勒补测','']
    if len(a['dev_aggregate']) and all(str(s) in a['dev'] for s in (17,29,41,53,67)) and 'rsdr' in a['dev']:
        d=a['dev_aggregate'];r=a['dev']['rsdr']['means']
        keys=['u_nmse','damage','background_nmse','no_jammer_pooled_nmse','noise_nmse','amplitude','sjrif_db','false_peak_suppression_db','pd','frame_pfa_jammer','frame_pfa_noise']
        lines +=[table(['指标','Hybrid 五种子 mean ± SD','Stage 1 五种子 mean ± SD','RSDR 单检查点'],[[k,ms(d['main__'+k]['seed_means']),ms(d['stage1__'+k]['seed_means']),fmt(r['main__'+k])] for k in keys]),'',
          table(['seed','E_U','E_S','Pd','干扰帧Pfa','SJRI-F dB','幅度比'],[[s]+[fmt(a['dev'][str(s)]['means']['main__'+k]) for k in ['u_nmse','damage','pd','frame_pfa_jammer','sjrif_db','amplitude']] for s in (17,29,41,53,67)]), '',
          '以下相干指标统一对照纯目标脉压输出，取真实目标的距离单元。operational 为实际 u(S+J+E)，保留噪声和干扰残留；increment 为 D，用于分离模型目标响应。两种指标不能混称。相干功率 dB 是有符号功率比，负值代表下降；它不是所有距离单元最大峰值的损失，也不能直接等同旧报告的 absCPL。相位 RMSE 去除公共相位后计算，多普勒峰值误差用相同 4096 点零填充 FFT，只有 8 个实际脉冲，零填充没有提高物理分辨率。','',
          table(['脉压/多普勒指标','Hybrid mean ± seed SD','RSDR','Hybrid−RSDR [parent95%CI]'],[
            [k,ms(d['main__'+k]['seed_means']),fmt(r['main__'+k]),f"{fmt(a['dev_paired_vs_rsdr']['main__'+k]['mean'])} [{fmt(a['dev_paired_vs_rsdr']['main__'+k]['parent_bootstrap_ci95'][0])}, {fmt(a['dev_paired_vs_rsdr']['main__'+k]['parent_bootstrap_ci95'][1])}]"]
            for k in ['operational_coherent_power_db','operational_phase_rmse_deg','operational_peak_error_hz','increment_coherent_power_db','increment_phase_rmse_deg','increment_peak_error_hz']]),'',
          '![DEV 与多普勒](figures/13_dev_and_doppler.png)', '',
          'DEV 五种子 E_U=0.04285，反而高于 RSDR 的 0.03343；E_S=0.05806，低于 RSDR 的 0.09470，下降约 38.69%。目标幅度比从 0.89052 提高到 0.92172。检测率 87.578% 对 86.979%、干扰帧虚警 5.651% 对 5.078%，两项配对区间均包含零，不能判定检测或虚警全面胜出。', '',
          'Stage 2 将 DEV 的 E_U 从 Stage 1 的 0.39580 降到 0.04285，E_S 从 0.12468 降到 0.05806；但 Stage 1 的幅度比更接近 1，Stage 2 带来了额外幅度衰减。低目标损伤、幅度保真和残余抑制必须分别评价，不能以单一幅度指标概括整个阶段的作用。', '',
          '实际输出的相干功率相对纯目标平均为 −0.7780 dB（seed SD 0.0410 dB），RSDR 为 −1.1010 dB，Hybrid 少损失约 0.3230 dB；但尚未达到旧设想的 0.3–0.4 dB 损失。目标增量的相位 RMSE 为 0.9786° 对 1.5627°，插值多普勒峰值误差为 8.6339 对 12.1858 Hz，两项配对区间均支持改善。实际含噪输出的相位 RMSE 为 16.9083° 对 16.8483°、峰值误差为 203.6921 对 200.1339 Hz，差值区间均包含零。目标响应更保真不等于最终含噪多普勒估计已改善。', '',
          f"RSDR 本次复算与原 E7 冻结 DEV 的共同指标最大绝对差为 {max(abs(x) for x in a.get('rsdr_dev_reproduction',{'none':0}).values()):.3e}；检测和帧虚警完全一致。", '',
          table(['DEV 干扰类型','E_U','E_S','Pd','干扰帧Pfa','SJRI-F','幅度比'],[[v]+[fmt(d['main__'+k]['variant_means'][i]) for k in ['u_nmse','damage','pd','frame_pfa_jammer','sjrif_db','amplitude']] for i,v in enumerate(['direct','repeat','cyclic'])]),'',
          'DEV 与独立扫描上的排名变化是分布依赖的实际证据。下结论时应同时引用本节与第 6 节，而非只选有利结果。ANALYSIS.json 还给出每个指标的 parent 中位数、p95、配对区间及逐干扰类型均值。','']
        controls=a.get('dev_convolutional_baselines',{})
        if controls:
            lines +=['### 7.1 全部 12 类卷积对照','',table(['方法（各3 seed）','E_U↓','E_S↓','Pd↑','干扰帧Pfa↓','SJRI-F dB↑','幅度比'],[
                [m]+[fmt(v['mean'][k]) for k,_ in METRICS] for m,v in controls.items()]),'',
                '在这套 DEV 上，Hybrid 的 E_U 均值和干扰帧虚警均低于全部 12 类卷积对照；但 E_S 高于 CN-NoDechirp、UNet、UNet-Tail 和 UNet-Dilation，幅度保真也不是所有对照中最好。不能把恢复误差、虚警控制的优势改写为每一维都最好。全部 Hybrid−对照的 parent 配对区间保存在 dev_comparisons.csv；未按相同 seed 号码强行配对。','',
                '![DEV全部卷积对照](figures/14_dev_convolutional_controls.png)','']
        lines +=['### 7.2 场景分布的尾部','',table(['指标（先在parent内平均3类干扰）','Hybrid parent中位数','Hybrid parent p95'],[
            [k,fmt(d['main__'+k]['parent_median']),fmt(d['main__'+k]['parent_p95'])] for k in ['u_nmse','damage','operational_coherent_power_db','operational_phase_rmse_deg','operational_peak_error_hz','increment_phase_rmse_deg','increment_peak_error_hz']]),'',
            '以上分位数对应先平均五个 seed、再在每个 parent 内平均三类干扰后的分布；不是所有帧混合分位数。相干功率的 p95 是数值较高的一端，不能把它当作损失最严重的 5% 尾部。','']
        lines +=['### 7.3 五次训练的稳定性','',table(['DEV指标','五种子均值','seed SD','最小seed均值','最大seed均值'],[
            [k]+[fmt(d['main__'+k]['seed_means'][field]) for field in ['mean','sd','minimum','maximum']]
            for k in ['u_nmse','damage','amplitude','pd','frame_pfa_jammer','noise_nmse','operational_coherent_power_db']]),'',
            '所有五个 seed 均计入，不因为背景改动较大而删除 seed 67。E0 的均值和 SD 被该 seed 明显拉高，不能仅展示平均检测率就宣称各 seed 在全部指标上同样稳定。完整逐种子数值见 dev_metrics.csv，种子均值的 df=4 t 区间见 SEED_STATISTICS.csv。','']
    else:lines +=['补测尚未完成；本报告不得标记最终交付。','']
    lines+=['## 8. 零训练结构诊断（WP8）','',
      f"已读取 seed {', '.join(a['wp8'])} 的现成结果（本节 n={len(a['wp8'])}，与前述五种子主评测分开计数）。以下改变都是冻结模型的推理期干预，不能当作从头训练的架构消融，也不用于选择最终 K 或门控参数。",'']
    if a['wp8']:
        variants=['stage1_K4_learned','stage1_K8_learned','stage1_K12_learned','stage1_K16_learned','stage1_K16_noreg','final_K16_learned','final_K16_noreg','final_K16_alpha1']
        def wmean(v,k):return float(np.mean([x['means'][v+'__'+k] for x in a['wp8'].values()]))
        lines +=[table(['推理期变体','E_U','E_S','E0','Pd','干扰帧Pfa'],[[v]+[fmt(wmean(v,k)) for k in ['u_nmse','damage','noise_nmse','pd','frame_pfa_jammer']] for v in variants]),'',
          '前 k 个原子按贪心候选峰选择顺序截取，并非按距离排序。K 减少同时改变候选覆盖与后续精修输入；强行取消学习正则或令门控为 1 会造成分布偏移。它们能揭示冻结模型对部件的依赖，不能证明相应组件在重训练后的因果必要性。', '',
          'seed 17 重构与原 DEV 的 E_U 差约 7.3e-8、E_S 差约 1.84e-6、SJRI-F 差约 3.95e-5 dB；Pd 和帧虚警一致。这些小数值差异已保留，未重复旧回执中“所有指标在 1e-6 内”的不准确说法。','']
        lines +=['取消逐原子正则后，完整输出的 E_S 从约 0.058 升到 1.250，Pd 从 87.58% 降到 31.51%；其较低虚警不能解释为更好的干扰抑制。强令门控 α=1 后，E_U≈13.15、E_S≈13.48，说明训练得到的残差幅值控制对当前冻结模型非常关键。减小 Stage 1 的 K 会降低候选覆盖；K=8 的 E_S 小于 K=16，但恢复误差更大，体现预算、目标损伤和残余抑制之间的折衷。','',
            f"五个包装器 K=16 对照的最大相对修正差为 {max(v['sanity_max_rel_diff_stage1_K16_vs_model_correction'] for v in a['wp8'].values()):.3e}。",'',
            '![冻结推理干预](figures/15_frozen_interventions.png)','',
            'WP8 的参数核对给出：Stage 1 为 28,948 个有效实参数，Stage 2 为 28,577 个，总计 57,525。RSDR 适配器登记 3,017,877 个参数元素，训练数据与训练预算也不同。参数量不能代替同机耗时；本报告没有把带并发负载的零散推理耗时包装为公平速度基准，也没有补造未覆盖解析求解的 FLOPs。','']
    lines+=['## 9. 当前能下什么结论，下一步做什么','',
      '1. 可以确认：五种子在 5%/10% 校准工作点上稳定优于两种 Tail 卷积基线；Stage 2 对同次前向 Stage 1 的恢复、目标保真与检测有实证收益。',
      '2. 可以确认：在所注册的 SNR×JSR 网格上，Hybrid 的目标损伤和宏平均恢复误差优于第一篇冻结 RSDR；但严格低虚警检测、部分背景指标、平均 SJRI-F 仍有短板。',
      '3. 不能确认：已跨所有 SNR/JSR/宽度/未受扰比例获得全面优势；已彻底消灭 CFAR 掩蔽；已满足旧任务书 BVR01≤0.15 dB 或 JCR>16.5 dB。当前套件没有同定义的五种子 BVR01/JCR 汇总，不用 E0 或 SJRI-F 替代这些量。',
      '4. 不能确认：模型已完成 8 种子最终验证或一次独立确认集封口。报告不自动冻结最终模型、不追加种子、不改训练损失。保留既有剩余种子流水线，完成后按同一统计口径增量更新。',
      '5. 建议下一步：优先审计 seed 67 的 E0 异常和 1% 虚警端阈值/残余长尾；补齐宽度、支持比例扫描与公平同机计时，再依据预先约定的门槛决定是否微调。微调属于后续独立训练版本，不能与当前五种子混合汇总。', '',
      '## 10. 文件、复现与交接','',
      '- 本报告：REPORT.md；所有图在 figures/，同时提供 PNG 和矢量 PDF。',
      '- 数值与溯源：ANALYSIS.json（含原证据 SHA/服务器相对路径）、TRAINING.json、equal_pfa.csv、snr_jsr.csv、SEED_STATISTICS.csv。原始 parent 数组和权重只保留服务器，不进入 GitHub。',
      '- 逐对照比较：dev_comparisons.csv 给出对 RSDR 和 12 类卷积对照的 DEV 差值及 parent 95% 区间；dev_metrics.csv 给出逐种子主输出/Stage 1 和 RSDR 数值。',
      '- 重新作图：在本文件夹运行 `python build_hybrid_five_seed_report.py --package .`。只读汇总脚本和补测脚本副本也随包交付，执行推理仍要求服务器 B。',
      '- 项目备份：`G:/Project Code/ISRJ_222/reports/Hybrid_5seed_20261127/`。',
      '- 交接备份：`G:/Files/Markdown/ISRJ_2/opus对话/03_分析/Hybrid_5seed_20261127/`。主目录 README.md 的“Hybrid 五种子结果”固定段落持续更新，不另建主目录说明报告。',
      '- 统计仅使用本次列明的五个 seed。其他线程后续新增结果不会自动混入，需重新运行汇总并更新本包。', '',
      '## 图注索引','']
    lines +=[f'- `{name}`：{caption}' for name,caption in figlist]
    (p/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (p/'dev_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['label','epoch','metric','value'])
        for label,v in a['dev'].items():
            for k,x in v['means'].items():w.writerow([label,v['epoch'],k,x])
    with (p/'dev_comparisons.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['comparator','metric','hybrid_minus_baseline','parent_ci95_low','parent_ci95_high'])
        for k,v in a['dev_paired_vs_rsdr'].items():w.writerow(['RSDR-Net',k,v['mean'],*v['parent_bootstrap_ci95']])
        for family,meta in a.get('dev_convolutional_baselines',{}).items():
            for k,v in meta['hybrid_minus_baseline'].items():w.writerow([family,k,v['mean'],*v['parent_bootstrap_ci95']])
    with (p/'SEED_STATISTICS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['protocol','metric','n_seeds','mean','seed_sd','t_ci95_low','t_ci95_high','note'])
        for protocol,values in [('DEV',a['dev_aggregate'])]:
            for k,v in values.items():
                s=v['seed_means'];n=v['n_seeds'];half=2.7764451051977987*s['sd']/math.sqrt(n) if n==5 else float('nan')
                w.writerow([protocol,k,n,s['mean'],s['sd'],s['mean']-half,s['mean']+half,'t interval over independent training seeds on fixed DEV; not a scene CI'])
        for t,v in h['curves'].items():
            s=v['pd'];half=2.7764451051977987*s['sd']/np.sqrt(5)
            w.writerow(['EqualPFA','pd@'+t,5,s['mean'],s['sd'],s['mean']-half,s['mean']+half,'fixed eval+calibration, seed variation only'])
        for cell,v in a['sweep']['ReCoVP-Hybrid']['cells'].items():
            for k,s in v['metrics'].items():
                half=2.7764451051977987*s['sd']/np.sqrt(5)
                w.writerow(['SNR_JSR_'+str(cell),k,5,s['mean'],s['sd'],s['mean']-half,s['mean']+half,'fixed cell, df4 t interval over training seeds only'])
    print('rendered',p,len(figlist),'figures')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    build(parser.parse_args().package)
