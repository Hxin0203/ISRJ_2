"""Build an evidence-linked Chinese comparison report, with explicit coverage."""
from __future__ import annotations
import csv
import io
from pathlib import Path

from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_copy, atomic_json, atomic_write
from tools.compare_rsdr_unified import LIGHT, FREEZE
from tools.status_rsdr_unified import snapshot

REPORT=ROOT/'PAPER2_RSDR_UNIFIED_COMPARISON_20261115.md'
NAMES={'main_repeat':'重复转发干扰主比较','main_direct':'直接转发干扰','main_cyclic':'循环转发干扰',
       'main_all':'三类干扰完整主扫描','pd':'重复转发干扰密集 Pd 扫描','pd_jsr20':'JSR = 20 dB 的密集 Pd 曲线',
       'retention':'无干扰保持性','coherent':'相干性','sliceA':'固定周期宽度扫描','sliceB':'固定重复次数宽度扫描'}
METRICS={'u_nmse':'恢复波形 NMSE ↓','damage':'目标响应损伤 ↓','sjrif_db':'SJRIF 代理量 / dB ↑','pd':'检测概率 Pd ↑',
 'cell_pfa_noise':'噪声 H0 单元 Pfa','frame_pfa_noise':'噪声 H0 帧级 Pfa','cell_pfa_jammer':'干扰 H0 单元 Pfa','frame_pfa_jammer':'干扰 H0 帧级 Pfa',
 'no_jammer_pooled_nmse':'无干扰合并 NMSE ↓','pd_no_jammer':'无干扰 Pd ↑',
 'phase_rmse_deg':'相位 RMSE / ° ↓','coherent_power_db':'相干功率比 / dB（理想 0）','peak_error_hz':'多普勒峰误差 / Hz ↓'}


def num(v):
    if v==0:return '0'
    return f'{v:.3e}' if abs(v)<.0001 or abs(v)>=1000 else f'{v:.6f}'


def table(group,keys):
    lines=['| 指标 | 第二版 | 第一版 RSDR-Net | 第二版 − 第一版：配对差值 [95% CI] |',
           '|---|---:|---:|---:|']
    for k in keys:
        if k not in group['metrics']['model']:continue
        a,b=[group['metrics'][m][k]['mean'] for m in ('model','rsdr')]
        d=group['model_minus_rsdr'][k];lo,hi=d['parent_bootstrap_ci95']
        lines.append(f"| {METRICS[k]} | {num(a)} | {num(b)} | {num(d['mean'])} [{num(lo)}, {num(hi)}] |")
    return '\n'.join(lines)


def primary_interpretation(g):
    lines=[]
    for k in ('u_nmse','damage','sjrif_db','pd','frame_pfa_jammer'):
        d=g['model_minus_rsdr'][k];lo,hi=d['parent_bootstrap_ci95']
        favorable_negative=k in ('u_nmse','damage','frame_pfa_jammer')
        better=(hi<0 if favorable_negative else lo>0)
        worse=(lo>0 if favorable_negative else hi<0)
        verdict='第二版数值更好，点态区间未跨 0' if better else '第一版数值更好，点态区间未跨 0' if worse else '配对区间包含 0，未分辨出稳定差异'
        lines.append(f"{METRICS[k]}：{verdict}。")
    lines.append('上述判断只描述冻结模型在这份测试分布上的配对结果。Pd 受实际虚警水平影响；Pfa 较低也可能伴随过度抑制，因此不以单项数值宣布整体优胜。未做多重比较校正。')
    return '\n\n'.join(lines)


def primary_overview(g):
    a,b=[g['metrics'][m] for m in ('model','rsdr')]
    return ('冻结的重复转发干扰主比较已经完整。以下均按第二版／第一版排列：'
        f"恢复 NMSE 为 {a['u_nmse']['mean']:.6f}／{b['u_nmse']['mean']:.6f}，"
        f"目标响应损伤为 {a['damage']['mean']:.6f}／{b['damage']['mean']:.6f}，"
        f"SJRIF 为 {a['sjrif_db']['mean']:.4f}／{b['sjrif_db']['mean']:.4f} dB，"
        f"Pd 为 {100*a['pd']['mean']:.3f}%／{100*b['pd']['mean']:.3f}%，"
        f"干扰 H0 帧级虚警率为 {100*a['frame_pfa_jammer']['mean']:.3f}%／{100*b['frame_pfa_jammer']['mean']:.3f}%。"
        '这些指标存在权衡，不能概括为第二版全面超过第一版。'
        'NMSE 约束总波形能量误差，而 SJRIF 和 CFAR 对峰值及局部背景敏感；不同排序并不矛盾，当前比较尚未证明差异的具体机制。'
        '配对差值与区间见后表。')


def primary_snr_table():
    path=LIGHT/'CURVE_SOURCE_SUMMARIES.csv'
    if not path.exists():return ''
    with path.open(encoding='utf8',newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if r['figure']=='main_repeat']
    if not rows:return ''
    cells={(float(r['snr_db']),r['metric'],r['method']):float(r['mean']) for r in rows}
    lines=['### 重复转发干扰按 SNR 展开',
        '每行等权平均四个冻结 JSR，仍为同一批 256 个配对父样本。这是完整曲线的描述性展开，不能把跨 SNR 的总体均值解释为每个 SNR 都改善。总体区间见前表，逐条件事件数与区间见 CSV。',
        '| SNR / dB | 第二版 NMSE | 第一版 NMSE | 第二版 Pd / % | 第一版 Pd / % | 第二版干扰帧 Pfa / % | 第一版干扰帧 Pfa / % |\n|---:|---:|---:|---:|---:|---:|---:|']
    for snr in sorted({k[0] for k in cells}):
        vals=[cells[snr,k,m] for k in ('u_nmse','pd','frame_pfa_jammer') for m in ('model','rsdr')]
        lines.append(f'| {snr:g} | {vals[0]:.6f} | {vals[1]:.6f} | {100*vals[2]:.3f} | {100*vals[3]:.3f} | {100*vals[4]:.3f} | {100*vals[5]:.3f} |')
    return '\n\n'.join(lines[:2])+ '\n\n'+'\n'.join(lines[2:])


def report():
    a=read(LIGHT/'PAIRED_RESULTS.json');f=read(FREEZE);ids=read(LIGHT/'MODEL_IDENTITIES.json');s=snapshot(save=True)
    full=a['status']=='COMPLETE';primary=a['groups'].get('main_repeat',{})
    state='完整比较已完成' if full else '主比较已完成，扩展扫描仍在运行' if primary.get('complete') else '比较正在进行，主比较尚未完整'
    if not full and s['status'] not in ('RUNNING','COMPLETE'):state='扩展扫描已停止，已保存结果保留；完整覆盖尚未完成'
    lines=[f'# 第二版模型与第一版完整 RSDR-Net 的统一口径直接比较\n',f'状态：**{state}**。报告生成时间：{now()}。科学汇总快照：{a["checked_utc"]}。',
        f'已审计 {a["complete_conditions"]}/{a["expected_conditions"]} 个配对条件，共 {a["parent_evaluations"]:,} 次父样本条件评价。后者包含同一物理父样本的重复干预，不能作为独立样本量。',
        '本次新增的是第一版模型在第二版冻结 TEST 上的推理，第二版结果直接复用已完成、SHA 校验通过的同一样本结果。没有重新训练，也没有依据此次 TEST 重选 256 轮模型。旧目录只用于读取模型源码、权重和身份说明，没有读取旧 held-out Gate 数组。',
        '## 模型身份与可比性\n',
        '| 项目 | 第二版 | 第一版 |\n|---|---|---|',
        '| 部署模型 | CorrectionNet，seed 29，epoch 128 | 最终完整 F-R7 RSDR-Net，seed 40312102，epoch 32 |',
        f'| 可学习参数 | {ids["paper2_parameters"]:,} | {ids["rsdr_total_parameters"]:,} |',
        '| 训练预算 | 128 × 48 = 6,144 次更新 | 每个训练组件 32 × 512 = 16,384 次更新，且包含多阶段训练 |',
        '| 推理结构 | 预测干扰校正量 C，输出 x − C(x) | 两个恢复专家、固定凸锚点、三个路由器和有界脉冲组门控 |',
        f'第二版参数量约为第一版的 {ids["paper2_parameters"]/ids["rsdr_total_parameters"]*100:.2f}%，减少约 {(1-ids["paper2_parameters"]/ids["rsdr_total_parameters"])*100:.2f}%。这是部署图参数量比较，不是延迟或吞吐量测量。',
        '第一版按原有输入状态顺序运行后映射回统一顺序；输入均为原始 IQ，不增加外部归一化。移植推理已与原运行时做数值一致性检查，详见 `evidence/outputs/RSDR_UNIFIED_20261115/ADAPTER_PARITY.json` 和 `EXTENDED_PARITY.json`。工程一致性样本不进入科学检验。',
        '第二版 epoch 128 来自原三 seed 的平均 DEV 恢复误差选轮，seed 29 为预先固定的中间 seed，并非 TEST 最优 seed。选择规则形成于 DEV 结果之后、TEST 之前。8 个 seed 的 epoch 256 是 DEV 结果，本次不将其当作新的独立 TEST 证据。两版训练分布和训练成本不同。',
        '## 统一的数据、处理链和统计方法\n',
        '两版使用逐父样本完全相同的 S、J、E 和元数据，以及四个状态 E、S+E、J+E、S+J+E。原有 TEST 每个块与此次 RSDR-Net 块均核验 SHA、有限值、结果均值及父样本身份。不同模型没有使用各自的测试集。',
        '主扫描 SNR 为 −30、−28、−26、−25、−24、−23、−22、−20、−15、−10、−5、0、5、10 dB，JSR 为 0、10、20、35 dB，三类干扰各 56 个条件、每条件 256 个配对物理父样本。预先固定的主要比较范围是 repeat 的 56 个条件。',
        '密集 Pd 扫描 SNR 为 −30 至 0 dB、步长 1 dB，JSR 为 −10 至 35 dB、步长 5 dB，共 310 个条件、每条件 512 个父样本；另外包含 42 个无干扰保持性、3 个相干性和 41 个宽度干预条件。所有网格沿用原冻结实验。',
        '脉压使用 600 点 LFM 模板和完整线性匹配滤波；保留 0 至 1200 的有效距离时延。8 脉冲做慢时间 FFT。二维 CA-CFAR 的距离保护／参考半宽为 12／24，多普勒为 1／2（周期边界），参考单元数 436，每帧 9032 个有效 CUT，名义 Pfa = 10⁻⁶。检测命中在真实最近多普勒 bin、目标距离 ±1 单元判断。真实值只用于离线计分。',
        '同一套检测器和名义 Pfa 并不保证相同的经验 Pfa。本报告同时列出噪声 H0 与干扰 H0 的单元及帧级虚警率，因此 Pd 不应脱离虚警表单独解读。没有用 TEST 校准检测阈值。',
        '统计时先在每个物理父样本内对指定网格等权平均，再对两模型差值做 4,000 次父样本配对 bootstrap，固定 seed 20261114。无干扰保持性的三个独立父样本组分层重采样、等权合并。不能把同一父样本的多个 SNR／JSR 单元或 8 个脉冲当作独立重复。区间为点态 95%，未校正多重比较，也不包含训练 seed 方差。逐条件二项事件同时保存事件数、分母与 Clopper–Pearson 区间；观察到零事件不能解释为真实概率为零。',
        '## 指标的数学含义\n',
        '令 T 为部署的恢复算子，U = S+E。恢复波形误差为',
        '$$\\mathrm{NMSE}_U=\\frac{\\|T(S+J+E)-(S+E)\\|_2^2}{\\|S+E\\|_2^2}.$$',
        '它同时惩罚残余干扰和对目标／原噪声的改变。目标响应损伤用共享干扰及噪声的有限差分定义：',
        '$$D_S=\\frac{\\|T(S+J+E)-T(J+E)-S\\|_2^2}{\\|S\\|_2^2}.$$',
        '两者均在父样本的全部脉冲和快时间上计算能量比，越小越好。对非线性模型，有限差分衡量加入目标后的响应变化，不应解释为可物理分离出的真实目标分量。整体恢复误差与目标响应损伤可以给出不同排序。',
        '无干扰保持性将 T(E) 对 E、T(S+E) 对 S+E 的误差能量相加后除以对应总能量。它衡量无干扰时改动多大；在数据的物理生成假设下，理想恒等映射为 0。',
        '本次 SJRIF 沿用统一脉压指标：令 P(r) 为 8 脉冲平均距离功率，rₜ 为 target-only 脉压峰，R(P)=10 log₁₀[P(rₜ)/max_{r≠rₜ}P(r)]，SJRIF=R(Pout)−R(Pin)。这是目标 bin 对最强其他 bin 的改善代理量，含旁瓣与噪声影响，并非使用 oracle 干扰能量的物理 SJR 增益。',
        'Pd 是目标存在时的检测事件均值；Pfa 在目标缺席的 E 和 J+E 上分别计分。相干性按相同目标 bin 的慢时间相位 RMSE、相干功率比和谱峰频率误差衡量；4096 点零填充仅插值频谱，不提高 8 脉冲的物理分辨率。',
        '## 当前覆盖\n','| 扫描 | 已审计 / 计划 |\n|---|---:|']
    if primary.get('complete'):lines.insert(4,primary_overview(primary))
    for name,v in a['coverage'].items():lines.append(f'| {NAMES.get(name,name)} | {v["complete"]} / {v["expected"]} |')
    for name,g in a['groups'].items():
        if not g['complete']:continue
        lines+=['\n## '+NAMES[name]+'\n',f'完整范围：{g["conditions"]} 个条件；独立物理父样本 {g["independent_parents"]} 个。']
        keys=('no_jammer_pooled_nmse','damage','pd_no_jammer','cell_pfa_noise','frame_pfa_noise') if name=='retention' else ('phase_rmse_deg','coherent_power_db','peak_error_hz','damage','pd') if name=='coherent' else ('u_nmse','damage','sjrif_db','pd','cell_pfa_noise','frame_pfa_noise','cell_pfa_jammer','frame_pfa_jammer')
        lines.append(table(g,keys))
        if name=='main_repeat':lines+=['\n'+primary_interpretation(g),primary_snr_table()]
    if not primary.get('complete'):lines.append('\n主比较尚未完整，因此此快照不据局部低 SNR 结果给出总体排序。')
    if (LIGHT/'CONDITION_COMPARISONS.csv').exists():
        with (LIGHT/'CONDITION_COMPARISONS.csv').open(encoding='utf8',newline='') as stream:rows=list(csv.DictReader(stream))
        co=[r for r in rows if r['suite']=='coherent' and r['metric'] in ('phase_rmse_deg','coherent_power_db','peak_error_hz')]
        if co:
            lines+=['\n## 相干性逐条件结果\n','每条件 16 个父样本，样本量较小；不能只凭平均值推广到所有多普勒场景。',
                '| SNR / dB | JSR / dB | 指标 | 第二版 | 第一版 |\n|---:|---:|---|---:|---:|']
            for r in co:lines.append(f'| {r["snr_db"]} | {r["jsr_db"]} | {METRICS[r["metric"]]} | {num(float(r["paper2_mean"]))} | {num(float(r["rsdr_mean"]))} |')
    lines+=['\n## 图表和可核验文件\n',
        '图表目录为 `evidence/outputs/RSDR_UNIFIED_20261115/figures/`，当前已生成文件以 `FIGURE_MANIFEST.json` 为准。曲线只在对应范围完整后生成，阴影为父样本点态 95% 区间；密集网格的未完成单元保留灰色空缺。全扫描完成后，相干性示例按原冻结 parent 0 的全部三个条件生成，不挑选有利示例；脉压与多普勒采用共同归一化和颜色范围。',
        '模型身份：`MODEL_IDENTITIES.json`；配对汇总：`PAIRED_RESULTS.json`；逐条件指标、事件数与区间：`CONDITION_COMPARISONS.csv`；逐块 SHA 与样本配对审计：`RESULT_INTEGRITY_AUDIT.json`；工程移植记录：`ENGINEERING_EVENTS.json`；图来源与 SHA：`FIGURE_MANIFEST.json`。上述文件均在 `evidence/outputs/RSDR_UNIFIED_20261115/`。',
        f'比较冻结清单：`configs/rsdr_unified_20261115.json`，SHA-256：`{sha(FREEZE)}`。原始数组、模型权重、缓存、完整日志只保存在本地 `outputs/rsdr_unified_20261115/`，不上传 GitHub。',
        '## 结论的适用边界\n',
        '本次是已知第二版 TEST 后追加的冻结基线比较，未用于调参或模型选择；不将其描述为在所有结果未知前预注册的独立确认。比较仅覆盖当前仿真器分布、每版一个冻结训练 seed。8 seed 的训练稳定性不能代替两版在共同独立 TEST 上的 8 seed 比较。',
        '如果某项 Pd 改善伴随更高 Pfa，只能表述为当前固定检测器工作点发生变化，不能声称在等虚警水平下提高检测率。若后续需要等经验 Pfa 的结论，应先用独立校准集冻结阈值，再使用全新的确认数据；本次不据 TEST 调阈值。',
        '如 NMSE、目标损伤和 SJRIF 给出不同排序，应保留这种权衡，不能将它们合并成事后设计的总分。此次没有启动 MCA-ADMM、FNet、消融或重新训练。']
    if not full:lines.append(f'\n扩展扫描状态：{s["status"]}。尚未完成的条件不能当作零值、不能外推为完成结果；全部结束后控制器会重新审计并刷新本报告。')
    chunks=[]
    for part in lines:
        # Keep Markdown table rows contiguous while separating prose blocks.
        sep='\n' if chunks and chunks[-1].rstrip().endswith('|') and part.lstrip().startswith('|') else '\n\n'
        chunks.append((sep if chunks else '')+part)
    raw=(''.join(chunks)+'\n').encode('utf8')
    if REPORT.exists():
        backup=ROOT/'outputs/rsdr_unified_20261115/report_history'/f'{REPORT.stem}_{sha(REPORT)}.md'
        if not backup.exists():atomic_copy(REPORT,backup)
    atomic_write(REPORT,lambda stream:stream.write(raw))
    atomic_json(LIGHT/'REPORT_RECEIPT.json',dict(created_utc=now(),report=str(REPORT.relative_to(ROOT)),sha256=sha(REPORT),
        paired_results_sha256=sha(LIGHT/'PAIRED_RESULTS.json'),status=a['status'],primary_complete=primary.get('complete',False)))
    print(dict(report=str(REPORT),status=state),flush=True)
    return REPORT


if __name__=='__main__':report()
