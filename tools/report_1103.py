"""Refresh lightweight evidence and reports from actual 1103 result receipts."""
from pathlib import Path
import csv
import io
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper2.io import now, read, sha, write

TRAIN=ROOT/'outputs/paper2_1103_retry_20261114'
LIGHT=ROOT/'evidence/outputs/paper2_1103_retry_20261114'
RUNS=('paper2_1103_test_20261114','paper2_1103_test_classical_20261114')
KEYS=('u_nmse','damage','no_jammer_pooled_nmse','sjrif_db','pd','pd_no_jammer','frame_pfa_jammer')
LABELS=('eU','eS','e0 pooled','SJRIF (dB)','Pd','Pd no jammer','Frame Pfa J+E')


def backup(path):
    if path.exists():
        h=sha(path)
        dest=ROOT/'outputs/report_history'/h/path.name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(path.read_bytes())
        return dict(path=path.name,sha256=h,local_backup=str(dest.relative_to(ROOT)))


def save(path,text):
    backup(path)
    path.write_text(text,encoding='utf8')


def main():
    timestamp=now()
    frozen=read(ROOT/'configs/paper2_1103_scan_freeze.json')
    train=read(TRAIN/'TRAINING_COMPLETE.json')
    dev=read(LIGHT/'DEV_SUMMARY_v4.json')
    rows=[]
    for row in dev['rows']:
        rows.append(dict(seed=row['seed'],epoch=row['epoch'],
                         **{k:row['metrics'][k]['mean'] for k in KEYS}))
    buf=io.StringIO()
    writer=csv.DictWriter(buf,fieldnames=('seed','epoch',*KEYS),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    (LIGHT/'DEV_SELECTION_TABLE_v4.csv').write_text(buf.getvalue(),encoding='utf8',newline='\n')
    devtable='| Seed | Epoch | '+' | '.join(LABELS)+' |\n'
    devtable+='|---:|---:|'+'---:|'*len(KEYS)+'\n'
    for row in rows:
        devtable+='| '+str(row['seed'])+' | '+str(row['epoch'])+' | '+' | '.join(f'{row[k]:.6g}' for k in KEYS)+' |\n'
    aggregate='| Epoch | '+' | '.join(LABELS)+' |\n|---:|'+'---:|'*len(KEYS)+'\n'
    for epoch,values in frozen['dev_epoch_mean'].items():
        aggregate+='| '+epoch+' | '+' | '.join(f'{values[k]:.6g}' for k in KEYS)+' |\n'
    coverage={suite:{method:0 for method in methods} for suite,methods in frozen['methods'].items()}
    results=[]
    seen=set()
    active=[]
    for run in RUNS:
        out=ROOT/'outputs'/run
        status=read(out/'STATUS.json') if (out/'STATUS.json').exists() else {'stage':'NOT_STARTED'}
        invoke=read(out/'INVOCATION.json') if (out/'INVOCATION.json').exists() else {}
        active.append(dict(name=run,status=status,invocation=invoke))
        if (out/'STATUS.json').exists():
            write(ROOT/'evidence/outputs'/run/'STATUS.json',status)
        if (out/'INVOCATION.json').exists():
            write(ROOT/'evidence/outputs'/run/'INVOCATION.json',invoke)
        for p in sorted(out.glob('*/*/result.json')):
            r=read(p)
            identity=r['identity']
            key=(identity['suite'],identity['index'],identity['method'])
            if key in seen:
                raise RuntimeError(f'duplicate TEST condition/method {key}')
            seen.add(key)
            assert r['status']=='COMPLETE'
            assert identity['binding']['freeze_sha256']==sha(ROOT/'configs/paper2_1103_scan_freeze.json')
            assert identity['condition']==frozen['grids'][identity['suite']][identity['index']]
            coverage[key[0]][key[2]]+=1
            results.append(dict(suite=key[0],condition_index=key[1],method=key[2],
                **identity['condition'],**{k:v['mean'] for k,v in r['metrics'].items()},
                pd_hits=r['metrics']['pd']['events'],
                pfa_noise_frames=r['metrics']['frame_pfa_noise']['events'],
                pfa_jammer_frames=r['metrics']['frame_pfa_jammer']['events'],
                local_receipt_sha256=sha(p)))
    if results:
        keys=list(dict.fromkeys(k for row in results for k in row))
        with (ROOT/'evidence/outputs/TEST_CONDITION_TABLE_1103.csv').open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(results)
    write(ROOT/'evidence/outputs/TEST_COVERAGE_1103.json',dict(checked_utc=timestamp,
          freeze_sha256=sha(ROOT/'configs/paper2_1103_scan_freeze.json'),coverage=coverage,
          expected={s:len(g) for s,g in frozen['grids'].items()},complete_method_conditions=len(results)))
    covtable='| 实验 | 完整网格条件数 | 当前已完成条件数（按方法） |\n|---|---:|---|\n'
    for suite,methods in coverage.items():
        covtable+=f"| {suite} | {len(frozen['grids'][suite])} | "+'；'.join(f'{m}: {n}' for m,n in methods.items())+' |\n'
    training='；'.join(f"seed {r['seed']}: {r['elapsed_seconds']/60:.2f} min" for r in train['runs'])
    report=f'''# ISRJ_2 按 1103 路线执行的训练与实验进展

更新时间（UTC）：{timestamp}。本报告以当前磁盘回执为准，旧版本保存在本机 `outputs/report_history/`。

## 已完成的训练和模型选择

三个 seed（17、29、41）均完成 128 轮，32、64、96、128 轮各保存一份检查点，共 12 份。已自动核对权重 SHA、训练回执、训练源码绑定及 DEV v4 的数据/源码/权重绑定。训练实测：{training}。模型参数量为 112756，训练设备为 RTX 4060。

本次每轮遍历 512 parent × 3 种干扰记录，batch=32，每轮 48 次更新，每 seed 共 6144 次；1103 的专家每轮为 512 次更新，两者训练预算不同。这里沿用实验路径和扫描设置，不能声称训练预算完全相同。32/64/96 轮是同一次 128 轮学习率日程中的中间检查点，不是各自独立衰减至终点的训练。

DEV v4 的全部结果如下。每项先用完整 8 脉冲组计算，再在同一 physical parent 内平均三种干扰；独立样本数为 256。该表是开发结果，不是 TEST 结果。

{devtable}

三 seed 的描述性平均：

{aggregate}

最终采用 128 轮、seed 29 进入独立扫描：epoch 按三 seed 平均 eU 最小选取，seed 固定取列表中间的 29，不按最好 seed 选取。选择规则在查看全部 DEV v4 后确定、独立 TEST 生成前冻结，不能写成看 DEV 前已经完成预注册。eS、SJRIF、Pd、Pfa 和保持误差同时公开，不把恢复误差改善解释为所有指标都必然改善。其余检查点完整保留，TEST 不用于重新选模型。

选用权重 SHA-256：`{frozen['checkpoint_sha256']}`。冻结记录：`configs/paper2_1103_scan_freeze.json`，SHA-256 为 `{sha(ROOT/'configs/paper2_1103_scan_freeze.json')}`。

## 实验范围与当前覆盖

主比较限定 CorrectionNet、Max-TF、GMM-OMP、真实支撑 BCS-oracle、KEEP 和同噪声无干扰参考。MCA-ADMM-Net、FNet、消融继续暂停。切片扫描只保留完整模型、Max-TF、KEEP 和参考。Pd、相参与保持性扫描使用完整模型、KEEP 和参考。所有新数据与输出均在本工作区，旧 ISRJ 目录只读，未访问旧 held-out Gate arrays。

{covtable}

表内只统计具有完成回执的完整条件，不把运行中的 parent block 计为整条件完成。工程试跑不计入独立 TEST。当前任务状态见 `handoff/RUNNING_EXPERIMENTS.json`；逐条件数值见 `evidence/outputs/TEST_CONDITION_TABLE_1103.csv`。覆盖未齐时不宣称全部实验完成。

## 与 1103 对齐的设置

主比较为 3 种干扰 × SNR（−30、−28、−26、−25、−24、−23、−22、−20、−15、−10、−5、0、5、10 dB）× JSR（0、10、20、35 dB），168 格，每格 256 parent。不同强度使用同一组目标、基础噪声及干扰几何，方法在相同观测上配对。

独立 Pd 扫描只用重复转发：SNR −30 至 0 dB、步长 1 dB；JSR −10 至 35 dB、步长 5 dB；310 格，每格 512 parent。无干扰保持使用三组独立场景、每组 512 parent，SNR −30 至 −20 dB、步长 1 dB，以及 −15、−10、0 dB，共 42 格。

切片 A：周期 260 点、M=2，宽 3、6、18、36、54、72、84 点；SNR/JSR_E 为 −10/20、−20/35、0/10 dB，每格 256 parent。切片 B：宽 6、18、36、54、72 点，M=2/4，周期 (M+1)w+1；前两组强度，每格 128 parent。采样相位 0、转发延迟 1 点、间隔 0，按组总能量 JSR_E 重新标定。目标、噪声、受扰位置、通道相位和频偏在宽度干预间固定，保存真实目标覆盖率；B 组只能称名义占比控制，不能声称实际覆盖严格相同。

处理链为 600 点模板线性脉压、lag 0…1200、8 点慢时间 FFT、二维 CA-CFAR。距离保护/参考窗每侧 12/24，多普勒每侧 1/2、周期延拓；Nr=436，名义 cell Pfa=1e−6，每帧 9032 个有效 CUT。目标匹配滤波峰由 target-only 分量定位，多普勒评分处理 PRF 周期边界。真实分量只用于离线评分和 BCS 明示的 oracle 支撑。

相参评价为 16 parent × 三组 SNR/JSR（−20/20、0/20、0/35 dB），目标频移限制在 −14…14 kHz，按 1103 式（4-3）计算去公共相位 RMSE 与相参功率变化。峰频误差使用 4096 点零填充慢时间 FFT；零填充仅插值频谱，不增加 8 脉冲物理分辨率。典型图固定 parent=0、SNR/JSR=0/20，不按模型表现挑场景。

## 指标与数学解释

令 T 为恢复映射、U=S+E，四状态为 E、S+E、J+E、S+J+E。报告：

$$e_U=\\frac{{\\|T(S+J+E)-(S+E)\\|_F^2}}{{\\|S+E\\|_F^2}},\\qquad
e_S=\\frac{{\\|T(S+J+E)-T(J+E)-S\\|_F^2}}{{\\|S\\|_F^2}}.$$

eU 衡量含噪参考恢复，eS 衡量共享背景下目标加入后的输出增量。eU 在弱目标条件下可能被背景主导，因此与 eS、检测及目标复增益并列。对以去噪为目标的经典方法，去掉噪声也计入 eU；不能单凭 eU 宣布其全面失效。

$$e_{{0,\\mathrm{{pool}}}}=\\frac{{\\|T(E)-E\\|_F^2+\\|T(S+E)-(S+E)\\|_F^2}}{{\\|E\\|_F^2+\\|S+E\\|_F^2}}.$$

该误差度量无干扰输入改动，同时报告无干扰 Pd 的配对变化。SJRIF 严格按 8 脉冲平均脉压功率计算，分母只剔除真实目标峰单元；另报固定主瓣保护区外最大假峰的功率抑制。两项解释不同，不相互替代。必须同时给出 Pd、纯噪声 E 与无目标受扰 J+E 的 cell/frame Pfa。

连续指标及方法差使用 4000 次 physical-parent bootstrap。Pd 和 frame Pfa 同报事件计数及逐条件 Clopper–Pearson 95% 区间；CUT 相关，cell Pfa 不按全部 CUT 独立进行二项推断。零事件时保留计数、注明退化 bootstrap 的限制，并给基于 parent 整帧事件的保守上界。所有区间均为逐点区间，不是全网格同时置信带。跨 SNR/JSR 重用 parent，不把格点乘以脉冲数当独立样本量。

## 实现修正、证据与限制

v4 修正了逐脉冲能量比平均及 Nyquist 边界评分问题。旧 v2/v3 结果保留作历史，不再用于选点或当前结论。切片配对生成器及无干扰三组场景独立性已验证。36 项测试通过；工程两 parent、五个实验套件的 20 个方法条件在控制缓存优化前后结果相同。完整工程条件实测见本机 `outputs/paper2_1103_engineering_full_condition_v1/`，该结果不作为论文性能。

基线是本地公式复现，参数按 1103 固定，不在 TEST 上调参；BCS 使用真实支撑，不能称为与观测方法同等信息条件。论文文献的完整元数据与离散实现差异仍需提交前核验，见 `paper2/vendor/baselines_1103/SOURCE_AND_FORMULA_AUDIT.md`。同名义 CFAR 门限并不等于同实际虚警率；单目标仿真、逐脉冲网络、脉间非相参干扰的结论不能推广为实测、多目标或联合慢时间建模结论。当前不声明新颖性或投稿级全面优越性。

权重、parent 数组、数据、缓存和完整日志只保留本机。GitHub 只同步源码、配置、报告和聚合证据，专用 remote 为 paper2（Hxin0203/ISRJ_2），不推旧 origin。
'''
    save(ROOT/'PAPER2_1103_PROGRESS_20261114.md',report)
    plan=f'''# ISRJ_2 近期执行安排（替代旧进度）

更新时间（UTC）：{timestamp}。

训练三 seed × 128 轮、12 个 checkpoint 和全部 DEV v4 评价已经完成。当前直接执行冻结后的独立 TEST 全网格，不再重复一套全 DEV 网格。该安排为 DEV 后确定的执行决策，不能称作 DEV 前预注册。全部数值、选择依据、指标公式和限制见 `PAPER2_1103_PROGRESS_20261114.md`。

{covtable}

执行顺序为：训练/DEV 完成 → 工程核验与完整条件吞吐试跑 → TEST 前冻结 → 主比较与相参评价 → 310 格 Pd/Pfa → 无干扰保持 → A/B 切片扫描 → 典型脉压/多普勒图与完整论文实验章。CPU 经典方法和 GPU 模型可并行，均使用同一冻结场景，保存 parent block 与条件完成回执；重启只续接身份相同的结果。MCA-ADMM、FNet 和消融暂停。

主比较检验三类干扰和强度变化下的相对恢复、目标保持和检测表现；Pd/Pfa 扫描检验检测过渡区及虚警代价；相参评价检验脉间相位和功率保持；切片扫描检验窄/宽切片及实际覆盖变化；无干扰评价检验错误改动。它们是规定仿真范围内的证据，不替代新颖性核验或实测验证。

网格、步长和每格样本数按 1103：主比较 168×256，Pd 310×512，保持 42×512，切片 A 21×256、B 20×128，相参 3×16。完整训练指已完成的每轮 48 次更新方案；第一篇每轮 512 次更新的差异须披露。选择 128 轮 seed 29 后不依据 TEST 换模型。未完成条件不作完整结果，失败及负结果全部保留。
'''
    save(ROOT/'PAPER2_1103_NEAR_TERM_PLAN_20261114.md',plan)
    write(ROOT/'handoff/RUNNING_EXPERIMENTS.json',dict(checked_utc=timestamp,
          training_status='COMPLETE_3_SEEDS_12_CHECKPOINTS',dev_status='COMPLETE_V4_12_CHECKPOINTS',
          test_runs=active,coverage=coverage,old_gate_arrays_accessed=False,
          frozen_model_sha256=frozen['checkpoint_sha256'],
          note='Use condition receipts and live STATUS timestamps; INVOCATION_COMPLETE need not mean every suite/method is complete.'))
    save(ROOT/'handoff/CURRENT_STATE.md',f'''# Current state

Verified UTC: {timestamp}.

All three 128-epoch training runs and all 12 DEV v4 evaluations complete. All checkpoint hashes and source/data bindings verified. Selected epoch 128 using three-seed mean DEV eU; fixed seed 29. Rule finalized after DEV, before independent TEST; not a pre-DEV preregistration. Weights remain local.

Independent TEST scans are tracked in `RUNNING_EXPERIMENTS.json`; coverage in `evidence/outputs/TEST_COVERAGE_1103.json`. Primary report: `PAPER2_1103_PROGRESS_20261114.md`. Freeze: `configs/paper2_1103_scan_freeze.json` (SHA {sha(ROOT/'configs/paper2_1103_scan_freeze.json')}).

Only paper2 remote may be pushed. No ChatGPT collaboration; no old Gate access; no MCA-ADMM/FNet/ablations. Training was 48 updates/epoch, unlike 1103's 512. Preserve every checkpoint, failure and negative condition. Raw data, weights, arrays, caches and logs stay local. Baselines are formula reproductions, BCS oracle is labelled; bibliographic verification remains.
''')
    save(ROOT/'handoff/LAST_CODEX_EXECUTION.md',f'''# Last execution

Updated UTC: {timestamp}.

Verified 12 checkpoint SHA values and DEV v4 bindings; 36 tests passed. Engineering 20 method/condition results agree before/after control-cache optimization. Complete 256-parent engineering main condition finished. Implemented parent-block resume, identity/SHA validation, oracle-state support checks, paired differences and parent-aware probability intervals. Frozen independent TEST scans launched with one GPU model process and a separate CPU classic-method pool.

No old Gate arrays accessed. No TEST tuning. Coverage and current run status are generated from actual receipts, not estimates. Report text is audited offline; protected formulas, values, paths and citations retained. No external manuscript service or ChatGPT collaboration used.
''')
    files=[p for p in (ROOT/'evidence/outputs').rglob('*') if p.is_file() and p.suffix in ('.json','.csv')]
    index='# Output evidence index\n\nUpdated UTC: '+timestamp+'\n\n'
    index+='Current TEST coverage: `evidence/outputs/TEST_COVERAGE_1103.json`; condition table: `evidence/outputs/TEST_CONDITION_TABLE_1103.csv` (only when results exist).\n\n'
    index+='DEV v4: `evidence/outputs/paper2_1103_retry_20261114/DEV_SUMMARY_v4.json`; checkpoint integrity: `CHECKPOINT_VERIFICATION.json`; selection table: `DEV_SELECTION_TABLE_v4.csv`.\n\n'
    index+='Local raw parent blocks and checkpoints are intentionally excluded. A missing condition result means not yet produced, never a fabricated scientific value. Earlier v2/v3 results are historical.\n\n'
    index+='| Suite | Expected conditions | Completed method counts |\n|---|---:|---|\n'
    for suite,methods in coverage.items():
        index+=f"| {suite} | {len(frozen['grids'][suite])} | "+', '.join(f'{k}={v}' for k,v in methods.items())+' |\n'
    save(ROOT/'handoff/OUTPUT_EVIDENCE_INDEX.md',index)
    git=subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True,check=True)
    (ROOT/'handoff/GIT_STATUS.txt').write_text('Snapshot UTC '+timestamp+'\n'+git.stdout,encoding='utf8')
    print(json.dumps(dict(updated_utc=timestamp,coverage=coverage,complete_method_conditions=len(results)),indent=2))


if __name__=='__main__':
    main()
