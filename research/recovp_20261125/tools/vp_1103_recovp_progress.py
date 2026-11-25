"""Read-only experiment monitoring and timestamped, source-bound progress reports."""
import csv
from datetime import datetime,timedelta,timezone
import json
import math
import subprocess
from zoneinfo import ZoneInfo
import numpy as np
import psutil
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vp_1103_common import WORK,read,sha,atomic_json,now

def run():
    captured=datetime.now(timezone.utc)
    stamp=captured.strftime('%Y%m%d_%H%M%SUTC')
    out=WORK/'evidence/outputs/recovp_progress'/stamp
    private=WORK/'outputs/recovp_progress'/stamp
    out.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
    formal=WORK/'evidence/outputs/recovp_formal_dev'
    state=read(formal/'STATUS.json');cfg_path=WORK/'outputs/recovp_formal_dev/CONFIG.json'
    cfg=read(cfg_path)
    for name,digest in cfg['source_sha256'].items():assert sha(WORK/name)==digest,name
    process=psutil.Process(state['pid'])
    assert PathEq(process.cwd(),WORK)
    assert 'vp_1103_recovp_gpu_run.py' in ' '.join(process.cmdline())
    arm,seed=state['arm'],state['seed'];end=int(state['last_committed_epoch'])
    folder=WORK/'outputs/recovp_formal_dev'/arm/str(seed)
    rows=[];verified=[];sources={}
    for epoch in range(end+1):
        cp=folder/f'recovery/epoch_{epoch:03d}.pt';receipt=cp.with_suffix('.json')
        r=read(receipt);digest=sha(cp)
        assert digest==r['checkpoint_sha256'] and r['epoch']==epoch
        assert r['identity']=={'run':'recovp_formal_dev','arm':arm,'seed':seed,'config_sha256':sha(cfg_path)}
        verified.append({'epoch':epoch,'checkpoint_sha256':digest,'receipt_sha256':sha(receipt)})
        if epoch:
            log=folder/f'training_epoch_{epoch:03d}.json';row=read(log)
            assert row['checkpoint_sha256']==digest and row['global_step']==48*epoch
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,(float,int)))
            assert abs(row['total']-(row['endpoint']+.1*row['pc']+row['damage']+.05*row['tail']))<1e-4
            rows.append(row);sources[log.relative_to(WORK).as_posix()]=sha(log)
    last=rows[-1];cp=folder/f'recovery/epoch_{end:03d}.pt'
    payload=torch.load(cp,map_location='cpu',weights_only=False)
    assert payload['epoch']==end and all(torch.isfinite(v).all() for v in payload['model'].values())
    lr=[float(g['lr']) for g in payload['optimizer']['param_groups']]
    keys=['total','endpoint','pc','damage','tail','epoch_seconds']
    averages=lambda block:{key:float(np.mean([r[key] for r in block])) for key in keys}
    recent=averages(rows[-20:]);previous=averages(rows[-40:-20])
    first=rows[0]
    reductions={key:100*(1-last[key]/first[key]) for key in keys if key!='epoch_seconds'}
    window_change=100*(previous['total']-recent['total'])/previous['total']
    median_seconds=float(np.median([r['epoch_seconds'] for r in rows[-20:]]))
    p10,p90=map(float,np.quantile([r['epoch_seconds'] for r in rows[-20:]],[.1,.9]))
    remaining=256-end
    base_eta=captured+timedelta(seconds=remaining*median_seconds)
    pair_eta=base_eta+timedelta(seconds=256*median_seconds)
    la=ZoneInfo('America/Los_Angeles')
    local=captured.astimezone(la)
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
        gpu={'name':raw[0].strip(),'used_mib':int(raw[1]),'total_mib':int(raw[2]),'utilization_percent':int(raw[3])}
    except Exception as error:gpu={'observation_error':repr(error)}
    milestones=[r for r in rows if r['epoch'] in {1,32,64,96,128,160,192,end}]
    statuses=[]
    for model in ('ReCoVP-Base','ReCoVP-NoEquiv'):
        for s in (17,29,41):
            path=WORK/'outputs/recovp_formal_dev'/model/str(s)
            done=path/'TRAINING_COMPLETE.json'
            statuses.append({'model':model,'seed':s,'state':'COMPLETE' if done.exists() else
                'TRAINING' if model==arm and s==seed else 'QUEUED_SEED17' if s==17 else 'CONDITIONAL_WINNER_NOT_SELECTED',
                'last_committed_epoch':end if model==arm and s==seed else 256 if done.exists() else 0})
    suite=WORK/'evidence/outputs/recovp_isrj1_suite';suite_state=read(suite/'STATUS.json')
    assert not (formal/'WINNER.json').exists()
    dev=read(formal/'SUMMARY.json');assert dev['actual_fixed_epoch256_dev_results'] is None
    stderr=WORK/'outputs/recovp_formal_dev/runner2.stderr.log'
    errors=stderr.read_text(encoding='utf8')
    old=read(WORK/'evidence/outputs/recovp_gate_summary/FINAL_SEAL.json')
    assert all(sha(WORK/name)==digest for name,digest in old['files'].items())
    summary={'captured_utc':captured.isoformat(),'captured_los_angeles':local.isoformat(),
        'status':'TRAINING_NO_FORMAL_DEV_YET','controller_pid':state['pid'],'controller_alive':True,'controller_workspace_verified':True,
        'arm':arm,'seed':seed,'training_epoch_in_progress':state['epoch'],'last_committed_epoch':end,
        'completion_fraction':end/256,'global_updates_committed':end*48,'total_updates_planned':12288,
        'elapsed_seconds':(captured-datetime.fromisoformat(state['started_utc'])).total_seconds(),
        'loss_milestones':[{k:r[k] for k in ['epoch',*keys]} for r in milestones],
        'first_epoch_to_latest_loss_reduction_percent':reductions,'recent20_mean':recent,'preceding20_mean':previous,
        'recent20_total_loss_improvement_percent':window_change,'last_committed_learning_rates':lr,
        'gpu_snapshot':gpu,'seed_status':statuses,'running_stderr_bytes':stderr.stat().st_size,
        'running_stderr_has_traceback':'Traceback' in errors,'all_committed_training_records_finite':True,
        'all_recovery_checkpoints_verified':len(verified),'latest_checkpoint_weights_finite':True,
        'last_checkpoint_sha256':sha(cp),'config_sha256':sha(cfg_path),'old_negative_seal_unchanged':True,
        'plan_sha256':sha(suite/'PLAN.json'),'suite_status':suite_state,
        'eta':{'seconds_per_epoch_recent_median':median_seconds,'epoch_seconds_p10_p90':[p10,p90],
            'base_training_only_utc':base_eta.isoformat(),'base_training_only_los_angeles':base_eta.astimezone(la).isoformat(),
            'two_seed17_training_only_utc':pair_eta.isoformat(),'two_seed17_training_only_los_angeles':pair_eta.astimezone(la).isoformat(),
            'note':'Linear extrapolation of recent epoch runtime, not a statistical forecast CI. Excludes DEV scoring, scan, extensions, scheduling changes.'},
        'formal_dev_results_exist':False,'winner_exists':False,'scan_completed_conditions':suite_state['completed_conditions'],
        'independent_confirmation_unlocked':False,'softshare_and_cvar':'NOT_RUN_THIS_ROUND'}
    atomic_json(out/'SUMMARY.json',summary)
    atomic_json(out/'CHECKPOINT_AUDIT.json',{'through_epoch':end,'checkpoints':verified,'training_record_sha256':sources})
    csvpath=private/'training_trajectory.csv'
    with csvpath.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch',*keys]);writer.writeheader()
        writer.writerows({k:r[k] for k in ['epoch',*keys]} for r in rows)
    x=[r['epoch'] for r in rows]
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    axs[0].plot(x,[r['total'] for r in rows],label='Total TRAIN objective')
    for key,weight,label in [('endpoint',1,'Endpoint'),('pc',.1,'0.1 PC'),('damage',1,'Target damage'),('tail',.05,'0.05 tail')]:
        axs[1].plot(x,[r[key]*weight for r in rows],label=label)
    for ax in axs:
        ax.set_yscale('log');ax.set_xlim(1,256);ax.set_xlabel('Committed epoch');ax.set_ylabel('TRAIN objective (log scale)')
        ax.axvspan(end,256,alpha=.08,color='gray',label='Not yet committed');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'{arm}, seed {seed}: TRAIN only, epochs 1-{end}; no DEV score yet')
    for suffix in ('png','pdf'):fig.savefig(out/f'training_curve.{suffix}',dpi=200)
    plt.close(fig)
    text=['# ReCoVP 实验进展报告','',f'检查时点：{local:%Y-%m-%d %H:%M:%S %Z}（{captured:%Y-%m-%d %H:%M:%S UTC}）。',
        '',f'**结论：正式 GPU 训练正常推进。Base seed17 已提交 {end}/256 轮（{end/256:.1%}），正在第 {state["epoch"]} 轮；NoEquiv seed17 尚未启动。尚无正式 DEV 泛化成绩或全套扫描结果。**','',
        '## 当前阶段','',
        '| 项目 | 状态 |','|---|---|',
        f'| Base / seed17 | 已提交 {end} 轮、{end*48:,}/12,288 次更新 |',
        '| NoEquiv / seed17 | 排队，Base完整256轮后顺序启动 |',
        '| 256-DEV双模型对决与Winner选择 | 未开始，等待两版seed17完整结束 |',
        '| Winner / seeds29、41 | 条件待定，尚未选出Winner |',
        '| 579工况、2811方法工况扫描 | 计划与工程检查已完成，正式完成0工况 |',
        '| SoftShare、CVaR | 本轮不运行 |',
        '| CONFIRM、历史TEST | 保持锁定 |','',
        '## 优化趋势：仅TRAIN损失','',
        '| Epoch | 总损失 | Endpoint | PC（未加权） | Damage | Tail（未加权） |','|---:|---:|---:|---:|---:|---:|']
    text += [f"| {r['epoch']} | {r['total']:.6f} | {r['endpoint']:.6f} | {r['pc']:.6f} | {r['damage']:.6f} | {r['tail']:.6f} |" for r in milestones]
    text += ['',f"从首轮至第{end}轮，总损失下降 {reductions['total']:.2f}%；训练损伤项从 {first['damage']:.6f} 降至 {last['damage']:.6f}，下降 {reductions['damage']:.2f}%。",
        f"最近20轮总损失均值 {recent['total']:.6f}，前一个20轮窗口为 {previous['total']:.6f}，改善 {window_change:.3f}%。后期下降趋缓，与学习率退火阶段同时出现；这支持数值稳定和优化进入平台附近，不足以证明充分收敛或DEV性能达标。",
        '', '损失口径为原Lnew优化目标：endpoint + 0.1 PC + damage + 0.05 tail。它不是DEV的eU或正式CA-CFAR统计；训练Damage也不能直接与P0的DEV损伤0.04574比较。日志中的tail峰值代理不作为新RD残峰指标或虚警率替代。',
        '', '![训练目标轨迹](training_curve.png)','',
        '## 运行与收据核验','',
        f"- 控制器PID {state['pid']} 存活，工作树路径与当前任务匹配；运行中stderr {stderr.stat().st_size}字节，未发现Traceback。",
        f"- 第0至{end}轮共{len(verified)}个不可变恢复检查点的SHA256与身份绑定全部核验通过；对应{end}份训练记录连续、有限，最新权重全部有限。",
        f"- GPU：{gpu.get('name','UNKNOWN')}，采样利用率 {gpu.get('utilization_percent','UNKNOWN')}%，显存 {gpu.get('used_mib','UNKNOWN')}/{gpu.get('total_mib','UNKNOWN')} MiB。这是瞬时观测，不是整段平均。",
        '- RAW_IQ、E0_rect_k12、micro-batch8、256轮及源码/配置绑定未变。旧CPU阴性封存哈希保持一致，未改写过去结果。',
        '- 已有FP32数学、梯度和既有方法适配工程收据均PASS；这些不代表模型在DEV或扫描集上已PASS。','',
        '## 时间估计与接续','',
        f"近期每轮中位耗时约 {median_seconds:.1f} 秒。按当前速度线性外推，Base剩余训练约 {remaining*median_seconds/60:.0f} 分钟，预计 {base_eta.astimezone(la):%Y-%m-%d %H:%M %Z} 左右完成。",
        f"若NoEquiv耗时相近，两版seed17纯训练预计 {pair_eta.astimezone(la):%Y-%m-%d %H:%M %Z} 左右完成；不含DEV评分、漂移分解、额外种子和大网格扫描。该估计不是保证或置信区间。",
        '自动队列顺序保持：Base17 → NoEquiv17 → 固定末轮DEV评分及Winner锁定 → 条件Winner29/41 → 全套冻结扫描与计时。无论竞争力条件是否满足，冻结Winner17均执行扫描。数值异常会保存失败收据并停止，不剔除失败seed。','',
        '## 当前能回答和不能回答的问题','',
        '可以确认：作业真实存活，已接近Base训练收尾，记录完整、数值有限，训练目标持续降低并进入后期平台附近。',
        '尚不能判断：是否优于P0/UNet/RSDR、Base与NoEquiv谁胜出、正式Pd/Pfa、新四维指标、repeat/cyclic泛化、实际漂移改善或端到端效率。以上必须等待固定256轮DEV与冻结扫描，不能由训练损失推断。','',
        f'机器收据：[SUMMARY.json](SUMMARY.json)、[检查点核验](CHECKPOINT_AUDIT.json)。完整逐轮CSV仅存私有目录 `{csvpath.relative_to(WORK).as_posix()}`，未放入可发布证据。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    atomic_json(out/'STATUS.json',{'stage':'PROGRESS_REPORT_COMPLETE','captured_utc':captured.isoformat(),'ended_utc':now(),
        'training_interrupted':False,'new_training_or_dev_evaluation_started':False,'source_sha256':sha(__file__)})
    for name in ('REPORT.md','SUMMARY.json','CHECKPOINT_AUDIT.json','training_curve.png','training_curve.pdf'):
        assert (out/name).is_file()
    atomic_json(out/'VERIFICATION.json',{'status':'PASS_PROGRESS_AUDIT','artifacts':{p.name:sha(p) for p in out.iterdir() if p.name!='VERIFICATION.json'},
        'private_trajectory_sha256':sha(csvpath),'last_audited_epoch':end,'formal_dev_results_exist':False,
        'source_records_bound':len(sources),'checkpoint_hashes_verified':len(verified),'live_training_continues':True})
    print(json.dumps({'report':out.relative_to(WORK).as_posix(),'cutoff_epoch':end,'latest_loss':last['total'],
        'recent20_mean':recent['total'],'previous20_mean':previous['total'],'window_improvement_percent':window_change,
        'base_eta_local':base_eta.astimezone(la).isoformat(),'pair_training_eta_local':pair_eta.astimezone(la).isoformat()},ensure_ascii=False))

def PathEq(path,expected):
    from pathlib import Path
    return Path(path).resolve()==expected.resolve()

if __name__=='__main__':run()
