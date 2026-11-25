"""Freeze a text-only publication bundle; never writes to live experiment receipts."""
from datetime import datetime,timezone
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

WORK=Path(__file__).resolve().parents[1]
ROOT=WORK/'outputs/recovp_publication_20261125'
DEST=ROOT/'bundle'
PREFIX='research/recovp_20261125'
FORBIDDEN={'.npz','.npy','.pt','.pth','.pkl','.pickle','.log','.csv','.b64','.zlib','.env'}
METRICS={'pd','pd_no_jammer','frame_pfa_jammer','cell_pfa_jammer','frame_pfa_noise','cell_pfa_noise',
 'jcr_db','rd_residual_peak_db','cpl_db','abs_cpl_db','sin2_theta_s','bvr00_db','abs_bvr00_db',
 'bvr01_db','abs_bvr01_db','damage','no_jammer_pooled_nmse','u_nmse','e_U','e_S','e_0_pool',
 'sjrif_db','c_s_real','c_s_imag','drift_over_target_energy','leak_over_target_energy',
 'cross_term_over_target_energy','noise_gain_H0','noise_gain_H1'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,obj):
    path=DEST/path;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')
def text(path,value):
    path=DEST/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding='utf8')

def compact(obj,historical=False):
    if isinstance(obj,list):return [compact(v,historical) for v in obj]
    if not isinstance(obj,dict):return obj
    if 'mean' in obj and 'parents' in obj:
        return {k:v for k,v in obj.items() if k in {'parents','mean','parent_bootstrap_ci95','events','trials','false_cells','valid_cuts','binomial_cp_ci95'}}
    if historical and 'global' in obj and any('/' in k for k in obj):
        return {k:compact(v,historical) for k,v in obj.items() if k in {'global','direct/all','repeat/all','cyclic/all'}}
    skip={'source_sha256','worst_cases','worst_case_attribution','failure_atoms','private_failure_heads','estimated_head_diagnosis',
          'chi_conflicts','unet_tail_worst_cases','fit_trajectory','trajectory','parent_rows','parent_values'}
    return {k:compact(v,historical) for k,v in obj.items() if k not in skip and not (historical and 'jnr' in k.lower())}

def levels(obj,all_strata=True):
    return {label:{k:compact(v) for k,v in row.items() if k in METRICS} for label,row in obj.items()
            if all_strata or label in {'global','direct/all','repeat/all','cyclic/all'}}

def main():
    if DEST.exists():raise FileExistsError('Immutable publication bundle already exists')
    DEST.mkdir(parents=True);now=datetime.now(timezone.utc).isoformat();bindings={}
    sources=sorted((WORK/'tools').glob('vp_1103*.py'))+[WORK/'tools/benchmark_latency_unified.py']+sorted((WORK/'tools/recovp_gpu').glob('*.py'))
    for source in sources:
        ast.parse(source.read_text(encoding='utf8'))
        target=DEST/source.relative_to(WORK);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        bindings[source.relative_to(WORK).as_posix()]=sha(source)
    stages=['p0_snr_jsr_reference','p1a_oracle_vp_audit','p1b_mismatch_drift_audit','p1c_structure_ambiguity_audit',
      'p1d1_replay_dict_sensitivity','p1d2_replay_first_drift_audit','p1d3_hankel_noisy_audit',
      'p2_head_substitution_audit','p2_smoothvp_localization_review','p2_smoothvp_preflight',
      'p2_smoothvp_v2_preflight','p2_smoothvp_v21_preflight','exp0_anchor_audit','exp1_recovp_base_preflight','exp2a_softshare_audit']
    historical=set(stages[1:12])
    for name in stages:
        path=WORK/'evidence/outputs'/name/'SUMMARY.json';source=read(path);digest=sha(path)
        value=compact(source,name in historical)
        if name=='p0_snr_jsr_reference':
            value={'methods':{method:{'seed_average':levels(row['seed_average']),
                'seeds':{s:compact(v) for s,v in row['seeds'].items()}} for method,row in source['methods'].items()}}
        write(Path('results/audits')/f'{name}.json',{'original_summary_sha256':digest,'historical_global_only':name in historical,
            'summary_type':'Aggregate-only compact export; full original kept locally, no historical axis relabeling','results':value})
        bindings[path.relative_to(WORK).as_posix()]=digest
        # Preserve original Markdown only after the native reporting migration.
        for filename in ('REPORT.md','DESIGN.md','PAIRED_COMPARISON.md'):
            src=path.parent/filename
            if src.exists() and not re.search(r'jnr',src.read_text(encoding='utf8'),re.I):
                target=DEST/'reports'/name/filename;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    formal_dir=WORK/'evidence/outputs/recovp_formal_dev';formal=read(formal_dir/'SUMMARY.json')
    ver=read(formal_dir/'DEV_VERIFICATION.json')
    assert sha(formal_dir/'SUMMARY.json')==ver['summary_sha256']
    assert sha(formal_dir/'WINNER.json')==ver['winner_sha256']
    for name,digest in ver['source_sha256'].items():assert sha(WORK/name)==digest,name
    for method,row in formal['methods'].items():
        for slot,score in row['seeds'].items():
            raw=WORK/'outputs/recovp_formal_dev/scoring'/method/slot/'metrics.npz'
            assert sha(raw)==score['metrics_sha256']
            result={'method':method,'native_seed':score['seed'],'epoch':score['epoch'],
                'model_binding':score['identity']['method'],'dev_binding':score['identity']['dev'],
                'metrics_all_native_strata':levels(score['metrics']),
                'scaling':{k:levels(v,False) if k in {'scaled','scaled_minus_standard'} else v for k,v in score['scaling'].items()},
                'three_layer_drift':{k:levels(v,False) for k,v in score.get('three_layer_drift',{}).items()},
                'worst_cases_native_snr_jsr':score['worst_cases'],'private_parent_metrics_sha256':score['metrics_sha256']}
            write(Path('results/formal_dev')/f'{method}_seed_{score["seed"]}.json',result)
    write(Path('results/formal_dev/PAIRED_DELTAS.json'),{name:levels(v) for name,v in formal['paired_winner_seed17_minus_controls'].items()})
    # Copy readonly release/plan/math documentation, not the 29MB live result file.
    docs={'recovp_formal_dev':['REPORT.md','REPRODUCE.md','WINNER.json','AUTHORIZATION.json','FP32_QUICKCHECK.json','METRIC_UNIT_TESTS.json','DEV_VERIFICATION.json'],
          'recovp_isrj1_suite':['REPRODUCE.md','PLAN.json','HARNESS_CHECKS.json'],
          'recovp_gate_summary':['REPORT.md','REPRODUCE.md','NATIVE_POWER_CONVENTION.json','BOUNDARY_NORM_TEST.json','RANDOM_MATH_FINAL.json','CVAR_UNIT_TEST.json'],
          'snr_jsr_protocol':['CONFIG.json']}
    for folder,names in docs.items():
        for name in names:
            source=WORK/'evidence/outputs'/folder/name
            target=DEST/'reports'/folder/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            bindings[source.relative_to(WORK).as_posix()]=sha(source)
    state=read(formal_dir/'STATUS.json');suite=read(WORK/'evidence/outputs/recovp_isrj1_suite/STATUS.json')
    completed=[]
    for arm in ('ReCoVP-Base','ReCoVP-NoEquiv'):
        for seed in (17,29,41):
            p=WORK/'outputs/recovp_formal_dev'/arm/str(seed)/'TRAINING_COMPLETE.json'
            if p.exists():
                r=read(p);cp=p.parent/'epoch_256.pt';assert sha(cp)==r['checkpoint_sha256']
                completed.append({'arm':arm,'seed':seed,'epochs':r['epochs'],'updates':r['updates'],
                    'checkpoint_sha256':r['checkpoint_sha256'],'ended_utc':r['ended_utc']})
    sys.path.insert(0,str(WORK/'tools'))
    from vp_1103_unet_matched_model import MatchedUNet
    import torch
    torch.set_num_threads(1);torch.manual_seed(17);unet=MatchedUNet()
    unet(torch.randn(2,2,1800)).square().mean().backward()
    active=sum(p.numel() for p in unet.parameters() if p.grad is not None)
    assert active==sum(p.numel() for p in unet.parameters())==58943
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in unet.parameters())
    progress={'captured_utc':now,'completed_training_runs':completed,
        'current_training':{k:state[k] for k in ['stage','arm','seed','epoch','last_committed_epoch','updated_utc']},
        'formal_scored_seeds':{k:v['completed_seeds'] for k,v in formal['methods'].items()},
        'winner':formal['winner'],'scan_completed_conditions':suite.get('completed_conditions',0),
        'scan_planned_conditions':579,'scan_planned_method_conditions':2811,'independent_confirmation_unlocked':False,
        'unet_matched58943':{'widths':[21,40,74],'registered_and_active_real_parameters':active,
            'gradient_cpu_check':'PASS','formal_training':'NOT_STARTED','controller':'NOT_IMPLEMENTED_YET'},
        'note':'Snapshot, not a live dashboard. Completed training is distinct from completed DEV scoring.'}
    write(Path('PROGRESS.json'),progress)
    g=lambda name,key:formal['methods'][name]['seed_average']['global'][key]['mean']
    rows=['| 模型 | 帧虚警↓ | Pd↑ | JCR↑ | absCPL↓ | 损伤↓ |','|---|---:|---:|---:|---:|---:|']
    for name in formal['methods']:
        rows.append(f"| {name} | {100*g(name,'frame_pfa_jammer'):.3f}% | {100*g(name,'pd'):.3f}% | {g(name,'jcr_db'):.3f} dB | {g(name,'abs_cpl_db'):.3f} dB | {g(name,'damage'):.6f} |")
    text(Path('README.md'),'# ISRJ_2：ReCoVP 研究快照\n\n'+f'UTC快照：{now}。这是开发/机制证据，不是论文创新或独立确认结论。\n\n'+
        '## 已核验的正式 DEV 结果\n\n同一256个DEV parent；以下轻量模型为seed17，RSDR保留原冻结seed40312102/epoch32及不同训练预算。三类型在parent内平均，不能扩为768独立样本。\n\n'+
        '\n'.join(rows)+'\n\nReCoVP-NoEquiv按预设虚警优先规则获选，但相对Base的虚警差仅-0.260 pp，配对95%区间[-1.172,+0.651] pp跨零，不是显著全面优胜。\n\n'+
        '相对第一篇RSDR：NoEquiv的Pd低3.125 pp（配对区间[-5.339,-0.781] pp）、JCR低5.663 dB；absCPL低0.852 dB，目标增益保持更好。相对P0：虚警低3.646 pp，但Pd也低3.776 pp、旧损伤更高。不能只用虚警或参数量宣称全面跑赢。\n\n'+
        '## 当前执行状态\n\n详细完成种子和当前轮次见 [PROGRESS.json](PROGRESS.json)。本快照的正式DEV仅包含已评分种子，不把完成训练但未评分的seed29/41冒充三种子均值。579工况、2811方法工况计划已冻结，未执行部分明确待完成。\n\n'+
        '新增精确同参U-Net宽度(21,40,74)，实参数58943、CPU全参数梯度检查通过；正式训练与调度尚未开始，不能把代码实现写成已获得消融结果。既有UNet-Tail为59714参数，比ReCoVP多1.31%，其真实对照已保留。\n\n'+
        '## 阅读入口\n\n- [历史路线审计与阴性结果](HISTORY.md)\n- [正式训练复现协议](reports/recovp_formal_dev/REPRODUCE.md)\n- [第一篇对齐扫描协议](reports/recovp_isrj1_suite/REPRODUCE.md)\n- [正式DEV分层结果与配对区间](results/formal_dev/)\n- [机制审计汇总](results/audits/)\n- [CPU预检NO_GO原报告](reports/recovp_gate_summary/REPORT.md)：这是后续GPU人工放行之前的历史结果，不是当前训练状态。\n\n'+
        '## 复现与边界\n\n源码基线5723605，工作树锚点fcf9245。将本目录tools/与所需evidence布局叠加到独立的基线工作树根目录后运行，不要在旧ISRJ目录写入。现有脚本中的历史SHA绑定保持原样；reports/为只读发布副本，原始本地evidence路径由快照收据追溯。\n\n'+
        '私有TRAIN/DEV数据、历史模型权重需在本地按原收据配置；本仓库不提供这些原始文件。所有主统计采用原生snr_db/jsr_db有效支撑功率定义，宽度扫描明确单列JSR_E。历史审计只导出全局/类型汇总，不把旧强度分档重新命名为JSR。\n\n'+
        '数据集、逐parent数组、权重、传输包、原始训练日志及任何CONFIRM/历史TEST数组均未上传。小JSON是明确标注的聚合压缩导出，完整原始本地JSON的SHA256在清单中保留。所有负结果及已完成seed均保留。SoftShare/CVaR尚未正式训练；CONFIRM继续锁定。\n')
    text(Path('HISTORY.md'),'# 路线演进与保留的负结果\n\n'+
        '1. P0完整收尾并锁定RAW_IQ。原生SNR/JSR重整后的三种子P0-Tail均值虚警10.156%、Pd87.326%、损伤0.045742；与当前seed17表不混作同一统计。\n'+
        '2. P1-A已知几何下存在很大解析抵消上限；这是Oracle边界，不是盲推理成绩。P1-B目标优先硬候选暴露算子漂移和误保护；不再作为正式路线。\n'+
        '3. P1-C没有触发独立滞后或Flow主线；Hankel纯分量与含噪混合结论分开。不同profile似然子集与判别定义的结果不可混写成同一个歧义率，原始聚合值均保留在对应JSON。\n'+
        '4. P1-D支持紧凑字典与连续精度，但Oracle/受控邻域不能替代盲定位；二阶段保护未获支持。\n'+
        '5. SmoothVP初版、v2、v2.1与ReCoVP CPU预检的未达标结果保留。早期相位不变ReCoVP数学检查通过，但TRAIN16全流程漂移约0.04392、高JSR拟合残差仍超门槛，曾明确NO_GO。\n'+
        '6. 用户于2026-11-24明确覆盖性能预检暂停，允许FP32特征/CNN、complex128求解正式GPU验证；该授权不删除历史失败、不开放CONFIRM。\n'+
        '7. 正式DEV显示：小参数与目标增益保持优势已出现，检测/抵消尚未超过第一篇。NoEquiv扩种来自预定门槛，不是全面优胜结论；三种子与冻结扫描仍须完整完成。\n\n'+
        '各阶段aggregate-only JSON包括所有配置的保留结果，不筛掉失败lambda/seed；因体积和发布边界，仅省略逐parent数组、冗余分位数及历史旧轴分档。完整本地收据SHA在PUBLICATION_MANIFEST.json中。\n')
    text(Path('.gitignore'),'/data/\n/outputs/\n__pycache__/\n*.pyc\n*.log\n*.npz\n*.npy\n*.pt\n*.pth\n*.b64\n*.zlib\n.env\n')
    files=[];credential=re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|sk-proj-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    for path in sorted(DEST.rglob('*')):
        if not path.is_file():continue
        assert path.suffix not in FORBIDDEN,path
        value=path.read_text(encoding='utf8');assert not credential.search(value),path
        assert path.stat().st_size<2_000_000,(path,path.stat().st_size)
        files.append({'path':path.relative_to(DEST).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    manifest={'captured_utc':now,'repository':'Hxin0203/ISRJ_2','prefix':PREFIX,
        'scope':'Source + aggregate results + negative outcomes; no datasets, per-parent arrays, weights or raw logs',
        'source_local_commit':'fcf9245','numeric_baseline_commit':'5723605','local_original_receipt_sha256':bindings,'files':files}
    write(Path('PUBLICATION_MANIFEST.json'),manifest)
    payload=[{'path':PREFIX+'/'+p.relative_to(DEST).as_posix(),'content':p.read_text(encoding='utf8')} for p in sorted(DEST.rglob('*')) if p.is_file()]
    (ROOT/'FILES.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf8')
    print(json.dumps({'bundle':str(DEST),'files':len(payload),'payload_bytes':(ROOT/'FILES.json').stat().st_size,
        'bundle_bytes':sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),'manifest_sha256':sha(DEST/'PUBLICATION_MANIFEST.json')}))

if __name__=='__main__':main()
