"""Three paired seeds, fixed epoch256, all native radar metrics and parent CIs."""
import numpy as np
import torch
from vp_1103_common import WORK,atomic_json,read,sha,validate_role,provenance,now
from vp_1103_unet_matched_model import MatchedUNet
from vp_1103_recovp_formal_score import evaluate
from vp_1103_snr_jsr import frame_axes,cases
from vp_1103_recovp_formal_stats import summarize
from paper2.evaluation_1103_v4 import restore_model
from tools.durable_io import atomic_write

OUT=WORK/'evidence/outputs/unet_matched58943';PRIVATE=WORK/'outputs/unet_matched58943'
CORE=('pd','frame_pfa_jammer','jcr_db','rd_residual_peak_db','abs_cpl_db','sin2_theta_s','damage','no_jammer_pooled_nmse')

class Adapter:
    def __init__(self,model):self.model=model.cuda().eval()
    def restore(self,x):return restore_model(self.model,x,'cuda',batch_pulses=64)

def run():
    path,binding=validate_role('dev')
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    axes=frame_axes(data);unet=[];recovp=[];rows={}
    for seed in (17,29,41):
        folder=PRIVATE/'UNet-Matched58943'/str(seed);cp=folder/'epoch_256.pt';done=read(folder/'TRAINING_COMPLETE.json')
        assert done['epochs']==256 and done['checkpoint_sha256']==sha(cp)
        rec=WORK/'outputs/recovp_formal_dev/scoring/ReCoVP-NoEquiv'/str(seed)
        source=read(rec/'SCORE.json');assert sha(rec/'metrics.npz')==source['metrics_sha256']
        assert source['identity']['dev']['sha256']==binding['sha256']
        with np.load(rec/'metrics.npz',allow_pickle=False) as z:
            assert np.array_equal(z['parent_id'],data['parent_id'])
            baseline={k:z[k] for k in z.files if k!='parent_id'}
        raw=folder/'dev_metrics.npz';score=folder/'DEV_SCORE.json'
        if score.exists():
            old=read(score);assert old['checkpoint_sha256']==sha(cp) and old['private_metrics_sha256']==sha(raw)
            with np.load(raw,allow_pickle=False) as z:values={k:z[k] for k in z.files if k!='parent_id'}
        else:
            model=MatchedUNet();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
            adapter=Adapter(model);values=evaluate(adapter,data)
            assert all(np.isfinite(v).all() for v in values.values())
            atomic_write(raw,lambda f:np.savez_compressed(f,parent_id=data['parent_id'],**values))
            del model,adapter;torch.cuda.empty_cache()
        deltas={k:baseline[k]-values[k] for k in values}
        row={'seed':seed,'epoch':256,'checkpoint_sha256':sha(cp),'private_metrics_sha256':sha(raw),
            'recovp_checkpoint_sha256':source['identity']['method']['checkpoint_sha256'],
            'metrics':summarize(values,axes),'recovp_minus_unet':summarize(deltas,axes),
            'worst_cases':cases(values,data['parent_id'],axes)}
        atomic_json(score,row);rows[str(seed)]=row;unet.append(values);recovp.append(baseline)
    avg=lambda matrices:{k:np.mean([m[k] for m in matrices],axis=0) for k in matrices[0]}
    u,r=avg(unet),avg(recovp)
    delta=summarize({k:r[k]-u[k] for k in u},axes)
    # No single-metric "winner" declaration: preserve simultaneous improvements and harms.
    directions={'pd':1,'frame_pfa_jammer':-1,'jcr_db':1,'rd_residual_peak_db':-1,
        'abs_cpl_db':-1,'sin2_theta_s':-1,'damage':-1,'no_jammer_pooled_nmse':-1}
    decisions={}
    for key,direction in directions.items():
        m=delta['global'][key];lo,hi=m['parent_bootstrap_ci95']
        decisions[key]={'recovp_minus_unet':m['mean'],'ci95':[lo,hi],
            'significant_recovp_benefit':lo>0 if direction==1 else hi<0,
            'significant_recovp_harm':hi<0 if direction==1 else lo>0}
    harms=[k for k,v in decisions.items() if v['significant_recovp_harm']]
    benefits=[k for k,v in decisions.items() if v['significant_recovp_benefit']]
    conclusion='TRADEOFF' if harms and benefits else 'NO_UNIFORM_WIN_ESTABLISHED' if not benefits or harms else 'BENEFITS_WITHOUT_DETECTED_HARM_NOT_CONFIRMATION'
    summary={**provenance(__file__,WORK/'tools/vp_1103_unet_matched_model.py'),'status':'COMPLETE_DEV_COMPARISON',
        'parameters':{'UNet-Matched58943':58943,'ReCoVP-NoEquiv':58943},'seeds':[17,29,41],'epochs':256,'micro_batch':8,'data':binding,
        'per_seed':rows,'seed_average':{'UNet-Matched58943':summarize(u,axes),'ReCoVP-NoEquiv':summarize(r,axes)},
        'paired_recovp_minus_unet':delta,'criterion_details':decisions,'descriptive_conclusion':conclusion,
        'inference_scope':'Parent-cluster paired CI after averaging the same three seeds; not training-seed uncertainty or independent confirmation',
        'ablation_scope':'Capacity-matched architecture control; does not isolate VP alone because full architectures differ',
        'confirmation_unlocked':False}
    atomic_json(OUT/'SUMMARY.json',summary)
    columns=CORE;lines=['# 精确同参 U-Net 与 ReCoVP-NoEquiv 对照','',
        '两模型均58943有效实参数，同512 TRAIN/256 DEV，seeds17/29/41、256轮、micro8与同Lnew。所有配置按事前协议固定。',
        '这是同预算架构对照，不将完整架构差异归因于单个VP算子。扫描后不重选模型。','',
        '| Model | Seed | '+' | '.join(columns)+' |','|---|---|'+'---:|'*len(columns)]
    for seed in (17,29,41):
        for name,g in [('UNet-Matched58943',rows[str(seed)]['metrics']['global']),('ReCoVP-NoEquiv',summarize(recovp[(17,29,41).index(seed)],axes)['global'])]:
            lines.append('| '+name+' | '+str(seed)+' | '+' | '.join(f"{g[k]['mean']:.6g}" for k in columns)+' |')
    for name,levels in summary['seed_average'].items():
        lines.append('| '+name+' | 3-seed mean | '+' | '.join(f"{levels['global'][k]['mean']:.6g}" for k in columns)+' |')
    lines += ['',f'描述性结论：**{conclusion}**。收益项：{benefits}；损害项：{harms}。',
        '不能只用虚警更低宣称整体跑赢；同步解释Pd、JCR、目标保真和无扰保持。全部原生SNR/JSR分层与配对95%CI在SUMMARY.json。',
        '未读取CONFIRM、历史TEST或旧Gate数组。']
    atomic_write(OUT/'REPORT.md',lambda f:f.write(('\n'.join(lines)+'\n').encode('utf8')))
    atomic_json(OUT/'VERIFICATION.json',{'status':'PASS_FIXED_EPOCH_PAIRED_DEV','ended_utc':now(),
        'summary_sha256':sha(OUT/'SUMMARY.json'),'data':binding,'scored_seeds':[17,29,41],'confirmation_unlocked':False})

if __name__=='__main__':run()
