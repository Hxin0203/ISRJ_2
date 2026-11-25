"""Fixed epoch256 DEV competition using paired physical radar metrics."""
import numpy as np
import torch
from vp_1103_common import WORK,atomic_json,read,sha,now,provenance,validate_role
from vp_1103_recovp_methods import Method
from vp_1103_recovp_radar_metrics import metrics,PRIMARY,DEFINITION
from vp_1103_snr_jsr import frame_axes,cases
from vp_1103_recovp_formal_stats import summarize
from vp_1103_recovp_drift import three_layers
from paper2.evaluation_1103_v4 import states_from_components
from tools.unet_ablation.runtime import configure_precision
from tools.durable_io import atomic_write,ProcessLock

OUT=WORK/'evidence/outputs/recovp_formal_dev'
PRIVATE=WORK/'outputs/recovp_formal_dev'
SOURCES=[WORK/'tools'/f for f in ('vp_1103_recovp_formal_score.py','vp_1103_recovp_methods.py',
 'vp_1103_recovp_radar_metrics.py','vp_1103_recovp_fp32.py','vp_1103_recovp_fp32_ablation.py',
 'vp_1103_recovp_fp32_anchors.py','vp_1103_recovp_drift.py','vp_1103_snr_jsr.py','vp_1103_recovp_formal_stats.py')]

def evaluate(method,data,scales=None):
    rows={};n=len(data['parent_id'])
    for lo in range(0,n,4):
        hi=min(n,lo+4)
        s,j,e=(data[key][lo:hi].astype(np.complex128) for key in ('s','j','epsilon'))
        if scales is not None:
            c=scales[lo:hi,None,None,None];s,j,e=(a*c for a in (s,j,e))
        s,j,e=(a.reshape(-1,8,1800) for a in (s,j,e))
        observed=states_from_components(s,j,e);u=method.restore(observed)
        values=metrics(s,j,e,observed,u,np.repeat(data['doppler_hz'][lo:hi],3))
        for key,value in values.items():rows.setdefault(key,[]).append(value.reshape(hi-lo,3))
    return {k:np.concatenate(v) for k,v in rows.items()}

def score(name,seed,data,binding):
    method=Method(name,seed);folder=PRIVATE/'scoring'/name/str(seed)
    folder.mkdir(parents=True,exist_ok=True)
    file=folder/'metrics.npz';receipt=folder/'SCORE.json'
    identity={**provenance(*SOURCES),'method':method.binding,'dev':binding}
    if receipt.exists():
        old=read(receipt);assert old['identity']==identity and sha(file)==old['metrics_sha256']
        method.close();return old
    values=evaluate(method,data);axes=frame_axes(data)
    rng=np.random.default_rng(20261124)
    c=rng.uniform(.2,5.,len(data['parent_id']))*np.exp(1j*rng.uniform(0,2*np.pi,len(data['parent_id'])))
    scaled=evaluate(method,data,c)
    scaling={'scope':'Same DEV parents, paired global complex scaling; native metadata unchanged',
        'random_seed':20261124,'amplitude_range':[.2,5.],'phase_range':[0,float(2*np.pi)],
        'scaled':summarize(scaled,axes),'scaled_minus_standard':summarize({k:scaled[k]-values[k] for k in values},axes)}
    extra={}
    if name.startswith('ReCoVP-'):
        layer,error=three_layers(method.model,data)
        extra={'three_layer_drift':{k:summarize(v,axes) for k,v in layer.items()},'identity_errors':error}
    atomic_write(file,lambda f:np.savez_compressed(f,parent_id=data['parent_id'],**values))
    atomic_write(folder/'scaled_metrics.npz',lambda f:np.savez_compressed(f,parent_id=data['parent_id'],**scaled))
    worst=cases(values,data['parent_id'],axes)
    row={'identity':identity,'seed':method.binding['seed'],'epoch':method.binding.get('epoch',256),
        'comparison_seed_slot':seed,'metrics':summarize(values,axes),
        'metrics_sha256':sha(file),'scaling':scaling,'worst_cases':worst,**extra}
    if name.startswith('ReCoVP-'):
        from vp_1103_recovp_failures import failure_heads
        row['private_failure_heads']=failure_heads(method.model.cpu(),data,worst,folder)
    atomic_json(receipt,row);method.close();return row

def run():
    configure_precision();torch.set_num_threads(2)
    started=now();path,binding=validate_role('dev')
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    axes=frame_axes(data);rows={}
    for name in ('ReCoVP-Base','ReCoVP-NoEquiv','P0-NoDechirp-Tail','UNet-Tail','RSDR-Net'):
        seeds=[17]
        if name.startswith('ReCoVP-'):
            seeds += [s for s in (29,41) if (PRIVATE/name/str(s)/'epoch_256.pt').exists()]
        rows[name]={str(seed):score(name,seed,data,binding) for seed in seeds}
    def selection_key(name):
        g=rows[name]['17']['metrics']['global']
        return (g['no_jammer_pooled_nmse']['mean']>5e-4,g['frame_pfa_jammer']['mean'],-g['pd']['mean'],
                g['rd_residual_peak_db']['mean'],-g['jcr_db']['mean'],g['abs_cpl_db']['mean'],g['sin2_theta_s']['mean'],name)
    winner=min(('ReCoVP-Base','ReCoVP-NoEquiv'),key=selection_key)
    g=rows[winner]['17']['metrics']['global']
    extend=g['frame_pfa_jammer']['mean']<=.115 and (g['abs_cpl_db']['mean']<=.8 or g['sin2_theta_s']['mean']<=.05) and g['no_jammer_pooled_nmse']['mean']<=5e-4
    winning={'arm':winner,'seed':17,'seed17_checkpoint_sha256':rows[winner]['17']['identity']['method']['checkpoint_sha256'],
        'extend_seeds':bool(extend),'suite_required':True,'independent_confirmation_unlocked':False,
        'selection_rule':'no-jammer safety, then Pfa asc, Pd desc, RD peak asc, JCR desc, absCPL asc, angle asc; seed17 only',
        'extension_rule':'Pfa<=.115 AND (mean_abs_CPL<=.8 OR angle<=.05) AND no-jammer-NMSE<=5e-4',
        'dev_sha256':binding['sha256']}
    winpath=OUT/'WINNER.json'
    if winpath.exists():assert read(winpath)==winning,'Frozen winner changed'
    else:atomic_json(winpath,winning)
    methods={}
    for name,seeds in rows.items():
        arrays=[]
        for seed,row in seeds.items():
            with np.load(PRIVATE/'scoring'/name/seed/'metrics.npz',allow_pickle=False) as z:
                arrays.append({k:z[k] for k in z.files if k!='parent_id'})
        mean={k:np.mean([a[k] for a in arrays],0) for k in arrays[0]}
        methods[name]={'seeds':seeds,'seed_average':summarize(mean,axes),
            'completed_seeds':[r['seed'] for r in seeds.values()],
            'comparison_slots':list(seeds),'note':'RSDR retains native seed40312102/epoch32; comparison slot17 is not a new RSDR training seed.' if name=='RSDR-Net' else ''}
    paired={}
    for control in ('ReCoVP-Base','ReCoVP-NoEquiv','P0-NoDechirp-Tail','UNet-Tail','RSDR-Net'):
        if control==winner:continue
        with np.load(PRIVATE/'scoring'/winner/'17/metrics.npz',allow_pickle=False) as a,np.load(PRIVATE/'scoring'/control/'17/metrics.npz',allow_pickle=False) as b:
            paired[control]=summarize({k:a[k]-b[k] for k in a.files if k!='parent_id'},axes)
    summary={'stage':'DEV_SCORED','coordinates':'RAW_IQ','winner':winning,'methods':methods,
        'paired_winner_seed17_minus_controls':paired,'metric_definition':DEFINITION,'independent_confirmation_unlocked':False,
        'seed_uncertainty_note':'Parent cluster intervals are distinct from training-seed variability; report each completed seed.'}
    atomic_json(OUT/'SUMMARY.json',summary)
    columns=('pd','frame_pfa_jammer',*PRIMARY,'no_jammer_pooled_nmse')
    lines=['# ReCoVP 正式 DEV 对决','',DEFINITION,'',f'Frozen winner: **{winner}**. Scan evaluation runs regardless of competitiveness.','',
        '| Method | Seed | '+' | '.join(columns)+' |','|---|---|'+'---:|'*len(columns)]
    for name,seeds in rows.items():
        for seed,row in seeds.items():
            lines.append('| '+name+' | '+str(row['seed'])+' | '+' | '.join(f"{row['metrics']['global'][k]['mean']:.6g}" for k in columns)+' |')
    lines+=['','完整原生 SNR/JSR 分层、逐 parent 配对区间、复缩放与三层漂移见 SUMMARY.json。',
        'RSDR-Net 保留原论文训练口径，不宣称同训练预算。扫描不用于重选模型、epoch 或超参数。']
    atomic_write(OUT/'REPORT.md',lambda f:f.write(('\n'.join(lines)+'\n').encode('utf8')))
    atomic_json(OUT/'DEV_VERIFICATION.json',{**provenance(*SOURCES),'started_utc':started,'ended_utc':now(),
        'status':'PASS','dev':binding,'summary_sha256':sha(OUT/'SUMMARY.json'),'winner_sha256':sha(winpath),
        'checkpoints':{name:{seed:r['identity']['method'].get('checkpoint_sha256',r['identity']['method'].get('assets')) for seed,r in seeds.items()} for name,seeds in rows.items()},
        'independent_confirmation_unlocked':False})

if __name__=='__main__':
    with ProcessLock(PRIVATE/'SCORE.lock'):run()
