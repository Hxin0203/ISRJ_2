"""CPU-only, server-B summary of frozen per-parent evaluation evidence.

Bootstrap unit is physical parent, after averaging seed predictions/metrics.
Intervals condition on fitted checkpoints and calibration thresholds. Raw arrays
stay on server. No model selection or fitting is performed.
"""
from pathlib import Path
import hashlib
import json
import platform
import time
import numpy as np

ROOT=Path('D:/code/Xin/ISRJ_222')
SUITE=ROOT/'outputs/paper2_baseline_suite_20261126'
OUT=ROOT/'outputs/hybrid_five_seed_review_20261127'
SEEDS=(17,29,41,53,67)
REPS=4000
RNG=20261127


def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def stats(x):
    x=np.asarray(x,dtype=float)
    return dict(mean=float(x.mean()),sd=float(x.std(ddof=1)) if len(x)>1 else None,
                minimum=float(x.min()),maximum=float(x.max()),values=x.tolist())


def ci(x):
    x=np.asarray(x,dtype=float)
    rng=np.random.default_rng(RNG)
    means=[]
    for _ in range(REPS//100):
        means.extend(x[rng.integers(len(x),size=(100,len(x)))].mean(axis=1))
    return dict(mean=float(x.mean()),parent_bootstrap_ci95=np.quantile(means,[.025,.975]).tolist(),parents=len(x))


def main():
    if platform.node().lower()!='desktop-rfferm3':raise RuntimeError('server B only')
    provenance=[]
    def track(p):
        provenance.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    result={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'bootstrap_reps':REPS,'bootstrap_rng':RNG,
            'ci_scope':'physical-parent paired bootstrap conditional on fitted checkpoints and calibration; seeds averaged first; seed SD reported separately',
            'equal_pfa':{},'sweep':{},'dev':{},'wp8':{}}
    methods={}
    folders=list((SUITE/'E3').glob('*/seed_*'))+list((SUITE/'E6/equal_pfa').glob('*'))+list((SUITE/'E7/equal_pfa').glob('*'))
    for folder in folders:
        rowp,raw=folder/'ROW.json',folder/'SCORES.npz'
        if not rowp.exists() or not raw.exists():continue
        row=load(rowp);family=row['family']
        if family.startswith('ReCoVP') and row.get('seed') not in SEEDS:continue
        track(rowp);track(raw)
        with np.load(raw,allow_pickle=False) as z:
            ids=z['eval_parent_id'].copy()
            arrays={k:z[k].copy() for k in z.files if k.startswith('eval_')}
        methods.setdefault(family,[]).append((row,ids,arrays))
    ids_ref=methods['ReCoVP-Hybrid'][0][1]
    equal_parent={}
    for family,items in methods.items():
        if any(not np.array_equal(ids_ref,item[1]) for item in items):raise RuntimeError('E3 parent order mismatch')
        summary={'seeds':[x[0].get('seed') for x in items],'curves':{},'diagnostics':{}}
        equal_parent[family]={}
        for target in (.001,.002,.005,.01,.02,.05,.1):
            pd,pfa,noise=[],[],[]
            for row,ids,a in items:
                threshold=next(x['threshold_ratio'] for x in row['curve'] if x['target_frame_pfa']==target)
                pd.append(a['eval_target']>threshold);pfa.append(a['eval_h0']>threshold);noise.append(a['eval_h0_noise']>threshold)
            pd,pfa,noise=(np.asarray(x,dtype=float) for x in (pd,pfa,noise))
            summary['curves'][str(target)]={
                'pd':stats(pd.mean(axis=(1,2))),'achieved_pfa':stats(pfa.mean(axis=(1,2))),
                'noise_pfa':stats(noise.mean(axis=(1,2))),
                'pd_by_variant':dict(zip(['direct','repeat','cyclic'],pd.mean(axis=(0,1)).tolist())),
                'pd_parent_ci':ci(pd.mean(axis=(0,2))),
                'seed_pd_by_variant':pd.mean(axis=1).tolist()}
            equal_parent[family][str(target)]=pd.mean(axis=(0,2))
        for k in ['resid_peak_noise_units','resid_concentration','resid_energy_ratio','gain_error','orth_error','e0_modification']:
            arr=np.asarray([a['eval_'+k] for _,_,a in items])
            summary['diagnostics'][k]={'seed_means':stats(arr.mean(axis=(1,2))),
                  'pooled_quantiles':dict(zip(['p50','p90','p95','p99','p999'],np.quantile(arr,[.5,.9,.95,.99,.999]).tolist()))}
            if k=='resid_peak_noise_units':
                grid=np.geomspace(.01,1e4,90)
                summary['residual_ccdf']={'x':grid.tolist(),'survival':[(arr>x).mean().item() for x in grid]}
        result['equal_pfa'][family]=summary
    result['equal_pfa_paired_differences']={fam:{target:ci(equal_parent['ReCoVP-Hybrid'][target]-v) for target,v in rows.items()}
            for fam,rows in equal_parent.items() if fam!='ReCoVP-Hybrid'}
    sweep=load(SUITE/'E5/SWEEP_SUMMARY.json');track(SUITE/'E5/SWEEP_SUMMARY.json')
    perparent={}
    for family,meta in sweep['methods'].items():
        runs=sorted((SUITE/'E5/results'/family).glob('*'))
        if family=='ReCoVP-Hybrid':runs=[r for r in runs if r.name in {f'seed_{s}' for s in SEEDS}]
        runs=[r for r in runs if (r/'DONE.json').exists()]
        perparent[family]={}
        for cell,cellmeta in meta['cells'].items():
            vals=[]
            for run in runs:
                p=run/f'cell_{cell}.npz';track(p)
                with np.load(p,allow_pickle=False) as z:
                    vals.append({k:z[k].copy() for k in z.files})
            if not vals:continue
            ref=vals[0]['parent_id']
            if any(not np.array_equal(ref,v['parent_id']) for v in vals):raise RuntimeError('sweep seed parent mismatch')
            means={k:np.asarray([v[k] for v in vals]).mean(axis=0) for k in vals[0] if k!='parent_id'}
            perparent[family][cell]=(ref,means)
            result['sweep'].setdefault(family,{'runs':[r.name for r in runs],'cells':{}})['cells'][cell]={
                'snr_db':cellmeta['snr_db'],'jsr_db':cellmeta['jsr_db'],
                'metrics':{k:stats([v[k].mean() for v in vals]) for k in means}}
    result['sweep_paired_vs_rsdr']={}
    for cell,(ids,a) in perparent['ReCoVP-Hybrid'].items():
        bid,b=perparent['RSDR-Net'][cell]
        if not np.array_equal(ids,bid):raise RuntimeError('sweep cross-model parent mismatch')
        result['sweep_paired_vs_rsdr'][cell]={k:ci(a[k]-b[k]) for k in a}
    for family,cells in perparent.items():
        result['sweep'][family]['uniform_grid_mean']={k:float(np.mean([v[k] for _,v in cells.values()])) for k in next(iter(cells.values()))[1]}
    for folder in OUT.iterdir():
        if not folder.is_dir() or not (folder/'SUMMARY.json').exists():continue
        p=folder/'SUMMARY.json';track(p)
        result['dev'][folder.name]=load(p)
    # DEV arrays include variant information; the common parent order is verified.
    dev_arrays={}
    for label in result['dev']:
        p=OUT/label/'PARENT_VARIANT_VALUES.npz';track(p)
        with np.load(p,allow_pickle=False) as z:dev_arrays[label]={k:z[k].copy() for k in z.files}
    result['dev_aggregate']={}
    for output in ['main','stage1']:
        selected=[dev_arrays[str(s)] for s in SEEDS if str(s) in dev_arrays]
        if not selected:continue
        keys=[k for k in selected[0] if k.startswith(output+'__')]
        for k in keys:
            if any(not np.array_equal(v['parent_id'],selected[0]['parent_id']) for v in selected):raise RuntimeError('DEV seed order mismatch')
            a=np.asarray([v[k] for v in selected])
            result['dev_aggregate'][k]={'n_seeds':len(selected),'seed_means':stats(a.mean(axis=(1,2))),
                'parent_ci':ci(a.mean(axis=(0,2))),'variant_means':a.mean(axis=(0,1)).tolist(),
                'parent_median':float(np.median(a.mean(axis=(0,2)))),'parent_p95':float(np.quantile(a.mean(axis=(0,2)),.95))}
    result['dev_paired_vs_rsdr']={}
    if 'rsdr' in dev_arrays and all(str(s) in dev_arrays for s in SEEDS):
        r=dev_arrays['rsdr'];hy=[dev_arrays[str(s)] for s in SEEDS]
        if any(not np.array_equal(r['parent_id'],h['parent_id']) for h in hy):raise RuntimeError('DEV paired mismatch')
        for k in r:
            if k=='parent_id':continue
            result['dev_paired_vs_rsdr'][k]=ci((np.mean([h[k] for h in hy],axis=0)-r[k]).mean(axis=1))
    # Reuse all 12 convolutional controls; replace the historical seed-17
    # reference with the current five-seed mean without pairing seed numbers.
    result['dev_convolutional_baselines']={}
    e1=SUITE/'E1/PAIRED_DEV_STATS.json'
    e1raw=SUITE/'E1/SEEDAVG_PARENT_VALUES.npz'
    if e1.exists() and e1raw.exists() and all(str(s) in dev_arrays for s in SEEDS):
        track(e1);track(e1raw)
        original=load(e1)
        with np.load(e1raw,allow_pickle=False) as z:
            if not np.array_equal(z['parent_id'],dev_arrays['17']['parent_id']):
                raise RuntimeError('E1 five-seed parent mismatch')
            for family,meta in original['families'].items():
                paired={}
                for metric in meta['seed_mean']:
                    hv=np.mean([dev_arrays[str(s)]['main__'+metric].mean(axis=1) for s in SEEDS],axis=0)
                    paired[metric]=ci(hv-z[family+'__'+metric+'__seedavg'])
                result['dev_convolutional_baselines'][family]={
                    'n_seeds':3,'mean':meta['seed_mean'],'seed_sd':meta['seed_sd'],
                    'hybrid_minus_baseline':paired,
                    'scope':'same DEV256; baseline seed averages from frozen E1; reference is five Hybrid seeds, not historical seed17'}
    # A reproducibility check against the already frozen RSDR DEV receipt.
    prior=SUITE/'E7/dev256/rsdr_net/SUMMARY.json'
    if prior.exists() and 'rsdr' in result['dev']:
        track(prior)
        before=load(prior)['means'];after=result['dev']['rsdr']['means']
        result['rsdr_dev_reproduction']={k:after['main__'+k.removeprefix('rsdr_net__')]-v
            for k,v in before.items() if 'main__'+k.removeprefix('rsdr_net__') in after}
    for seed in SEEDS:
        p=SUITE/f'WP8/seed_{seed}/SUMMARY.json'
        if p.exists():track(p);result['wp8'][str(seed)]=load(p)
    result['provenance']=provenance
    OUT.mkdir(exist_ok=True)
    (OUT/'ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=1),encoding='utf-8')
    print('wrote',OUT/'ANALYSIS.json','dev',list(result['dev']),'wp8',list(result['wp8']))


if __name__=='__main__':main()
