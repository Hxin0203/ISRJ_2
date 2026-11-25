"""Fresh-namespace, resumable first-paper-aligned frozen-model scans."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from vp_1103_common import WORK,atomic_json,now,read,sha,provenance,distribution
from vp_1103_recovp_scan_scenes import scene
from vp_1103_recovp_methods import Method
from vp_1103_recovp_radar_metrics import metrics,DEFINITION,unit_tests
from paper2.evaluation_1103_v4 import states_from_components
from tools.durable_io import atomic_write,ProcessLock
from tools.unet_ablation.runtime import configure_precision,user_control

OUT=WORK/'evidence/outputs/recovp_isrj1_suite'
PRIVATE=WORK/'outputs/recovp_isrj1_suite'
SNRS=[-30,-28,-26,-25,-24,-23,-22,-20,-15,-10,-5,0,5,10]
JSRS=[0,10,20,35]
METHODS=['Unprocessed','Max-TF','GMM-OMP','RSDR-Net','UNet-Tail','P0-NoDechirp-Tail','ReCoVP-Winner']
SOURCE_NAMES=('vp_1103_recovp_isrj1_suite.py','vp_1103_recovp_scan_scenes.py','vp_1103_recovp_methods.py',
 'vp_1103_recovp_radar_metrics.py','vp_1103_recovp_fp32.py','vp_1103_recovp_fp32_ablation.py',
 'vp_1103_recovp_fp32_anchors.py','vp_1103_recovp_isrj1_plots.py','vp_1103_recovp_formal_latency.py')

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def plan():
    rows=[]
    def add(suite,family,snr,jsr,count,methods,width=None):
        row={'suite':suite,'jam_type':family,'snr_db':snr,'jsr_db':jsr,'count':count,'methods':methods}
        if width is not None:row.update(width=width,repeats=2,strength_definition='group_energy_JSR_E')
        else:row['strength_definition']='native_active_power'
        rows.append(row)
    neural=['Unprocessed','UNet-Tail','P0-NoDechirp-Tail','RSDR-Net','ReCoVP-Winner']
    add('typical','repeat',0,20,1,neural)
    for s,r in ((-20,20),(0,20),(0,35)):add('coherent','repeat',s,r,16,neural)
    # The first-paper producer has 31*10=310 cells. The plotted full-range
    # 5dB and transition 1dB views are subsets, not 310 distinct cells by themselves.
    for s in range(-30,1):
        for r in range(-10,36,5):add('pd','repeat',s,r,512,['Unprocessed','Reference','P0-NoDechirp-Tail','ReCoVP-Winner'])
    for w in (6,18,36,54,72):
        for s in range(-20,1,2):add('sliceB','repeat',s,20,128,['Unprocessed','P0-NoDechirp-Tail','ReCoVP-Winner'],w)
    for family in ('repeat','direct','cyclic'):
        for s in SNRS:
            for r in JSRS:add('main',family,s,r,256,METHODS)
    for family in ('direct','repeat','cyclic'):
        for s in SNRS:add('retention',family,s,0,512,neural)
    return rows

def prepare():
    value={'protocol_version':1,'created_for':'Explicit user GPU release 2026-11-24','scan_seed':2026112407,
        'new_scene_role':'scan','role_code':8,'old_dataset_arrays_read':False,'conditions':plan(),
        'grid_note':'310=31 SNR values step1 x 10 JSR values; both requested plotted subsets exported',
        'typical_parent':0,'coherent_fft_points':4096,'parent_block':4,'cpu_workers':4,
        'bootstrap_reps':4000,'bootstrap_seed':20261122,
        'grouping':'Physical parent reused across methods and cells; bootstrap cells averaged within parent first. Retention groups independent.',
        'metric_definition':DEFINITION,'source_sha256':provenance(*[WORK/'tools'/f for f in SOURCE_NAMES])['source_sha256'],
        'confirmation_unlocked':False,'scan_use':'Frozen-model characterization only, never model/epoch/threshold selection.'}
    path=OUT/'PLAN.json'
    if path.exists():assert read(path)==value,'Frozen scan plan/source changed'
    else:atomic_json(path,value)
    return value

def build(case,ids):
    # Typical figure uses the independent coherent-scene namespace but a
    # separate fixed parent index. No outcome-based example selection.
    suite='coherent' if case['suite']=='typical' else case['suite']
    indexes=[1000000+p for p in ids] if case['suite']=='typical' else ids
    kwargs={k:case[k] for k in ('jam_type','snr_db','jsr_db','width','repeats') if k in case}
    scenes=[scene('scan',suite,p,**kwargs) for p in indexes]
    s,j,e=(np.stack([v[k] for v in scenes]) for k in range(3))
    if case['suite']=='retention':j=np.zeros_like(j)
    meta=[v[3] for v in scenes]
    return s,j,e,meta

def measure_case(case,ids,method):
    s,j,e,meta=build(case,ids);y=states_from_components(s,j,e)
    if method.name=='Reference':u=states_from_components(s,np.zeros_like(j),e)
    else:u=method.restore(y)
    values=metrics(s,j,e,y,u,np.array([v['doppler_hz'] for v in meta]),coherent=case['suite'] in ('typical','coherent'))
    if case['suite']=='typical':
        path=PRIVATE/'typical_waveforms'/f'{method.name}.npz'
        atomic_write(path,lambda f:np.savez_compressed(f,s=s,j=j,e=e,y=y,restored=u))
    return values,meta

_worker_method=None
def init_worker(name):
    global _worker_method
    torch.set_num_threads(1);_worker_method=Method(name,device='cpu')
def cpu_parent(job):return measure_case(job[0],[job[1]],_worker_method)

def evaluate_case(case,index,method,update,pool=None):
    folder=PRIVATE/f"{case['suite']}_{index:03d}"/method.name
    public=OUT/'cells'/f"{index:03d}_{case['suite']}"/f'{method.name}.json'
    identity={'case':case,'index':index,'method':method.binding,'plan_sha256':sha(OUT/'PLAN.json')}
    if public.exists():
        result=read(public);assert result['identity']==identity
        for block in result['blocks']:assert sha(WORK/block['file'])==block['sha256']
        return result
    blocks=[];collected={};allmeta=[]
    for lo in range(0,case['count'],4):
        user_control(PRIVATE);ids=list(range(lo,min(lo+4,case['count'])))
        raw=folder/f'parents_{lo:04d}.npz';receipt=raw.with_suffix('.json')
        token=digest({'identity':identity,'parents':ids})
        if receipt.exists():
            saved=read(receipt);assert saved['identity_sha256']==token and sha(raw)==saved['sha256']
            with np.load(raw,allow_pickle=False) as z:values={k:z[k] for k in z.files}
            meta=saved['metadata']
        else:
            if raw.exists():raw.rename(raw.with_suffix(f'.unreceipted_{time.time_ns()}.npz'))
            if pool is not None:
                parts=list(pool.map(cpu_parent,[(case,p) for p in ids]))
                values={k:np.concatenate([v[0][k] for v in parts]) for k in parts[0][0]}
                meta=[m for v in parts for m in v[1]]
            else:values,meta=measure_case(case,ids,method)
            for k,v in values.items():
                if k=='jcr_db' and case['suite']=='retention':assert np.isnan(v).all()
                else:assert np.isfinite(v).all(),(method.name,case,k)
            atomic_write(raw,lambda f:np.savez_compressed(f,**values))
            saved={'identity_sha256':token,'sha256':sha(raw),'metadata':meta,'parent_indexes':ids}
            atomic_json(receipt,saved)
        for k,v in values.items():collected.setdefault(k,[]).append(v)
        allmeta+=meta
        blocks.append({'file':raw.relative_to(WORK).as_posix(),'sha256':saved['sha256']})
        update(stage='SCANNING',condition=index,suite=case['suite'],method=method.name,parents_done=lo+len(ids),parents=case['count'])
    values={k:np.concatenate(v) for k,v in collected.items()}
    result={'identity':identity,'metrics':{k:distribution(v) for k,v in values.items()},'blocks':blocks,
        'paired_scene_invariants':[m['paired_invariant_sha256'] for m in allmeta],
        'parent_ids':[m['parent_id'] for m in allmeta],'snr_db':case['snr_db'],'jsr_db':case['jsr_db'],
        'strength_definition':case['strength_definition'],'status':'COMPLETE'}
    from scipy.stats import beta
    for k in ('pd','pd_no_jammer','frame_pfa_noise','frame_pfa_jammer'):
        count=int(values[k].sum());n=len(values[k])
        result['metrics'][k].update(events=count,trials=n,
            binomial_cp_ci95=[0. if count==0 else float(beta.ppf(.025,count,n-count+1)),1. if count==n else float(beta.ppf(.975,count+1,n-count))])
    for k in ('cell_pfa_noise','cell_pfa_jammer'):
        count=int(round(values[k].sum()*9032));n=len(values[k])
        result['metrics'][k].update(false_cells=count,valid_cuts=n*9032,
            note='CUTs are correlated; uncertainty clusters physical parents. Reused grid cells are not additional independent trials.')
        if count==0:result['metrics'][k]['zero_event_frame_based_upper95']=float(beta.ppf(.975,1,n))
    atomic_json(public,result);return result

def run():
    configure_precision();torch.set_num_threads(2);frozen=prepare()
    winner=read(WORK/'evidence/outputs/recovp_formal_dev/WINNER.json')
    started=now();state={'stage':'STARTING','started_utc':started,'pid':os.getpid(),'winner':winner,
        'plan_sha256':sha(OUT/'PLAN.json'),'confirmation_unlocked':False,'condition_count':len(frozen['conditions'])}
    def update(**kw):state.update(kw,updated_utc=now());atomic_json(OUT/'STATUS.json',state)
    results={};models={};pools={}
    try:
        for index,case in enumerate(frozen['conditions']):
            rows={}
            for name in case['methods']:
                if name not in models:
                    if name=='Reference':
                        m=Method('Unprocessed');m.name='Reference';m.binding={'name':'Reference','oracle':'S+E reference only; not a blind method'}
                    else:m=Method(name,device='cpu' if name in ('Max-TF','GMM-OMP') else 'cuda')
                    models[name]=m
                if name in ('Max-TF','GMM-OMP') and name not in pools:
                    pools[name]=ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(name,))
                rows[name]=evaluate_case(case,index,models[name],update,pools.get(name))
            controls=rows.get('Unprocessed')
            if controls:
                assert all(row['paired_scene_invariants']==controls['paired_scene_invariants'] for row in rows.values())
            results[str(index)]=rows
            atomic_json(OUT/'PROGRESS.json',{'completed_conditions':len(results),'total_conditions':len(frozen['conditions']),
                'last_condition':case,'winner':winner,'updated_utc':now(),'confirmation_unlocked':False})
        atomic_json(OUT/'SUMMARY.json',{'status':'SCANS_COMPLETE','plan_sha256':sha(OUT/'PLAN.json'),'winner':winner,
            'conditions':frozen['conditions'],'results':results,'metric_definition':DEFINITION,'confirmation_unlocked':False})
    except Exception as error:
        update(stage='FAILED_EXECUTION',error=repr(error),traceback=traceback.format_exc(),ended_utc=now());raise
    finally:
        for pool in pools.values():pool.shutdown()
        for model in models.values():model.close()
    from vp_1103_recovp_isrj1_plots import render
    from vp_1103_recovp_formal_latency import run as latency
    render();latency()
    update(stage='COMPLETE_GATE_REVIEW',ended_utc=now())
    atomic_json(OUT/'VERIFICATION.json',{'status':'PASS','completed_conditions':len(results),'planned_conditions':len(frozen['conditions']),
        'completed_method_cells':sum(map(len,results.values())),'plan_sha256':sha(OUT/'PLAN.json'),
        'summary_sha256':sha(OUT/'SUMMARY.json'),'figures':read(OUT/'FIGURE_MANIFEST.json'),
        'latency_summary_sha256':sha(OUT/'LATENCY.json'),'source_sha256':frozen['source_sha256'],
        'confirmation_unlocked':False,'review_required':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:print({'conditions':len(prepare()['conditions'])})
    else:
        PRIVATE.mkdir(parents=True,exist_ok=True)
        with ProcessLock(PRIVATE/'PROCESS.lock'):run()
