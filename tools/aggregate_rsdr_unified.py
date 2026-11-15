"""Audit and paired aggregation for the additive frozen RSDR benchmark."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import beta

from paper2.io import ROOT, now, read, sha
from paper2.scan_1103 import digest
from paper2.evaluation_1103_v4 import summarize
from tools.durable_io import atomic_json, atomic_write
from tools.compare_rsdr_unified import DEST,LIGHT,FREEZE,OLD_FREEZE,verify_freeze

METRICS = ('u_nmse','damage','sjrif_db','false_peak_suppression_db','pd','pd_no_jammer',
           'no_jammer_pooled_nmse','noise_nmse','h0_nmse','background_nmse','amplitude','phase_deg',
           'cell_pfa_noise','frame_pfa_noise','cell_pfa_jammer','frame_pfa_jammer')


def load_records():
    f=verify_freeze(); old=read(OLD_FREEZE)
    records={}; audit=[]
    for job in f['plan']:
        suite,index=job['suite'],job['index']
        rp=DEST/f'{suite}_{index:03d}/rsdr/result.json'
        if not rp.exists():continue
        r=read(rp)
        assert r['status']=='COMPLETE' and r['identity']==dict(freeze_sha256=sha(FREEZE),**job)
        original=ROOT/f'outputs/paper2_1103_test_20261114/{suite}_{index:03d}/model/result.json'
        o=read(original)
        assert o['status']=='COMPLETE'
        assert o['identity']['condition']==job['condition']
        assert o['identity']['binding']['freeze_sha256']==sha(OLD_FREEZE)
        data={}; metadata={}; block_counts={}
        keys=METRICS+(('phase_rmse_deg','coherent_power_db','peak_error_hz') if suite=='coherent' else ())
        for name,res,root in [('rsdr',r,DEST),('model',o,ROOT/'outputs/paper2_1103_test_20261114')]:
            parts=[]; metas=[]
            for b in res['blocks']:
                p=root/b['file']; receipt=read(p.with_suffix('.json'))
                assert sha(p)==b['sha256']==receipt['sha256'],p
                if name=='rsdr':
                    lo=len(metas); ids=list(range(lo,lo+len(receipt['metadata'])))
                    assert receipt['identity_sha256']==digest(dict(**r['identity'],parent_indices=ids))
                metas.extend(receipt['metadata'])
                with np.load(p,allow_pickle=False) as z:
                    parts.append({k:z[k].copy() for k in keys})
            data[name]={k:np.concatenate([p[k] for p in parts]) for k in keys}
            metadata[name]=metas;block_counts[name]=len(res['blocks'])
            expected=r['metric_means'] if name=='rsdr' else {k:v['mean'] for k,v in o['metrics'].items()}
            for k,v in data[name].items():
                assert len(v)==job['condition']['count'] and np.isfinite(v).all()
                assert np.isclose(v.mean(),expected[k],rtol=1e-12,atol=1e-14),(suite,index,name,k)
        assert metadata['rsdr']==metadata['model'],(suite,index,'metadata differs')
        ids=[m['parent_id'] for m in metadata['rsdr']]
        assert len(ids)==len(set(ids))==job['condition']['count']
        records[suite,index]=dict(job=job,data=data,ids=ids)
        audit.append(dict(suite=suite,index=index,parents=len(ids),blocks=block_counts,
            rsdr_result_sha256=sha(rp),paper2_result_sha256=sha(original)))
    return f,records,audit


def paired_group(records,suite,indexes):
    have=[i for i in indexes if (suite,i) in records]
    if not have:return None
    # Retention has separately drawn parents for each jammer-labelled group.
    strata={}
    for i in have:
        rec=records[suite,i]
        label=rec['job']['condition']['jam_type'] if suite=='retention' else 'shared_physical_parents'
        strata.setdefault(label,[]).append(i)
    summaries={};arrays={m:{} for m in ('model','rsdr')}
    for label,items in strata.items():
        ids=records[suite,items[0]]['ids']
        assert all(records[suite,i]['ids']==ids for i in items)
        for method in arrays:
            arrays[method][label]={k:np.mean([records[suite,i]['data'][method][k] for i in items],axis=0)
                                   for k in records[suite,items[0]]['data'][method]}
    if len(strata)==1:
        label=next(iter(strata))
        for method in arrays:summaries[method]={k:summarize(v) for k,v in arrays[method][label].items()}
        diff={k:summarize(arrays['model'][label][k]-arrays['rsdr'][label][k]) for k in arrays['model'][label]}
    else:
        # Fixed equal weight for each independent retention stratum; resample
        # physical parents within strata, never pulses or repeated SNR cells.
        labels=list(strata);rng=np.random.default_rng(20261114)
        draws={l:rng.integers(len(arrays['model'][l]['pd']),size=(4000,len(arrays['model'][l]['pd']))) for l in labels}
        def stat(vals):
            boot=np.mean([vals[l][draws[l]].mean(1) for l in labels],axis=0)
            x=np.concatenate(list(vals.values()))
            return dict(mean=float(np.mean([v.mean() for v in vals.values()])),
                        parent_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),parents=len(x))
        for method in arrays:
            summaries[method]={k:stat({l:arrays[method][l][k] for l in labels}) for k in arrays[method][labels[0]]}
        diff={k:stat({l:arrays['model'][l][k]-arrays['rsdr'][l][k] for l in labels}) for k in arrays['model'][labels[0]]}
    return dict(complete=len(have)==len(indexes),expected_conditions=len(indexes),conditions=len(have),
                independent_parents=sum(len(arrays['model'][l]['pd']) for l in strata),
                weighting='Equal condition weighting within parent; retention strata equally weighted with stratified bootstrap.',
                metrics=summaries,model_minus_rsdr=diff)


def aggregate(require_complete=False,return_records=False):
    f,recs,audit=load_records();old=read(OLD_FREEZE)
    complete=len(recs)==len(f['plan'])
    if require_complete:assert complete,(len(recs),len(f['plan']))
    groups={}
    for suite,grid in old['grids'].items():
        if suite=='main':
            for jam in ('repeat','direct','cyclic','all'):
                indexes=[i for i,c in enumerate(grid) if jam=='all' or c['jam_type']==jam]
                groups[f'main_{jam}']=paired_group(recs,suite,indexes)
        else:
            groups[suite]=paired_group(recs,suite,list(range(len(grid))))
    groups['pd_jsr20']=paired_group(recs,'pd',[i for i,c in enumerate(old['grids']['pd']) if c['jsr_db']==20])
    groups={k:v for k,v in groups.items() if v is not None}
    coverage={s:dict(complete=sum((s,i) in recs for i in range(len(g))),expected=len(g)) for s,g in old['grids'].items()}
    result=dict(checked_utc=now(),status='COMPLETE' if complete else 'PARTIAL_IN_PROGRESS',
                freeze_sha256=sha(FREEZE),complete_conditions=len(recs),expected_conditions=len(f['plan']),
                parent_evaluations=sum(a['parents'] for a in audit),coverage=coverage,groups=groups,
                interval_note='Pointwise paired 95% physical-parent bootstrap intervals, 4000 resamples. No multiplicity correction; no claim of training-seed generality.',
                limitations=f['limitations'])
    atomic_json(LIGHT/'PAIRED_RESULTS.json',result)
    atomic_json(LIGHT/'RESULT_INTEGRITY_AUDIT.json',dict(checked_utc=now(),status=result['status'],
        freeze_sha256=sha(FREEZE),records=audit,blocks_verified={m:sum(a['blocks'][m] for a in audit) for m in ('model','rsdr')},
        metadata_and_parent_identity_exact=True,all_checked_finite=True))
    rows=[]
    for (suite,index),rec in sorted(recs.items()):
        for k in rec['data']['model']:
            a,b=rec['data']['model'][k],rec['data']['rsdr'][k]
            row=dict(suite=suite,index=index,**rec['job']['condition'],metric=k,
                     paper2_mean=float(a.mean()),rsdr_mean=float(b.mean()),mean_difference=float((a-b).mean()))
            if k in ('pd','pd_no_jammer','frame_pfa_noise','frame_pfa_jammer'):
                for name,v in [('paper2',a),('rsdr',b)]:
                    n=len(v);events=int(v.sum());assert np.isin(v,[0,1]).all()
                    row[f'{name}_events']=events;row[f'{name}_parents']=n
                    row[f'{name}_cp_lower']=0. if events==0 else float(beta.ppf(.025,events,n-events+1))
                    row[f'{name}_cp_upper']=1. if events==n else float(beta.ppf(.975,events+1,n-events))
                ci=summarize(a-b)['parent_bootstrap_ci95'];row['paired_ci_lower'],row['paired_ci_upper']=ci
            rows.append(row)
    if rows:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        import io
        stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
        atomic_write(LIGHT/'CONDITION_COMPARISONS.csv',lambda dest:dest.write(stream.getvalue().encode('utf8')))
    print(dict(status=result['status'],complete_conditions=len(recs),coverage=coverage),flush=True)
    for name in ('main_repeat','main_all','pd_jsr20'):
        if name in groups:
            g=groups[name];print(name,g['complete'],{k:(g['metrics']['model'][k]['mean'],g['metrics']['rsdr'][k]['mean'],g['model_minus_rsdr'][k]['parent_bootstrap_ci95']) for k in ('u_nmse','damage','sjrif_db','pd','frame_pfa_jammer')},flush=True)
    return (result,recs) if return_records else result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args();aggregate(a.require_complete)
