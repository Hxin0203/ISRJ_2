"""Verify a live, resumable release without claiming completed experiment results."""
import ast
from pathlib import Path
import psutil
import torch
from vp_1103_common import WORK,read,sha,atomic_json,now,validate_role
from vp_1103_recovp_isrj1_suite import prepare

def run():
    out=WORK/'evidence/outputs/recovp_formal_dev';suite=WORK/'evidence/outputs/recovp_isrj1_suite'
    q=read(out/'FP32_QUICKCHECK.json');assert q['status']=='PASS'
    for name,h in q['source_sha256'].items():assert sha(WORK/name)==h,name
    cfg=read(WORK/'outputs/recovp_formal_dev/CONFIG.json')
    for name,h in cfg['source_sha256'].items():assert sha(WORK/name)==h,name
    assert cfg['epochs']==256 and cfg['micro_batch']==8 and cfg['extractor']['k']==12
    bindings={role:validate_role(role)[1] for role in ('train','dev')}
    frozen=prepare();assert len(frozen['conditions'])==579
    h=read(suite/'HARNESS_CHECKS.json');assert h['status']=='PASS'
    assert h['adapters']['P0-NoDechirp-Tail']['binding']['effective_real_parameters']==59714
    assert h['adapters']['UNet-Tail']['binding']['effective_real_parameters']==59714
    for pattern in ('*recovp*fp32*.py','*recovp*gpu*.py','*recovp*formal*.py','*recovp*isrj1*.py','*recovp*radar*.py'):
        for path in (WORK/'tools').glob(pattern):ast.parse(path.read_text(encoding='utf8'))
    state=read(out/'STATUS.json');pid=state['pid'];process=psutil.Process(pid)
    assert 'vp_1103_recovp_gpu_run.py' in ' '.join(process.cmdline())
    assert Path(process.cwd()).resolve()==WORK.resolve()
    epoch=int(state.get('last_committed_epoch',0));arm=state['arm'];seed=state['seed']
    cp=WORK/'outputs/recovp_formal_dev'/arm/str(seed)/f'recovery/epoch_{epoch:03d}.pt'
    receipt=read(cp.with_suffix('.json'));assert sha(cp)==receipt['checkpoint_sha256']
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    assert saved['epoch']==epoch and saved['identity']['config_sha256']==sha(WORK/'outputs/recovp_formal_dev/CONFIG.json')
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    old=read(WORK/'evidence/outputs/recovp_gate_summary/FINAL_SEAL.json')
    assert all(sha(WORK/name)==digest for name,digest in old['files'].items())
    value={'status':'PASS_RELEASE_INTEGRITY','verified_utc':now(),'owner_pid':pid,'owner_workspace_verified':True,
        'training_state':state['stage'],'arm':arm,'seed':seed,'last_verified_committed_epoch':epoch,
        'checkpoint_sha256':sha(cp),'config_sha256':sha(WORK/'outputs/recovp_formal_dev/CONFIG.json'),
        'fp32_math_status':'PASS','harness_status':'PASS','source_and_data_bindings':'PASS','old_negative_seal_unchanged':True,
        'scan_plan_sha256':sha(suite/'PLAN.json'),'planned_conditions':579,'planned_method_conditions':2811,
        'data':bindings,'formal_dev_results_completed':False,'scan_results_completed':False,
        'independent_confirmation_unlocked':False,'note':'Live-training release verification; not a performance PASS or final Gate result.'}
    atomic_json(out/'RELEASE_VERIFICATION.json',value)
    print({k:value[k] for k in ('status','owner_pid','training_state','arm','seed','last_verified_committed_epoch','scan_results_completed')})

if __name__=='__main__':run()
