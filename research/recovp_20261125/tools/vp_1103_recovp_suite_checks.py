"""CPU-only fixture/harness tests; never model selection or held-out evaluation."""
import ast
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from vp_1103_common import WORK,atomic_json,now,provenance
from vp_1103_recovp_isrj1_suite import plan,build,measure_case
from vp_1103_recovp_methods import Method
from vp_1103_recovp_radar_metrics import unit_tests
from vp_1103_recovp_formal_latency import RSDRGroup

def run():
    started=now();torch.set_num_threads(1);rows=plan();counts=dict(Counter(c['suite'] for c in rows))
    assert counts=={'typical':1,'coherent':3,'pd':310,'sliceB':55,'main':168,'retention':42}
    a={'suite':'main','jam_type':'repeat','snr_db':0,'jsr_db':20,'count':1,'strength_definition':'native_active_power'}
    ids=[10000000];s,j,e,meta=build(a,ids)
    b={**a,'snr_db':-20,'jsr_db':35};ss,jj,ee,mm=build(b,ids)
    assert np.array_equal(s,ss) and np.allclose(ee,10*e,atol=1e-12,rtol=1e-12)
    assert np.allclose(jj,j*10**(.75),atol=1e-12,rtol=1e-12)
    assert meta[0]['paired_invariant_sha256']==mm[0]['paired_invariant_sha256']
    active_s=(abs(s)>0).sum(-1);active_j=(abs(j)>0).sum(-1);valid=active_j>0
    native_jsr=10*np.log10((abs(j)**2).sum(-1)[valid]/active_j[valid]/((abs(s)**2).sum(-1)/active_s)[valid])
    assert np.max(abs(native_jsr-20))<1e-8
    narrow={**a,'suite':'sliceB','width':6,'repeats':2,'strength_definition':'group_energy_JSR_E'}
    ws,wj,we,wm=build(narrow,ids)
    energy_jsr=float(10*np.log10(np.sum(abs(wj)**2)/np.sum(abs(ws)**2)))
    assert abs(energy_jsr-20)<1e-10 and wm[0]['period_samples']==19
    adapters={}
    for name in ('Unprocessed','Max-TF','GMM-OMP','UNet-Tail','P0-NoDechirp-Tail','RSDR-Net'):
        m=Method(name,device='cpu');values,_=measure_case(a,ids,m)
        assert all(np.isfinite(v).all() for v in values.values())
        row={'binding':m.binding,'finite':True,'scope':'One fixed independent engineering fixture; no winner or hyperparameter selection'}
        if name=='RSDR-Net':
            from paper2.evaluation_1103_v4 import states_from_components
            y=states_from_components(s,j,e);x=y[0,3]
            iq=torch.from_numpy(np.stack((x.real,x.imag),1).astype(np.float32))
            with torch.inference_mode():actual=RSDRGroup(m.adapter)(iq).numpy()
            expected=m.restore(y)[0,3]
            error=float(np.max(abs(actual[:,0]+1j*actual[:,1]-expected)))
            assert error<1e-5;row['tensor_group_parity_max_error']=error
        adapters[name]=row;m.close();print(name,'PASS',flush=True)
    for pattern in ('*recovp*fp32*.py','*recovp*gpu*.py','*recovp*formal*.py','*recovp*isrj1*.py','*recovp*radar*.py'):
        for path in (WORK/'tools').glob(pattern):ast.parse(path.read_text(encoding='utf8'))
    result={**provenance(__file__),'status':'PASS','started_utc':started,'ended_utc':now(),'metric_units':unit_tests(),
        'grid_counts':counts,'common_random_number_checks':'PASS','native_jsr_max_error':float(np.max(abs(native_jsr-20))),
        'width_energy_jsr_db':energy_jsr,'adapters':adapters,'test_fixture_parent':ids,'old_dataset_arrays_read':False}
    atomic_json(WORK/'evidence/outputs/recovp_isrj1_suite/HARNESS_CHECKS.json',result)

if __name__=='__main__':run()
